import torch
from torch import optim
from argparse import ArgumentParser
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat
from modules import EnTransformer
from tqdm import tqdm


class EnDenoiser(pl.LightningModule):

    def __init__(self,
                 dim=32,
                 dim_head=64,
                 heads=4,
                 depth=4,
                 rel_pos_emb=True,
                 neighbors=16,
                 beta_small=2e-4,
                 beta_large=0.02,
                 timesteps=100,
                 lr=1e-3):
        super().__init__()

        torch.set_default_dtype(torch.float64)
        self.save_hyperparameters()
        self.transformer = EnTransformer(
            num_tokens=23,
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            depth=depth,
            rel_pos_emb=rel_pos_emb,
            neighbors=neighbors
        )

        self.lr = lr
        self.b1 = beta_small
        self.b2 = beta_large
        self.timesteps = timesteps

        # precompute all betas
        self.betas = self.linear_beta_schedule()

        # precompute all alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def linear_beta_schedule(self):
        return torch.linspace(self.b1, self.b2, self.timesteps)

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start).to(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape).to(x_start)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ).to(x_start)

        noised_x = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return noised_x, noise

    @torch.no_grad()
    def p_sample(self, coords, seqs, masks, t, t_index):
        betas_t = self.extract(self.betas, t, coords.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, coords.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, coords.shape)
        _, prediction = self.transformer(seqs, coords, t, mask=masks)
        model_mean = sqrt_recip_alphas_t * (
            coords - betas_t * prediction / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean

        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, coords.shape)
            noise = torch.randn_like(coords)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, x, timesteps):
        coords, seqs, masks = self.prepare_inputs(x)
        b = coords.size(0)

        # start with random gaussian noise
        res = torch.randn_like(coords)
        results = []

        # iterate over timesteps with p_sample
        desc = 'sampling loop time step'
        for i in tqdm(range(timesteps - 1, 0, -1), desc=desc, total=timesteps):
            ts = torch.full((b,), i)  # all samples same t
            res = self.p_sample(res, seqs, masks, ts, i)
            results.append(res)

        return results

    def prepare_inputs(self, x):
        # extract input
        seqs, coords, masks = x.residue_token, x.atom_coord, x.atom_mask

        # type matching for transformer
        coords = coords.type(torch.float64)

        # keeping only the backbone coordinates
        coords = coords[:, :, 0:3, :]
        masks = masks[:, :, 0:3]
        coords = rearrange(coords, 'b l s c -> b (l s) c')
        masks = rearrange(masks, 'b l s -> b (l s)')

        seq = repeat(seqs, 'b n -> b (n c)', c=3)

        return coords, seq, masks

    def step(self, x):
        coords, seq, masks = self.prepare_inputs(x)

        # noise with random amount
        ts = torch.randint(0, self.timesteps, [coords.shape[0]]).to(coords).type(torch.int64)

        # forward diffusion
        noised_coords, noise = self.q_sample(coords, ts)
        ts = ts.type(torch.float64)

        # predict noisy input with transformer
        feats, prediction = self.transformer(seq, noised_coords, ts, mask=masks)

        # loss between original and prediction
        loss = F.mse_loss(prediction[masks], coords[masks])

        return feats, prediction, loss

    def training_step(self, batch, batch_idx):
        # run model on inputs
        batch_size = batch.atom_coord.shape[0]
        feats, denoised, loss = self.step(batch)

        # compute loss
        self.log("train_loss", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        # run model on inputs
        batch_size = batch.atom_coord.shape[0]
        feats, denoised, loss = self.step(batch)

        self.log("val_loss", loss, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        # run model on inputs
        feats, denoised, loss = self.step(batch)
        batch_size = batch.atom_coord.shape[0]

        self.log("test_loss", loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--beta_small', type=float, default=0.02)
        parser.add_argument('--beta_large', type=float, default=0.2)
        parser.add_argument('--dim', type=int, default=32)
        parser.add_argument('--dim_head', type=int, default=64)
        parser.add_argument('--depth', type=int, default=4)
        parser.add_argument('--timesteps', type=int, default=100)
        return parser
