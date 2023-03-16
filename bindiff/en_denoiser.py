import torch
from torch import optim
from argparse import ArgumentParser
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat
from en_transformer import EnTransformer
from tqdm import tqdm
import beta_schedule


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
                 schedule='linear',
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
        if schedule == 'linear':
            self.betas = beta_schedule.linear_beta_schedule(
                timesteps, beta_small, beta_large
            )

        elif schedule == 'cosine':
            self.betas = beta_schedule.cosine_beta_schedule(timesteps)

        elif schedule == 'quadratic':
            self.betas = beta_schedule.quadratic_beta_schedule(
                timesteps, beta_small, beta_large
            )

        else:
            allowed = beta_schedule.SCHEDULES
            err = f"Schedule must be one of: {allowed}. receieved: {schedule}"
            raise AttributeError(err)

        # precompute all alphas
        a, ac, sqac, sq1ac, pv, sra = beta_schedule.compute_alphas(self.betas)
        self.alphas = a
        self.alphas_cumprod = ac
        self.sqrt_alphas_cumprod = sqac
        self.sqrt_one_minus_alphas_cumprod = sq1ac
        self.posterior_variance = pv
        self.sqrt_recip_alphas = sra

    def extract(self, a, t, x_shape):
        t = t.type(torch.int64)
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out.to(t.device)

    def q_sample(self, x_start, t, noise=None):
        # generate random noise
        if noise is None:
            noise = torch.randn_like(x_start).to(x_start)

        # calculate alpha values for rescaling
        s = x_start.shape

        sqrt_alphas_cumprod_t = self.extract(
            self.sqrt_alphas_cumprod, t, s
        ).to(x_start)

        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, s
        ).to(x_start)

        # rescale noise and input
        scaled_noise = sqrt_one_minus_alphas_cumprod_t * noise
        scaled_input = sqrt_alphas_cumprod_t * x_start
        noised_x = scaled_input + scaled_noise

        return noised_x, scaled_noise

    @torch.no_grad()
    def p_sample(self, coords, seqs, masks, t, t_index):
        s = coords.shape

        # extract alhpas
        betas_t = self.extract(self.betas, t, s)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, s
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, s)

        # inference from the model
        _, prediction = self.transformer(seqs, coords, t, mask=masks)
        pred_noise = prediction - coords

        # calculate mean based on the model prediction
        model_mean = sqrt_recip_alphas_t * (
            coords - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean

        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, s)
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
        for i in tqdm(range(timesteps, -1, -1), desc=desc, total=timesteps):
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
        coords = coords[:, :, 0:4, :]
        masks = masks[:, :, 0:4]
        coords = rearrange(coords, 'b l s c -> b (l s) c')
        masks = rearrange(masks, 'b l s -> b (l s)')

        # assign sequence token to each of the backbone atoms
        seq = repeat(seqs, 'b n -> b (n c)', c=4)

        return coords, seq, masks

    def step(self, x):
        coords, seq, mask = self.prepare_inputs(x)

        # noise with random amount
        s = coords.shape[0]
        ts = torch.randint(0, self.timesteps, [s]).to(coords)

        # forward diffusion
        noised_coords, noise = self.q_sample(coords, ts)
        noised_coords = noised_coords * mask.type(torch.float64)[..., None]
        ts = ts.type(torch.float64)

        # predict noisy input with transformer
        feats, prediction = self.transformer(seq, noised_coords, ts, mask=mask)
        predicted_noise = prediction - noised_coords

        # loss between original noise and prediction
        loss = F.mse_loss(predicted_noise[mask], noise[mask])

        return feats, prediction, loss

    def training_step(self, batch, batch_idx):
        batch_size = batch.atom_coord.shape[0]
        feats, denoised, loss = self.step(batch)
        self.log("train_loss", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch.atom_coord.shape[0]
        feats, denoised, loss = self.step(batch)
        self.log("val_loss", loss, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        feats, denoised, loss = self.step(batch)
        batch_size = batch.atom_coord.shape[0]
        self.log("test_loss", loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--beta_small', type=float, default=2e-4)
        parser.add_argument('--beta_large', type=float, default=0.02)
        parser.add_argument('--dim', type=int, default=64)
        parser.add_argument('--dim_head', type=int, default=64)
        parser.add_argument('--depth', type=int, default=8)
        parser.add_argument('--timesteps', type=int, default=100)
        return parser
