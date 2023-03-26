import torch
from torch import optim
from argparse import ArgumentParser
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat
from models.equitransformer import EnTransformer
from sampling.diffusion import Diffusion


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
                 lr=1e-4):
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

        self.diffusion = Diffusion(
            beta_small=beta_small,
            beta_large=beta_large,
            timesteps=timesteps,
            schedule=schedule
        )

        self.lr = lr

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
        ts = torch.randint(0, self.diffusion.timesteps, [s]).to(coords)

        # forward diffusion
        noised_coords, noise = self.diffusion.q_sample(coords, ts)
        noised_coords = noised_coords * mask.type(torch.float64)[..., None]
        ts = ts.type(torch.float64)

        # predict noisy input with transformer
        feats, prediction = self.transformer(seq, noised_coords, ts, mask=mask)

        # loss between original noise and prediction
        loss = F.mse_loss(prediction[mask], noise[mask])

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
        return optim.AdamW(self.parameters(), lr=self.lr)

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
        parser.add_argument('--schedule', type=str, default='linear')
        return parser
