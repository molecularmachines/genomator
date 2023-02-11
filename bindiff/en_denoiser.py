import torch
from torch import optim
from argparse import ArgumentParser
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat
from en_transformer.en_transformer import EnTransformer


class EnDenoiser(pl.LightningModule):

    def __init__(self,
                 dim=32,
                 dim_head=64,
                 heads=4,
                 depth=4,
                 rel_pos_emb=True,
                 neighbors=16,
                 denoise_step=1e-3,
                 lr=1e-3):
        super().__init__()

        torch.set_default_dtype(torch.float64)
        self.transformer = EnTransformer(
            num_tokens=21,
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            depth=depth,
            rel_pos_emb=rel_pos_emb,
            neighbors=neighbors
        )

        self.lr = lr
        self.lamb = denoise_step

    def denoise(self, seq, coords, masks):
        noised_coords = coords + self.lamb * torch.randn_like(coords)
        feats, denoised_coords = self.transformer(seq, noised_coords, mask=masks)
        loss = F.mse_loss(denoised_coords[masks], coords[masks])
        return feats, denoised_coords, loss

    def step(self, x):
        # extract input
        seqs, coords, masks = x.seqs, x.crds, x.msks

        # type matching for transformer
        seqs = seqs.argmax(dim=-1)
        coords = coords.type(torch.float64)
        masks = masks.bool()

        # arrange in residues atom format (14 atoms max)
        coords = rearrange(coords, 'b (l s) c -> b l s c', s=14)

        # keeping only the backbone coordinates
        coords = coords[:, :, 0:3, :]
        coords = rearrange(coords, 'b l s c -> b (l s) c')

        seq = repeat(seqs, 'b n -> b (n c)', c=3)
        masks = repeat(masks, 'b n -> b (n c)', c=3)

        return self.denoise(seq, coords, masks)

    def training_step(self, batch, batch_idx):
        # run model on inputs
        feats, denoised, loss = self.step(batch)

        # compute loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # run model on inputs
        feats, denoised, loss = self.step(batch)

        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # run model on inputs
        feats, denoised, loss = self.step(batch)

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--step', type=float, default=0.0001)
        parser.add_argument('--freeze', type=bool, default=True)
        return parser
