import torch
from torch import nn, optim
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
                 neighbors=16):
        super().__init__()

        self.transformer = EnTransformer(
            num_tokens=21,
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            depth=depth,
            rel_pos_emb=rel_pos_emb,
            neighbors=neighbors
        )

    def step(self, x):
        seqs, coords, masks = x.seqs, x.crds, x.msks

        seqs = seqs.argmax(dim=-1)
        coords = coords.type(torch.float64)
        masks = masks.bool()

        l = seqs.shape[1]
        coords = rearrange(coords, 'b (l s) c -> b l s c', s = 14)

        # keeping only the backbone coordinates
        coords = coords[:, :, 0:3, :]
        coords = rearrange(coords, 'b l s c -> b (l s) c')

        seq = repeat(seqs, 'b n -> b (n c)', c=3)
        masks = repeat(masks, 'b n -> b (n c)', c=3)

        noised_coords = coords + torch.randn_like(coords)

        feats, denoised_coords = self.transformer(seq, noised_coords, mask=masks)

        return feats, denoised_coords

    def training_step(self, batch, batch_idx):
        # input variables
        masks, coords = batch.msks, batch.crds

        # run model on inputs
        feats, denoised = self.step(batch)

        # compute loss
        loss = F.mse_loss(denoised[masks], coords[masks])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # input variables
        masks, coords = batch.msks, batch.crds

        # run model on inputs
        feats, denoised = self.step(batch)

        # compute loss
        loss = F.mse_loss(denoised[masks], coords[masks])
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # input variables
        masks, coords = batch.msks, batch.crds

        # run model on inputs
        feats, denoised = self.step(batch)

        # compute loss
        loss = F.mse_loss(denoised[masks], coords[masks])
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--freeze', type=bool, default=True)
        return parser
