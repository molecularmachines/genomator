import time
import random
import os
import argparse
import torch
from torch import optim
from argparse import ArgumentParser
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat
from models.equitransformer import EnTransformer
from sampling.diffusion import Diffusion
from visualize import pred_to_pdb
from utils import calc_tm_score, calc_distmap_loss


class EnDenoiser(pl.LightningModule):

    def __init__(self,
                 dim=32,
                 dim_head=64,
                 heads=4,
                 depth=4,
                 rel_pos_emb=True,
                 neighbors=0,
                 beta_small=2e-4,
                 beta_large=0.02,
                 timesteps=100,
                 bb_start=1,
                 bb_end=2,
                 trim=128,
                 ckpt_path='',
                 schedule='linear',
                 verbose=False,
                 context=False,
                 lr=1e-4):
        super().__init__()

        torch.set_default_dtype(torch.float64)
        self.save_hyperparameters()

        if trim:
            if neighbors >= trim and neighbors > 0 and trim > 0:
                neighbors = trim - 1

        self.transformer = EnTransformer(
            num_tokens=6,
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

        self.bb_start = bb_start
        self.bb_end = bb_end
        self.trim = trim
        self.lr = lr
        self.ckpt_path = ckpt_path
        self.verbose = verbose
        self.context = context
        self.start_epoch_time = time.time()

    def prepare_inputs(self, x):
        # extract input
        seqs, coords, masks = x.atom_token, x.atom_coord, x.atom_mask

        # type matching for transformer
        coords = coords.type(torch.float64)
        seqs = seqs.type(torch.int64)

        if not self.context:
            # keeping only the backbone coordinates
            coords = coords[:, :self.trim, self.bb_start:self.bb_end, :]
            masks = masks[:, :self.trim, self.bb_start:self.bb_end]
            coords = rearrange(coords, 'b l s c -> b (l s) c')
            masks = rearrange(masks, 'b l s -> b (l s)')

            # assign sequence token to each of the backbone atoms
            seqs = seqs[:, :self.trim]
            reps = self.bb_end - self.bb_start
            seqs = repeat(seqs, 'b n -> b (n c)', c=reps)

        return coords, seqs, masks

    def step(self, x):
        coords, seq, mask = self.prepare_inputs(x)

        # noise with random amount
        s = coords.shape[0]
        t = random.randint(0, self.diffusion.timesteps - 1)
        ts = torch.full((s,), t).to(coords)  # all samples same t

        # forward diffusion
        noised_coords, noise = self.diffusion.q_sample(coords, mask, ts)
        if not self.context:
            noised_coords = noised_coords * mask.type(torch.float64)[..., None]
        ts = ts.type(torch.float64)

        # predict noisy input with transformer
        feats, prediction = self.transformer(noised_coords,
                                             ts,
                                             context=seq,
                                             mask=mask)

        # loss between original noise and prediction
        loss = F.mse_loss(prediction[mask], noise[mask])

        return feats, prediction, loss, t

    def score(self, x):
        # sample with diffusion
        coords, seqs, masks = self.prepare_inputs(x)
        model = self.transformer
        timesteps = self.diffusion.timesteps
        samples = self.diffusion.sample(model, coords, seqs, masks, timesteps)
        last_sample = samples[-1]
        dna = None
        if self.context:
            dna = str(x.dna_sequence[0])

        # save prediction tensor
        epoch = self.current_epoch + 1
        filename = f"pred_{epoch}.pt"
        filepath = os.path.join(self.ckpt_path, filename)
        torch.save(last_sample, filepath)

        # save prediction as PDB file
        pdb_filename = f"pred_{epoch}.pdb"
        pdb_filepath = os.path.join(self.ckpt_path, pdb_filename)
        pred_coord = last_sample[0]
        pred_seq = str(x.sequence[0][:self.trim])
        bb_start, bb_end = self.bb_start, self.bb_end
        pred_to_pdb(pred_coord, pred_seq, pdb_filepath, bb_start, bb_end, dna=dna)

        # save reference
        ref_fname = "ref.pt"
        ref_filepath = os.path.join(self.ckpt_path, ref_fname)
        torch.save(coords, ref_filepath)

        # calculate validation metrics
        ground = coords[0][:len(pred_seq)]
        pred = last_sample[0][:len(pred_seq)]
        dist_loss = calc_distmap_loss(ground, pred)
        tm1, tm2 = calc_tm_score(ground, pred, pred_seq, pred_seq)
        tm_score = max(tm1, tm2)
        return dist_loss, tm_score

    def training_step(self, batch, batch_idx):
        batch_size = batch.atom_coord.shape[0]
        feats, denoised, loss, t = self.step(batch)
        quantile = int((t / self.diffusion.timesteps * 100) // 10)
        self.log(f"train_loss_q{quantile}", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch.atom_coord.shape[0]
        feats, denoised, loss, t = self.step(batch)
        distmap_loss, tm_score = self.score(batch)
        self.log("val_loss", loss, batch_size=batch_size)
        self.log("val_distmap_loss", distmap_loss, batch_size=batch_size)
        self.log("val_tm_score", tm_score, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        feats, denoised, loss, t = self.step(batch)
        batch_size = batch.atom_coord.shape[0]
        self.log("test_loss", loss, batch_size=batch_size)
        return loss

    def on_train_epoch_start(self):
        self.start_epoch_time = time.time()

    def on_train_epoch_end(self):
        now = time.time()
        elapsed = now - self.start_epoch_time
        if self.verbose:
            print(f"Epoch took {elapsed:.2f} seconds")

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--beta_small', type=float, default=2e-4)
        parser.add_argument('--beta_large', type=float, default=0.02)
        parser.add_argument('--dim', type=int, default=128)
        parser.add_argument('--dim_head', type=int, default=64)
        parser.add_argument('--depth', type=int, default=8)
        parser.add_argument('--timesteps', type=int, default=250)
        parser.add_argument('--trim', type=int, default=None)
        parser.add_argument('--schedule', type=str, default='linear')
        parser.add_argument('--context', action=argparse.BooleanOptionalAction)
        parser.add_argument('--verbose', action=argparse.BooleanOptionalAction)
        return parser
