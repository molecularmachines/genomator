import torch
from torch import optim
import torch.nn.functional as F
from argparse import ArgumentParser
import pytorch_lightning as pl
from einops import rearrange, repeat

from models.edm_models import EGNN_dynamics_QM9
from models.en_diffusion import EnVariationalDiffusion
from models.losses import compute_loss_and_nll


class EGNNDenoiser(pl.LightningModule):

    def __init__(self,
                 nodes_dist,
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

        # torch.set_default_dtype(torch.float64)
        self.save_hyperparameters()
        self.nodes_dist = nodes_dist
        self.num_tokens = 23  # number of amino acids and special tokens
        in_node_nf = self.num_tokens
        dynamincs_in_node_nf = self.num_tokens + 1

        net_dynamics = EGNN_dynamics_QM9(
            in_node_nf=dynamincs_in_node_nf,
            context_node_nf=0,
            n_dims=3
        )

        self.model = EnVariationalDiffusion(
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3
        )

        self.lr = lr

    def prepare_inputs(self, x):
        # extract input
        seq, coord, node_mask = x.residue_token, x.atom_coord, x.atom_mask

        # type matching for transformer
        coord = coord.type(torch.float32)

        # keeping only the backbone coordinates
        coord = coord[:, :, 0:4, :]
        node_mask = node_mask[:, :, 0:4]
        coord = rearrange(coord, 'b l s c -> b (l s) c')
        node_mask = rearrange(node_mask, 'b l s -> b (l s)')
        node_mask = node_mask.unsqueeze(-1).int()

        # compute edge mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

        # assign sequence token to each of the backbone atoms
        seq = repeat(seq, 'b n -> b (n c)', c=4)

        return coord, seq, node_mask, edge_mask

    def step(self, x):
        coord, seq, node_mask, edge_mask = self.prepare_inputs(x)
        one_hot_seq = F.one_hot(seq, num_classes=self.num_tokens)
        ode_regularization = 1e-3
        charges = torch.zeros(0).to(coord)

        # prepare input to EGNN
        x = coord
        h = {'categorical': one_hot_seq, 'integer': charges}
        context = None

        # compute NLL
        nll, reg_term, mean_abs_z = compute_loss_and_nll(
            self.model, self.nodes_dist, x, h, node_mask, edge_mask, context
        )

        # standard nll from forward KL
        loss = nll + ode_regularization * reg_term
        return loss

    def training_step(self, batch, batch_idx):
        batch_size = batch.atom_coord.shape[0]
        loss = self.step(batch)
        self.log("train_loss", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch.atom_coord.shape[0]
        loss = self.step(batch)
        self.log("val_loss", loss, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        batch_size = batch.atom_coord.shape[0]
        loss = self.step(batch)
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
