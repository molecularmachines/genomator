import torch
from torch import nn, optim
from argparse import ArgumentParser
import torch.nn.functional as F
import pytorch_lightning as pl


class ESMLinear(pl.LightningModule):

    def __init__(self,
                 esm_model,
                 out_size,
                 lr=1e-3,
                 freeze=True,
                 esm_size=1280):
        super().__init__()

        # ESM model
        self.esm, self.alphabet = esm_model
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm.eval()

        if freeze:
            for param in self.esm.parameters():
                param.requires_grad = False

        # linear fine tune layer
        self.out_size = out_size
        self.out = nn.Linear(esm_size, out_size)

        # model args
        self.lr = lr
        self.freeze = freeze

    def _esm_inference(self, x, y):
        # encode ESM tokens
        _, _, tokens = self.batch_converter(x)
        tokens = tokens.to(y)  # move to same device as input
        lens = (tokens != self.alphabet.padding_idx).sum(1)

        # ESM forward
        with torch.no_grad():
            results = self.esm(tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]

        # one representation for the entire sequence
        sequence_representations = []
        for i, tokens_len in enumerate(lens):
            seq_repr = token_representations[i, 1:tokens_len - 1].mean(0)
            sequence_representations.append(seq_repr)
        sequence_representations = torch.stack(sequence_representations)
        return sequence_representations

    def step(self, x, y):
        # ESM layers
        sequence_rep = self._esm_inference(x, y)

        # linear layer
        y_hat = self.out(sequence_rep)
        return y_hat

    def training_step(self, batch, batch_idx):
        # unpack batch
        x, y = batch
        x = list(zip(x[0], x[1]))

        # run model on inputs
        y_hat = self.step(x, y)

        # compute loss
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # unpack batch
        x, y = batch
        x = list(zip(x[0], x[1]))

        # run model on inputs
        y_hat = self.step(x, y)

        # compute loss
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

        # compute accuracy
        preds = torch.argmin(y_hat, dim=1)
        acc = (preds == y).sum() / len(y)
        self.log("val_acc", acc)

        return loss

    def test_step(self, batch, batch_idx):
        # unpack batch
        x, y = batch
        x = list(zip(x[0], x[1]))

        # run model on inputs
        y_hat = self.step(x, y)

        # compute loss
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)

        # compute accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).sum() / len(y)
        self.log("test_acc", acc)

        return loss


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--freeze', type=bool, default=True)
        return parser
