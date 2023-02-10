import esm
from argparse import ArgumentParser
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from esm_classifier import ESMLinear
from dataset import ProtDNADataset, labels

TRAIN_FILE = "data/restriction_enzymes_train.txt"
VAL_FILE = "data/restriction_enzymes_val.txt"
TEST_FILE = "data/restriction_enzymes_test.txt"


def cli_main():
    pl.seed_everything(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=2, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ESMLinear.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    train_dataset = ProtDNADataset(TRAIN_FILE)
    val_dataset = ProtDNADataset(VAL_FILE)
    test_dataset = ProtDNADataset(TEST_FILE)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    num_labels = len(labels)
    esm_model = esm.pretrained.esm2_t33_650M_UR50D()
    model = ESMLinear(esm_model, num_labels, args.lr, args.freeze)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
