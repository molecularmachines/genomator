import esm
from argparse import ArgumentParser
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from en_denoiser import EnDenoiser
import sidechainnet as scn


def cli_main():
    pl.seed_everything(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--accumulate_grad_batches', default=16, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = EnDenoiser.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    def cycle(loader, len_thres=200):
        while True:
            for data in loader:
                if data.seqs.shape[1] > len_thres:
                    continue
                yield data

    data = scn.load(
        casp_version=12,
        thinning=30,
        with_pytorch='dataloaders',
        batch_size=args.batch_size,
        dynamic_batching=False
    )

    train_loader = cycle(data['train'])
    val_loader = cycle(data['train-eval'])
    test_loader = cycle(data['test'])

    # ------------
    # model
    # ------------
    model = EnDenoiser()

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
