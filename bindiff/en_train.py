import torch
from argparse import ArgumentParser
import pytorch_lightning as pl

from en_denoiser import EnDenoiser
import sidechainnet as scn
from collate import prepare_dataloaders


def cli_main():
    pl.seed_everything(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=2, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = EnDenoiser.add_model_specific_args(parser)
    args = parser.parse_args()
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # ------------
    # data
    # ------------
    data = scn.load(
        casp_version=12,
        thinning=30,
        batch_size=args.batch_size,
        dynamic_batching=False
    )

    data_loaders = prepare_dataloaders(data, True, batch_size=args.batch_size)

    train_loader = data_loaders['train']
    val_loader = data_loaders['train-eval']
    test_loader = data_loaders['test']

    sanity_loader = data_loaders['valid-10']
    sanity_val_loader = data_loaders['valid-20']

    # ------------
    # model
    # ------------
    model = EnDenoiser(denoise_step=args.step)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, accelerator=device, devices=1)
    trainer.fit(model, sanity_loader, sanity_val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
