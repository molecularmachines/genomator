import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from preprocess import trim_dataset, standardize_dataset
from en_denoiser import EnDenoiser
import sidechainnet as scn
from collate import prepare_dataloaders


def cli_main():
    # TODO: does this affect random noise?
    pl.seed_everything(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--device', default='1', type=str)
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

    # preprocess data
    dataset_prefixes = ('train', 'valid', 'test')
    datasets = [d for d in data.keys() if d.startswith(dataset_prefixes)]
    for d in datasets:
        dataset = trim_dataset(data[d])
        dataset = standardize_dataset(dataset)
        data[d] = dataset
    data_loaders = prepare_dataloaders(data, True, batch_size=args.batch_size)

    # data loaders
    train_loader = data_loaders['train']
    val_loader = data_loaders['train-eval']
    test_loader = data_loaders['test']

    sanity_loader = data_loaders['valid-10']
    sanity_val_loader = data_loaders['valid-20']

    # ------------
    # model
    # ------------
    model = EnDenoiser()

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, accelerator=device, devices=args.device)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
