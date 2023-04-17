import os
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from preprocess import StandardizeTransform
from models.en_denoiser import EnDenoiser
from moleculib.protein.dataset import ProteinDataset
from moleculib.protein.batch import PadBatch
from torch.utils.data import DataLoader
from aim.pytorch_lightning import AimLogger
from datetime import datetime


def cli_main():
    # TODO: does this affect random noise?
    pl.seed_everything(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--device', default='1', type=str)
    parser.add_argument('--train_path', default="data/single", type=str)
    parser.add_argument('--val_path', default="data/single", type=str)
    parser.add_argument('--test_path', default="data/single", type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = EnDenoiser.add_model_specific_args(parser)
    args = parser.parse_args()
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # ------------
    # data
    # ------------
    transform = StandardizeTransform()

    # train
    train_dataset = ProteinDataset(args.train_path, transform=[transform], preload=True)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=PadBatch.collate,
        batch_size=args.batch_size
    )

    # validation
    val_dataset = ProteinDataset(args.val_path, transform=[transform], preload=True)
    val_loader = DataLoader(
        val_dataset,
        collate_fn=PadBatch.collate,
        batch_size=args.batch_size
    )

    # test
    test_dataset = ProteinDataset(args.test_path, transform=[transform], preload=True)
    test_loader = DataLoader(
        test_dataset,
        collate_fn=PadBatch.collate,
        batch_size=args.batch_size
    )

    # ------------
    # logging
    # ------------
    now = datetime.now()
    date = now.strftime("%Y%m%d_%H%M%S")
    ex_name = f'ex_{date}'
    logger = AimLogger(
        experiment=ex_name,
        train_metric_prefix='train_',
        val_metric_prefix='val_'
    )

    # checkpoint
    checkpoint_path = os.path.join("checkpoints", ex_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_top_k=2,
        monitor="val_distmap_loss",
        mode="min"
    )

    # ------------
    # model
    # ------------
    model = EnDenoiser(
        dim=args.dim,
        dim_head=args.dim_head,
        beta_small=args.beta_small,
        beta_large=args.beta_large,
        lr=args.lr,
        depth=args.depth,
        schedule=args.schedule,
        timesteps=args.timesteps,
        trim=args.trim,
        ckpt_path=checkpoint_path
    )

    # ------------
    # training
    # ------------
    devices = [int(x) for x in args.device.split(',')]
    if device == 'cpu':
        devices = devices[0]
    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=checkpoint_path,
        accelerator=device,
        devices=devices,
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
