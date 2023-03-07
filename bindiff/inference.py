import os
import sidechainnet as scn
from en_denoiser import EnDenoiser
from collate import prepare_dataloaders
from preprocess import trim_dataset, standardize_dataset

CHECKPOINT_DIR = "lightning_logs/checkpoints"
CHECKPOINT_NAME = "ex2_epoch80"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME + ".ckpt")


def sample():
    # TODO: params should be loaded from checkpoint
    dim = 128
    dim_head = 128
    heads = 4
    depth = 8
    timesteps = 100

    # load model from checkpoint
    model = EnDenoiser.load_from_checkpoint(CHECKPOINT_FILE,
                                            dim=dim,
                                            dim_head=dim_head,
                                            heads=heads,
                                            depth=depth,
                                            timesteps=timesteps)
    # load data for reference
    batch_size = 1
    data = scn.load(
        casp_version=12,
        thinning=30,
        batch_size=batch_size,
        dynamic_batching=False
    )

    # preprocess data
    dataset_prefixes = ('train')
    datasets = [d for d in data.keys() if d.startswith(dataset_prefixes)]
    for d in datasets:
        dataset = trim_dataset(data[d])
        dataset = standardize_dataset(dataset)
        data[d] = dataset
    data_loaders = prepare_dataloaders(data, True, batch_size=batch_size)
    train_loader = data_loaders['train']

    # create initial structure
    batch = next(iter(train_loader))

    # sample from model
    results = model.sample(batch, timesteps)
    res = {
        "original": batch,
        "results": results
    }
    return res


if __name__ == "__main__":
    results = sample()
