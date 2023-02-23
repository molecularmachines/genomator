import torch
import sidechainnet as scn
from en_denoiser import EnDenoiser
from collate import prepare_dataloaders
from preprocess import trim_dataset, standardize_dataset

if __name__ == "__main__":
    bs = 4
    data = scn.load(12, thinning=30, batch_size=bs)
    denoiser = EnDenoiser()

    # preprocess data
    dataset_prefixes = ('train', 'valid', 'test')
    datasets = [d for d in data.keys() if d.startswith(dataset_prefixes)]
    for d in datasets:
        dataset = trim_dataset(data[d])
        dataset = standardize_dataset(dataset)
        data[d] = dataset
    data_loaders = prepare_dataloaders(data, True, batch_size=bs)

    # data loaders
    train_loader = data_loaders['train']
    val_loader = data_loaders['train-eval']
    test_loader = data_loaders['test']

    sanity_loader = data_loaders['valid-10']
    sanity_val_loader = data_loaders['valid-20']

    # fetch one batch
    b = next(iter(sanity_loader))

    # visualize
    seq_batch = torch.argmax(b.seqs, dim=-1)
    builder = scn.BatchedStructureBuilder(seq_batch=seq_batch, crd_batch=b.crds)
    struct = builder.build()
    builder.to_3Dmol(0)

    # sample like batch
    timesteps = 10
    samples = denoiser.sample(b, timesteps)
