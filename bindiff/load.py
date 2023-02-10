import sidechainnet as scn
from collate import prepare_dataloaders

if __name__ == "__main__":
    bs = 4
    data = scn.load(12, thinning=30, batch_size=bs)
    data_loaders = prepare_dataloaders(data, True, batch_size=bs)
