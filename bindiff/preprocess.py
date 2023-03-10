import numpy as np
from tqdm import tqdm
from moleculib.protein.transform import ProteinTransform


def trim_dataset(dataset, length=300):
    seq_dataset = dataset['seq']
    idx = -1

    # find first occurrence of long protein (list is sorted)
    for i in range(len(seq_dataset)):
        if len(seq_dataset[i]) > length:
            idx = i
            break

    # trim all attributes of dataset
    for k in dataset.keys():
        dataset[k] = dataset[k][:idx]

    return dataset


def center_coords(coords):
    # processing each protein coords at a time
    assert len(coords.shape) == 3  # [SEQ_LEN, MAX_ATOMS, 3]
    assert coords.shape[-1] == 3

    # mask zero coords
    mask = coords != 0

    # calculate masked mean and std
    # coords_mean = (mask * coords).sum(axis=0) / mask.sum(axis=0)
    coords_mean = coords.mean(axis=0)
    # coords_std = np.sqrt(((coords - coords_mean * mask) ** 2).sum(axis=0) / mask.sum(axis=0))
    coords_std = 9.0

    # standardize proteins
    centered_coords = coords - coords_mean * mask
    return centered_coords, coords_std


def standardize_dataset(dataset):
    coords_dataset = dataset['crd']
    std_const = 9.0

    # center and rescale dataset
    for i, coords in tqdm(enumerate(coords_dataset)):
        centered_coords, coords_std = center_coords(coords)
        coords_dataset[i] = centered_coords / std_const

    return dataset


class StandardizeTransform(ProteinTransform):

    def __init__(self, std_const=9.0):
        super().__init__()
        self.std_const = std_const

    def transform(self, datum):
        centered_coords, coords_std = center_coords(datum.atom_coord)
        standardized = centered_coords / self.std_const
        datum.atom_coord = standardized
        return datum
