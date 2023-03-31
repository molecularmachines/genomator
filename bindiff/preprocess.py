import torch
from torch.distributions.categorical import Categorical
import numpy as np
from tqdm import tqdm
from moleculib.protein.transform import ProteinTransform


class DistributionNodes:
    def __init__(self, histogram):

        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob/np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys.get(i.item(), 0) for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)
        log_p = torch.log(self.prob + 1e-30)
        log_p = log_p.to(batch_n_nodes.device)
        log_probs = log_p[idcs]

        return log_probs


def get_dataset_info(dataset):
    n_nodes = dict()
    for i in range(len(dataset)):
        protein = dataset[i]
        seqlen = len(protein.sequence)
        n_nodes[seqlen] = n_nodes.get(seqlen, 0) + 1

    return {
        'n_nodes': n_nodes
    }


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
