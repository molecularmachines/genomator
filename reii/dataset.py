import random
import torch
import torch.nn.functional as F
from random import shuffle
from torch.utils.data import Dataset
from parse import parse

random.seed(42)


labels = [
    'O',
    'ACGT',
    'CATG',
    'CCGG',
    'CCTC',
    'CGCG',
    'CTAG',
    'GAAC',
    'GATC',
    'GCGC',
    'GGCC',
    'GTAC',
    'TCGA',
    'TGCA'
]

label2idx = {x: i for i, x in enumerate(labels)}
idx2label = {i: x for i, x in enumerate(labels)}


class ProtDNADataset(Dataset):

    def __init__(self, data_filepath):
        super().__init__()

        # parse data from dataset file
        data = self.parse_dataset_file(data_filepath)

        # format data for ESM tokenizer
        O_idx = label2idx['O']
        for i, (dna, aa) in enumerate(data):
            dna_label = torch.tensor(label2idx.get(dna, O_idx))
            aa = (str(i), aa)
            data[i] = (aa, dna_label)

        self.data = data

    @staticmethod
    def parse_dataset_file(dataset_file):
        data = []
        with open(dataset_file, "r") as f:
            for line in f:
                line = line.rstrip()
                if len(line):
                    dna, aa = line.split()
                    data.append((dna, aa))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def split_data(data_filepath, train=0.8, val=0.1, test=0.1):
    DNA_LENGTH = 4
    OCC_RATE = 20

    # parse data from file
    data = parse(data_filepath, expand=False)

    # only data of certain length
    data = [(d, aa) for (d, aa) in data if len(d) == DNA_LENGTH]

    # filter sequences with low occurrence rate
    dna_seqs = [dna for (dna, aa) in data]
    dna_occ = dict()
    for dna in dna_seqs:
        dna_occ[dna] = dna_occ.get(dna, 0) + 1
    data = [(dna, aa) for (dna, aa) in data if dna_occ[dna] > OCC_RATE]

    # prepare data for split
    shuffle(data)
    train_split_idx = int(train * len(data))
    val_split_idx = int(train_split_idx + val * len(data))
    train_data = data[:train_split_idx]
    val_data = data[train_split_idx:val_split_idx]
    test_data = data[val_split_idx:]

    # save datasets to file
    filename, extension = data_filepath.split(".")

    def save_dataset(dataset, name):
        dataset_fp = f"{filename}_{name}.{extension}"
        f = open(dataset_fp, "w", encoding="utf-8")
        for dna, aa in dataset:
            f.write(f"{dna} {aa}\n")
        f.close()
        print(f"file {dataset_fp} has been saved.")
        return dataset_fp

    train_file = save_dataset(train_data, "train")
    val_file = save_dataset(val_data, "val")
    test_file = save_dataset(test_data, "test")

    return train_file, val_file, test_file


if __name__ == "__main__":
    fp = "data/restriction_enzymes.txt"
    split_data(fp)
    train_fp = "data/restriction_enzymes_train.txt"
    train_dataset = ProtDNADataset(train_fp)
