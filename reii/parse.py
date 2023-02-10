import itertools
import string

DATA_FILE = 'data/restriction_enzymes.txt'

symbols = {
    'A': ['A'],
    'G': ['G'],
    'C': ['C'],
    'T': ['T'],
    'R': ['G', 'A'],
    'Y': ['C', 'T'],
    'K': ['G', 'T'],
    'M': ['A', 'C'],
    'S': ['G', 'C'],
    'W': ['A', 'T'],
    'B': ['G', 'T', 'C'],
    'D': ['G', 'A', 'T'],
    'H': ['A', 'C', 'T'],
    'V': ['G', 'C', 'A'],
    'N': ['A', 'G', 'C', 'T']
}


def strip_punct(x):
    return x.translate(str.maketrans('', '', string.punctuation))


def parse(filename: str, expand=True):
    entries = []
    curr_entry_dna = None
    curr_entry_aa = ''
    with open(DATA_FILE) as f:
        for line in f:
            # new entry
            if line.startswith('>'):
                entry_comps = line[1:].split()
                if ord(entry_comps[1][0]) < 65:  # not alphabet
                    continue
                else:
                    curr_entry_dna = entry_comps[1]  # the DNA sequence
            else:
                if line == '\n':
                    # add new entry
                    if curr_entry_dna and len(curr_entry_aa):
                        curr_entry_dna = strip_punct(curr_entry_dna)
                        entries.append((curr_entry_dna, curr_entry_aa))
                        curr_entry_dna = None
                        curr_entry_aa = ''
                else:
                    if curr_entry_dna:
                        curr_entry_aa += ''.join(line.split())

    # expand multiple binding DNA seqs to individual samples
    if expand:
        new_entries = []
        for entry in entries:
            # collate all possible basepairs
            entry_possible_lists = []
            dna, aa = entry
            for x in dna:
                entry_possible_lists.append(symbols[x])

            # calculate all combinatorial sequences
            entry_dna_seqs = itertools.product(*entry_possible_lists)
            entry_dna_seqs = [''.join(x) for x in entry_dna_seqs]
            new_entries += [(dna_seq, aa) for dna_seq in entry_dna_seqs]
            entry_possible_lists = []

        entries = new_entries

    return entries


if __name__ == '__main__':
    entries = parse(DATA_FILE)
