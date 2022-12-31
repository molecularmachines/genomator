import itertools

comp = {
    'A': 'T',
    'T': 'A',
    'C': 'G',
    'G': 'C'
}


def is_palindromic(dna: str):
    # all palindromic sequences are of even length
    if len(dna) % 2:
        return False
    length = len(dna)
    left_half = dna[:length // 2]
    rev_left_half = left_half[::-1]
    right_half = dna[length // 2:]
    for i, x in enumerate(rev_left_half):
        if x != comp[right_half[i]]:
            return False
    return True


def is_specific(dna: str):
    for b in dna:
        if b not in "ACGT":
            return False
    return True


def comp_seq(seq: str):
    rev_seq = seq[::-1]
    return seq + "".join([comp[x] for x in rev_seq])


def generate_palindroms(length: int):
    assert length % 2 == 0, "Palindroms are of even length"
    possible_lists = [['A', 'C', 'G', 'T'] for _ in range(length // 2)]
    combinatorial_seqs = list(itertools.product(*possible_lists))
    seqs = ["".join(x) for x in combinatorial_seqs]
    return [comp_seq(seq) for seq in seqs]
