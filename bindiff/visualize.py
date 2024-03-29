import torch
from sidechainnet.utils.sequence import ProteinVocabulary
from biotite.sequence import ProteinSequence
from einops import rearrange
from tmtools import tm_align
import numpy as np


def backbone_to_pdb(coords, seq, pdb_fname, chain="A", bb_start=1, bb_end=2, dna=None, save=True):
    start_idx = 0
    dna_lines = None
    if dna is not None:
        dna_len = 2 * len(dna)
        centroids = coords[-dna_len:]
        dna_lines, chain, start_idx = dna_to_pdb(dna, centroids)
        chain = chr(ord(chain) + 1)
        coords = coords[:-dna_len]

    num_backbone_atoms = bb_end - bb_start
    assert num_backbone_atoms <= 4 and num_backbone_atoms > 0
    vocab = ProteinVocabulary()
    backbone_atoms = ['N', 'CA', 'C', 'O'][bb_start:bb_end]
    seq_str = seq if type(seq) is str else vocab.ints2str(seq)
    assert len(coords) == len(seq_str) * num_backbone_atoms

    def _coords_to_atom_line(i, crd, start_idx=0):
        # setup params
        atom_type = backbone_atoms[i % num_backbone_atoms]
        res_id = i // num_backbone_atoms
        res = ProteinSequence.convert_letter_1to3(seq_str[res_id])
        if res == "UNK":
            return ""
        res_atom = atom_type[0]
        x, y, z = crd

        # construct atom line from coords
        line = list(" " * 80)
        line[0:6] = "ATOM".ljust(6)
        line[6:11] = str(start_idx + i + 1).rjust(5)
        line[12:16] = atom_type.ljust(4)
        line[17:20] = res.ljust(3)
        line[21] = chain
        line[22:26] = str(res_id + 1).rjust(4)
        line[30:38] = f'{x:.3f}'.rjust(8)
        line[38:46] = f'{y:.3f}'.rjust(8)
        line[46:54] = f'{z:.3f}'.rjust(8)
        line[54:60] = "1.00".rjust(6)
        line[60:66] = "0.00".rjust(6)
        line[76:78] = res_atom.rjust(2)
        line = "".join(line) + "\n"

        return line

    # write as PDB file
    if save:
        with open(pdb_fname, 'w') as f:
            if dna:
                f.write(dna_lines)
            for i, coord in enumerate(coords):
                line = _coords_to_atom_line(i, coord, start_idx)
                f.write(line)
        print(f"File {pdb_fname} has been saved.")
        return pdb_fname

    # return as PDB string
    else:
        pdb_str = ""
        if dna:
            pdb_str += dna_lines
        for i, coord in enumerate(coords):
            pdb_str += _coords_to_atom_line(i, coord, start_idx)
        return pdb_str


def dna_to_pdb(seq, centroids):

    def _coords_to_atom_line(i, crd, res, chain):
        atom_type = "P"
        x, y, z = crd
        res = "D" + res

        # construct atom line from coords
        line = list(" " * 80)
        line[0:6] = "ATOM".ljust(6)
        line[6:11] = str(i + 1).rjust(5)
        line[12:16] = atom_type.ljust(4)
        line[17:20] = res.ljust(3)
        line[21] = chain
        line[22:26] = str(i + 1).rjust(4)
        line[30:38] = f'{x:.3f}'.rjust(8)
        line[38:46] = f'{y:.3f}'.rjust(8)
        line[46:54] = f'{z:.3f}'.rjust(8)
        line[54:60] = "1.00".rjust(6)
        line[60:66] = "0.00".rjust(6)
        line[76:78] = atom_type.rjust(2)
        line = "".join(line) + "\n"

        return line

    def _ter_line(i):
        line = list("" * 80)
        line[0:6] = "TER".ljust(6)
        line[6:11] = str(i + 1).rjust(5)
        line = "".join(line) + "\n"
        return line

    dna_lines = ""
    idx = 0
    chain = "A"
    for i, crd in enumerate(centroids):
        if i == len(seq):
            ter_line = _ter_line(idx)
            dna_lines += ter_line
            chain = chr(ord(chain) + 1)
            idx += 1

        atom_line = _coords_to_atom_line(idx, crd, seq[i % len(seq)], chain)
        dna_lines += atom_line
        idx += 1

    dna_lines += _ter_line(idx)
    idx += 1
    return dna_lines, chain, idx


def backbones_to_animation(coords_list, seq, pdb_fname, bb_start=1, bb_end=2, dna=None):
    with open(pdb_fname, "w") as f:
        for i, coords in enumerate(coords_list):
            f.write(f"MODEL {i+1}\n")
            pdb_str = backbone_to_pdb(coords, seq, pdb_fname, bb_start=bb_start, bb_end=bb_end, save=False, dna=dna)
            f.write(pdb_str)
            f.write("ENDMDL\n")
    print(f"File {pdb_fname} has been saved.")
    return pdb_fname


def rescale_protein(coords, std_const=9.0):
    return std_const * coords


def rearrange_coords(coords, bb_start, bb_end):
    new_coords = coords[:, bb_start:bb_end, :]
    new_coords = rearrange(new_coords, "s b c -> (s b) c")
    return new_coords


def align_coords(coords, ref, seq):
    align = tm_align(coords, ref, seq, seq)
    return np.matmul(coords, align.u) + align.t


def pred_to_pdb(coord, seq, pdb_fname, bb_start, bb_end, rearrange=False, dna=None):
    if rearrange:
        coord = rearrange_coords(coord, bb_start, bb_end)
    coord = rescale_protein(coord)
    backbone_to_pdb(coord, seq, pdb_fname, bb_start=bb_start, bb_end=bb_end, dna=dna)


def preds_to_pdb(crds, seq, pdb_fname, bb_start, bb_end, rearrange=False, align=True):
    chains = "ABCDEFGHI"
    pdb = ""
    ref = crds[0]
    for i, coord in enumerate(crds):
        chain = chains[i]
        if align and i > 0:
            coord = align_coords(coord, ref, seq)
        if rearrange:
            coord = rearrange_coords(coord, bb_start, bb_end)
        coord = rescale_protein(coord)
        pdb += backbone_to_pdb(coord, seq, pdb_fname, chain, bb_start, bb_end, save=False)

    with open(pdb_fname, "w") as f:
        f.write(pdb)
        print(f"File {pdb_fname} has been saved.")


if __name__ == "__main__":
    seq = "ATV"
    num_backbone_atoms = 4
    coords = torch.rand((num_backbone_atoms * len(seq), 3))
    pdb_fname = "test.pdb"
    backbone_to_pdb(coords, seq, pdb_fname, num_backbone_atoms)
