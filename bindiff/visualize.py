import torch
from sidechainnet.utils.sequence import ProteinVocabulary, ONE_TO_THREE_LETTER_MAP


def backbone_to_pdb(coords, seq, pdb_fname, num_backbone_atoms=4, save=True):
    assert num_backbone_atoms <= 4 and num_backbone_atoms > 0
    vocab = ProteinVocabulary()
    backbone_atoms = ['N', 'CA', 'C', 'O'][:num_backbone_atoms]
    seq_str = seq if type(seq) is str else vocab.ints2str(seq)
    assert len(coords) == len(seq_str) * num_backbone_atoms

    def _coords_to_atom_line(i, crd):
        # setup params
        atom_type = backbone_atoms[i % num_backbone_atoms]
        res_id = i // num_backbone_atoms
        res = ONE_TO_THREE_LETTER_MAP[seq_str[res_id]]
        res_atom = atom_type[0]
        x, y, z = crd

        # construct atom line from coords
        line = list(" " * 80)
        line[0:6] = "ATOM".ljust(6)
        line[6:11] = str(i + 1).rjust(5)
        line[12:16] = atom_type.ljust(4)
        line[17:20] = res.ljust(3)
        line[21] = "A"
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
            for i, coord in enumerate(coords):
                line = _coords_to_atom_line(i, coord)
                f.write(line)
        print(f"File {pdb_fname} has been saved.")
        return pdb_fname

    # return as PDB string
    else:
        pdb_str = ""
        for i, coord in enumerate(coords):
            pdb_str += _coords_to_atom_line(i, coord)
        return pdb_str


def backbones_to_animation(coords_list, seq, pdb_fname, num_backbone_atoms=4):
    with open(pdb_fname, "w") as f:
        for i, coords in enumerate(coords_list):
            f.write(f"MODEL {i+1}\n")
            pdb_str = backbone_to_pdb(coords, seq, pdb_fname, num_backbone_atoms, save=False)
            f.write(pdb_str)
            f.write("ENDMDL\n")
    print(f"File {pdb_fname} has been saved.")
    return pdb_fname


if __name__ == "__main__":
    seq = "ATV"
    num_backbone_atoms = 4
    coords = torch.rand((num_backbone_atoms * len(seq), 3))
    pdb_fname = "test.pdb"
    backbone_to_pdb(coords, seq, pdb_fname, num_backbone_atoms)
