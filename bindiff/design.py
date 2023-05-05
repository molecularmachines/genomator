import os
import subprocess
from biotite.sequence.io import fasta


OUTPUT_PATH = "pipeline"


def process(args):
    ret = -1
    num_tries = 0
    while ret < 0:
        try:
            process = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )
            ret = process.wait()
        except Exception as e:
            num_tries += 1
            print(f'Failed ProteinMPNN. Attempt {num_tries}/5')
            if num_tries > 4:
                raise e
    return True


def fname_from_path(path):
    fname = os.path.basename(pdb_path)
    return os.path.splitext(fname)[0]


def process_pdb_for_mpnn(pdb_path):
    print(f"Processing file {pdb_path}")
    fname = fname_from_path(pdb_path)
    out_file = os.path.join(OUTPUT_PATH, f'{fname}.jsonl')
    process_args = [
        'python',
        'ProteinMPNN/helper_scripts/parse_multiple_chains.py',
        f'--input_path {pdb_path}',
        f'--output_path {out_file}',
        '--ca_only'
    ]
    process(process_args)
    print(f"Created file {out_file}")
    return out_file


def mpnn(jsonl_path):
    print("Running ProteinMPNN")
    out_dir = "pmpnn_test"
    pmpnn_args = [
        'python',
        'ProteinMPNN/protein_mpnn_run.py',
        '--out_folder',
        f'{out_dir}',
        '--jsonl_path',
        'ProteinMPNN/test.jsonl',
        '--ca_only'
    ]

    ret = -1
    num_tries = 0
    while ret < 0:
        try:
            process = subprocess.Popen(
                pmpnn_args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )
            ret = process.wait()
        except Exception as e:
            num_tries += 1
            print(f'Failed ProteinMPNN. Attempt {num_tries}/5')
            if num_tries > 4:
                raise e

    fname = fname_from_path(jsonl_path)
    out_fasta = os.path.join(out_dir, 'seqs', f'{fname}.fa')
    print(f"Produced sequences in file {out_fasta}")
    return out_fasta


def esmfold(fasta_path):
    fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)

def pipeline(pdb_path):
    # process PDB file into jsonl for MPNN
    processed_file = process_pdb_for_mpnn(pdb_path)
    # run MPNN to get sequences
    sequence_fasta = mpnn(processed_file)
    # use ESMFold to fold sequences


if __name__ == "__main__":
    pdb_path = "notebooks/ca.pdb"
    pipeline(pdb_path)