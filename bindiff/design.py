import os
import subprocess
from biotite.sequence.io import fasta
import torch
import esm

from moleculib.protein.dataset import ProteinDNADataset
from moleculib.protein.batch import PadComplexBatch
from preprocess import StandardizeTransform
from torch.utils.data import DataLoader
from visualize import (
    backbone_to_pdb,
    backbones_to_animation,
    rescale_protein
)
from models.en_denoiser import EnDenoiser

OUTPUT_PATH = "pipeline"
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()


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
    process(pmpnn_args)
    fname = fname_from_path(jsonl_path)
    out_fasta = os.path.join(out_dir, 'seqs', f'{fname}.fa')
    print(f"Produced sequences in file {out_fasta}")
    return out_fasta


def esmfold(fasta_path):
    out_path = os.path.join(OUTPUT_PATH, 'out.pdb')
    fasta_seqs = fasta.FastaFile.read(fasta_path)

    # run ESMFold on all sequences in FASTA
    for i, (header, string) in enumerate(fasta_seqs.items()):
        with torch.no_grad():
            output = model.infer_pdb(string)

        with open(out_path, "w") as f:
            f.write(output)

    print("Folded structure saved to: {out_path}")
    return out_path


def load_model_and_loader(checkpoint, data_dir):
    transform = [StandardizeTransform()]
    train_dataset = ProteinDNADataset(data_dir, transform=transform, preload=True)
    loader = DataLoader(train_dataset, collate_fn=PadComplexBatch.collate, batch_size=2, shuffle=False)
    model = EnDenoiser.load_from_checkpoint(checkpoint).eval()
    return model, loader


def inference(model, loader):
    sample_path = os.path.join(OUTPUT_PATH, "sample.pdb")
    animation_path = os.path.join(OUTPUT_PATH, "animation.pdb")

    # load first batch
    batch = next(iter(loader))
    crd, seq, msk = model.prepare_inputs(batch)
    seq_str = str(batch.sequence[0][:model.trim])
    dna_seq = str(batch.dna_sequence[0])
    cmask = batch.complex_mask

    # run diffusion
    results = model.diffusion.sample(model.transformer, crd, seq, msk, model.diffusion.timesteps)
    results = [x[0][cmask[0]].squeeze(0) for x in results]

    # save PDB files for diffusion steps
    last_result = rescale_protein(results[-1])
    backbone_to_pdb(last_result, seq_str, sample_path, dna=dna_seq)
    rescaled_results = [rescale_protein(x) for x in results]
    backbones_to_animation(rescaled_results, seq_str, animation_path, dna=dna_seq)
    return sample_path


def pipeline(checkpoint, data_dir):
    # load the model and data loader
    model, loader = load_model_and_loader(checkpoint, data_dir)

    # run diffusion model to produce sample PDB
    sample_pdb = inference(model, loader)

    # process PDB file into jsonl for MPNN
    processed_file = process_pdb_for_mpnn(sample_pdb)

    # run MPNN to get sequences
    sequence_fasta = mpnn(processed_file)

    # use ESMFold to fold sequences
    esmfold(sequence_fasta)


if __name__ == "__main__":
    data_dir = "data/protdna_8"
    checkpoint = "protdna_double.ckpt"
    pipeline(checkpoint, data_dir)
