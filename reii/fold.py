import os
import torch
import esm
from parse import parse

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
# model.set_chunk_size(128)

if __name__ == "__main__":
    DATA_FILE = "data/restriction_enzymes.txt"
    OUTPUT_DIR = "folds"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # parse entries from data file
    entries = parse(DATA_FILE, expand=False)

    # fold all entries in dataset
    for i, entry in enumerate(entries):
        dna, aa = entry
        filename = f"re_{dna}_{i+1}.pdb"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with torch.no_grad():
            output = model.infer_pdb(aa)

        with open(filepath, "w") as f:
            f.write(output)

        print(f"file {filepath} has been successfully saved")
