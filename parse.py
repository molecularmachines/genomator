DATA_FILE = 'data/restriction_enzymes.txt'
if __name__ == '__main__':
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
                        entries.append((curr_entry_dna, curr_entry_aa))
                        curr_entry_dna = None
                        curr_entry_aa = ''
                else:
                    if curr_entry_dna:
                        curr_entry_aa += ''.join(line.split())
