import os
from Bio import PDB
from tqdm import tqdm


file_path = '/home/v-yantingli/mmp/data/test_surface_data'
pdb_path = '/home/v-yantingli/mmp/data/test_data'
pdb_list = [name.split('.')[0] for name in os.listdir(os.path.join(file_path)) if name.endswith('.vert')]

amino_acid_dict = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
                   "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
                   "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


with open(os.path.join(file_path, 'fasta.txt'), 'w') as f:
    for filename in tqdm(pdb_list):
        filepath = os.path.join(pdb_path, f'{filename}.pdb')
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure(filename, filepath)
        first_aa_id = None
        valid = True
        sequence = ""
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == ' ' and PDB.is_aa(residue):  # ensures that it is amino acid
                        sequence += amino_acid_dict[residue.get_resname()]
                    else:
                        valid = False
                    if first_aa_id is None:
                        first_aa_id = f"{residue.get_id()[1]}"

        if valid and len(sequence)<=500:
            f.write(f'>{filename}_{first_aa_id}\n{sequence}\n')