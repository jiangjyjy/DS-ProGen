import subprocess
import os
from tqdm import tqdm
from Bio import PDB

def check_residue_ids_continuity(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)

    for model in structure:
        for chain in model:
            previous_residue_id = None
            for residue in chain:
                current_residue_id = residue.id[1]

                if previous_residue_id is not None:
                    if current_residue_id != previous_residue_id + 1:
                        return False  
                
                previous_residue_id = current_residue_id
                
    return True  


def generate_surface(pdb_file, output_dir):
    pdb_name = os.path.basename(pdb_file)
    prefix = pdb_name.split(".")[0]
    xyzr_file = os.path.join(output_dir, prefix + ".xyzr")
    pdb_to_xyzrn_command = f"pdb_to_xyzrn {pdb_file} > {xyzr_file}"
    subprocess.run(pdb_to_xyzrn_command, shell=True)
    msms_command = ["msms", "-if", xyzr_file, "-of", os.path.join(output_dir, prefix)]
    subprocess.run(msms_command, check=True)


pdb_dir = '/home/v-yantingli/mmp/data/test_data'
output_dir = "/home/v-yantingli/mmp/data/test_surface_data"

for pdb_file in tqdm(os.listdir(pdb_dir)):
    if check_residue_ids_continuity(os.path.join(pdb_dir, pdb_file)):
        generate_surface(os.path.join(pdb_dir, pdb_file), output_dir)
