import subprocess
import os
from tqdm import tqdm
from Bio import PDB
from multiprocessing import Pool, cpu_count

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
    xyzr_file = os.path.join(output_dir, prefix + ".xyzrn")
    pdb_to_xyzrn_command = f"pdb_to_xyzrn {pdb_file} > {xyzr_file}"
    msms_command = ["msms", "-if", xyzr_file, "-of", os.path.join(output_dir, prefix)]
    try:
        subprocess.run(pdb_to_xyzrn_command, shell=True)
        subprocess.run(msms_command, check=True)
        os.remove(xyzr_file)
        os.remove(os.path.join(output_dir, prefix + ".face"))
    except:
        if os.path.exists(xyzr_file):
            os.remove(xyzr_file)
        if os.path.exists(os.path.join(output_dir, prefix + ".face")):
            os.remove(os.path.join(output_dir, prefix + ".face"))

def process_pdb_file(pdb_file):
    pdb_path = os.path.join(pdb_dir, pdb_file)
    if check_residue_ids_continuity(pdb_path):
        generate_surface(pdb_path, output_dir)

if __name__ == "__main__":
    pdb_dir = '/home/v-yantingli/mmp/afdb_v4'
    output_dir = "/home/v-yantingli/mmp/afdb_v4_surface"
    
    # Limit the files to the first 100000 for this example
    pdb_files = os.listdir(pdb_dir)[:100000]

    # Set up a progress bar and pool of workers
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(process_pdb_file, pdb_files), total=len(pdb_files)))

