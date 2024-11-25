import requests
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

data_list_path = 'data/data_list/train&valid'
save_path = 'data/cath43_data'
os.makedirs(save_path,exist_ok=True)

# Load data list 
train_list = os.listdir(data_list_path)
train_lines = []
for file_name in train_list:
    with open(os.path.join(data_list_path, file_name), 'r') as file:
        for line in file:
            train_lines.append(line.strip())

# Search data not exsist
download_set = set()
for line in train_lines:
    download_set.add(line.split(' ')[0][:4])


def download_pdb(pdb_id, save_path):
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    response = requests.get(url)
    if response.status_code == 200:
        # Save as .cif
        with open(os.path.join(save_path, f"{pdb_id}.cif"), "wb") as cif_file:
            cif_file.write(response.content)

def download_all_pdbs(unseen, save_path, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Wrap the executor in tqdm for progress
        list(tqdm(executor.map(lambda pdb_id: download_pdb(pdb_id.strip(), save_path), unseen), desc='Download:', total=len(unseen)))


download_all_pdbs(download_set, save_path)