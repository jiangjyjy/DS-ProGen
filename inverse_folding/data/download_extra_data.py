import requests
import os
from tqdm import tqdm

data_list_path = '/home/v-yantingli/mmp/data/data_list/train&valid'
all_data_path = '/home/v-yantingli/pdb_mmcif/mmcif_files'
save_path = '/home/v-yantingli/mmp/data/extra_data'
os.makedirs(save_path,exist_ok=True)

# Load data list 
train_list = os.listdir(data_list_path)
train_lines = []
for file_name in train_list:
    with open(os.path.join(data_list_path, file_name), 'r') as file:
        for line in file:
            train_lines.append(line.strip())

# Search data not exsist
cif_list = os.listdir(all_data_path)
unseen = []
for line in train_lines:
    expected_cif = f"{line.split(' ')[0][:4]}.cif"
    if expected_cif not in cif_list:
        unseen.append(line.split(' ')[0][:4])

# Download cif
for line in tqdm(unseen, desc='Download:'):
    pdb_id = line.strip()
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    response = requests.get(url)
    
    # save cif
    with open(os.path.join(save_path, f"{pdb_id}.cif"), "wb") as cif_file:
        cif_file.write(response.content)