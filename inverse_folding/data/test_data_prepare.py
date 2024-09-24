import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from data.util import load_structure, extract_coords_from_structure, load_model
from tqdm import tqdm
import pickle


model_path = '/home/v-yantingli/mmp/ckpt/coords_encoder'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
raw_data_path = '/home/v-yantingli/mmp/data/test_data'
cif_list = os.listdir(raw_data_path)
data_list_path = '/home/v-yantingli/mmp/data/data_list/test'
data_list = os.listdir(data_list_path)
save_path = '/home/v-yantingli/mmp/data/processed_data'
os.makedirs(save_path,exist_ok=True)


# Load esm2 to get coords embeddings
model = load_model(model_path, device=device)
print(f'model loaded on {device}')

for f in data_list:
    file_path = os.path.join(data_list_path, f)
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line.strip())
    save_name = f"test_{f.split('_')[-1]}.pkl"
    processed_data = []
    for line in tqdm(lines, desc=f'Processing {save_name}'):
        fpath = os.path.join(raw_data_path, f'{line}_native.pdb')
        structure = load_structure(fpath)
        coords, native_seq = extract_coords_from_structure(structure)
        coords_rep = model.encode(coords, device=device)
        temp_dict = {
            'seq': native_seq,
            'coords': coords,
            'rep': coords_rep.detach().to('cpu')
        }
        processed_data.append(temp_dict)
    with open(os.path.join(save_path, save_name), 'wb') as f:
        pickle.dump(processed_data, f)
    print(f'{save_name} saved')