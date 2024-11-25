import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from data.util import process_line, load_model
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

model_path = 'ckpt/coords_encoder'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path = '/home/v-yantingli/pdb_mmcif/mmcif_files'
cif_list = os.listdir(data_path)
data_list_path = 'data/data_list/train&valid'
data_list = sorted(os.listdir(data_list_path))
valid_set_num = [5] # valid_set_num[i] is between 0-9
save_path = 'data/processed_data_new'
os.makedirs(save_path,exist_ok=True)

# Load data list
valid_data_list = []

for num in valid_set_num:
    valid_data_list.append(data_list[num])
train_data_list = data_list
for valid_set in valid_data_list:
    train_data_list.remove(valid_set)
    
train_lines, valid_lines = [], []

for file_name in train_data_list:
    with open(os.path.join(data_list_path, file_name), 'r') as file:
        for line in file:
            train_lines.append(line.strip())
print(f'train data length: {len(train_lines)}')

for file_name in valid_data_list:
    with open(os.path.join(data_list_path, file_name), 'r') as file:
        for line in file:
            valid_lines.append(line.strip())
print(f'valid data length: {len(valid_lines)}')

# Check if all raw data is download
unseen = []
for line in train_lines+valid_lines:
    expected_cif = f"{line.split(' ')[0][:4]}.cif"
    if expected_cif not in cif_list:
        unseen.append(line.split(' ')[0][:4])
assert not unseen, f'There are several file you have not downloaded: {unseen}'

# Load esm2 to get coords embeddings
model = load_model(model_path, device=device)
print(f'model loaded on {device}')

# Use multiprocess
def process_data(lines: list[str], data_path: str, use: str):
    processed_data, unprocessed_data = [], []
    with ProcessPoolExecutor(max_workers=None) as executor:  # when max_workers=None will do os.cpu_count()
        futures = [executor.submit(process_line, line, data_path) for line in lines]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {use} Data"):
            temp_list, unprocessed = future.result()
            if temp_list:
                processed_data.extend(temp_list)
            if unprocessed:
                unprocessed_data.extend(unprocessed)
    return processed_data, unprocessed_data

train_data, unprocessed_train = process_data(train_lines, data_path, 'Training')
valid_data, unprocessed_valid = process_data(valid_lines, data_path, 'Validation')

# Add the coords rep into dict
def get_rep(data: list[dict]):
    for d in tqdm(data, desc='Processing coords rep'): 
        coords_rep = model.encode(d['coords'],device=device)
        d['rep'] = coords_rep.detach().to('cpu')

get_rep(train_data)
get_rep(valid_data)

# Save processed data
with open(os.path.join(save_path, 'train.pkl'), 'wb') as f:
    pickle.dump(train_data, f)
print('Training data saved')
 
with open(os.path.join(save_path, 'valid.pkl'), 'wb') as f:
    pickle.dump(valid_data, f)
print('Validation data saved')

with open(os.path.join(save_path, 'unprocessed_train.pkl'), 'wb') as f:
    pickle.dump(unprocessed_train, f)
with open(os.path.join(save_path, 'unprocessed_valid.pkl'), 'wb') as f:
    pickle.dump(unprocessed_valid, f)
print('Unprocessed list saved')
