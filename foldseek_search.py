import subprocess
import os
from tqdm import tqdm


with open('results/split/valid_248', 'r') as f:
    test_248 = [os.path.join('data/test_data',line.strip()) for line in f.readlines()]
with open('results/split/valid_hme', 'r') as f:
    test_493 = [os.path.join('data/test_data',line.strip()) for line in f.readlines()]

for pdb_path in tqdm(test_248):
    output_file = os.path.basename(pdb_path).replace(".pdb", "") + ".m8"
    command1 = [
        'foldseek', 'easy-search',
        pdb_path,
        'foldseek_db/train_7k_db',
        f'results/foldseek_search/test248_train7k/{output_file}',
        'tmpFolder'
    ]
    result1 = subprocess.run(command1, check=True, capture_output=True, text=True)
    command2 = [
        'foldseek', 'easy-search',
        pdb_path,
        'foldseek_db/train_80k_db',
        f'results/foldseek_search/test248_train80k/{output_file}',
        'tmpFolder'
    ]
    result2 = subprocess.run(command2, check=True, capture_output=True, text=True)

for pdb_path in tqdm(test_493):
    output_file = os.path.basename(pdb_path).replace(".pdb", "") + ".m8"
    command1 = [
        'foldseek', 'easy-search',
        pdb_path,
        'foldseek_db/train_29k_db',
        f'results/foldseek_search/test493_train29k/{output_file}',
        'tmpFolder2'
    ]
    result1 = subprocess.run(command1, check=True, capture_output=True, text=True)
    command2 = [
        'foldseek', 'easy-search',
        pdb_path,
        'foldseek_db/train_4m_db',
        f'results/foldseek_search/test493_train4m/{output_file}',
        'tmpFolder'
    ]
    result2 = subprocess.run(command2, check=True, capture_output=True, text=True)

