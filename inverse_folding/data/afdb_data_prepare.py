import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.util import process_afdb, load_model, obj2bstr
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import gc
import lmdb
import zlib
warnings.filterwarnings("ignore")
gc.collect()

raw_data_path = '/home/v-yantingli/mmp/afdb_v4'
cif_list = os.listdir(raw_data_path)
# cif_list = cif_list[:1000]  # for test
save_path = '/home/v-yantingli/mmp/data/afdb_data/afdb.lmdb'
env = lmdb.open(save_path, map_size=8*1024**4)

# Use multiprocess
def process_data(lines: list[str], raw_data_path: str, use: str):
    processed_data = []
    with ProcessPoolExecutor(max_workers=None) as executor:  # when max_workers=None will do os.cpu_count()
        futures = [executor.submit(process_afdb, line, raw_data_path) for line in lines]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {use} Data"):
            temp_dict = future.result()
            if temp_dict:
                processed_data.append(temp_dict)
    return processed_data


batch_size = 100000
for i in range(0, len(cif_list), batch_size):
    batch = cif_list[i:i+batch_size]
    train_data = process_data(batch, raw_data_path, f'{i//batch_size+1}/{len(cif_list)//batch_size+1}')
    with env.begin() as txn:
        with txn.cursor() as curs:
            datapoint_pickled = curs.get('__metadata__'.encode())
            if datapoint_pickled:
                keys = pickle.loads(zlib.decompress(datapoint_pickled))['keys']
            else:
                keys = []
    with env.begin(write=True) as txn:
        for idx, data_dict in enumerate(tqdm(train_data, desc='Saving data')):
            key = f'{i//batch_size}_{idx}'
            keys.append(key)
            key = key.encode() 
            value = obj2bstr(data_dict)
            txn.put(key, value)

    with env.begin(write=True) as txn:
        txn.put('__metadata__'.encode(), obj2bstr({'keys':keys}))

    print(f'Training data saved for batch {i//batch_size}')
    del train_data
    # for file in tqdm(batch):
    #     os.remove(os.path.join(raw_data_path, file))
    gc.collect()
    