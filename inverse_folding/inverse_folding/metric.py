import pickle
from tqdm import tqdm


data_path = '/home/v-yantingli/mmp/inference/progen2-small-onlycoords/e5/inference_test_hm.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

def calculate_recovery(original_seq: str, designed_seq: str):
    if len(original_seq) != len(designed_seq):
        raise ValueError("length is not match.")

    matches = sum(1 for orig, desig in zip(original_seq, designed_seq) if orig == desig)
    recovery = (matches / len(original_seq)) * 100
    return recovery

all_recovery = []
for d in tqdm(data):
    true_seq = d['seq']
    recovery = 0
    for pred_seq in d['pred_seq']:
        if len(pred_seq) > len(true_seq):
            pred_seq = pred_seq[:len(true_seq)]
        elif len(pred_seq) < len(true_seq):
            add = 'X' * (len(true_seq) - len(pred_seq))
            pred_seq += add
        recovery += calculate_recovery(true_seq, pred_seq)
    recovery = recovery/len(d['pred_seq'])
    all_recovery.append(recovery)

print(f'recovery:{sum(all_recovery)/len(all_recovery)}')

