import pickle
from tqdm import tqdm


data_path = 'inference/progen2-small-surface/e8/inference_surface.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)
true_path = 'data/processed_surface_data/test/seq.txt'
with open(true_path, 'r') as f:
    true_seqs = f.readlines()
true_seqs = [seq.strip() for seq in true_seqs]

def calculate_recovery(original_seq: str, designed_seq: str):
    if len(original_seq) != len(designed_seq):
        raise ValueError("length is not match.")

    matches = sum(1 for orig, desig in zip(original_seq, designed_seq) if orig == desig)
    recovery = (matches / len(original_seq)) * 100
    return recovery

all_recovery = []
all_recovery_0_100 = []
all_recovery_100_300 = []
all_recovery_300_500 = []
for d in tqdm(zip(true_seqs, data), total=len(data)):
    true_seq, all_pred_seq = d
    recovery = []
    for pred_seq in all_pred_seq:
        if len(pred_seq) > len(true_seq):
            pred_seq = pred_seq[:len(true_seq)]
        elif len(pred_seq) < len(true_seq):
            add = 'X' * (len(true_seq) - len(pred_seq))
            pred_seq += add
        recovery.append(calculate_recovery(true_seq, pred_seq))
    recovery = sum(recovery) / len(recovery)
    all_recovery.append(recovery)
    if len(true_seq) < 100:
        all_recovery_0_100.append(recovery)
    elif 100 <= len(true_seq) < 300:
        all_recovery_100_300.append(recovery)
    elif 300 <= len(true_seq) < 500:
        all_recovery_300_500.append(recovery)

print(f'recovery_0_100:{sum(all_recovery_0_100)/len(all_recovery_0_100)}')
print(f'recovery_100_300:{sum(all_recovery_100_300)/len(all_recovery_100_300)}')
print(f'recovery_300_500:{sum(all_recovery_300_500)/len(all_recovery_300_500)}')
print(f'recovery:{sum(all_recovery)/len(all_recovery)}')
