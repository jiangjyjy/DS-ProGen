import os
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
import pickle


class Surface_dataset(Dataset):
    def __init__(self, data_path, tokenizer: Tokenizer, filter_seq_len: int):
        self.data_path = data_path
        self.tokenizer = tokenizer
        seqs = self.read_seq_data(os.path.join(self.data_path, 'seq.txt'))
        aas, aa_ids = self.read_aa_data(os.path.join(self.data_path, 'atom.txt'))
        coords = self.read_coor_data(os.path.join(self.data_path, 'coor.txt'))
        first_ids = self.read_first_id(os.path.join(self.data_path, 'pdb.txt'))
        with open(os.path.join(self.data_path, 'rep.pkl'), 'rb') as f:
            rep = pickle.load(f)
        assert len(seqs) == len(aas) == len(aa_ids) == len(coords) == len(first_ids) == len(rep), f"len(seqs) {len(seqs)}, len(aas) {len(aas)}, len(aa_ids) {len(aa_ids)}, len(coords) {len(coords)}, len(first_ids) {len(first_ids)}, len(rep) {len(rep)}"
        for i in range(len(seqs)):
            assert aa_ids[i][0] >= first_ids[i], f"aa_ids[i][0] {aa_ids[i][0]} should be greater than or equal to first_ids[i] {first_ids[i]}"
            assert aa_ids[i][-1] < first_ids[i] + len(seqs[i]), f"aa_ids[i][-1] {aa_ids[i][-1]} should be less than first_ids[i] {first_ids[i]} + len(seqs[i]) {len(seqs[i])}"
            aa_ids[i] -= first_ids[i]
        self.lines = [(x, y, z, r, p) for x, y, z, r, p in zip(seqs, aas, coords, aa_ids, rep) if (len(x) <= filter_seq_len and len(x) == p.size(0))]

    def __len__(self):
        return len(self.lines)

    
    def read_aa_dict(self):
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        aa2index = dict()
        for aa in alphabet:
            aa2index[aa] = len(aa2index)
        return aa2index

    def read_aa_data(self, path):
        amino_acid_dict = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
                   "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
                   "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
        aa2index = self.read_aa_dict()
        tokens_list = []
        id_list = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                words = line.strip().split()
                aas = [amino_acid_dict[word.strip().split("_")[-3].strip()] for word in words]
                tokens = []
                for word in aas:
                    tokens.append(aa2index[word])
                tokens_list.append(torch.IntTensor(tokens))
                aa_id = [int(word.strip().split("_")[-2].strip()) for word in words]
                id_list.append(torch.IntTensor(aa_id))

        return tokens_list, id_list
    
    def read_coor_data(self, path):
        tokens_list = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                words = line.strip().split()
                tokens = []
                for i in range(0, len(words), 3):
                    coor = [float(words[i].strip()), float(words[i+1].strip()), float(words[i+2].strip())]
                    tokens.extend(coor)   # [L * 3]

                tokens_list.append(torch.tensor(tokens))
        return tokens_list

    def read_seq_data(self, path):
        tokens_list = []
        with open(path, "r", encoding="utf-8") as f:
            tokens_list = f.readlines()
        tokens_list = [x.strip() for x in tokens_list]         
        return tokens_list

    def read_first_id(self, path):
        tokens_list = []
        with open(path, "r", encoding="utf-8") as f:
            tokens_list = f.readlines()
        tokens_list = [int(x.strip().split('_')[-1]) for x in tokens_list]         
        return tokens_list

    
    def __getitem__(self, idx):
        item = dict()
        seq, aa, coord, aa_ids, rep = self.lines[idx]
        assert 3 * len(aa) == len(coord), f"len(aa) {len(aa)} should be equal to len(coord) {len(coord)}"
        item['aa_len'] = len(aa)
        item['seq_len'] = len(seq)
        item['input_aa'] = aa
        item['input_coord'] = coord
        item['aa_res_ids'] = aa_ids
        item['input_rep'] = rep
        seq_id = torch.tensor(self.tokenizer.encode(f'1{seq}2').ids)
        rep_x =  torch.zeros(len(seq))
        item['input_ids'] = torch.cat((rep_x, seq_id)).to(torch.int32)
        rep_mask = torch.ones(len(seq)+seq_id.shape[0])
        rep_mask[len(seq):] = 0
        item['input_rep_mask'] = rep_mask.to(torch.int32)
        label = item['input_ids'].clone()
        label[:len(seq)] = -100
        item['label'] = label.long()
        return item


def collate_features(samples):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        values = [s for s in samples]
        size = max(v.size(0) for v in values)

        batch_size = len(values)
        res = values[0].new(batch_size, size).fill_(0.0)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][: len(v)])
        return res


def collate_fn(batch):
    # Get all input_ids and input_rep_mask from the batch
    input_ids = [item['input_ids'] for item in batch]
    input_rep_mask = [item['input_rep_mask'] for item in batch]
    input_aa = [item['input_aa'] for item in batch]
    input_coord = [item['input_coord'] for item in batch]
    input_rep = [item['input_rep'].to('cpu') for item in batch]
    labels = [item['label'] for item in batch]
    aa_ids  = [item['aa_res_ids'] for item in batch]
    seq_len = torch.tensor([item['seq_len'] for item in batch])
    aa_len = torch.tensor([item['aa_len'] for item in batch])
    
    # Find the max length for padding
    max_len = max([x.size(0) for x in input_ids])
    max_rep_len = max([x.size(0) for x in input_rep])
    
    # Pad input_ids, input_rep_mask, and labels to the same length
    padded_input_ids = torch.stack([torch.cat([x, torch.zeros(max_len - x.size(0), dtype=torch.int32)]) for x in input_ids])
    padded_input_rep_mask = torch.stack([torch.cat([x, torch.zeros(max_len - x.size(0), dtype=torch.int32)]) for x in input_rep_mask])
    padded_labels = torch.stack([torch.cat([x, torch.full((max_len - x.size(0),), -100, dtype=torch.int32)]) for x in labels])
    padded_input_rep = torch.stack([torch.cat([x, torch.zeros(max_rep_len - x.size(0), x.size(1))]) for x in input_rep])
    padded_aa_ids = collate_features(aa_ids)
    padded_inout_aa = collate_features(input_aa)
    padded_input_coord = collate_features(input_coord)
    
    # Return the batch as a dictionary
    return {
        'input_ids': padded_input_ids,
        'input_rep_mask': padded_input_rep_mask,
        'input_rep': padded_input_rep,
        'input_aa': padded_inout_aa,
        'input_coord': padded_input_coord,
        'label': padded_labels,
        'seq_len': seq_len,
        'aa_len': aa_len,
        'aa_res_ids': padded_aa_ids

    }

