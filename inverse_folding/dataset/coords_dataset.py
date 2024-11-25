import os
import pickle
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from typing import List, Tuple
import lmdb
import zlib
from data.util import load_model

class Protein_dataset(Dataset):
    def __init__(self, lines, tokenizer: Tokenizer, filter_seq_len: int, sec: bool):
        self.lines = [line for line in lines if len(line['seq']) < filter_seq_len]
        self.tokenizer = tokenizer
        self.sec = sec

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        item = dict()
        line = self.lines[idx]
        item['seq_len'] = len(line['seq'])
        item['input_rep'] = line['rep']
        seq = line['seq']
        seq = torch.tensor(self.tokenizer.encode(f'1{seq}2').ids)
        if self.sec:
            item['sec_struc'] = True
            sec_struc = ''.join([f'<{s}>' for s in line['sec_struc']])
            sec_struc =  torch.tensor(self.tokenizer.encode(sec_struc).ids)
            assert len(sec_struc) == len(line['seq'])
            item['input_ids'] = torch.cat((sec_struc, seq)).to(torch.int32)
        else:
            item['sec_struc'] = False
            rep_x =  torch.zeros(len(line['seq']))
            item['input_ids'] = torch.cat((rep_x, seq)).to(torch.int32)
        rep_mask = torch.ones(len(line['seq'])+seq.shape[0])
        rep_mask[len(line['seq']):] = 0
        item['input_rep_mask'] = rep_mask.to(torch.int32)
        label = item['input_ids'].clone()
        label[:len(line['seq'])] = -100
        item['label'] = label.long()
        return item


class Protein_Large_dataset(Dataset):
    def __init__(self, lmdb_path, tokenizer: Tokenizer, sec: bool):
        self.tokenizer = tokenizer
        self.sec = sec
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        with self.env.begin() as txn:  
            with txn.cursor() as curs:  
                datapoint_pickled = curs.get('__metadata__'.encode())
                keys = pickle.loads(zlib.decompress(datapoint_pickled))['keys']
        self.keys = keys[:4000000]
        self.rep_model = load_model('ckpt/coords_encoder','cuda')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            with txn.cursor() as curs:
                datapoint_pickled = curs.get(self.keys[idx].encode())
                line = pickle.loads(zlib.decompress(datapoint_pickled))
                item = dict()
                item['seq_len'] = len(line['seq'])
                coords_rep = self.rep_model.encode(line['coords'],device='cuda')
                item['input_rep'] = coords_rep.detach()
                seq = line['seq']
                seq = torch.tensor(self.tokenizer.encode(f'1{seq}2').ids)
                if self.sec:
                    item['sec_struc'] = True
                    sec_struc = ''.join([f'<{s}>' for s in line['sec_struc']])
                    sec_struc =  torch.tensor(self.tokenizer.encode(sec_struc).ids)
                    assert len(sec_struc) == len(line['seq'])
                    item['input_ids'] = torch.cat((sec_struc, seq)).to(torch.int32)
                else:
                    item['sec_struc'] = False
                    rep_x =  torch.zeros(len(line['seq']))
                    item['input_ids'] = torch.cat((rep_x, seq)).to(torch.int32)
                rep_mask = torch.ones(len(line['seq'])+seq.shape[0])
                rep_mask[len(line['seq']):] = 0
                item['input_rep_mask'] = rep_mask.to(torch.int32)
                label = item['input_ids'].clone()
                label[:len(line['seq'])] = -100
                item['label'] = label.long()
        return item
        


def collate_fn(batch):
    # Get all input_ids and input_rep_mask from the batch
    input_ids = [item['input_ids'] for item in batch]
    input_rep_mask = [item['input_rep_mask'] for item in batch]
    input_rep = [item['input_rep'].to('cpu') for item in batch]
    labels = [item['label'] for item in batch]
    seq_len = torch.tensor([item['seq_len'] for item in batch])
    sec_struc = torch.tensor(any([item['sec_struc'] for item in batch]))
    
    # Find the max length for padding
    max_len = max([x.size(0) for x in input_ids])
    max_rep_len = max([x.size(0) for x in input_rep])
    
    # Pad input_ids, input_rep_mask, and labels to the same length
    padded_input_ids = torch.stack([torch.cat([x, torch.zeros(max_len - x.size(0), dtype=torch.int32)]) for x in input_ids])
    padded_input_rep_mask = torch.stack([torch.cat([x, torch.zeros(max_len - x.size(0), dtype=torch.int32)]) for x in input_rep_mask])
    padded_labels = torch.stack([torch.cat([x, torch.full((max_len - x.size(0),), -100, dtype=torch.int32)]) for x in labels])
    padded_input_rep = torch.stack([torch.cat([x, torch.zeros(max_rep_len - x.size(0), x.size(1))]) for x in input_rep])
    
    # Return the batch as a dictionary
    return {
        'input_ids': padded_input_ids,
        'input_rep_mask': padded_input_rep_mask,
        'input_rep': padded_input_rep,
        'label': padded_labels,
        'seq_len': seq_len,
        'sec_struc': sec_struc
    }


def load_data(file: str, sec: bool) -> Tuple[List[str], List[str]]:
    lines = []
    prefixes = set()
    with open(file, "rb") as f:
        lines = pickle.load(f)
        if sec:
            for line in lines:
                for s in line['sec_struc']:
                    prefixes.add(f'<{s}>')
    if sec:
        prefixes = sorted(list(prefixes))
    else:
        prefixes = []
    return lines, prefixes