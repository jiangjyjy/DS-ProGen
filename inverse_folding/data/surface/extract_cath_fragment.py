from biotite.structure.io.pdb import PDBFile
from biotite.structure.io import pdbx, pdb
from biotite.structure import get_chains
from typing import List
from CifFile import ReadCif
import os
# from util import get_res_id
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

data_list_path = 'data/data_list/train&valid'
data_list = os.listdir(data_list_path)
data_path = 'data/cath43_data'
save_path = 'data/cath43_filter_pdb'
lines = []
for file_name in data_list:
    with open(os.path.join(data_list_path, file_name), 'r') as file:
        for line in file:
            lines.append(line.strip())


def get_res_id(fpath: str, chain_ids: List[str], start_pos, end_pos):
    id = os.path.basename(fpath).replace('.cif', '')
    cif_data = ReadCif(fpath)
    chain_list = cif_data[id]['_pdbx_poly_seq_scheme.pdb_strand_id']
    res_id_list = cif_data[id]['_pdbx_poly_seq_scheme.pdb_seq_num']
    seq_ld_list = cif_data[id]['_pdbx_poly_seq_scheme.seq_id']
    seq_num = []
    start_id, end_id = None, None
    for c, r, s in zip(chain_list, res_id_list, seq_ld_list):
        if c in chain_ids:
            seq_num.append((int(r),int(s)))
            if int(s) == start_pos:
                start_id = int(r)
            elif int(s) == end_pos:
                end_id = int(r)
    start_id = min(seq_num) if start_id is None else start_id
    end_id = max(seq_num) if end_id is None else end_id
    
    return start_id, end_id


def load_structure_all(fpath, chain=None, start_pos=None, end_pos=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if fpath.endswith('cif'):
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    elif fpath.endswith('pdb'):
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain] 
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    if start_pos is not None or end_pos is not None:
        start_id, end_id = get_res_id(fpath, chain_ids, start_pos, end_pos)
        res_fliter = [start_id<=a.res_id<=end_id for a in structure]
        structure = structure[res_fliter]
    return structure


def line_to_pdb(line, data_path):
    line_elements = line.split(' ')
    cif = f'{line_elements[0][:4]}.cif'
    fpath = os.path.join(data_path, cif)
    chain_id = line_elements[2].split('.')[0]
    all_pos = line_elements[2].split('.')[1].split(',')
    for i, p in enumerate(all_pos):
        start_pos, end_pos = p.split('-')
        start_pos, end_pos = int(start_pos), int(end_pos)
        try:
            structure = load_structure_all(fpath, chain_id, start_pos, end_pos)
            if structure:
                pdb_file = PDBFile()
                pdb_file.set_structure(structure)
                pdb_file.write(os.path.join(save_path, f"{line_elements[0]}_{i}.pdb"))
        except:
            pass


def process_data(lines: list[str], data_path: str, use: str):
    with ProcessPoolExecutor(max_workers=None) as executor:  # when max_workers=None will do os.cpu_count()
        futures = [executor.submit(line_to_pdb, line, data_path) for line in lines]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {use} Data"):
            pass

process_data(lines, data_path, '')



