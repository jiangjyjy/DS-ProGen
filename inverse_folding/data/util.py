import biotite.structure
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_backbone
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
from typing import List
import numpy as np
import re
from CifFile import ReadCif
import os
import json
import torch
import esm
from argparse import Namespace
from models.coords_encoder import CoordsEncoder
from Bio import PDB


_canonical_aa_list = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS",
                      "ILE","LEU","LYS","MET","PHE","PRO","PYL","SER","THR",
                      "TRP","TYR","VAL", "SEC"]

def load_structure(fpath, chain=None, start_pos=None, end_pos=None):
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
    bbmask = filter_backbone(structure)
    structure = structure[bbmask]
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
        if len(structure) != 3*(end_id-start_id+1):
            return None
    return structure


def extract_coords_from_structure(structure: biotite.structure.AtomArray):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    coords = get_atom_coords_residuewise(["N", "CA", "C"], structure)
    residue_identities = get_residues(structure)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r) if r in _canonical_aa_list else 'X' for r in residue_identities])
    return coords, seq


def get_atom_coords_residuewise(atoms: List[str], struct: biotite.structure.AtomArray):
    """
    Example for atoms argument: ["N", "CA", "C"]
    """
    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        sum = filters.sum(0)
        if not np.all(sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[sum == 0] = float("nan")
        return coords

    return biotite.structure.apply_residue_wise(struct, struct, filterfn)

def find_missing_res_id(fpath: str):
    missing_residues = []

    with open(fpath, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("REMARK 465"):
            match = re.search(r"(\w{3})\s+(\w)\s+(\d+)", line)
            if match:
                residue_name = match.group(1)
                chain_id = match.group(2)
                residue_number = match.group(3)
                missing_residues.append((residue_name, chain_id, residue_number))

    return missing_residues

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


def process_line(line, cif_list2, extra_data_path, raw_data_path):
    line_elements = line.split(' ')
    cif = f'{line_elements[0][:4]}.cif'
    if cif in cif_list2:
        fpath = os.path.join(extra_data_path, cif)
    else:
        fpath = os.path.join(raw_data_path, cif)
    chain_id = line_elements[2].split('.')[0]
    all_pos = line_elements[2].split('.')[1].split(',')
    temp_list, unprocessed = [], []
    for p in all_pos:
        start_pos, end_pos = p.split('-')
        start_pos, end_pos = int(start_pos), int(end_pos)
        try:
            structure = load_structure(fpath, chain_id, start_pos, end_pos)
            sec_structure = get_sec_structure(fpath, chain_id, start_pos, end_pos)
            if len(sec_structure)*3 != len(structure):
                continue
            if not structure:
                continue
            coords, native_seq = extract_coords_from_structure(structure)
            line_elements.append((start_pos, end_pos))
            temp_dict = {
                'seq': native_seq,
                'coords': coords,
                'info': line_elements,
                'sec_struc': sec_structure
            }
            temp_list.append(temp_dict)
        except:
            unprocessed.append(line)

    return temp_list, unprocessed

def load_model(path: str, device: str = 'cpu'):
    config_path = os.path.join(path, 'config.json')
    model_weights_path = os.path.join(path, 'pytorch_model.pt')
    with open(config_path, 'r') as json_file:
        model_args = json.load(json_file)
    alphabet = esm.Alphabet.from_architecture(model_args["arch"])
    model = CoordsEncoder(Namespace(**model_args), alphabet)
    model.to(device)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()
    return model

def get_sec_structure(fpath, chain=None, start_pos=None, end_pos=None):
    if fpath.endswith('cif'):
        parser = PDB.MMCIFParser()
    elif fpath.endswith('pdb'):
        parser = PDB.PDBParser()
    structure = parser.get_structure("structure", fpath)
    dssp = PDB.DSSP(structure[0], fpath)
    if start_pos is not None or end_pos is not None:
        start_id, end_id = get_res_id(fpath, chain, start_pos, end_pos)
        for i, k in enumerate(dssp.keys()):
            if k[1][1] == start_id and k[0] == chain:
                i_s = i
            elif k[1][1] == end_id and k[0] == chain:
                i_e = i
        sec_sturc = []
        for i, res in enumerate(dssp):
            if i_s <= i <= i_e:
                sec_sturc.append(res[2])
    else:
        sec_sturc = [res[2] for res in dssp]
    return sec_sturc

def add_cryst1_if_missing(pdb_file):
    with open(pdb_file, 'r') as file:
        lines = file.readlines()

    has_cryst1 = any(line.startswith('CRYST1') for line in lines)

    if not has_cryst1:
        cryst1_line = "CRYST1\n"
        lines.insert(0, cryst1_line)

        with open(pdb_file, 'w') as file:
            file.writelines(lines)
