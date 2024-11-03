import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import logging
import torch
from tqdm import tqdm
import re
from typing import Optional, Union
from tokenizers import Tokenizer, Encoding
from models.surf_progen import ProGenSurfForCausalLM
from typing import List
from data.util import load_model
import pickle
import math

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def inference(
    model: ProGenSurfForCausalLM,
    tokenizer: Tokenizer,
    device: torch.device,
    aa: Optional[torch.Tensor],
    coord: Optional[torch.Tensor],
    seq_len,
    aa_res_id,
    max_length: int,
    num_return_sequences: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    sec_struc: Optional[list] = None
) -> List[str]:
    """
    Generate samples from the model given a prompt. Using top-k sampling with temperature.
    """
    model.eval()

    encoding: Encoding = tokenizer.encode('1')
    ids = torch.tensor(encoding.ids)                             # (T,)
    ids = ids[:len(torch.nonzero(ids))]
    rep_ids =  torch.zeros(seq_len)
    ids = torch.cat((rep_ids, ids))
    x = torch.zeros((num_return_sequences, ids.shape[0]))        # (B, T)
    x = x + ids
    x = x.to(device).to(torch.int32)
    
    past_key_values = None
    generated = x

    rep_mask = torch.zeros((num_return_sequences, ids.shape[0])) 
    rep_mask[:, :seq_len] = 1
    rep_mask = rep_mask.to(device).to(torch.int32)
    seq_len = torch.full((num_return_sequences,), seq_len).to(device).to(torch.int32)

    input_dict = {'input_ids': x,
                'input_aa': aa.unsqueeze(0).expand(num_return_sequences, -1, -1).to(device),
                'input_coord': coord.unsqueeze(0).expand(num_return_sequences, -1, -1).to(device),
                'input_rep_mask': rep_mask,
                'seq_len': seq_len,
                'aa_len': torch.full((num_return_sequences,), len(aa)).to(device).to(torch.int32),
                'aa_res_ids': aa_res_id.unsqueeze(0).expand(num_return_sequences, -1).to(device),
                }
    while generated.shape[-1] < min(seq_len[0]*2 + 1, max_length):
        # using cached attn outputs from previous iterations
        output = model(input_batch=input_dict, past_key_values=past_key_values)
        past_key_values = output.past_key_values
        logits = output.logits                                       # (B, T, V)
        # get logits only for the last token
        logits = logits[:, -1, :]                                    # (B, V)
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k, dim=-1)                 # (B, k)
            logits = torch.where(logits >= v[:, -1:], logits, torch.tensor(-1e9, dtype=torch.float).to('cuda'))  # (B, V)
        probs = torch.softmax(logits, dim=-1)                        # (B, V)
        x = torch.multinomial(probs, num_samples=1)                 # (B, 1)
        input_dict = {'input_ids': x}                  
        generated = torch.cat([generated, x], dim=-1)                # (B, T+1)

    decoded = [tokenizer.decode(row.detach().cpu().numpy().tolist()) for row in generated]
    return decoded


def truncate(seq: str) -> str:
    """
    Remove family special tokens, initial 1 or 2 token and truncate
    the sequence to the first 1 or 2 token found.
    
    Sequences begginning with 2 (C -> N generation) are reversed.
    """

    # remove family token
    seq = re.sub(r'<.*?>', "", seq)

    # remove initial terminus
    terminus = seq[0]
    seq = seq[1:]

    min_1 = seq.find("1")
    if min_1 == -1:
        min_1 = len(seq)

    min_2 = seq.find("2")
    if min_2 == -1:
        min_2 = len(seq)

    # truncate the sequence to next terminus token
    seq = seq[: min(min_1, min_2)]
    if terminus == "1":
        return seq
    else:
        return seq[::-1]


def read_aa_dict():
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        aa2index = dict()
        for aa in alphabet:
            aa2index[aa] = len(aa2index)
        return aa2index

def read_aa_data(path):
    amino_acid_dict = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
                "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
                "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
    aa2index = read_aa_dict()
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


def read_coor_data(path):
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

def read_seq_len(path):
        tokens_list = []
        with open(path, "r", encoding="utf-8") as f:
            tokens_list = f.readlines()
        tokens_list = [len(x.strip()) for x in tokens_list]         
        return tokens_list

def read_first_id(path):
        tokens_list = []
        with open(path, "r", encoding="utf-8") as f:
            tokens_list = f.readlines()
        tokens_list = [int(x.strip().split('_')[-1]) for x in tokens_list]         
        return tokens_list

def main(args):
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    if str(device) == "cpu" and args.batch_size > 1:
        logger.warning(f"You are using CPU for inference with a relatively high batch size of {args.batch_size}, therefore inference might be slow. Consider using a GPU or smaller batch.")

    logger.info(f"Loading model from {args.model}")
    model = ProGenSurfForCausalLM.from_pretrained(args.model).to(device)
    logger.debug("Model loaded.")

    logger.info("Loading tokenizer")
    def create_tokenizer_custom(file):
        with open(file, 'r') as f:
            return Tokenizer.from_str(f.read())
    tokenizer = create_tokenizer_custom(file=os.path.join(args.model,'tokenizer.json'))
    # tokenizer: Tokenizer = Tokenizer.from_pretrained(args.model)
    tokenizer.no_padding()
    logger.debug("Tokenizer loaded.")
    logger.debug(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    logger.debug(f"Tokenizer vocab: {tokenizer.get_vocab()}")

    aa_list, aa_id_list = read_aa_data(os.path.join(args.test_data_dir, 'atom.txt'))
    coord_list = read_coor_data(os.path.join(args.test_data_dir, 'coor.txt'))
    seq_len_list = read_seq_len(os.path.join(args.test_data_dir, 'seq.txt'))
    first_id_list = read_first_id(os.path.join(args.test_data_dir, 'pdb.txt'))
    for i in range(len(aa_id_list)):
        aa_id_list[i] -= first_id_list[i]

    output_dir = os.path.join("inference", args.model.split("/")[-2], args.model.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"inference_surface.pkl")

    if args.k == 0 or args.k > model.config.vocab_size:
        args.k = None

    logger.debug(f"Sampling parameters: top_k={args.k}, temperature={args.t}")

    pred_seq_list = []
    for i, d in tqdm(enumerate(zip(aa_list, aa_id_list, coord_list, seq_len_list)), total=len(aa_list)):
        aa, aa_res_id, coord, seq_len = d
        # sec_struc = d.get('sec_struc', None)
        sec_struc = None

        pred_seq = inference(
            model=model,
            tokenizer=tokenizer,
            device=device,
            aa=aa,
            coord=coord,
            seq_len=seq_len,
            aa_res_id=aa_res_id,
            num_return_sequences=args.batch_size,
            temperature=args.t,
            max_length=args.max_length,
            top_k=args.k,
            sec_struc=sec_struc
        )
        for j in range(len(pred_seq)):
            pred_seq[j] = truncate(pred_seq[j])
        pred_seq_list.append(pred_seq)
    with open(output_file, 'wb') as f:
        pickle.dump(pred_seq_list, f)
    logger.info(f"saved to file {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="/home/v-yantingli/mmp/checkpoints/progen2-small",
        help="Hugging Face model name or path to the model directory. If path, should contain tokenizer.json, config.json and pytorch_model.bin.",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=True,
        help="Path to test data file. Must contain preprocessed data.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="How many sequences to generate at one iteration. Default: 64",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum length of the generated sequence. Default: 1024",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=15,
        help="Top-k sampling parameter. 0 means no top-k sampling. Default: 15",
    )
    parser.add_argument(
        "--t", type=float, default=1.0, help="Temperature for sampling. Default: 1.0"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging level.")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    main(args)
    logger.info("Inference finished.")