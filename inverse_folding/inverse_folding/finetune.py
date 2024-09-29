import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import argparse
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from tqdm import tqdm
import logging
from typing import List, Tuple
from models.progen import ProGenForCausalLM
import wandb


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class Protein_dataset(Dataset):
    def __init__(self, lines, tokenizer: Tokenizer, filter_seq_len: int):
        self.lines = [line for line in lines if len(line['seq']) < filter_seq_len]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        item = dict()
        line = self.lines[idx]
        item['seq_len'] = len(line['seq'])
        item['input_rep'] = line['rep']
        seq = line['seq']
        seq = torch.tensor(self.tokenizer.encode(f'1{seq}2').ids)
        sec_struc = ''.join([f'<{s}>' for s in line['sec_struc']])
        sec_struc =  torch.tensor(self.tokenizer.encode(sec_struc).ids)
        assert len(sec_struc) == len(line['seq'])
        item['input_ids'] = torch.cat((sec_struc, seq)).to(torch.int32)
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
        'seq_len': seq_len
    }


def load_data(file: str) -> Tuple[List[str], List[str]]:
    lines = []
    prefixes = set()
    with open(file, "rb") as f:
        lines = pickle.load(f)
        for line in lines:
            for s in line['sec_struc']:
                prefixes.add(f'<{s}>')
    prefixes = sorted(list(prefixes))
    return lines, prefixes


def init_new_embeddings(model: ProGenForCausalLM, prefixes: List[str]):
    if len(prefixes) <= 2:
        logger.info("No new embeddings to initialize.")
        return
    new_embs = torch.zeros((len(prefixes) - 2, model.config.n_embd)).to(model.transformer.wte.weight.device)

    unk_token_emb: torch.Tensor = model.transformer.wte.weight[-1].detach()
    mean_unk_emb = torch.zeros_like(new_embs) + unk_token_emb.mean()
    std_unk_emb = torch.zeros_like(new_embs) + unk_token_emb.std()

    # initialize new embeddings with normal distribution same as untrained embeddings
    torch.normal(mean_unk_emb, std_unk_emb, out=new_embs)
    new_embs = torch.cat([model.transformer.wte.weight, new_embs], dim=0)
    logger.debug(f"New embeddings shape: {new_embs.shape}")
    model.transformer.wte.weight = torch.nn.Parameter(new_embs, requires_grad=True)
    model.config.vocab_size = new_embs.shape[0]


def get_lr_schedule(
    optimizer: torch.optim.Optimizer, args: argparse.Namespace, train_steps: int
):
    if args.decay == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=train_steps,
        )
    elif args.decay == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=train_steps,
        )
    elif args.decay == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.9, last_epoch=-1
        )
    elif args.decay == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
        )
    else:
        raise ValueError(
            f"Invalid learning rate decay type. Must be 'cosine', 'linear', 'exponential', or 'constant'. Got: {args.decay}"
        )
    return scheduler


def train_epoch(
    model: ProGenForCausalLM,
    dataset: Protein_dataset,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    args: argparse.Namespace,
):
    model.train()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    total_loss = 0
    pbar = tqdm(total=len(dataloader) // args.accumulation_steps)
    batch: dict
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        loss: torch.Tensor = model(batch, labels=batch['label']).loss
        loss = loss / args.accumulation_steps
        loss.backward()
        total_loss = total_loss + loss.item()
        # using gradient accumulation to save memory
        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.update()
    pbar.close()
    logger.info(f"TRAIN epoch {epoch}: loss: {total_loss / len(dataloader)}")
    logger.debug(f"Last learning rate: {scheduler.get_last_lr()}")
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: ProGenForCausalLM,
    dataset: Protein_dataset,
    args: argparse.Namespace,
    before_train: bool = False,
):
    model.eval()
    total_loss = 0
    if before_train:
        # batch_size needs to be 1 so that we dont have different lengths of rows in the tensor
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    total_length = len(dataloader)
    pbar = tqdm(total=total_length)
    batch: torch.Tensor
    for batch in dataloader:
        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        loss: torch.Tensor = model(batch, labels=batch['label']).loss
        total_loss += loss.item()
        pbar.update()
    pbar.close()
    logger.info(f"EVAL loss: {total_loss / total_length}")
    return total_loss / total_length


def train(
    model: ProGenForCausalLM,
    tokenizer: Tokenizer,
    train_dataset: Protein_dataset,
    valid_dataset: Protein_dataset,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    args: argparse.Namespace,
):
    train_losses = []
    eval_losses = []
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Start time of epoch {epoch}: {datetime.now()}")
        train_loss = train_epoch(model, train_dataset, optimizer, scheduler, epoch, args)
        train_losses.append(train_loss)
        wandb.log({"epoch": epoch,
                   "train_loss": train_loss,
                   "learning rate": scheduler.get_last_lr()[0],
                  })
        if epoch % args.eval_steps == 0 or epoch == args.epochs:
            logger.info(f"Running test set evaluation after {epoch} epochs:")
            eval_loss = evaluate(model, valid_dataset, args)
            eval_losses.append(eval_loss)
            wandb.log({"eval_loss": eval_loss,
                      })
        model_name = args.model.strip(os.sep).split(os.sep)[-1]
        if epoch % args.checkpoint_steps == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join("ckpt", f"{model_name}-finetuned", f"e{epoch}")
            os.makedirs(checkpoint_path, exist_ok=True)
            
            model.save_pretrained(checkpoint_path)
            tokenizer.save(os.path.join(checkpoint_path, "tokenizer.json"), pretty=True)

            if args.save_optimizer:
                logger.info("Saving optimizer and scheduler...")
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))

            logger.info(f"Model saved at: {checkpoint_path}")
    return model, train_losses, eval_losses


def main(args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args_dict = vars(args)
    wandb.init(project="mmp-if")
    wandb.config.update(args_dict)

    # loading data and tokenizer
    def create_tokenizer_custom(file):
        with open(file, 'r') as f:
            return Tokenizer.from_str(f.read())
    tokenizer = create_tokenizer_custom(file='/home/v-yantingli/mmp/checkpoints/tokenizer.json')
    tokenizer.enable_truncation(max_length=1024)

    train_data, prefixes = load_data(args.train_file)
    valid_data, prefixes_test = load_data(args.test_file)
    logger.info(f"Found prefixes: {prefixes}")
    assert prefixes == prefixes_test, "Prefixes in train and test data must be the same"
    tokenizer.add_tokens(prefixes)

    train_data = Protein_dataset(train_data, tokenizer, filter_seq_len=512)
    valid_data = Protein_dataset(valid_data, tokenizer, filter_seq_len=512)
    logger.debug(f"Train data size: {len(train_data)}")
    logger.debug(f"Test data size: {len(valid_data)}")

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU. Please consider using a GPU for training.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    logger.debug(f"hyperparameters: effective batch={args.batch_size * args.accumulation_steps}, {args.batch_size=}, {args.accumulation_steps=}, {args.epochs=}, {args.lr=}, {args.warmup_steps=}, {args.checkpoint_steps=}")

    # loading model
    logger.info(f"Loading model: {args.model}...")
    if not args.model_parallel:
        model = ProGenForCausalLM.from_pretrained(args.model).to(device)
    else:
        model = ProGenForCausalLM.from_pretrained(args.model).parallelize()
    logger.info(f"Model loaded. Parameter count: {model.num_parameters() // 1e6} M")
    init_new_embeddings(model, prefixes)

    # creating optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    training_steps = (
        args.epochs * len(train_data) // (args.batch_size * args.accumulation_steps)
    )
    if training_steps > 0:
        logger.debug(f"Weight updates per epoch: {training_steps / args.epochs}")
    logger.debug(f"Total weight updates: {training_steps}")
    scheduler = get_lr_schedule(optimizer, args, training_steps)

    if args.eval_before_train:
        logger.info("Runnning evaluation on test set before training...")
        eval_loss_0 = evaluate(model, valid_data, args, before_train=True)
        wandb.log({"epoch": 0,
                    "eval_loss": eval_loss_0,
                        })
        

    # training loop
    model, train_losses, test_losses = train(
        model,
        tokenizer,
        train_data,
        valid_data,
        optimizer,
        scheduler,
        args,
    )

    logger.info("Finetuning finished.")
    logger.info(f"Train losses: {train_losses}")
    logger.info(f"Test losses: {test_losses}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="hugohrban/progen2-small",
        help="Name of the model checkpoint to be finetuned.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on. Default: \"cuda\"",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to training data file. Must contain preprocessed data (includes prefixes and one protein per line, e.g. not fasta format).",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to test data file. Must contain preprocessed data (includes prefixes and one protein per line, e.g. not fasta format).",
    )
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
        help="How many steps to accumulate gradients before updating weights. Default: 4",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate. Check out also the '--decay' argument. Default: 1e-4",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=200,
        help="Number of warmup steps for learning rate scheduler. Linearly increasing form 0 to --lr. Default: 200",
    )
    parser.add_argument(
        "--checkpoint_steps", type=int, default=5, help="Save model checkpoint every n epochs. Default: 5"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=1, help="Save model checkpoint every n epochs. Default: 5"
    )
    parser.add_argument(
        "--decay",
        type=str,
        choices=["cosine", "linear", "constant"],
        default="cosine",
        help="Learning rate decay. Default: \"cosine\"",
    )
    parser.add_argument(
        "--save_optimizer",
        action="store_true",
        default=False,
        help="Should we also save the optimizer and scheduler at every checkpoint",
    )
    parser.add_argument(
        "--eval_before_train",
        action="store_true",
        default=False,
        help="Run evaluation on test set before training. default: False",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging level.",
    )
    parser.add_argument(
        "--model_parallel",
        action="store_true",
        default=False,
        help="Train on multi gpu. default: False",
    )
    args = parser.parse_args()

    main(args)
