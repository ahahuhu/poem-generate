import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from tqdm import tqdm
from transformers import BertTokenizer
from einops import rearrange

from datasets import (
  PoemDataset,
)
from model.poem_gpt import PoemGPT

from transformers import AdamW


def save_model(model, optimizer, args, filepath):
    save_info = {
      'model': model.state_dict(),
      'optim': optimizer.state_dict(),
      'args': args,
      'system_rng': random.getstate(),
      'numpy_rng': np.random.get_state(),
      'torch_rng': torch.random.get_rng_state(),
    }
    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'val-batch:{num_batches}'):
            b_ids, b_mask = batch["token_ids"], batch["attention_mask"]
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            logits = model(b_ids, b_mask)
            logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')
            labels = b_ids[:, 1:].contiguous().flatten() 
            loss = F.cross_entropy(logits, labels, reduction='mean')
            total_loss += loss.item()
            num_batches += 1
    return total_loss/num_batches

def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    poem_dataset = PoemDataset(args.poem_path, args.model_name)
    poem_dataloader = DataLoader(poem_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=poem_dataset.collate_fn)

    args = add_arguments(args)
    model = PoemGPT(args)
    model = model.to(device)
  
    optimizer = AdamW(model.parameters(), lr=args.lr)
  
    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
  
        for batch in tqdm(poem_dataloader, desc=f'train-{epoch}', disable=False):
            # Get the input and move it to the gpu (I do not recommend training this model on CPU).
            b_ids, b_mask = batch['token_ids'], batch['attention_mask']
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            # Compute the loss, gradients, and update the model's parameters.
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            # 忽略最后一个预测出来的token
            logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d') 
            # 忽略第一个token
            labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
            loss = F.cross_entropy(logits, labels, reduction='mean')
            loss.backward()

            # Apply gradient clipping
            max_norm = 10.0  # Set the maximum norm for the gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
    
        train_loss = train_loss / num_batches
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
  
    save_model(model, optimizer, args, f'{args.epochs}_{args.filepath}')
      
def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--poem_path", type=str, default="data/clear_data.npy")

  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.", default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
  parser.add_argument("--model_name", type=str, help="The model size as specified on hugging face.", default='uer/gpt2-chinese-cluecorpussmall')

  args = parser.parse_args()
  return args


def add_arguments(args):
    if args.model_name == 'uer/gpt2-chinese-cluecorpussmall':
        args.d = 768
        args.l = 12
        args.num_heads = 12
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
    train(args)