# -*- coding: utf-8 -*-
import os
import re
import gc
import json
import math
import copy
import random
import pickle
import argparse
from typing import List, Tuple
import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import transformer_decoder
import utils

parser = argparse.ArgumentParser('Iterative Transformer training (checkpoint-only)')
parser.add_argument('--logdir', default='logdir', help='Folder to store everything/load')

parser.add_argument('--dim', default=3, type=int, help='Dimension of the chaotic systems')
parser.add_argument('--input-size', default=4, type=int, help='Transformer input dimension (3 states + k)')
parser.add_argument('--output-size', default=3, type=int, help='Transformer output dimension')
parser.add_argument('--hidden-size', default=256, type=int, help='Transformer hidden layer dimension')
parser.add_argument('--nhead', default=4, type=int, help='Transformer number of heads')
parser.add_argument('--num-layers', default=4, type=int, help='Transformer number of layers')
parser.add_argument('--d-model', default=128, type=int, help='Transformer projection dimension')
parser.add_argument('--dropout', default=0.2, type=float, help='Transformer drop out ratio')
parser.add_argument('--noise-level', default=0.05, type=float, help='Noise level added to the training data (only on states)')

parser.add_argument('--sequence-length', default=512, type=int, help='Input sequence length')
parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
parser.add_argument('--num-epochs', default=100, type=int, help='Epochs per iteration')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')

parser.add_argument('--iter-start', default=0, type=int, help='First iteration index (inclusive)')
parser.add_argument('--iter-end', default=12, type=int, help='Last iteration index (exclusive)')
parser.add_argument('--base-seed', default=1234, type=int, help='Base random seed; per-iteration seed = base + iter_index')

parser.add_argument('--save-dir', default='./save_models', type=str, help='Directory to write model checkpoints')
parser.add_argument('--data-dir', default='./save_data', type=str, help='Directory containing {foodchain_k*.pkl}')

args = parser.parse_args()
print(args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.save_dir, exist_ok=True)

torch.cuda.empty_cache(); gc.collect()

data_read_length = 200000

system = 'foodchain'     # used in file names previously; not embedded here to keep filenames simple

ks = [0.97, 0.98, 0.99]
ks_length = len(ks)

system_list = [f'foodchain_k{k}' for k in ks]
train_set = copy.deepcopy(system_list)

interval = round(args.sequence_length / 50)
print('interval', interval)

_k_rx = re.compile(r'k([0-9.]+)')

def parse_k_from_name(name: str) -> float:
    m = _k_rx.search(name)
    if not m:
        raise ValueError(f'Cannot parse k from name: {name}')
    return float(m.group(1))

def prepare_full_sequences(data: np.ndarray, seq_length: int, interval: int) -> np.ndarray:
    seqs = []
    N = len(data)
    stop = N - seq_length
    for i in range(0, stop, interval):
        seqs.append(data[i:i+seq_length])
    if not seqs:
        return np.empty((0, seq_length, data.shape[-1]), dtype=data.dtype)
    return np.stack(seqs, axis=0)

raw_data = {}
for sysname in train_set:
    fp = os.path.join(args.data_dir, f'{sysname}.pkl')
    with open(fp, 'rb') as fh:
        bundle = pickle.load(fh)
    traj = np.asarray(bundle['traj'])

    if len(traj) < data_read_length + 20000:
        start = max(0, len(traj) - data_read_length)
    else:
        start = random.randint(10000, len(traj) - data_read_length - 10000)
    raw_data[sysname] = traj[start:start + data_read_length, :]

train_data_raw = {}
for sysname in train_set:
    full = raw_data[sysname]
    split_point = len(full) - 1000 - args.sequence_length
    train_data_raw[sysname] = full[:split_point]

train_sequences = []
train_k_channels = []
for sysname in train_set:
    seqs = prepare_full_sequences(train_data_raw[sysname], args.sequence_length + 1, interval)  # (N, L+1, 3)
    k_val = parse_k_from_name(sysname)
    k_seq = np.full((seqs.shape[0], args.sequence_length, 1), k_val, dtype=np.float32)

    train_sequences.append(seqs)
    train_k_channels.append(k_seq)

if len(train_sequences) == 0:
    raise RuntimeError('No training sequences were generated. Check data sizes and --sequence-length/interval.')

train_sequences = np.concatenate(train_sequences, axis=0)   # (N, L+1, 3)
train_k_channels = np.concatenate(train_k_channels, axis=0) # (N, L, 1)

train_inputs_states = train_sequences[:, :-1, :]             # (N, L, 3)
train_targets = train_sequences[:, 1:, :]                    # (N, L, 3)
train_inputs = np.concatenate([train_inputs_states, train_k_channels], axis=-1)  # (N, L, 4)

train_dataset = TensorDataset(
    torch.tensor(train_inputs, dtype=torch.float32),
    torch.tensor(train_targets, dtype=torch.float32)
)
    
def run_single_training(iter_index: int):
    seed = args.base_seed + int(iter_index)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    model = transformer_decoder.TimeSeriesTransformer(
        args.input_size, args.output_size, args.d_model,
        args.nhead, args.num_layers, args.hidden_size, args.dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # simple training loop (no logging/plots)
    for epoch in range(args.num_epochs):
        model.train()
        losses = []
        # for inputs, targets in tqdm.tqdm(train_loader):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            noisy_inputs = inputs.clone()
            if args.noise_level > 0:
                noise = torch.normal(0.0, args.noise_level, size=inputs[:, :, :3].shape, device=inputs.device)
                noisy_inputs[:, :, :3] = inputs[:, :, :3] + inputs[:, :, :3] * noise

            outputs = model(noisy_inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        # step LR on epoch mean loss
        if losses:
            scheduler.step(float(np.mean(losses)))
    # Save checkpoint only (no results/figures)
    ckpt_path = os.path.join(args.save_dir, f'model_foodchain_iter_{iter_index:02d}.pth')
    torch.save(model.state_dict(), ckpt_path)
    print(f'[iter {iter_index}] saved -> {ckpt_path}')


# -------------------------
# Main: iterate
# -------------------------
if __name__ == '__main__':
    start = int(args.iter_start)
    end = int(args.iter_end)
    if end <= start:
        raise ValueError('--iter-end must be greater than --iter-start')

    print(f'Running iterations [{start}, {end}) ...')
    for it in range(start, end):
        run_single_training(it)
    print('All iterations completed.')
