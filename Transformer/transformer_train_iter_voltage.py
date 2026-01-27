# -*- coding: utf-8 -*-
import os
import re
import gc
import glob
import copy
import math
import json
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

parser = argparse.ArgumentParser('Voltage Transformer (iteration saver, no eval)')
parser.add_argument('--logdir', default='logdir', help='Folder to store logs (not heavily used here)')

parser.add_argument('--dim', default=4, type=int, help='Number of state dimensions in voltage data')
parser.add_argument('--input-size', default=5, type=int, help='Transformer input dimension (4 states + Q1)')
parser.add_argument('--output-size', default=4, type=int, help='Transformer output dimension (states only)')
parser.add_argument('--hidden-size', default=256, type=int, help='Transformer hidden layer dimension')
parser.add_argument('--nhead', default=4, type=int, help='Transformer number of heads')
parser.add_argument('--num-layers', default=4, type=int, help='Transformer number of layers')
parser.add_argument('--d-model', default=128, type=int, help='Transformer projection dimension')
parser.add_argument('--dropout', default=0.2, type=float, help='Transformer dropout ratio')
parser.add_argument('--noise-level', default=0.05, type=float, help='Multiplicative noise level added to state dims during training')

parser.add_argument('--sequence-length', default=512, type=int, help='Input sequence length')
parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
parser.add_argument('--num-epochs', default=100, type=int, help='Epochs per iteration')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')

parser.add_argument('--iter-start', default=0, type=int, help='First iteration index (inclusive)')
parser.add_argument('--iter-end', default=12, type=int, help='Last iteration index (exclusive)')
parser.add_argument('--base-seed', default=1234, type=int, help='Base random seed; per-iteration seed = base + iter')

parser.add_argument('--save-dir', default='./save_models', type=str, help='Directory to write model checkpoints')
parser.add_argument('--data-dir', default='./save_data', type=str, help='Directory containing voltage_Q1*.pkl')

parser.add_argument('--data-read-length', default=200000, type=int, help='Random contiguous segment length per system')
args = parser.parse_args()
print(args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.save_dir, exist_ok=True)

torch.cuda.empty_cache(); gc.collect()


Q1s = [2.989680, 2.989730, 2.989780]
system_list = []
for Q1 in Q1s:
    system_list.append(f'voltage_Q1{Q1:.6f}')

interval = round(args.sequence_length / 50)
print('interval', interval)

_k_rx = re.compile(r'Q1([0-9.]+)')

def parse_Q1_from_name(name: str) -> float:
    m = _k_rx.search(name)
    if not m:
        raise ValueError(f'Cannot parse Q1 from: {name}')
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
for sysname in system_list:
    fp = os.path.join(args.data_dir, f'{sysname}.pkl')
    with open(fp, 'rb') as fh:
        bundle = pickle.load(fh)
    traj = np.asarray(bundle['traj'])

    if len(traj) < args.data_read_length + 20000:
        start = max(0, len(traj) - args.data_read_length)
    else:
        start = random.randint(10000, len(traj) - args.data_read_length - 10000)
    raw_data[sysname] = traj[start:start + args.data_read_length, :]

train_data_raw = {}
for sysname in system_list:
    full = raw_data[sysname]
    split_point = len(full) - 1000 - args.sequence_length
    train_data_raw[sysname] = full[:split_point]

train_sequences = []
train_Q1_channels = []
for sysname in system_list:
    seqs = prepare_full_sequences(train_data_raw[sysname], args.sequence_length + 1, interval)  # (N, L+1, dim)
    Q1_val = parse_Q1_from_name(sysname)
    q1_seq = np.full((seqs.shape[0], args.sequence_length, 1), Q1_val, dtype=np.float32)

    train_sequences.append(seqs)
    train_Q1_channels.append(q1_seq)

if len(train_sequences) == 0:
    raise RuntimeError('No training sequences were generated. Check data sizes and --sequence-length/interval.')

train_sequences = np.concatenate(train_sequences, axis=0)   # (N, L+1, dim)
train_Q1_channels = np.concatenate(train_Q1_channels, axis=0) # (N, L, 1)

train_inputs_states = train_sequences[:, :-1, :]             # (N, L, dim)
train_targets = train_sequences[:, 1:, :]                    # (N, L, dim)
train_inputs = np.concatenate([train_inputs_states, train_Q1_channels], axis=-1)  # (N, L, dim+1)

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

    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    model = transformer_decoder.TimeSeriesTransformer(
        args.input_size, args.output_size, args.d_model,
        args.nhead, args.num_layers, args.hidden_size, args.dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for epoch in range(args.num_epochs):
        model.train()
        losses = []
        for inputs, targets in (loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            noisy_inputs = inputs.clone()
            if args.noise_level > 0:
                noise = torch.normal(0.0, args.noise_level, size=inputs[:, :, :args.dim].shape, device=inputs.device)
                noisy_inputs[:, :, :args.dim] = inputs[:, :, :args.dim] + inputs[:, :, :args.dim] * noise

            outputs = model(noisy_inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        if losses:
            scheduler.step(float(np.mean(losses)))

    ckpt_path = os.path.join(args.save_dir, f'model_voltage_iter_{iter_index:02d}.pth')
    torch.save(model.state_dict(), ckpt_path)
    print(f'[iter {iter_index}] saved -> {ckpt_path}')
    
if __name__ == '__main__':
    start = int(args.iter_start)
    end = int(args.iter_end)
    if end <= start:
        raise ValueError('--iter-end must be greater than --iter-start')

    print(f'Running iterations [{start}, {end}) over {len(system_list)} voltage systems ...')
    for it in range(start, end):
        run_single_training(it)
    print('All iterations completed.')
