# -*- coding: utf-8 -*-
import os
import re
import gc
import argparse
import pickle
import random
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import transformer_decoder  
import utils  # kept for parity, not strictly required here

parser = argparse.ArgumentParser('Iterative simulation over saved models (foodchain)')
parser.add_argument('--models-dir', type=str, default='./save_models')
parser.add_argument('--data', type=str, default='./save_data/foodchain_k0.99.pkl')
parser.add_argument('--save-dir', type=str, default='./save_for_plot')

parser.add_argument('--iters', type=int, default=5, help='Number of model checkpoints (expect 0..iters-1)')
parser.add_argument('--runs-per-iter', type=int, default=2, help='Random segments per model')

# Model arch (must mirror training)
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--input-size', type=int, default=4)   # 3 states + k
parser.add_argument('--output-size', type=int, default=3)
parser.add_argument('--hidden-size', type=int, default=256)
parser.add_argument('--nhead', type=int, default=4)
parser.add_argument('--num-layers', type=int, default=4)
parser.add_argument('--d-model', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.2)

# Rollout config
parser.add_argument('--sequence-length', type=int, default=512)
parser.add_argument('--rollout-steps', type=int, default=1000)

# k sweep
parser.add_argument('--k-min', type=float, default=0.99)
parser.add_argument('--k-max', type=float, default=1.35)
parser.add_argument('--k-step', type=float, default=0.02)

# Critical criteria
parser.add_argument('--z-threshold', type=float, default=0.4, help='z < threshold triggers critical')
parser.add_argument('--const-window', type=int, default=100, help='rolling window for const detection')
parser.add_argument('--const-std', type=float, default=1e-3, help='std threshold for near-constant detection')

# Randomness
parser.add_argument('--base-seed', type=int, default=777)
parser.add_argument('--guard', type=int, default=10000, help='guard samples when choosing random starts')

args = parser.parse_args()
print(args)

# -------------------------
# Setup
# -------------------------
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.save-dir if hasattr(args, 'save-dir') else args.save_dir, exist_ok=True)
save_dir = getattr(args, 'save_dir', None) or getattr(args, 'save-dir')

# Load data (only foodchain_k0.99)
with open(args.data, 'rb') as fh:
    bundle = pickle.load(fh)
traj = np.asarray(bundle['traj'])  # (N,3)

SEQ = args.sequence_length
STEPS = args.rollout_steps

random.seed(args.base_seed)
np.random.seed(args.base_seed)

def _choose_random_segment(total_len: int, seq_len: int, steps: int, guard: int) -> int:
    need = seq_len + steps
    lo = max(0, guard)
    hi = max(lo + 1, total_len - need - guard)
    if hi <= lo:
        return max(0, total_len - need)  # fall back: last valid start
    return random.randint(lo, hi)

# -------------------------
# Model helpers
# -------------------------

def build_model() -> nn.Module:
    model = transformer_decoder.TimeSeriesTransformer(
        args.input_size, args.output_size, args.d_model,
        args.nhead, args.num_layers, args.hidden_size, args.dropout
    ).to(DEVICE)
    return model

_ckpt_tpl = os.path.join(args.models_dir, 'model_foodchain_iter_{:02d}.pth')

# -------------------------
# Rollout and detection
# -------------------------

def rollout_with_k(model: nn.Module, init_states: np.ndarray, k_val: float, steps: int) -> np.ndarray:
    """Iterative rollout given an initial state sequence (L,3) and constant k channel."""
    model.eval()
    with torch.no_grad():
        k_seq = np.full((init_states.shape[0], 1), k_val, dtype=np.float32)
        cur = np.concatenate([init_states.astype(np.float32), k_seq], axis=-1)  # (L,4)
        cur_t = torch.tensor(cur, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        preds = []
        for _ in range(steps):
            out = model(cur_t)              # (1,L,3)
            next_val = out[:, -1, :]        # (1,3)
            preds.append(next_val.squeeze(0).cpu().numpy())
            k_next = torch.tensor([[k_val]], dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,1,1)
            nxt = torch.cat([next_val.unsqueeze(1), k_next], dim=-1)  # (1,1,4)
            cur_t = torch.cat([cur_t[:, 1:, :], nxt], dim=1)
        return np.asarray(preds)  # (steps,3)


def detect_critical(pred_traj: np.ndarray, z_threshold: float, const_window: int, const_std: float) -> Optional[int]:
    """Return the first index t where z < z_threshold."""
    z = pred_traj[:, 2]
    below = np.where(z < z_threshold)[0]
    return int(below[0]) if below.size > 0 else None

# -------------------------
# Main sweep
# -------------------------

k_values = np.arange(args.k_min, args.k_max + 1e-12, args.k_step)

all_critical_ks: List[float] = []
records = []

for mi in range(args.iters):
    ckpt_path = _ckpt_tpl.format(mi)
    if not os.path.exists(ckpt_path):
        print(f'[warn] missing checkpoint: {ckpt_path} (skip)')
        continue

    # Build and load model
    model = build_model()
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)

    # For reproducibility, shift seed per model index
    random.seed(args.base_seed + mi)
    np.random.seed(args.base_seed + mi)

    for ri in range(args.runs_per_iter):
        
        start = _choose_random_segment(len(traj), SEQ, STEPS, args.guard)
        window = traj[start:start+SEQ, :]

        critical_k = None
        critical_idx = None

        for k_val in k_values:
            print('model: ', mi, 'iter: ', ri, 'k_val:', k_val)
            preds = rollout_with_k(model, window, float(k_val), STEPS)
            idx = detect_critical(preds, args.z_threshold, args.const_window, args.const_std)
            if idx is not None:
                critical_k = float(k_val)
                critical_idx = int(idx)
                break

        records.append({
            'model_index': mi,
            'run_index': ri,
            'start': int(start),
            'critical_k': critical_k,
            'critical_idx': critical_idx,
        })
        if critical_k is not None:
            all_critical_ks.append(critical_k)

    # free GPU per model
    del model
    torch.cuda.empty_cache(); gc.collect()

# -------------------------
# Save + Plot
# -------------------------
res = {
    'config': {
        'iters': args.iters,
        'runs_per_iter': args.runs_per_iter,
        'sequence_length': args.sequence_length,
        'rollout_steps': args.rollout_steps,
        'k_min': args.k_min,
        'k_max': args.k_max,
        'k_step': args.k_step,
        'z_threshold': args.z_threshold,
        'const_window': args.const_window,
        'const_std': args.const_std,
        'data': args.data,
        'models_dir': args.models_dir,
    },
    'records': records,
    'critical_k_values': all_critical_ks,
}

os.makedirs(save_dir, exist_ok=True)
res_pkl = os.path.join(save_dir, 'foodchain_k099_critical_points.pkl')
with open(res_pkl, 'wb') as fh:
    pickle.dump(res, fh)
print(f'[save] results -> {res_pkl} (N={len(all_critical_ks)})')

# Histogram
plt.figure(figsize=(7,4))
valid = np.array([x for x in all_critical_ks if x is not None], dtype=float)
if valid.size > 0:
    bins_edges = np.arange(args.k_min, args.k_max + args.k_step*0.5, args.k_step, dtype=float)

    plt.hist(valid, bins=bins_edges)
    plt.xlabel('Predicted critical k')
    plt.ylabel('Frequency')
    plt.title('Distribution of predicted critical points (foodchain k0.99)')

    # put ticks at the same k grid (optional: comment out if too dense)
    plt.xticks(bins_edges, rotation=45)

    fig_path = os.path.join(save_dir, 'foodchain_k099_critical_hist.png')
    plt.tight_layout()
    # plt.savefig(fig_path, dpi=150)
    plt.show()
    print(f'[save] histogram -> {fig_path}')
else:
    print('[info] No critical points detected; histogram skipped.')
plt.close()
