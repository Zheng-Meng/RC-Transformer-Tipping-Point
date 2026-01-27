# -*- coding: utf-8 -*-
import os
import gc
import argparse
import pickle
import random
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import transformer_decoder
import utils

parser = argparse.ArgumentParser('Iterative simulation over saved VOLTAGE models')
parser.add_argument('--models-dir', type=str, default='./save_models_11132025')
parser.add_argument('--data', type=str, default='./save_data/voltage_Q12.989780.pkl')
parser.add_argument('--save-dir', type=str, default='./save_for_plot')

parser.add_argument('--iters', type=int, default=50, help='Number of model checkpoints (expect 0..iters-1)')
parser.add_argument('--runs-per-iter', type=int, default=10, help='Random segments per model')

# Model arch
parser.add_argument('--dim', type=int, default=4)
parser.add_argument('--input-size', type=int, default=5)   # 4 states + Q1
parser.add_argument('--output-size', type=int, default=4)
parser.add_argument('--hidden-size', type=int, default=256)
parser.add_argument('--nhead', type=int, default=4)
parser.add_argument('--num-layers', type=int, default=4)
parser.add_argument('--d-model', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.2)

# Rollout config
parser.add_argument('--sequence-length', type=int, default=512)
parser.add_argument('--rollout-steps', type=int, default=1000)

# Q1 sweep
parser.add_argument('--q1-min', type=float, default=2.989780)
parser.add_argument('--q1-max', type=float, default=3.05)
parser.add_argument('--q1-step', type=float, default=0.003)

# Critical criteria
parser.add_argument('--v-threshold', type=float, default=0.75, help='V < threshold triggers critical')

# Randomness
parser.add_argument('--base-seed', type=int, default=777)
parser.add_argument('--guard', type=int, default=10000)

args = parser.parse_args()
print(args)

p0 = -3.1

# -------------------------
# Setup
# -------------------------
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.save_dir, exist_ok=True)
save_dir = args.save_dir

with open(args.data, 'rb') as fh:
    bundle = pickle.load(fh)
traj = np.asarray(bundle['traj'])  # (N,4)

SEQ = args.sequence_length
STEPS = args.rollout_steps

random.seed(args.base_seed)
np.random.seed(args.base_seed)

def _choose_random_segment(total_len: int, seq_len: int, steps: int, guard: int) -> int:
    need = seq_len + steps
    lo = max(0, guard)
    hi = max(lo + 1, total_len - need - guard)
    if hi <= lo:
        return max(0, total_len - need)
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

_ckpt_tpl = os.path.join(args.models_dir, 'transformer_model_train_voltage_iter{}_p-3.1.pth')

# -------------------------
# Rollout and detection
# -------------------------

def rollout_with_q1(model: nn.Module, init_states: np.ndarray, q1_val: float, steps: int) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        q1_val = q1_val + p0
        q1_seq = np.full((init_states.shape[0], 1), q1_val, dtype=np.float32)
        cur = np.concatenate([init_states.astype(np.float32), q1_seq], axis=-1)
        cur_t = torch.tensor(cur, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        preds = []
        for _ in range(steps):
            out = model(cur_t)
            next_val = out[:, -1, :]
            preds.append(next_val.squeeze(0).cpu().numpy())
            q1_next = torch.tensor([[q1_val]], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            nxt = torch.cat([next_val.unsqueeze(1), q1_next], dim=-1)
            cur_t = torch.cat([cur_t[:, 1:, :], nxt], dim=1)
        return np.asarray(preds)

def detect_critical(pred_traj: np.ndarray, v_threshold: float) -> Optional[int]:
    """Return the first index t where V (4th dim) < v_threshold."""
    v = pred_traj[:, 3]
    below = np.where(v < v_threshold)[0]
    return int(below[0]) if below.size > 0 else None

# -------------------------
# Main sweep
# -------------------------
q1_values = np.arange(args.q1_min, args.q1_max + 1e-12, args.q1_step)

all_critical_q1: List[float] = []
records = []

for mi in range(args.iters):
    ckpt_path = _ckpt_tpl.format(mi)
    if not os.path.exists(ckpt_path):
        print(f'[warn] missing checkpoint: {ckpt_path} (skip)')
        continue

    model = build_model()
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)

    random.seed(args.base_seed + mi)
    np.random.seed(args.base_seed + mi)

    for ri in range(args.runs_per_iter):
        start = _choose_random_segment(len(traj), SEQ, STEPS, args.guard)
        window = traj[start:start+SEQ, :]

        critical_q1 = None
        critical_idx = None

        for q1_val in q1_values:
            print('model: ', mi, 'iter: ', ri, 'q1_val:', q1_val)
            preds = rollout_with_q1(model, window, float(q1_val), STEPS)
            idx = detect_critical(preds, args.v_threshold)
            if idx is not None:
                critical_q1 = float(q1_val)
                critical_idx = int(idx)
                break

        records.append({
            'model_index': mi,
            'run_index': ri,
            'start': int(start),
            'critical_q1': critical_q1,
            'critical_idx': critical_idx,
        })
        if critical_q1 is not None:
            all_critical_q1.append(critical_q1)

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
        'q1_min': args.q1_min,
        'q1_max': args.q1_max,
        'q1_step': args.q1_step,
        'v_threshold': args.v_threshold,
        'data': args.data,
        'models_dir': args.models_dir,
    },
    'records': records,
    'critical_q1_values': all_critical_q1,
}

os.makedirs(save_dir, exist_ok=True)
res_pkl = os.path.join(save_dir, 'voltage_Q1_critical_points_bias.pkl')
with open(res_pkl, 'wb') as fh:
    pickle.dump(res, fh)
print(f'[save] results -> {res_pkl} (N={len(all_critical_q1)})')

# Histogram
plt.figure(figsize=(7,4))
valid = np.array([x for x in all_critical_q1 if x is not None], dtype=float)
if valid.size > 0:
    plt.hist(valid, bins=np.arange(args.q1_min, args.q1_max + args.q1_step, args.q1_step))
    plt.xlabel('Predicted critical Q1')
    plt.ylabel('Frequency')
    plt.title('Distribution of predicted critical points (Voltage system)')
    fig_path = os.path.join(save_dir, 'voltage_Q1_critical_hist_bias.png')
    plt.tight_layout()
    plt.show()
    # plt.savefig(fig_path, dpi=150)
    print(f'[save] histogram -> {fig_path}')
else:
    print('[info] No critical points detected; histogram skipped.')
plt.close()
