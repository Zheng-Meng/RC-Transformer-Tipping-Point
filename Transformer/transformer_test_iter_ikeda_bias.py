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
import utils  # kept for parity

# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser('Iterative simulation over saved IKEDA models')
parser.add_argument('--models-dir', type=str, default='./save_models_11132025')
parser.add_argument('--data', type=str, default='./save_data/ikeda_mu0.97.pkl')
parser.add_argument('--save-dir', type=str, default='./save_for_plot')

parser.add_argument('--iters', type=int, default=50, help='Number of model checkpoints (expect 0..iters-1)')
parser.add_argument('--runs-per-iter', type=int, default=10, help='Random segments per model')

# Model arch
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--input-size', type=int, default=3)   # 2 states + mu
parser.add_argument('--output-size', type=int, default=2)
parser.add_argument('--hidden-size', type=int, default=256)
parser.add_argument('--nhead', type=int, default=4)
parser.add_argument('--num-layers', type=int, default=4)
parser.add_argument('--d-model', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.2)

# Rollout config
parser.add_argument('--sequence-length', type=int, default=512)
parser.add_argument('--rollout-steps', type=int, default=2000)

# mu sweep
parser.add_argument('--mu-min', type=float, default=0.97)
parser.add_argument('--mu-max', type=float, default=1.2)
parser.add_argument('--mu-step', type=float, default=0.01)

# Collapse bounds
parser.add_argument('--x-min', type=float, default=-1.0)
parser.add_argument('--x-max', type=float, default=2.2)
parser.add_argument('--y-min', type=float, default=-2.5)
parser.add_argument('--y-max', type=float, default=2.0)

# Randomness
parser.add_argument('--base-seed', type=int, default=777)
parser.add_argument('--guard', type=int, default=10000)

args = parser.parse_args()
print(args)


p0 = 0.47

# -------------------------
# Setup
# -------------------------
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.save_dir, exist_ok=True)
save_dir = args.save_dir

with open(args.data, 'rb') as fh:
    bundle = pickle.load(fh)
traj = np.asarray(bundle['traj'])  # (N,2)

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

_ckpt_tpl = os.path.join(args.models_dir, 'transformer_model_train_ikeda_iter{}_p0.47.pth')

# -------------------------
# Rollout and detection
# -------------------------

def rollout_with_mu(model: nn.Module, init_states: np.ndarray, mu_val: float, steps: int) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        mu_val = mu_val + p0
        mu_seq = np.full((init_states.shape[0], 1), mu_val, dtype=np.float32)
        cur = np.concatenate([init_states.astype(np.float32), mu_seq], axis=-1)
        cur_t = torch.tensor(cur, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        preds = []
        for _ in range(steps):
            out = model(cur_t)
            next_val = out[:, -1, :]
            preds.append(next_val.squeeze(0).cpu().numpy())
            mu_next = torch.tensor([[mu_val]], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            nxt = torch.cat([next_val.unsqueeze(1), mu_next], dim=-1)
            cur_t = torch.cat([cur_t[:, 1:, :], nxt], dim=1)
        return np.asarray(preds)

def detect_critical(pred_traj: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> Optional[int]:
    """Return first index t where x out of [x_min, x_max] OR y out of [y_min, y_max]."""
    x = pred_traj[:, 0]
    y = pred_traj[:, 1]
    out_x = np.where((x < x_min) | (x > x_max))[0]
    out_y = np.where((y < y_min) | (y > y_max))[0]
    idx_x = int(out_x[0]) if out_x.size > 0 else None
    idx_y = int(out_y[0]) if out_y.size > 0 else None
    if idx_x is None and idx_y is None:
        return None
    if idx_x is None:
        return idx_y
    if idx_y is None:
        return idx_x
    return min(idx_x, idx_y)

# -------------------------
# Main sweep
# -------------------------
mu_values = np.arange(args.mu_min, args.mu_max + 1e-12, args.mu_step)

all_critical_mu: List[float] = []
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

        critical_mu = None
        critical_idx = None

        for mu_val in mu_values:
            print('model: ', mi, 'iter: ', ri, 'mu_val:', mu_val)
            preds = rollout_with_mu(model, window, float(mu_val), STEPS)
            idx = detect_critical(preds, args.x_min, args.x_max, args.y_min, args.y_max)
            if idx is not None:
                critical_mu = float(mu_val)
                critical_idx = int(idx)
                break

        records.append({
            'model_index': mi,
            'run_index': ri,
            'start': int(start),
            'critical_mu': critical_mu,
            'critical_idx': critical_idx,
        })
        if critical_mu is not None:
            all_critical_mu.append(critical_mu)

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
        'mu_min': args.mu_min,
        'mu_max': args.mu_max,
        'mu_step': args.mu_step,
        'x_min': args.x_min,
        'x_max': args.x_max,
        'y_min': args.y_min,
        'y_max': args.y_max,
        'data': args.data,
        'models_dir': args.models_dir,
    },
    'records': records,
    'critical_mu_values': all_critical_mu,
}

os.makedirs(save_dir, exist_ok=True)
res_pkl = os.path.join(save_dir, 'ikeda_mu_critical_points_bias.pkl')
with open(res_pkl, 'wb') as fh:
    pickle.dump(res, fh)
print(f'[save] results -> {res_pkl} (N={len(all_critical_mu)})')

# Histogram
plt.figure(figsize=(7,4))
valid = np.array([x for x in all_critical_mu if x is not None], dtype=float)
if valid.size > 0:
    plt.hist(valid, bins=np.arange(args.mu_min, args.mu_max + args.mu_step, args.mu_step))
    plt.xlabel('Predicted critical mu')
    plt.ylabel('Frequency')
    plt.title('Distribution of predicted critical points (Ikeda)')
    fig_path = os.path.join(save_dir, 'ikeda_mu_critical_hist_bias.png')
    plt.tight_layout()
    plt.show()
    # plt.savefig(fig_path, dpi=150)
    print(f'[save] histogram -> {fig_path}')
else:
    print('[info] No critical points detected; histogram skipped.')
plt.close()