# -*- coding: utf-8 -*-
import os
import gc
import argparse
import pickle
import random
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as sio

import transformer_decoder  


# -------------------------
# Arguments
# -------------------------
parser = argparse.ArgumentParser('Iterative KS sweep over saved Transformer models')

parser.add_argument('--models-dir', type=str, default='./save_models_11132025',
                    help='Directory containing transformer_model_train_KS_iter*.pth')
parser.add_argument('--data-mat', type=str, default='./save_data/KS_train_data.mat',
                    help='.mat file with KS training data, shape (3, T, 32)')
parser.add_argument('--save-dir', type=str, default='./save_for_plot_KS',
                    help='Directory to save statistics and plots')

# Iterations and runs
parser.add_argument('--iters', type=int, default=3,
                    help='Number of model checkpoints (iter1..iterN)')
parser.add_argument('--runs-per-iter', type=int, default=1,
                    help='Random warmup segments per model')

# Model architecture (must mirror transformer_train_KS_iter.py)
parser.add_argument('--dim', type=int, default=32,
                    help='KS spatial dimension (number of grid points)')
parser.add_argument('--input-size', type=int, default=33,
                    help='Transformer input size (dim + 1 parameter)')
parser.add_argument('--output-size', type=int, default=32,
                    help='Transformer output size (= dim)')
parser.add_argument('--hidden-size', type=int, default=256)
parser.add_argument('--nhead', type=int, default=4)
parser.add_argument('--num-layers', type=int, default=4)
parser.add_argument('--d-model', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.2)

# Rollout configuration
parser.add_argument('--sequence-length', type=int, default=1024,
                    help='Warmup context length')
parser.add_argument('--rollout-steps', type=int, default=20000,
                    help='Prediction length per scan (should be >= window_len)')
parser.add_argument('--dt', type=float, default=2e-5,
                    help='Time step between samples of KS data / Transformer rollouts')

# Alpha sweep (bifurcation parameter)
parser.add_argument('--alpha-min', type=float, default=200.0)
parser.add_argument('--alpha-max', type=float, default=203.0)
parser.add_argument('--alpha-step', type=float, default=0.2)

# Collapse detection parameters (frequency-based)
parser.add_argument('--window-len', type=int, default=2048,
                    help='Window length in time steps for FFT')
parser.add_argument('--step', type=int, default=200,
                    help='Step between windows')
parser.add_argument('--R-thresh', type=float, default=0.5,
                    help='Threshold on peak-power ratio for periodic regime')
parser.add_argument('--persist-win', type=int, default=5,
                    help='Number of consecutive periodic windows required')

# Which training trajectory to use as warmup
# Assuming KS_train_data.mat contains 3 trajectories for parameters [196, 197, 198]
parser.add_argument('--warmup-index', type=int, default=2,
                    help='Index of trajectory to use for warmup (0,1,2)')

# Randomness / segment selection
parser.add_argument('--base-seed', type=int, default=777)
parser.add_argument('--guard', type=int, default=10000,
                    help='Guard samples at beginning and end when choosing random starts')

args = parser.parse_args()
print(args)

iter_start = 1


# -------------------------
# Device and setup
# -------------------------
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.save_dir, exist_ok=True)

# Set random seeds for reproducibility
random.seed(args.base_seed)
np.random.seed(args.base_seed)
torch.manual_seed(args.base_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.base_seed)
    torch.cuda.manual_seed_all(args.base_seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"[seed] Random seed set to {args.base_seed}")


# -------------------------
# Data loading (KS .mat)
# -------------------------
def load_ks_data(mat_path: str, dim: int) -> np.ndarray:
    """
    Load KS data from .mat file.
    Expected main array shape: (3, T, dim), for parameters [196, 197, 198].
    Returns: data_all with shape (3, T, dim) as float32.
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Data file not found: {mat_path}")

    print(f"[data] Loading KS data from {mat_path}")
    mat_data = sio.loadmat(mat_path)

    # pick the first non-__ key
    keys = [k for k in mat_data.keys() if not k.startswith('__')]
    if not keys:
        raise ValueError(f"No valid data keys found in {mat_path}")

    key = keys[0]
    arr = mat_data[key]
    print(f"[data] Found key '{key}' with shape {arr.shape}")

    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[2] < dim:
        raise ValueError(f"Unexpected data shape {arr.shape}; expected (3, T, >=dim)")

    data_all = arr[:, ::2, :dim].astype(np.float32)  # (3,T,dim) - Keep every 2nd point (downsample)
    print(f"[data] After downsampling: shape {data_all.shape}")
    return data_all


def choose_random_segment(total_len: int, seq_len: int, guard: int) -> int:
    """
    Choose a random starting index for warmup of length seq_len,
    keeping 'guard' samples from both ends.
    """
    need = seq_len
    lo = max(0, guard)
    hi = max(lo + 1, total_len - need - guard)

    if hi <= lo:
        # fallback: last valid start
        return max(0, total_len - need)
    return random.randint(lo, hi)


# -------------------------
# Model helpers
# -------------------------
def build_model() -> nn.Module:
    model = transformer_decoder.TimeSeriesTransformer(
        args.input_size,
        args.output_size,
        args.d_model,
        args.nhead,
        args.num_layers,
        args.hidden_size,
        args.dropout,
    ).to(DEVICE)
    return model


# training script saves: transformer_model_train_{system}_iter{iter_index}.pth
# where system is "KS"
CKPT_TEMPLATE = os.path.join(args.models_dir,
                             'transformer_model_train_KS_down_iter{}.pth')


# -------------------------
# Collapse detection (frequency-based)
# -------------------------
def detect_KS_collapse_freq(
    U: np.ndarray,
    dt: float,
    window_len: int = 2048,
    step: int = 200,
    R_thresh: float = 0.5,
    persist_win: int = 5,
) -> Tuple[bool, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Python version of the MATLAB detect_KS_collapse_freq.

    Parameters
    ----------
    U : array (T, N)
        Time series of KS state (time x space).
    dt : float
        Time step between samples.
    window_len : int
        Window length (number of time steps).
    step : int
        Step between window starts.
    R_thresh : float
        Threshold on peak-power ratio for periodic regime.
    persist_win : int
        Require at least this many consecutive periodic windows.

    Returns
    -------
    is_collapse : bool
    t_collapse : float (NaN if none)
    R_series : array of raw peak-power ratios
    R_smooth : smoothed ratio (moving average)
    t_centers : time at center of each window
    """
    U = np.asarray(U)
    if U.ndim != 2:
        raise ValueError(f"U must be 2D (T,N), got shape {U.shape}")

    T, N = U.shape
    if T < window_len:
        # not enough data for even one window
        return False, float('nan'), np.array([]), np.array([]), np.array([])

    # 1. Representative time series: choose middle spatial point
    mid_idx = N // 2
    y = U[:, mid_idx].astype(float)
    y = y - np.mean(y)

    # 2. Sliding windows
    starts = np.arange(0, T - window_len + 1, step, dtype=int)
    nW = len(starts)
    if nW == 0:
        return False, float('nan'), np.array([]), np.array([]), np.array([])

    R_series = np.zeros(nW, dtype=float)
    t_centers = np.zeros(nW, dtype=float)

    for i, t0 in enumerate(starts):
        seg = y[t0: t0 + window_len]
        seg = seg - np.mean(seg)

        Y = np.fft.fft(seg)
        P = np.abs(Y) ** 2

        # use only positive frequencies, excluding DC (index 0)
        kpos = np.arange(1, window_len // 2, dtype=int)
        Ppos = P[kpos]

        if np.sum(Ppos) <= 0:
            R_series[i] = 0.0
        else:
            R_series[i] = float(np.max(Ppos) / np.sum(Ppos))

        t_centers[i] = ((t0 + t0 + window_len - 1) / 2.0) * dt

    # 3. Smooth R to reduce noise (moving average of length 3)
    if nW >= 3:
        kernel = np.ones(3, dtype=float) / 3.0
        R_smooth = np.convolve(R_series, kernel, mode='same')
    else:
        R_smooth = R_series.copy()

    # 4. Detect sustained periodic regime (high R)
    is_periodic = (R_smooth > R_thresh)

    # Find blocks of consecutive "True" of length >= persist_win
    if persist_win <= 1:
        idx_blocks = np.where(is_periodic)[0]
    else:
        # convolution trick: sum over sliding window of length persist_win
        conv = np.convolve(is_periodic.astype(int),
                           np.ones(persist_win, dtype=int),
                           mode='valid')
        idx_blocks = np.where(conv == persist_win)[0]

    if idx_blocks.size == 0:
        # No sustained periodic block
        return False, float('nan'), R_series, R_smooth, t_centers

    first_idx = int(idx_blocks[0])  # index in R_smooth (starting window)
    mean_before = np.mean(R_smooth[:max(first_idx, 1)])
    mean_after = np.mean(R_smooth[first_idx:])

    if (mean_after > R_thresh) and (mean_before < R_thresh):
        is_collapse = True
        t_collapse = t_centers[first_idx]
    else:
        is_collapse = False
        t_collapse = float('nan')

    return is_collapse, t_collapse, R_series, R_smooth, t_centers


# -------------------------
# Rollout with alpha
# -------------------------
def rollout_with_alpha(
    model: nn.Module,
    init_states: np.ndarray,
    alpha_val: float,
    steps: int,
) -> np.ndarray:
    """
    Iterative rollout for KS system with constant alpha channel.

    init_states: (L, dim) warmup window
    alpha_val : scalar (bifurcation parameter)
    steps     : number of prediction steps
    """
    model.eval()
    with torch.no_grad():
        L, dim = init_states.shape
        alpha_seq = np.full((L, 1), alpha_val, dtype=np.float32)
        cur = np.concatenate([init_states.astype(np.float32), alpha_seq], axis=-1)  # (L, dim+1)
        cur_t = torch.tensor(cur, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,L,dim+1)

        preds = []
        for _ in range(steps):
            out = model(cur_t)                 # (1,L,dim)
            next_val = out[:, -1, :]           # (1,dim)
            preds.append(next_val.squeeze(0).cpu().numpy())

            alpha_next = torch.tensor([[alpha_val]], dtype=torch.float32,
                                      device=DEVICE).unsqueeze(0)  # (1,1,1)
            nxt = torch.cat([next_val.unsqueeze(1), alpha_next], dim=-1)  # (1,1,dim+1)
            cur_t = torch.cat([cur_t[:, 1:, :], nxt], dim=1)

        preds = np.asarray(preds)  # (steps,dim)
    return preds


# -------------------------
# Main sweep
# -------------------------
def main():
    # Load KS training data
    data_all = load_ks_data(args.data_mat, args.dim)  # (3,T,dim)
    if not (0 <= args.warmup_index < data_all.shape[0]):
        raise ValueError(f"warmup_index must be in [0, {data_all.shape[0]-1}], got {args.warmup_index}")
    traj = data_all[args.warmup_index]  # (T,dim)

    SEQ = args.sequence_length
    STEPS = args.rollout_steps

    alpha_values = np.arange(args.alpha_min,
                             args.alpha_max + 1e-12,
                             args.alpha_step,
                             dtype=float)

    all_critical_alphas: List[float] = []
    records: List[Dict[str, Any]] = []

    for mi in range(iter_start , args.iters + iter_start):
        ckpt_path = CKPT_TEMPLATE.format(mi)
        if not os.path.exists(ckpt_path):
            print(f'[warn] missing checkpoint: {ckpt_path} (skip this model)')
            continue

        print(f'\n=== Model iteration {mi} ===')
        model = build_model()
        state = torch.load(ckpt_path, map_location=DEVICE)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)

        # For reproducibility, shift seeds per model index
        random.seed(args.base_seed + mi)
        np.random.seed(args.base_seed + mi)

        T = traj.shape[0]

        for ri in range(args.runs_per_iter):
            # Choose warmup segment
            start = choose_random_segment(T, SEQ, args.guard)
            window = traj[start: start + SEQ, :]  # (SEQ,dim)

            critical_alpha: Optional[float] = None
            t_collapse: Optional[float] = None

            for alpha_now in alpha_values:
                print(f'  model={mi}, run={ri}, alpha={alpha_now:.4f}')
                preds = rollout_with_alpha(model, window, float(alpha_now), STEPS)

                flag, t_col, _, _, _ = detect_KS_collapse_freq(
                    preds,
                    dt=args.dt,
                    window_len=args.window_len,
                    step=args.step,
                    R_thresh=args.R_thresh,
                    persist_win=args.persist_win,
                )

                if flag:
                    critical_alpha = float(alpha_now)
                    t_collapse = float(t_col)
                    print(f'    -> collapse detected at alpha={critical_alpha:.4f}, t_collapse={t_collapse:.4f}')
                    break

            records.append({
                'model_index': mi,
                'run_index': ri,
                'start': int(start),
                'critical_alpha': critical_alpha,
                't_collapse': t_collapse,
            })
            if critical_alpha is not None:
                all_critical_alphas.append(critical_alpha)

        # free GPU per model
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # -------------------------
    # Save results
    # -------------------------
    res: Dict[str, Any] = {
        'config': {
            'iters': args.iters,
            'runs_per_iter': args.runs_per_iter,
            'sequence_length': args.sequence_length,
            'rollout_steps': args.rollout_steps,
            'dt': args.dt,
            'alpha_min': args.alpha_min,
            'alpha_max': args.alpha_max,
            'alpha_step': args.alpha_step,
            'window_len': args.window_len,
            'step': args.step,
            'R_thresh': args.R_thresh,
            'persist_win': args.persist_win,
            'data_mat': args.data_mat,
            'models_dir': args.models_dir,
            'warmup_index': args.warmup_index,
        },
        'records': records,
        'critical_alpha_values': all_critical_alphas,
    }

    os.makedirs(args.save_dir, exist_ok=True)
    res_pkl = os.path.join(args.save_dir, 'KS_critical_alpha_points.pkl')
    with open(res_pkl, 'wb') as fh:
        pickle.dump(res, fh)
    print(f'[save] results -> {res_pkl} (N={len(all_critical_alphas)})')

    # -------------------------
    # Histogram
    # -------------------------
    plt.figure(figsize=(7, 4))
    valid = np.array(
        [x for x in all_critical_alphas if x is not None],
        dtype=float
    )
    if valid.size > 0:
        bins_edges = np.arange(
            args.alpha_min,
            args.alpha_max + args.alpha_step * 0.5,
            args.alpha_step,
            dtype=float
        )
        plt.hist(valid, bins=bins_edges)
        plt.xlabel('Predicted critical alpha')
        plt.ylabel('Frequency')
        plt.title('Distribution of predicted critical points (KS system)')
        plt.xticks(bins_edges, rotation=45)
        plt.tight_layout()
        # plt.savefig(os.path.join(args.save_dir, 'KS_critical_alpha_hist.png'), dpi=150)
        plt.show()
    else:
        print('[info] No critical points detected; histogram skipped.')
    plt.close()


if __name__ == '__main__':
    main()
