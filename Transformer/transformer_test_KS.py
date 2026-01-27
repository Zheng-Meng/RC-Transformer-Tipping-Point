# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import re
import scipy.io as sio

import transformer_decoder
import utils

paras_length = 3
system = 'KS'

model_path = f'./save_models_11132025/transformer_model_train_KS_down_iter1.pth'
data_dir = './save_data'
save_dir = './save_results'

paras = [196, 197, 198]
# paras = [198]
systems_train = [f'KS_para{p}' for p in paras]

dim = 32
input_size = 33  # 32 states + parameter channel
output_size = 32
hidden_size = 256
nhead = 4
num_layers = 4
d_model = 128
dropout = 0.2
sequence_length = 1024
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

short_term_steps = 250
long_term_steps = 20000
mse_threshold = 8
data_read_length = 50000

dt = 2e-5  # Time step between samples
window_len = 2048  # Window length for FFT
step = 200  # Step between windows
R_thresh = 0.5  # Threshold on peak-power ratio for periodic regime
persist_win = 5  # Number of consecutive periodic windows required

def parse_para_from_name(name):
    m = re.search(r'para([0-9]+)', name)
    if not m:
        raise ValueError(f'Cannot parse para from name: {name}')
    return float(m.group(1))

def detect_KS_collapse_freq(
    U,
    dt,
    window_len=2048,
    step=200,
    R_thresh=0.5,
    persist_win=5,
):
    """
    Detect collapse in KS system using frequency-based method.
    
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

data_file_path = os.path.join(data_dir, 'KS_train_data.mat')
print(f"Loading data from {data_file_path}...")
mat_data = sio.loadmat(data_file_path)

# Find the data key in the .mat file
data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
if data_keys:
    mat_key = data_keys[0]
    print(f"Found keys: {data_keys}. Using key: '{mat_key}'")
    all_trajectories = mat_data[mat_key]
else:
    raise ValueError(f"No data keys found in {data_file_path}.")

print(f"Loaded data with shape: {all_trajectories.shape}")

# Store data for each system
raw_data = {}
if len(all_trajectories.shape) == 3 and all_trajectories.shape[0] == len(paras):
    for i, system_name in enumerate(systems_train):
        data_para_i = all_trajectories[i, :, :dim]  # shape: (length, 32)
        actual_length = min(data_read_length, data_para_i.shape[0])
        raw_data[system_name] = data_para_i[:actual_length:2, :]  # Keep every 2nd point (downsample)
        print(f"System {system_name} (parameter {paras[i]}): data shape {raw_data[system_name].shape}")
else:
    raise ValueError(f"Unexpected data shape: {all_trajectories.shape}. Expected (3, length, 32)")

# -------------------------------
# Build model and load weights
# -------------------------------
model = transformer_decoder.TimeSeriesTransformer(input_size, output_size, d_model, nhead,
    num_layers, hidden_size, dropout).to(device)
state = torch.load(model_path, map_location=device)
if isinstance(state, dict) and 'state_dict' in state:
    state = state['state_dict']
model.load_state_dict(state)
model.eval()

os.makedirs(save_dir, exist_ok=True)

print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

# -------------------------------
# Evaluation function 
# -------------------------------
def eval_system(system_name, data, set_name):
    # short/long iterative rollout from the beginning (no random start)
    init_sequence_states = data[:sequence_length]
    true_future_short = data[sequence_length:sequence_length + short_term_steps]
    true_future_long = data[sequence_length:sequence_length + long_term_steps]

    para_val = parse_para_from_name(system_name)
    para_seq_init = np.full((sequence_length, 1), para_val, dtype=np.float32)

    current_seq = np.concatenate([init_sequence_states, para_seq_init], axis=-1)

    current_seq_t = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).to(device)
    preds_short = []
    preds_long = []

    with torch.no_grad():
        for step in range(long_term_steps):
            output = model(current_seq_t)
            next_value = output[:, -1, :]  # (1,32)

            if step < short_term_steps:
                preds_short.append(next_value.squeeze(0).cpu().numpy())
            preds_long.append(next_value.squeeze(0).cpu().numpy())

            # append next with parameter channel
            para_next = torch.tensor([[para_val]], dtype=torch.float32, device=device).unsqueeze(0)  # (1,1,1)
            next_with_para = torch.cat([next_value.unsqueeze(1), para_next], dim=-1)  # (1,1,33)
            current_seq_t = torch.cat([current_seq_t[:, 1:, :], next_with_para], dim=1)

    preds_short = np.array(preds_short)
    preds_long = np.array(preds_long)

    # metrics (using MSE as in training file)
    short_mse = float(np.mean((preds_short - true_future_short)**2))
    running_avg_mse = np.array([np.mean((preds_long[:i+1] - true_future_long[:i+1])**2) for i in range(len(preds_long))])
    horizon_steps = int(np.argmax(running_avg_mse > mse_threshold))
    if horizon_steps == 0 and running_avg_mse[0] > mse_threshold:
        prediction_horizon = 0
    else:
        prediction_horizon = horizon_steps

    # Collapse detection
    is_collapse, t_collapse, R_series, R_smooth, t_centers = detect_KS_collapse_freq(
        preds_long,
        dt=dt,
        window_len=window_len,
        step=step,
        R_thresh=R_thresh,
        persist_win=persist_win,
    )
    
    collapse_status = "COLLAPSE DETECTED" if is_collapse else "NO COLLAPSE"
    collapse_time = f"{t_collapse:.4f}" if is_collapse else "N/A"
    print(f"  Collapse Detection: {collapse_status} (t_collapse={collapse_time})")

    # Heatmap plots for high-dimensional data
    # Short-term prediction heatmap
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: True values heatmap
    im1 = axes[0].imshow(true_future_short.T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0].set_title(f'{system_name} - True Short-term (Time vs Dimensions)')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Dimensions')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot 2: Predicted values heatmap
    im2 = axes[1].imshow(preds_short.T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[1].set_title(f'{system_name} - Predicted Short-term (Time vs Dimensions)')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Dimensions')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot 3: Error heatmap
    error_short = np.abs(preds_short - true_future_short)
    im3 = axes[2].imshow(error_short.T, aspect='auto', cmap='hot', interpolation='nearest')
    axes[2].set_title(f'{system_name} - Absolute Error (Time vs Dimensions)')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Dimensions')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    # plt.savefig(f'{save_dir}/transformer_test_{system_name}_short_term_heatmap.png', dpi=150)
    plt.show()
    # plt.close()

    # Long-term prediction heatmap
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: True values heatmap
    im1 = axes[0].imshow(true_future_long.T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0].set_title(f'{system_name} - True Long-term (Time vs Dimensions)')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Dimensions')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot 2: Predicted values heatmap
    im2 = axes[1].imshow(preds_long.T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[1].set_title(f'{system_name} - Predicted Long-term (Time vs Dimensions)')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Dimensions')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot 3: Error heatmap
    error_long = np.abs(preds_long - true_future_long)
    im3 = axes[2].imshow(error_long.T, aspect='auto', cmap='hot', interpolation='nearest')
    axes[2].set_title(f'{system_name} - Absolute Error (Time vs Dimensions)')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Dimensions')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    # plt.savefig(f'{save_dir}/transformer_test_{system_name}_long_term_heatmap.png', dpi=150)
    plt.show()
    # plt.close()

    return init_sequence_states, short_mse, prediction_horizon, true_future_short, preds_short, true_future_long, preds_long, is_collapse, t_collapse

# -------------------------------
# Extrapolation evaluation function (use para 198 data but predict with para 200.4)
# -------------------------------
def eval_system_extrapolation(data_system_name, para_channel_value, data, set_name):
    """
    Evaluate extrapolation: use trajectory from data_system_name (para 198) but 
    feed para_channel_value (200.4) to the model for prediction.
    """
    # short/long iterative rollout from the beginning (no random start)
    init_sequence_states = data[:sequence_length]
    true_future_short = data[sequence_length:sequence_length + short_term_steps]
    true_future_long = data[sequence_length:sequence_length + long_term_steps]

    # Use the specified parameter channel value (200.4) instead of actual parameter (198)
    para_val = para_channel_value
    para_seq_init = np.full((sequence_length, 1), para_val, dtype=np.float32)

    current_seq = np.concatenate([init_sequence_states, para_seq_init], axis=-1)

    current_seq_t = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).to(device)
    preds_short = []
    preds_long = []

    with torch.no_grad():
        for step in range(long_term_steps):
            output = model(current_seq_t)
            next_value = output[:, -1, :]  # (1,32)

            if step < short_term_steps:
                preds_short.append(next_value.squeeze(0).cpu().numpy())
            preds_long.append(next_value.squeeze(0).cpu().numpy())

            # append next with parameter channel
            para_next = torch.tensor([[para_val]], dtype=torch.float32, device=device).unsqueeze(0)  # (1,1,1)
            next_with_para = torch.cat([next_value.unsqueeze(1), para_next], dim=-1)  # (1,1,33)
            current_seq_t = torch.cat([current_seq_t[:, 1:, :], next_with_para], dim=1)

    preds_short = np.array(preds_short)
    preds_long = np.array(preds_long)

    # metrics (using MSE as in training file)
    short_mse = float(np.mean((preds_short - true_future_short)**2))
    running_avg_mse = np.array([np.mean((preds_long[:i+1] - true_future_long[:i+1])**2) for i in range(len(preds_long))])
    horizon_steps = int(np.argmax(running_avg_mse > mse_threshold))
    if horizon_steps == 0 and running_avg_mse[0] > mse_threshold:
        prediction_horizon = 0
    else:
        prediction_horizon = horizon_steps

    # Collapse detection
    is_collapse, t_collapse, R_series, R_smooth, t_centers = detect_KS_collapse_freq(
        preds_long,
        dt=dt,
        window_len=window_len,
        step=step,
        R_thresh=R_thresh,
        persist_win=persist_win,
    )
    
    collapse_status = "COLLAPSE DETECTED" if is_collapse else "NO COLLAPSE"
    collapse_time = f"{t_collapse:.4f}" if is_collapse else "N/A"
    print(f"  Collapse Detection (Extrapolation): {collapse_status} (t_collapse={collapse_time})")

    label_name = f"{data_system_name}_as_para{para_val}"
    
    # Heatmap plots for high-dimensional data
    # Short-term prediction heatmap
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: True values heatmap
    im1 = axes[0].imshow(true_future_short.T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0].set_title(f'{label_name} - True Short-term (Time vs Dimensions)')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Dimensions')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot 2: Predicted values heatmap
    im2 = axes[1].imshow(preds_short.T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[1].set_title(f'{label_name} - Predicted Short-term (Time vs Dimensions)')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Dimensions')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot 3: Error heatmap
    error_short = np.abs(preds_short - true_future_short)
    im3 = axes[2].imshow(error_short.T, aspect='auto', cmap='hot', interpolation='nearest')
    axes[2].set_title(f'{label_name} - Absolute Error (Time vs Dimensions)')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Dimensions')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    # plt.savefig(f'{save_dir}/transformer_test_{label_name}_short_term_heatmap.png', dpi=150)
    plt.show()
    plt.close()

    # Long-term prediction heatmap
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: True values heatmap
    im1 = axes[0].imshow(true_future_long.T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0].set_title(f'{label_name} - True Long-term (Time vs Dimensions)')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Dimensions')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot 2: Predicted values heatmap
    im2 = axes[1].imshow(preds_long.T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[1].set_title(f'{label_name} - Predicted Long-term (Time vs Dimensions)')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Dimensions')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot 3: Error heatmap
    # error_long = np.abs(preds_long - true_future_long)
    # im3 = axes[2].imshow(error_long.T, aspect='auto', cmap='hot', interpolation='nearest')
    # axes[2].set_title(f'{label_name} - Absolute Error (Time vs Dimensions)')
    # axes[2].set_xlabel('Time Steps')
    # axes[2].set_ylabel('Dimensions')
    # plt.colorbar(im3, ax=axes[2])
    
    # plt.tight_layout()
    # # plt.savefig(f'{save_dir}/transformer_test_{label_name}_long_term_heatmap.png', dpi=150)
    # plt.show()
    # plt.close()

    return init_sequence_states, short_mse, prediction_horizon, true_future_short, preds_short, true_future_long, preds_long, is_collapse, t_collapse

# -------------------------------
# Loop through test systems
# -------------------------------
results = {}
model.eval()

with torch.no_grad():
    for system_name in systems_train:
        print(f'\nTesting system: {system_name}')
        data = raw_data[system_name]
        
        # Use data from sequence_length onward for testing
        # (matching the training file's approach where we start from beginning + sequence_length)
        start_idx = 0  # Start from beginning, eval_system will handle sequence extraction
        test_data = data[start_idx:]
        
        init_sequence_states, short_mse, prediction_horizon, true_future_short, preds_short, true_future_long, preds_long, is_collapse, t_collapse = eval_system(system_name, test_data, 'test')

        results[system_name] = {
            'eval_set': 'test',
            'init_sequence_states': init_sequence_states,
            'short_mse': short_mse,
            'prediction_horizon': prediction_horizon,
            'true_future_short': true_future_short,
            'preds_short': preds_short,
            'true_future_long': true_future_long,
            'preds_long': preds_long,
            'is_collapse': is_collapse,
            't_collapse': t_collapse,
        }

# -------------------------------
# Extrapolation test: Use para 198 data with para 200.4 channel
# -------------------------------
if 'KS_para198' in raw_data:
    print('\n' + '='*60)
    print('EXTRAPOLATION TEST: Using para 198 data with para 200.4 channel')
    print('='*60)
    
    extrapolation_para = 200.14
    data_198 = raw_data['KS_para198']
    start_idx = 0
    test_data = data_198[start_idx:]
    
    init_sequence_states, short_mse, prediction_horizon, true_future_short, preds_short, true_future_long, preds_long, is_collapse, t_collapse = eval_system_extrapolation(
        'KS_para198', extrapolation_para, test_data, 'extrapolation'
    )
    
    extrapolation_key = f'KS_para198_as_para{extrapolation_para}'
    results[extrapolation_key] = {
        'eval_set': 'extrapolation',
        'data_source': 'KS_para198',
        'para_channel': extrapolation_para,
        'init_sequence_states': init_sequence_states,
        'short_mse': short_mse,
        'prediction_horizon': prediction_horizon,
        'true_future_short': true_future_short,
        'preds_short': preds_short,
        'true_future_long': true_future_long,
        'preds_long': preds_long,
        'is_collapse': is_collapse,
        't_collapse': t_collapse,
    }
    print(f"\nExtrapolation Test Results:")
    print(f"  Data Source: para 198")
    print(f"  Parameter Channel: {extrapolation_para}")
    print(f"  Short-term MSE: {short_mse:.6f}")
    print(f"  Prediction Horizon: {prediction_horizon} steps")
    collapse_status = "YES" if is_collapse else "NO"
    collapse_time = f"{t_collapse:.4f}" if is_collapse else "N/A"
    print(f"  Collapse Detected: {collapse_status} (t={collapse_time})")
    
    # Save extrapolation data to .mat file for MATLAB plotting
    mat_save_dir = './save_for_mat'
    os.makedirs(mat_save_dir, exist_ok=True)
    
    mat_filename = f'KS_extrapolation_para{extrapolation_para}_predicted.mat'
    mat_filepath = os.path.join(mat_save_dir, mat_filename)
    
    mat_data = {
        'true_future_short': true_future_short,  # (short_term_steps, 32)
        'preds_short': preds_short,  # (short_term_steps, 32)
        'true_future_long': true_future_long,  # (long_term_steps, 32)
        'preds_long': preds_long,  # (long_term_steps, 32)
        'init_sequence_states': init_sequence_states,  # (sequence_length, 32)
        'para_channel': extrapolation_para,
        'data_source_para': 198,
        'short_term_steps': short_term_steps,
        'long_term_steps': long_term_steps,
        'sequence_length': sequence_length,
        'short_mse': short_mse,
        'prediction_horizon': prediction_horizon,
        'is_collapse': 1 if is_collapse else 0,
        't_collapse': t_collapse if is_collapse else np.nan,
        'dt': dt,
    }
    
    sio.savemat(mat_filepath, mat_data)
    print(f"\n  Saved extrapolation data to: {mat_filepath}")
else:
    print('\n' + '='*60)
    print('WARNING: Parameter 198 data not found. Skipping extrapolation test.')
    print('='*60)

# Print results 
print('\n' + '='*60)
print('SUMMARY OF ALL RESULTS')
print('='*60)
for system_name, metrics in results.items():
    collapse_status = "YES" if metrics.get('is_collapse', False) else "NO"
    collapse_time = f"{metrics.get('t_collapse', float('nan')):.4f}" if metrics.get('is_collapse', False) else "N/A"
    
    if metrics['eval_set'] == 'extrapolation':
        print(f"System: {system_name} [EXTRAPOLATION]\n"
              f"  Data Source: {metrics['data_source']}\n"
              f"  Parameter Channel: {metrics['para_channel']}\n"
              f"  Short-term MSE: {metrics['short_mse']:.6f}\n"
              f"  Prediction Horizon: {metrics['prediction_horizon']} steps\n"
              f"  Collapse Detected: {collapse_status} (t={collapse_time})\n")
    else:
        print(f"System: {system_name} [{metrics['eval_set']}]\n"
              f"  Short-term MSE: {metrics['short_mse']:.6f}\n"
              f"  Prediction Horizon: {metrics['prediction_horizon']} steps\n"
              f"  Collapse Detected: {collapse_status} (t={collapse_time})\n"
              )

# Save results
# with open(f'{save_dir}/transformer_test_{system}_bifurcation_{paras_length}.pkl', 'wb') as f:
#     pickle.dump(results, f)

print(f'\nResults saved to {save_dir}/transformer_test_{system}_bifurcation_{paras_length}.pkl')

