# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import MinMaxScaler

import transformer_decoder
import utils

# -------------------------------
# Settings
# -------------------------------
k_number = 3
# k_number = 27

if k_number == 3:
    model_path = f'./save_model/transformer_model_train_foodchain_k0.99_bifurcation_{k_number}_iter2.pth'
else:
    model_path = f'./save_model/transformer_model_train_foodchain_k0.996_bifurcation_{k_number}_iter0.pth'
data_dir = './save_data'
save_dir = './save_results'
if k_number == 3:
    systems_train = ['foodchain_k0.97', 'foodchain_k0.98', 'foodchain_k0.99']
else:
    ks = np.arange(0.970, 0.996, 0.001)
    systems_train = []
    for k in ks:
        systems_train.append(f'foodchain_k{k}')

systems_test = ['foodchain_k1.0']

# dim = 3
# input_size = 4  # 3 states + k channel (matching training file)
# output_size = 3
# hidden_size = 256
# nhead = 4
# num_layers = 4  # matching training file
# d_model = 128
# dropout = 0.2
# sequence_length = 512 
# start_point_test = 3500 - sequence_length
# long_term_steps = 2000
# max_points = 10000
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


dim = 3
input_size = 4  # 3 states + k channel (matching training file)
output_size = 3
hidden_size = 128
nhead = 2
num_layers = 2  # matching training file
d_model = 64
dropout = 0.2
sequence_length = 512
start_point_test = 3700 - sequence_length  # 3500 - sequence_length 
long_term_steps = 2000
max_points = 10000
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

short_term_steps = 1000
short_rmse_steps = 250
rmse_threshold = 0.1

def parse_k_from_name(name):
    m = re.search(r'k([0-9.]+)', name)
    if not m:
        raise ValueError(f'Cannot parse k from name: {name}')
    return float(m.group(1))

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

def eval_system(system, data, set_name):
    init_sequence_states = data[:sequence_length]
    true_future_short = data[sequence_length:sequence_length + short_term_steps]
    true_future_long = data[sequence_length:sequence_length + long_term_steps]

    k_val = parse_k_from_name(system)
    k_seq_init = np.full((sequence_length, 1), k_val, dtype=np.float32)

    current_seq = np.concatenate([init_sequence_states, k_seq_init], axis=-1)

    current_seq_t = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).to(device)
    preds_short = []
    preds_long = []

    with torch.no_grad():
        for step in range(long_term_steps):
            output = model(current_seq_t)
            next_value = output[:, -1, :]  # (1,3)

            if step < short_term_steps:
                preds_short.append(next_value.squeeze(0).cpu().numpy())
            preds_long.append(next_value.squeeze(0).cpu().numpy())

            # append next with k channel
            k_next = torch.tensor([[k_val]], dtype=torch.float32, device=device).unsqueeze(0)  # (1,1,1)
            next_with_k = torch.cat([next_value.unsqueeze(1), k_next], dim=-1)  # (1,1,4)
            current_seq_t = torch.cat([current_seq_t[:, 1:, :], next_with_k], dim=1)

    preds_short = np.array(preds_short)
    preds_long = np.array(preds_long)

    # metrics
    short_rmse = float(np.sqrt(np.mean((preds_short[:short_rmse_steps, :] - true_future_short[:short_rmse_steps, :])**2)))
    running_avg_rmse = np.array([np.sqrt(np.mean((preds_long[:i+1] - true_future_long[:i+1])**2)) for i in range(len(preds_long))])
    horizon_steps = int(np.argmax(running_avg_rmse > rmse_threshold))
    if horizon_steps == 0 and running_avg_rmse[0] > rmse_threshold:
        prediction_horizon = 0
    elif horizon_steps == 0:
        prediction_horizon = short_term_steps
    else:
        prediction_horizon = horizon_steps

    # plots (short-term time-series)
    plt.figure(figsize=(8, 10))
    for dim_idx in range(dim):
        plt.subplot(dim, 1, dim_idx+1)
        plt.plot(true_future_short[:, dim_idx], label='Actual', alpha=0.7)
        plt.plot(preds_short[:, dim_idx], label='Predicted', alpha=0.7)
        plt.title(f'{system} - Short-term dim {dim_idx+1}')
        plt.legend()
    plt.tight_layout()
    # plt.savefig(f'./save_results/transformer_test_{system}_short_term.png', dpi=150)
    plt.show()
    plt.close()

    # long-term 3D
    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(true_future_long[:, 0], true_future_long[:, 1], true_future_long[:, 2], label='Actual', alpha=0.7)
        ax.plot(preds_long[:, 0], preds_long[:, 1], preds_long[:, 2], label='Predicted', alpha=0.7)
        plt.legend()
        plt.title(f'{system} - Long-term 3D')
        # plt.savefig(f'./save_results/transformer_test_{system}_long_term.png', dpi=150)
        plt.show()
        plt.close()

    return init_sequence_states, short_rmse, prediction_horizon, true_future_short, preds_short, true_future_long, preds_long

# -------------------------------
# Loop through test systems
# -------------------------------
results = {}
model.eval()

with torch.no_grad():
    for system in systems_train:
        print(f'\nTesting system: {system}')
        file_path = os.path.join(data_dir, f'{system}.pkl')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        traj = data['traj']
        start_point_train = len(traj) - sequence_length - 10001
        traj = traj[start_point_train:, :]

        init_sequence_states, short_rmse, prediction_horizon, true_future_short, preds_short, true_future_long, preds_long = eval_system(system, traj, 'test')

        results[system] = {
            'eval_set': 'test',
            'init_sequence_states': init_sequence_states,
            'short_rmse': short_rmse,
            'prediction_horizon': prediction_horizon,
            'true_future_short': true_future_short,
            'preds_short': preds_short,
            'true_future_long': true_future_long,
            'preds_long': preds_long,
        }
    
    for system in systems_test:
        print(f'\nTesting system: {system}')
        file_path = os.path.join(data_dir, f'{system}.pkl')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        traj = data['traj'][start_point_test:, :]
        init_sequence_states, short_rmse, prediction_horizon, true_future_short, preds_short, true_future_long, preds_long = eval_system(system, traj, 'test')

        results[system] = {
            'eval_set': 'test',
            'init_sequence_states': init_sequence_states,
            'short_rmse': short_rmse,
            'prediction_horizon': prediction_horizon,
            'true_future_short': true_future_short,
            'preds_short': preds_short,
            'true_future_long': true_future_long,
            'preds_long': preds_long,
        }

# Print results 
for system, metrics in results.items():
    print(f"System: {system} [{metrics['eval_set']}]\n"
          f"  Short-term RMSE: {metrics['short_rmse']:.6f}\n"
          f"  Prediction Horizon: {metrics['prediction_horizon']} steps\n"
          )

# save as pkl file to save_for_plot
# with open(f'./save_for_plot/transformer_test_foodchain_bifurcation_{k_number}_example1.pkl', 'wb') as f:
#     pickle.dump(results, f)

