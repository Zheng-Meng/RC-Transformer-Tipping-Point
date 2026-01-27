# -*- coding: utf-8 -*-

import os
import re
import torch
import torch.nn as nn
import numpy as np
import transformer_decoder
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import argparse
import copy
import matplotlib.pyplot as plt
import utils
import gc
import pickle
import random
from sklearn.preprocessing import MinMaxScaler
import time
import scipy.io as sio

parser = argparse.ArgumentParser('Train transformer on chaos prediction (iterative)')
parser.add_argument('--logdir', default='logdir', help='Folder to store everything/load')

parser.add_argument('--dim', default=32, type=int, help='Dimension of the chaotic systems')
parser.add_argument('--input-size', default=33, type=int, help='Transformer input dimension (32 states + parameter)')
parser.add_argument('--output-size', default=32, type=int, help='Transformer output dimension')
parser.add_argument('--hidden-size', default=256, type=int, help='Transformer hidden layer dimension')
parser.add_argument('--nhead', default=4, type=int, help='Transformer number of heads')
parser.add_argument('--num-layers', default=4, type=int, help='Transformer number of layers')
parser.add_argument('--d-model', default=128, type=int, help='Transformer projection dimension')
parser.add_argument('--dropout', default=0.2, type=float, help='Transformer drop out ratio')
parser.add_argument('--noise-level', default=0.05, type=float, help='Noise level added to the training data')

parser.add_argument('--sequence-length', default=1024, type=int, help='Input sequence length')
parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
parser.add_argument('--num-epochs', default=100, type=int, help='Epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')

args = parser.parse_args()
print(args)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# -------------------------
# Utilities (unchanged)
# -------------------------
def parse_para_from_name(name):
    m = re.search(r'para([0-9]+)', name)
    if not m:
        raise ValueError(f'Cannot parse para from name: {name}')
    return float(m.group(1))

def prepare_full_sequences(data, seq_length, interval):
    sequences = []
    for i in range(0, len(data) - seq_length, interval):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

# -------------------------
# One full training+evaluation run
# -------------------------
def run_training(iter_index: int, args):
    torch.cuda.empty_cache()
    gc.collect()

    data_read_length = 400000
    interval = round(args.sequence_length / 50)
    interval = 10
    print('interval', interval)

    # Evaluation settings
    short_term_steps = 250
    long_term_steps = 10000
    mse_threshold = 8

    system = 'KS'

    paras = [196, 197, 198]
    paras_length = len(paras)
    system_list = []
    for p in paras:
        system_list.append(f'KS_para{p}')
    attractors = system_list

    # Train/test on the train systems
    attractors_target = system_list

    train_set = system_list
    test_set = copy.deepcopy(train_set)
    target_set = attractors_target

    # -------------------------
    # Read data from .mat file
    # -------------------------
    raw_data = {}
    data_dir = "./save_data"

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
    print("Expected shape: (3, length, 32) - 3 parameters, length time steps, 32 dimensions")

    # Data should be (3, length, 32) - each index corresponds to a different parameter
    if len(all_trajectories.shape) == 3 and all_trajectories.shape[0] == len(paras):
        for i, system_name in enumerate(system_list):
            data_para_i = all_trajectories[i, :, :args.dim]  # shape: (length, 32)
            actual_length = min(data_read_length, data_para_i.shape[0])
            raw_data[system_name] = data_para_i[:actual_length:2, :]  # Keep every 2nd point (downsample)
            print(f"System {system_name} (parameter {paras[i]}): data shape {raw_data[system_name].shape}")
    else:
        raise ValueError(f"Unexpected data shape: {all_trajectories.shape}. Expected (3, length, 32)")

    train_data_raw = {}
    test_data_raw = {}

    for system_name in train_set:
        full_data = raw_data[system_name]
        split_point = len(full_data) - short_term_steps - long_term_steps - 1000 - args.sequence_length
        train_data_raw[system_name] = full_data[:split_point]
        test_data_raw[system_name] = full_data[split_point:]

    # For target systems: keep full (optionally thin)
    for system_name in target_set:
        full_data = raw_data[system_name]
        raw_data[system_name] = full_data

    train_sequences = []
    train_para_channels = []

    for system_name in train_set:
        sequences = prepare_full_sequences(train_data_raw[system_name], args.sequence_length + 1, interval)  # (N,L+1,32)
        para_val = parse_para_from_name(system_name)
        para_seq = np.full((sequences.shape[0], args.sequence_length, 1), para_val, dtype=np.float32)

        train_sequences.append(sequences)
        train_para_channels.append(para_seq)

    train_sequences = np.concatenate(train_sequences, axis=0)  # (N,L+1,32)
    train_para_channels = np.concatenate(train_para_channels, axis=0)  # (N,L,1)

    train_inputs_states = train_sequences[:, :-1, :]           # (N,L,32)
    train_targets = train_sequences[:, 1:, :]                  # (N,L,32)
    train_inputs = np.concatenate([train_inputs_states, train_para_channels], axis=-1)  # (N,L,33)

    train_dataset = TensorDataset(
        torch.tensor(train_inputs, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    model = transformer_decoder.TimeSeriesTransformer(
        args.input_size, args.output_size, args.d_model, 
        args.nhead, args.num_layers, args.hidden_size, args.dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # print how many parameters the model has
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

    start_time = time.time()
    train_losses = []
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = []

        for inputs, targets in tqdm.tqdm(train_loader):
            inputs = inputs.to(device)   # (B,L,33)
            targets = targets.to(device) # (B,L,32)

            noisy_inputs = inputs.clone()
            if args.noise_level > 0:
                noise = torch.normal(0.0, args.noise_level, size=inputs[:, :, :32].shape, device=inputs.device)
                noisy_inputs[:, :, :32] = inputs[:, :, :32] + inputs[:, :, :32] * noise  # add noise only to state dims

            outputs = model(noisy_inputs)

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        scheduler.step(avg_loss)
        train_losses.append(avg_loss)
        print(f'Iter {iter_index} | Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_loss:.6f}')

    end_time = time.time()
    print(f'Iter {iter_index} | Training time: {end_time - start_time:.2f} seconds')

    os.makedirs('./save_results', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.semilogy(train_losses)
    plt.title(f'Iter {iter_index} - Training Loss Over Epochs (parameter-conditioned)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    # plt.savefig(f'./save_results/transformer_train_{system}_bifurcation_{paras_length}_iter{iter_index}_train_loss.png', dpi=150)
    plt.close()

    results = {}
    model.eval()

    def eval_system(system_name, data, set_name):
        # short/long iterative rollout from the beginning (no random start)
        init_sequence_states = data[:args.sequence_length]
        true_future_short = data[args.sequence_length:args.sequence_length + short_term_steps]
        true_future_long = data[args.sequence_length:args.sequence_length + long_term_steps]

        para_val = parse_para_from_name(system_name)
        para_seq_init = np.full((args.sequence_length, 1), para_val, dtype=np.float32)

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

        # metrics
        short_mse = float(np.mean((preds_short - true_future_short)**2))
        running_avg_mse = np.array([np.mean((preds_long[:i+1] - true_future_long[:i+1])**2) for i in range(len(preds_long))])
        horizon_steps = int(np.argmax(running_avg_mse > mse_threshold))
        if horizon_steps == 0 and running_avg_mse[0] > mse_threshold:
            prediction_horizon = 0
        else:
            prediction_horizon = horizon_steps

        # plots (short-term time-series) - only plot first 5 dimensions
        num_dims_to_plot = min(5, args.dim)
        plt.figure(figsize=(8, 2*num_dims_to_plot))
        for dim_idx in range(num_dims_to_plot):
            plt.subplot(num_dims_to_plot, 1, dim_idx+1)
            plt.plot(true_future_short[:, dim_idx], label='Actual', alpha=0.7)
            plt.plot(preds_short[:, dim_idx], label='Predicted', alpha=0.7)
            plt.title(f'{system_name} - Short-term dim {dim_idx+1} (iter {iter_index})')
            plt.legend()
        plt.tight_layout()
        # plt.savefig(f'./save_results/transformer_train_{system}_bifurcation_{paras_length}_iter{iter_index}_short_term_{system_name}.png', dpi=150)
        plt.close()

        return short_mse, prediction_horizon, true_future_short, preds_short, true_future_long, preds_long

    with torch.no_grad():
        for eval_set, systems in zip(['test'], [test_set]):
            for system_name in systems:
                data = test_data_raw[system_name]

                short_mse, prediction_horizon, true_future_short, preds_short, true_future_long, preds_long = eval_system(system_name, data, eval_set)

                results[system_name] = {
                    'eval_set': eval_set,
                    'short_term_mse': short_mse,
                    'prediction_horizon': prediction_horizon,
                    'true_future_short': true_future_short,
                    'preds_short': preds_short,
                    'true_future_long': true_future_long,
                    'preds_long': preds_long,
                }

    for system_name, metrics in results.items():
        print(f"[iter {iter_index}] System: {system_name} [{metrics['eval_set']}]\n"
              f"  Short-term MSE: {metrics['short_term_mse']:.6f}\n"
              f"  Prediction Horizon: {metrics['prediction_horizon']} steps\n"
              )

    os.makedirs('./save_results', exist_ok=True)
    os.makedirs('./save_model', exist_ok=True)

    combined_info = {
        'results': results,
        'args': vars(args),
        'data_read_length': data_read_length,
        'train_set': train_set,
        'test_set': test_set,
        'target_set': target_set,
        'interval': interval,
        'short_term_steps': short_term_steps,
        'long_term_steps': long_term_steps,
        'mse_threshold': mse_threshold,
        'iter_index': iter_index,
    }
    # downsampling data version
    with open(f'./save_results/transformer_train_{system}_down_iter{iter_index}.pkl', 'wb') as f:
        pickle.dump(combined_info, f)

    torch.save(model.state_dict(), f'./save_model/transformer_model_train_{system}_down_iter{iter_index}.pth')
    print(f'[iter {iter_index}] Saved model and results.')

# -------------------------
# Main loop: run 10 iterations
# -------------------------
if __name__ == "__main__":
    for i in range(2, 4):
        print(f"\n=== Running iteration {i} ===\n")
        run_training(i, args)
