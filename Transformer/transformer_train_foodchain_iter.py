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

parser = argparse.ArgumentParser('Train transformer on chaos prediction (iterative)')
parser.add_argument('--logdir', default='logdir', help='Folder to store everything/load')

parser.add_argument('--dim', default=3, type=int, help='Dimension of the chaotic systems')
parser.add_argument('--input-size', default=4, type=int, help='Transformer input dimension (3 states + k)')
parser.add_argument('--output-size', default=3, type=int, help='Transformer output dimension')
parser.add_argument('--hidden-size', default=512, type=int, help='Transformer hidden layer dimension')
parser.add_argument('--nhead', default=8, type=int, help='Transformer number of heads')
parser.add_argument('--num-layers', default=8, type=int, help='Transformer number of layers')
parser.add_argument('--d-model', default=256, type=int, help='Transformer projection dimension')
parser.add_argument('--dropout', default=0.2, type=float, help='Transformer drop out ratio')
parser.add_argument('--noise-level', default=0.05, type=float, help='Noise level added to the training data')

parser.add_argument('--sequence-length', default=512, type=int, help='Input sequence length')
parser.add_argument('--batch-size', default=128, type=int, help='Batch size')
parser.add_argument('--num-epochs', default=100, type=int, help='Epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')

setting = 5

args = parser.parse_args()
print(args)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def parse_k_from_name(name):
    m = re.search(r'k([0-9.]+)', name)
    if not m:
        raise ValueError(f'Cannot parse k from name: {name}')
    return float(m.group(1))

def prepare_full_sequences(data, seq_length, interval):
    sequences = []
    for i in range(0, len(data) - seq_length, interval):
        seq = data[i:i + seq_length]
        sequences.append(seq)
    return np.array(sequences)

def run_training(iter_index: int, args):
    torch.cuda.empty_cache()
    gc.collect()

    data_read_length = 400000  # previously 400000
    take_num = 30
    interval = round(args.sequence_length / 50)   # keep overlapping windows as in working config
    print('take num', take_num)
    print('interval', interval)

    short_term_steps = 250
    long_term_steps = 10000
    rmse_threshold = 0.1

    system = 'foodchain'

    ks = [0.97, 0.98, 0.99]
    ks_length = len(ks)
    system_list = [f'foodchain_k{k}' for k in ks]
    attractors = system_list  # kept for consistency

    attractors_target = system_list
    train_set = system_list
    test_set = copy.deepcopy(train_set)
    target_set = attractors_target

    all_systems = list(set(train_set + test_set + target_set))
    raw_data = {}
    data_dir = "./save_data"

    train_k_vals = [parse_k_from_name(s) for s in train_set]

    for sys_name in all_systems:
        file_path = os.path.join(data_dir, f'{sys_name}.pkl')
        with open(file_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        traj = data['traj']

        random_start = random.randint(10000, len(traj) - data_read_length - 10000)
        traj = traj[random_start:random_start + data_read_length, :]

        raw_data[sys_name] = traj

    train_data_raw = {}
    test_data_raw = {}

    for sys_name in train_set:
        full_data = raw_data[sys_name]
        split_point = len(full_data) - short_term_steps - long_term_steps - 1000 - args.sequence_length
        train_data_raw[sys_name] = full_data[:split_point]
        test_data_raw[sys_name] = full_data[split_point:]

    for sys_name in target_set:
        full_data = raw_data[sys_name]
        raw_data[sys_name] = full_data

    train_sequences = []
    train_k_channels = []

    for sys_name in train_set:
        sequences = prepare_full_sequences(train_data_raw[sys_name], args.sequence_length + 1, interval)  # (N,L+1,3)
        k_val = parse_k_from_name(sys_name)
        k_seq = np.full((sequences.shape[0], args.sequence_length, 1), k_val, dtype=np.float32)

        train_sequences.append(sequences)
        train_k_channels.append(k_seq)

    train_sequences = np.concatenate(train_sequences, axis=0)  # (N,L+1,3)
    train_k_channels = np.concatenate(train_k_channels, axis=0)  # (N,L,1)

    train_inputs_states = train_sequences[:, :-1, :]           # (N,L,3)
    train_targets = train_sequences[:, 1:, :]                  # (N,L,3)
    train_inputs = np.concatenate([train_inputs_states, train_k_channels], axis=-1)  # (N,L,4)

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

    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

    start_time = time.time()
    train_losses = []
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = []

        for inputs, targets in tqdm.tqdm(train_loader):
            inputs = inputs.to(device)   # (B,L,4)
            targets = targets.to(device) # (B,L,3)

            noisy_inputs = inputs.clone()
            if args.noise_level > 0:
                noise = torch.normal(0.0, args.noise_level, size=inputs[:, :, :3].shape, device=inputs.device)
                noisy_inputs[:, :, :3] = inputs[:, :, :3] + inputs[:, :, :3] * noise  # add noise only to state dims

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
    plt.title(f'Iter {iter_index} - Training Loss Over Epochs (k-conditioned)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    # plt.savefig(f'./save_results/transformer_train_{system}_bifurcation_{ks_length}_iter{iter_index}_train_loss.png', dpi=150)
    plt.close()

    results = {}
    model.eval()

    def eval_system(sys_name, data, set_name):
        # short/long iterative rollout from the beginning (no random start)
        init_sequence_states = data[:args.sequence_length]
        true_future_short = data[args.sequence_length:args.sequence_length + short_term_steps]
        true_future_long = data[args.sequence_length:args.sequence_length + long_term_steps]

        k_val = parse_k_from_name(sys_name)
        k_seq_init = np.full((args.sequence_length, 1), k_val, dtype=np.float32)

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
        short_rmse = float(np.sqrt(np.mean((preds_short - true_future_short)**2)))
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
        for dim_idx in range(args.dim):
            plt.subplot(args.dim, 1, dim_idx+1)
            plt.plot(true_future_short[:, dim_idx], label='Actual', alpha=0.7)
            plt.plot(preds_short[:, dim_idx], label='Predicted', alpha=0.7)
            plt.title(f'{sys_name} - Short-term dim {dim_idx+1} (iter {iter_index})')
            plt.legend()
        plt.tight_layout()
        # plt.savefig(f'./save_results/transformer_train_{sys_name}_bifurcation_{ks_length}_iter{iter_index}_short_term.png', dpi=150)
        plt.close()

        # long-term 3D
        if args.dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(true_future_long[:, 0], true_future_long[:, 1], true_future_long[:, 2], label='Actual', alpha=0.7)
            ax.plot(preds_long[:, 0], preds_long[:, 1], preds_long[:, 2], label='Predicted', alpha=0.7)
            plt.legend()
            plt.title(f'{sys_name} - Long-term 3D (iter {iter_index})')
            # plt.savefig(f'./save_results/transformer_train_{sys_name}_bifurcation_{ks_length}_iter{iter_index}_long_term.png', dpi=150)
            plt.close()

        return short_rmse, prediction_horizon, true_future_short, preds_short, true_future_long, preds_long

    with torch.no_grad():
        for eval_set, systems in zip(['test'], [test_set]):
            for sys_name in systems:
                data = test_data_raw[sys_name]

                short_rmse, prediction_horizon, true_future_short, preds_short, true_future_long, preds_long = eval_system(sys_name, data, eval_set)

                results[sys_name] = {
                    'eval_set': eval_set,
                    'short_term_rmse': short_rmse,
                    'prediction_horizon': prediction_horizon,
                    'true_future_short': true_future_short,
                    'preds_short': preds_short,
                    'true_future_long': true_future_long,
                    'preds_long': preds_long,
                }

    for sys_name, metrics in results.items():
        print(f"[iter {iter_index}] System: {sys_name} [{metrics['eval_set']}]\n"
              f"  Short-term RMSE: {metrics['short_term_rmse']:.6f}\n"
              f"  Prediction Horizon: {metrics['prediction_horizon']} steps\n"
              )

    # -------------------------
    # Save results and model (enabled)
    # -------------------------
    os.makedirs('./save_results', exist_ok=True)
    os.makedirs('./save_model', exist_ok=True)

    combined_info = {
        'results': results,
        'args': vars(args),
        'take_num': take_num,
        'data_read_length': data_read_length,
        'train_set': train_set,
        'test_set': test_set,
        'target_set': target_set,
        'interval': interval,
        'short_term_steps': short_term_steps,
        'long_term_steps': long_term_steps,
        'rmse_threshold': rmse_threshold,
        'iter_index': iter_index,
    }

    with open(f'./save_results/transformer_train_{system}_setting_{setting}_iter{iter_index}.pkl', 'wb') as f:
        pickle.dump(combined_info, f)

    torch.save(model.state_dict(), f'./save_model/transformer_model_train_{system}_setting_{setting}_iter{iter_index}.pth')
    print(f'[iter {iter_index}] Saved model and results.')

# -------------------------
# Main loop: run 10 iterations
# -------------------------
if __name__ == "__main__":
    for i in range(11, 21):
        print(f"\n=== Running iteration {i} ===\n")
        run_training(i, args)
