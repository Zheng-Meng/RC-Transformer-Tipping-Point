<h1 align="center">Can transformers predict system collapse in dynamical systems?</h1>

<p align="center">
<img src="images/rc_transformer_tipping.png" width="700">
</p>

Code for training **Transformer** digital twins on safe-regime chaotic time series and testing them near **collapse**, compared with **reservoir computing (RC)** in `RC/` (MATLAB). Transformers forecast well in-distribution but often miss collapse; RC tracks it more reliably (see figure).

## Quick start

```bash
conda create -n RC-Transformer python=3.10 -y && conda activate RC-Transformer
pip install -r requirements.txt      # includes torch; for CUDA wheels see https://pytorch.org

cd Transformer
mkdir -p save_data save_models save_for_plot
python generate_foodchain.py         # edit __main__ for ks=[0.97,0.98,0.99,1.0] (default loop is slow)
python transformer_train_iter_foodchain.py --iter-start 0 --iter-end 12
python transformer_test_iter_foodchain.py --iters 12
```

**Always run Python from `Transformer/`** so `./save_data` and `./save_models` resolve correctly.

## Layout

| Path | Contents |
|------|----------|
| `Transformer/` | `generate_*`, `transformer_train_*`, `transformer_test_*`, `transformer_decoder.py` |
| `Transformer/save_data/` | Trajectories (`foodchain_k*.pkl`, `ikeda_mu*.pkl`, `voltage_Q1*.pkl`, `KS_train_data.mat`) |
| `Transformer/save_models/` | Iter checkpoints, e.g. `model_foodchain_iter_00.pth` |
| `RC/` | MATLAB STP / reservoir baselines (`System_Food_Chain`, `System_Ikeda_Map`, …) |

## Food chain workflow

1. **Data** — `generate_foodchain.py` → `save_data/foodchain_k{k}.pkl` (train: k=0.97–0.99; test collapse: k=1.0).
2. **Train** — `transformer_train_iter_foodchain.py` → `save_models/model_foodchain_iter_XX.pth`.
3. **Test** — `transformer_test_iter_foodchain.py` sweeps **k**, detects collapse when species z drops below a threshold.

Single-run alternative: `transformer_train_foodchain.py` / `transformer_test_foodchain.py` (edit checkpoint paths in the script).

## Other systems

| System | Generate | Train | Test sweep |
|--------|----------|-------|------------|
| Ikeda | `generate_ikeda.py` | `transformer_train_ikeda.py` | `transformer_test_iter_ikeda.py` |
| Voltage | `generate_voltage.py` | `transformer_train_voltage.py` | `transformer_test_iter_voltage.py` |
| KS | add `save_data/KS_train_data.mat` | `transformer_train_KS.py` | `transformer_test_iter_KS.py` |

## Reservoir computing (MATLAB)

Run scripts under `RC/System_*` (e.g. `FoodChain_STP_10_train.m`, `FoodChain_STP_10_predict.m`) from MATLAB. Shared helpers live in `RC/Functions/`. No Python `requirements.txt` needed for RC.

## Optional baselines

| Script | Install |
|--------|---------|
| `chronos_test.py` | `chronos-forecasting` (in `requirements.txt`) |
| `panda_test.py` | `pip install "git+https://github.com/abao1999/panda.git"` — **not** `pip install panda` |

## Troubleshooting

- **`FileNotFoundError: save_data/...`** — `cd Transformer` and run `generate_*.py` first.
- **No `torch` / wrong CUDA build** — `torch` is in `requirements.txt`; if GPU is needed, install the matching build from [pytorch.org](https://pytorch.org) first, then `pip install -r requirements.txt`.
- **Bad predictions after load** — test script architecture must match training; check `--d-model`, `--nhead`, `--num-layers`.
- **Generator hangs** — default food-chain `ks` grid is huge; use a short `ks` list in `generate_foodchain.py`.

## Citation

If you use this code, please cite the publication associated with this repository and figure.
