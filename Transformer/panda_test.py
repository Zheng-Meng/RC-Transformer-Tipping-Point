import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Panda plotting utils (same as in panda_test1.py) ---
from panda.utils.plot_utils import (
    apply_custom_style,
    make_clean_projection,
)

from panda.patchtst.pipeline import PatchTSTPipeline

# Optional: use the same plotting style as the notebook
apply_custom_style("config/plotting.yaml")


def plot_model_prediction(
    model,
    context: np.ndarray,
    groundtruth: np.ndarray,
    prediction_length: int,
    title: str | None = None,
    save_path: str | None = None,
    show_plot: bool = True,
    figsize: tuple[int, int] = (6, 8),
    **kwargs,
):
    """
    context:     shape (D, T_context)
    groundtruth: shape (D, T_pred)
    """
    # convert to (B, T, D) tensor on model device
    context_tensor = torch.from_numpy(context.T).float().to(model.device)[None, ...]

    pred = model.predict(context_tensor, prediction_length, **kwargs)
    pred = pred.squeeze().cpu().numpy()  # (T_pred, D)

    total_length = context.shape[1] + prediction_length
    context_ts = np.arange(context.shape[1]) / total_length
    pred_ts = np.arange(context.shape[1], total_length) / total_length

    # Ensure continuity between context and groundtruth in the plot
    if context.shape[1] > 0 and groundtruth.shape[1] > 0:
        last_context_point = context[:, -1][:, np.newaxis]
        groundtruth = np.hstack((last_context_point, groundtruth))

    # Prepend last context point to prediction timeline and data for continuity
    pred_ts = np.concatenate(([context_ts[-1]], pred_ts))
    if pred.shape[0] + 1 == len(pred_ts):
        pred = np.vstack((context[:, -1], pred))

    # --- Plotting ---
    fig = plt.figure(figsize=figsize)

    outer_grid = fig.add_gridspec(2, 1, height_ratios=[0.65, 0.35], hspace=-0.1)
    gs = outer_grid[1].subgridspec(3, 1, height_ratios=[0.2] * 3, wspace=0, hspace=0)

    ax_3d = fig.add_subplot(outer_grid[0], projection="3d")

    # 3D trajectory (use first 3 dimensions)
    ax_3d.plot(*context[:3], alpha=0.5, color="black", label="Context")
    ax_3d.plot(*groundtruth[:3], linestyle="-", color="black", label="Groundtruth")
    ax_3d.plot(*pred.T[:3], color="red", label="Prediction")
    make_clean_projection(ax_3d)

    if title is not None:
        title_name = title.replace("_", " ")
        ax_3d.set_title(title_name, fontweight="bold")

    axes_1d = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for i, ax in enumerate(axes_1d):
        ax.plot(context_ts, context[i], alpha=0.5, color="black")
        ax.plot(pred_ts, groundtruth[i], linestyle="-", color="black")
        ax.plot(pred_ts, pred[:, i], color="red")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("auto")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"saving fig to: {save_path}")
        plt.savefig(save_path, bbox_inches="tight", dpi=600)

    if show_plot:
        plt.show()
    plt.close()


def load_foodchain_traj(system_name: str,
                        data_dir: str = "./save_data",
                        data_read_length: int = 50000) -> np.ndarray:
    """
    Load foodchain trajectory from the same .pkl files used in transformer_train_foodchain.py.

    Returns:
        traj: shape (D, T) with D=3 (x,y,z).
    """
    file_path = os.path.join(data_dir, f"{system_name}.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find {file_path}. Make sure your foodchain data is saved there.")

    with open(file_path, "rb") as f:
        data = pickle.load(f)
    traj = data["traj"]  # shape (T, 3)

    if traj.shape[0] > data_read_length:
        start = 10000
        end = start + data_read_length
        if end <= traj.shape[0]:
            traj = traj[start:end]
        else:
            traj = traj[-data_read_length:]

    # convert to (D, T)
    return traj.T


def run_foodchain_case(
    system_name: str,
    model: PatchTSTPipeline,
    context_length: int = 512,
    pred_length: int = 512,
    save_dir: str | None = None,
):
    """
    Run Panda prediction for one foodchain_k* system.
    """
    # Load trajectory (D, T)
    traj = load_foodchain_traj(system_name)  # (3, T)
    D, T = traj.shape

    if T < context_length + pred_length:
        raise ValueError(
            f"Not enough data in {system_name}: need at least "
            f"{context_length + pred_length}, but got {T}"
        )

    # Here we simply take the first window;
    start_time = 3700-512
    end_time = start_time + context_length

    context = traj[:, start_time:end_time]              # (3, context_length)
    groundtruth = traj[:, end_time:end_time + pred_length]  # (3, pred_length)

    # Simple RMSE just to have a number
    # (not sliding; just on the chosen window)
    context_tensor = torch.from_numpy(context.T).float().to(model.device)[None, ...]
    pred = model.predict(context_tensor, pred_length,
                         limit_prediction_length=False,
                         sliding_context=True)
    pred_np = pred.squeeze(0).cpu().numpy()  # (pred_length, 3)
    rmse = float(np.sqrt(np.mean((pred_np - groundtruth.T) ** 2)))
    print(f"{system_name}: RMSE over prediction window = {rmse:.6f}")

    save_path = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"panda_{system_name}.png")

    # Plot with the same helper as Lorenz
    plot_model_prediction(
        model,
        context,
        groundtruth,
        pred_length,
        limit_prediction_length=False,
        sliding_context=True,
        save_path=save_path,
        show_plot=True,
        figsize=(6, 8),
        title=system_name,
    )


def main():
    # 1. Load Panda model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pft_model = PatchTSTPipeline.from_pretrained(
        mode="predict",
        pretrain_path="GilpinLab/panda",
        device_map=device,
    )

    # 2. Foodchain systems (same naming as transformer_train_foodchain.py)
    ks = [0.99, 1.0]
    systems = [f"foodchain_k{k}" for k in ks]

    context_length = 512
    pred_length = 1024  # must be multiple of 128 for Panda
    assert pred_length % 128 == 0, "prediction length must be multiple of 128"

    for system_name in systems:
        print("\n========================")
        print(f"Predicting system: {system_name}")
        print("========================")
        run_foodchain_case(
            system_name,
            pft_model,
            context_length=context_length,
            pred_length=pred_length,
            save_dir="./save_figures_11132025",
        )


if __name__ == "__main__":
    main()
