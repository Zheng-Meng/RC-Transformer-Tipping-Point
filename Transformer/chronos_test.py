import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from chronos import Chronos2Pipeline  # pip install chronos-forecasting


# -----------------------------
# Data loading: foodchain
# -----------------------------
def load_foodchain_traj(
    system_name: str,
    data_dir: str = "./save_data",
    offset: int = 3700-512,
    total_length: int = 2048,
) -> np.ndarray:
    """
    Load foodchain trajectory from {data_dir}/{system_name}.pkl

    Returns:
        traj: shape (3, L) = (D, T_segment)
    """
    file_path = os.path.join(data_dir, f"{system_name}.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find file: {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    traj = np.asarray(data["traj"], dtype=np.float32)  # (T, 3)

    T = traj.shape[0]
    if T < offset + total_length:
        # Fall back if trajectory is shorter than expected
        offset = 0
        total_length = min(total_length, T)

    segment = traj[offset : offset + total_length, :]  # (L, 3)
    return segment.T  # (3, L)


# -----------------------------
# Chronos-2 helper: 1D series
# -----------------------------
def chronos_predict_1d(
    pipeline: Chronos2Pipeline,
    series: np.ndarray,
    context_length: int,
    pred_length: int,
    series_id: str,
):
    """
    Run Chronos-2 on a single 1D series.

    series: shape (L_total,)
    Returns:
        context: (context_length,)
        groundtruth: (pred_length,)
        pred: (pred_length,)
    """
    assert series.ndim == 1
    L_total = series.shape[0]
    needed = context_length + pred_length

    if L_total < needed:
        raise ValueError(
            f"Series too short for context={context_length}, pred={pred_length}: L={L_total}"
        )

    context = series[:context_length]
    groundtruth = series[context_length : context_length + pred_length]

    # Build DataFrame in the style of the Chronos-2 README example
    timestamps = np.arange(context_length, dtype=np.int64)
    context_df = pd.DataFrame(
        {
            "id": series_id,
            "timestamp": timestamps,
            "target": context.astype(np.float32),
        }
    )

    # Zero-shot forecast
    pred_df = pipeline.predict_df(
        context_df,
        future_df=None,
        prediction_length=pred_length,
        quantile_levels=[0.5],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    # Chronos-2 returns a column "predictions" for the median forecast
    pred = pred_df["predictions"].to_numpy(dtype=np.float32)
    return context, groundtruth, pred


# -----------------------------
# Plotting (similar layout to Panda)
# -----------------------------
def plot_foodchain_prediction(
    context: np.ndarray,
    groundtruth: np.ndarray,
    pred: np.ndarray,
    title: str | None = None,
    save_path: str | None = None,
    show_plot: bool = True,
):
    """
    context:     (3, T_context)
    groundtruth: (3, T_pred)
    pred:        (3, T_pred)
    """
    D, T_context = context.shape
    _, T_pred = groundtruth.shape

    # time axes
    context_ts = np.arange(T_context)
    pred_ts = np.arange(T_context, T_context + T_pred)

    fig = plt.figure(figsize=(6, 8))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.65, 0.35], hspace=0.2)
    bottom = outer[1].subgridspec(3, 1, hspace=0.0)

    # 3D attractor
    ax3d = fig.add_subplot(outer[0], projection="3d")
    ax3d.plot(context[0], context[1], context[2], color="black", alpha=0.5, label="context")
    ax3d.plot(
        groundtruth[0],
        groundtruth[1],
        groundtruth[2],
        color="black",
        linestyle="-",
        label="ground truth",
    )
    ax3d.plot(
        pred[0],
        pred[1],
        pred[2],
        color="red",
        linestyle="-",
        label="Chronos-2 pred",
    )
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])
    if title is not None:
        ax3d.set_title(title, fontweight="bold")

    # 1D panels for x,y,z
    labels = ["x", "y", "z"]
    for i in range(3):
        ax = fig.add_subplot(bottom[i])
        ax.plot(context_ts, context[i], color="black", alpha=0.5, label="context" if i == 0 else None)
        ax.plot(
            pred_ts,
            groundtruth[i],
            color="black",
            linestyle="-",
            label="ground truth" if i == 0 else None,
        )
        ax.plot(
            pred_ts,
            pred[i],
            color="red",
            linestyle="-",
            label="Chronos-2 pred" if i == 0 else None,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 2:
            ax.set_xticks([0, T_context, T_context + T_pred])
        ax.set_ylabel(labels[i])

    handles, labels_all = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_all, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.02))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving figure to: {save_path}")
        plt.savefig(save_path, bbox_inches="tight")

    if show_plot:
        plt.show()
    plt.close(fig)


# -----------------------------
# Run Chronos-2 on one foodchain system
# -----------------------------
def run_foodchain_case(
    pipeline: Chronos2Pipeline,
    system_name: str,
    context_length: int = 512,
    pred_length: int = 512,
    save_dir: str | None = "./figs_chronos_foodchain",
):
    # Load (3, L_segment)
    traj = load_foodchain_traj(system_name)
    D, L = traj.shape
    needed = context_length + pred_length
    if L < needed:
        raise ValueError(
            f"{system_name}: not enough points (L={L}) for context={context_length} + pred={pred_length}"
        )

    # Restrict to needed window
    traj_window = traj[:, :needed]  # (3, needed)

    context_list = []
    gt_list = []
    pred_list = []

    # Run Chronos-2 separately on each dimension (univariate)
    for i in range(3):
        series = traj_window[i]  # (needed,)
        c, gt, p = chronos_predict_1d(
            pipeline,
            series,
            context_length=context_length,
            pred_length=pred_length,
            series_id=f"{system_name}_dim{i}",
        )
        context_list.append(c)
        gt_list.append(gt)
        pred_list.append(p)

    context = np.stack(context_list, axis=0)  # (3, context_length)
    groundtruth = np.stack(gt_list, axis=0)   # (3, pred_length)
    pred = np.stack(pred_list, axis=0)        # (3, pred_length)

    # RMSE across all dims & pred steps
    rmse = float(np.sqrt(np.mean((pred - groundtruth) ** 2)))
    print(f"{system_name}: RMSE over prediction window = {rmse:.6f}")

    save_path = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"chronos2_{system_name}.png")

    plot_foodchain_prediction(
        context=context,
        groundtruth=groundtruth,
        pred=pred,
        title=f"Chronos-2: {system_name}",
        save_path=save_path,
        show_plot=True,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Chronos-2 model for zero-shot forecasting
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
    )

    ks = [1.0]
    systems = [f"foodchain_k{k}" for k in ks]

    context_length = 512
    pred_length = 1024

    for system_name in systems:
        print("\n========================")
        print(f"Chronos-2 predicting: {system_name}")
        print("========================")
        run_foodchain_case(
            pipeline,
            system_name,
            context_length=context_length,
            pred_length=pred_length,
            save_dir="./save_figures_11132025",
        )


if __name__ == "__main__":
    main()
