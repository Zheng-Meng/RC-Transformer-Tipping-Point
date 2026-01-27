
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

# Paper settings (can override via kwargs)
IKEDA_DEFAULTS = dict(gamma=0.9, kappa=0.4, nu=6.0)

def ikeda_step(z: complex, mu: float, gamma: float, kappa: float, nu: float) -> complex:
    # z_{n+1} = mu + gamma * z_n * exp( i*kappa - i*nu/(1+|z_n|^2) )
    phase = kappa - (nu / (1.0 + abs(z)**2))
    return mu + gamma * z * np.exp(1j * phase)

def simulate_ikeda(
    mu: float,
    n_burn: int = 8000,
    n_keep: int = 50000,
    z0: Optional[complex] = None,
    gamma: float = IKEDA_DEFAULTS["gamma"],
    kappa: float = IKEDA_DEFAULTS["kappa"],
    nu: float = IKEDA_DEFAULTS["nu"],
    escape_radius: float = 50.0,
    collect_until_escape: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterate the Ikeda map at parameter mu.
    Returns (t, traj, x0) where traj is shape (N,2) with columns [x, y].
    - Burn 'n_burn' steps, then keep up to 'n_keep' post-transient points.
    - If |z| exceeds escape_radius and collect_until_escape=True, stop early.
    """
    if z0 is None:
        z = 0.0 + 0.0j
    else:
        z = complex(z0)

    # burn-in
    for _ in range(n_burn):
        z = ikeda_step(z, mu, gamma, kappa, nu)
        if abs(z) > escape_radius and collect_until_escape:
            # escaped during burn; return empty post-transient
            return np.array([], dtype=float), np.empty((0, 2), dtype=float), np.array([z.real, z.imag], dtype=float)

    # collect post-transient
    pts: List[complex] = []
    for _ in range(n_keep):
        z = ikeda_step(z, mu, gamma, kappa, nu)
        if abs(z) > escape_radius:
            if collect_until_escape:
                break
            else:
                continue
        pts.append(z)

    if len(pts) == 0:
        t = np.array([], dtype=float)
        traj = np.empty((0, 2), dtype=float)
    else:
        traj = np.column_stack((np.real(pts), np.imag(pts)))  # (N,2): [x,y]
        t = np.arange(traj.shape[0], dtype=float)             # iteration index after burn

    x0_vec = np.array([traj[0,0], traj[0,1]], dtype=float) if traj.shape[0] else np.array([0.0, 0.0], dtype=float)
    return t, traj, x0_vec

def save_ikeda(
    mu: float,
    downsample_stride: int = 10,
    n_burn: int = 8000,
    n_keep: int = 50000,
    z0: Optional[complex] = None,
    gamma: float = IKEDA_DEFAULTS["gamma"],
    kappa: float = IKEDA_DEFAULTS["kappa"],
    nu: float = IKEDA_DEFAULTS["nu"],
    escape_radius: float = 50.0,
    collect_until_escape: bool = True,
    save_dir: str = "./save_data",
):
    """
    Simulate and save one Ikeda run to 'ikeda_mu{value}.pkl'
    - Quick preview plot for x(t), y(t).
    Bundle keys: {"t", "traj", "x0", "mu", "params"}.
    """
    t, traj, x0 = simulate_ikeda(
        mu=mu,
        n_burn=n_burn,
        n_keep=n_keep,
        z0=z0,
        gamma=gamma, kappa=kappa, nu=nu,
        escape_radius=escape_radius,
        collect_until_escape=collect_until_escape,
    )

    if downsample_stride > 1 and traj.shape[0] > 0:
        traj = traj[::downsample_stride, :]

    print("traj shape:", traj.shape)

    # quick preview
    # plot_len = min(10000, traj.shape[0]) if traj.shape[0] else 0
    # if plot_len > 0:
    #     fig, ax = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
    #     ax[0].plot(traj[:plot_len, 0])
    #     ax[0].set_ylabel("x = Re(z)")
    #     ax[1].plot(traj[:plot_len, 1])
    #     ax[1].set_ylabel("y = Im(z)")
    #     ax[1].set_xlabel("iteration (post-burn)")
    #     plt.tight_layout()
    #     plt.show()
    # else:
    #     print("[warn] No post-transient points collected (escaped too early).")

    # plot the 2d view, of real and imag in one figure
    plot_len = min(10000, traj.shape[0]) if traj.shape[0] else 0
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.scatter(traj[:plot_len, 0], traj[:plot_len, 1], label="Re(z) vs Im(z)")
    ax.set_ylabel("Im(z)")
    ax.set_xlabel("Re(z)")
    ax.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"./save_plot/ikeda_mu{round(mu, 4)}.png")
    # plt.close()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ikeda_mu{round(mu, 4)}.pkl")
    bundle = {
        "t": t,
        "traj": traj,       # shape (N,2)
        "x0": x0,           # [x0, y0]
        "mu": mu,
        "params": {
            "n_burn": n_burn,
            "n_keep": n_keep,
            "downsample_stride": downsample_stride,
            "gamma": gamma, "kappa": kappa, "nu": nu,
            "escape_radius": escape_radius,
            "collect_until_escape": collect_until_escape,
        },
    }
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    mus = np.arange(0.91, 1.01, 0.002)
    mus = [1.01]

    N_KEEP = 50000 * 10
    STRIDE = 1

    for mu in mus:
        save_ikeda(
            mu=mu,
            downsample_stride=STRIDE,
            n_burn=1,
            n_keep=N_KEEP,
            z0=1.0 - 1.0j,
            gamma=0.9, kappa=0.4, nu=6.0,   # paper settings
            escape_radius=50.0,
            collect_until_escape=True,
            save_dir="./save_data",
        )
