
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from scipy.integrate import odeint

# --- Fixed constants from the paper (see Phys. Rev. Research 3, 013090 (2021)) ---
Kpw = 0.4
Kpv = 0.3
Kqw = -0.03
Kqv = -2.8
Kqv2 = 2.1
T = 8.5
P0 = 0.6
P1 = 0.0
Q0 = 1.3

Y0 = 3.33
Ym = 5.0
Pm = 1.0
dm = 0.05
M  = 0.01464
Em = 1.05
E0 = 1.0
theta0 = 0.0
thetam = 0.0   # kept for completeness (not directly used)

# Adjusted Thévenin equivalents (E0', Y0', theta0')
def _thevenin_adjusted(C: float = 3.5):
    CY = C / Y0
    rad = np.sqrt(1.0 + CY**2 - 2.0*CY*np.cos(theta0))
    E0p = E0 / rad
    Y0p = Y0 * rad
    # robust arctan2 form for the angle shift:
    num = CY * np.sin(theta0)
    den = 1.0 - CY * np.cos(theta0)
    theta0p = theta0 + np.arctan2(num, den)
    return E0p, Y0p, theta0p

E0p, Y0p, theta0p = _thevenin_adjusted(C=3.5)

# --- RHS of the 4D ODE: x = [deltam, omega, delta, V] ---
def func_voltage(x: np.ndarray, t: float, Q1: float) -> np.ndarray:
    deltam, omega, delta, V = x

    # Network powers with adjusted Thévenin equivalents
    # P(δm, δ, V) and Q(δm, δ, V)
    P = -E0p * V * Y0p * np.sin(delta) + Em * V * Ym * np.sin(deltam - delta)
    Q =  E0p * V * Y0p * np.cos(delta) - (Y0p + Ym) * V**2 + Em * V * Ym * np.cos(deltam - delta)

    dx1 = omega
    dx2 = (-dm * omega + Pm - Em * V * Ym * np.sin(deltam - delta)) / M
    dx3 = (-Kqv2 * V**2 - Kqv * V + Q - Q0 - Q1) / Kqw
    dx4 = (Kpw*Kqv2*V**2 + (Kpw*Kqv - Kqw*Kpv)*V + Kqw*(P - P0 - P1) - Kpw*(Q - Q0 - Q1)) / (T*Kqw*Kpv)

    return np.array([dx1, dx2, dx3, dx4], dtype=float)

def simulate_voltage(
    Q1: float,
    dt: float = 0.05,
    Tmax: float = 2000.0,
    use_attempts: bool = True,
    max_attempts: int = 200,
    V_threshold: float = 0.6,         # ensure a “non-collapsed” segment if desired
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the voltage-collapse ODE for a given Q1. Returns (t, traj, x0).
    - If use_attempts=True, resamples x0 until min(V) > V_threshold (or attempts exhausted).
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(0.0, Tmax + 1e-12, dt)

    def rand_x0():
        # Reasonable initial ranges: small rotor angle/speed, load angle small, voltage near ~0.8
        return np.array([
            rng.uniform(-0.2, 0.2),    # deltam
            rng.uniform(-0.2, 0.2),    # omega
            rng.uniform(-0.2, 0.2),    # delta
            rng.uniform(0.75, 0.9),    # V
        ], dtype=float)

    if not use_attempts:
        x0 = rand_x0()
        traj = odeint(func_voltage, x0, t, args=(Q1,))
        return t, traj, x0

    attempts = 0
    last = None
    while attempts < max_attempts:
        x0 = rand_x0()
        traj = odeint(func_voltage, x0, t, args=(Q1,))
        last = (t, traj, x0)
        if np.min(traj[:, 3]) > V_threshold:
            return t, traj, x0
        attempts += 1
        print(f"[attempt {attempts}] min(V)={np.min(traj[:,3]):.4f} <= {V_threshold} — retrying...")

    # If none pass, return the last attempt anyway (may include collapse/transient)
    print("[warn] Could not satisfy V_threshold; returning last attempt (may include collapse).")
    return last

def save_voltage(
    Q1: float,
    dt: float = 0.05,
    Tmax: float = 2000.0,
    use_attempts: bool = True,
    max_attempts: int = 200,
    V_threshold: float = 0.6,
    downsample_stride: int = 10,
    preview_len: int = 1000,
    save_dir: str = "./save_data",
):
    """
    Simulate and save one run to 'voltage_Q1{value}.pkl' (one file per parameter).
    - Quick preview plot of the four state channels.
    """
    t, traj, x0 = simulate_voltage(
        Q1=Q1, dt=dt, Tmax=Tmax,
        use_attempts=use_attempts, max_attempts=max_attempts,
        V_threshold=V_threshold
    )

    if downsample_stride > 1 and traj.shape[0] > 0:
        traj = traj[::downsample_stride, :]

    print("traj shape:", traj.shape)

    # Quick preview
    pl = min(preview_len, traj.shape[0])
    if pl > 0:
        fig, ax = plt.subplots(4, 1, figsize=(9, 6), sharex=True)
        ax[0].plot(traj[:pl+1000, 0]); ax[0].set_ylabel("deltam")
        ax[1].plot(traj[:pl+1000, 1]); ax[1].set_ylabel("omega")
        ax[2].plot(traj[:pl+1000, 2]); ax[2].set_ylabel("delta")
        ax[3].plot(traj[:pl+1000, 3]); ax[3].set_ylabel("V"); ax[3].set_xlabel("step (post-burn)")
        plt.tight_layout(); 
        # plt.savefig(f"./save_plot/voltage_Q1{Q1:.6f}.png")
        plt.show()
        plt.close()
    else:
        print("[warn] No samples to preview (likely escaped/collapsed quickly).")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"voltage_Q1{Q1:.6f}.pkl")

    bundle = {
        "t": t,
        "traj": traj,           # shape (N, 4) with columns [deltam, omega, delta, V]
        "x0": traj[0, :].copy() if traj.shape[0] else np.array([0,0,0,0], dtype=float),
        "Q1": Q1,
        "params": {
            "dt": dt, "Tmax": Tmax,
            "use_attempts": use_attempts,
            "max_attempts": max_attempts,
            "V_threshold": V_threshold,
            "downsample_stride": downsample_stride,
        },
    }
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    q1s = [2.98983]

    DT = 0.03
    T_max = 50000.0 * 10 * DT
    
        
    for q in q1s:
        save_voltage(
            Q1=q,
            dt=DT,
            Tmax=T_max,
            use_attempts=False,
            max_attempts=300,
            V_threshold=0.6,       # relax to ~0.55 if it rejects too many initializations
            downsample_stride=1,
            preview_len=5000,
            save_dir="./save_data",
        )
