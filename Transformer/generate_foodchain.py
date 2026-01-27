# -*- coding: utf-8 -*-


import numpy as np
import pickle
from scipy.integrate import odeint
from typing import Tuple, Optional
import matplotlib.pyplot as plt

def func_foodchain(x: np.ndarray, t: float, k: float) -> np.ndarray:
    """Right-hand side of the 3-species food-chain ODE."""
    yc, yp = 2.009, 2.876
    xc, xp = 0.4, 0.08
    r0, c0 = 0.16129, 0.5
    x1, x2, x3 = x
    dx1 = x1 * (1 - x1 / k) - xc * yc * x2 * x1 / (x1 + r0)
    dx2 = xc * x2 * (yc * x1 / (x1 + r0) - 1) - xp * yp * x3 * x2 / (x2 + c0)
    dx3 = xp * x3 * (yp * x2 / (x2 + c0) - 1)
    return np.array([dx1, dx2, dx3])

def simulate_foodchain(
    k: float,
    dt: float = 0.1,
    Tmax: float = 12000.0,
    use_attempts: bool = True,
    max_attempts: int = 500,
    z_threshold: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the food-chain system for a given k. Returns (t, traj, x0)."""
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(0.0, Tmax + 1e-12, dt)

    def rand_x0():
        return np.array([
            0.6 + 0.4 * rng.random(),
            0.15 + 0.4 * rng.random(),
            0.3 + 0.5 * rng.random(),
        ])

    if not use_attempts:
        x0 = rand_x0()
        traj = odeint(func_foodchain, x0, t, args=(k,))
        return t, traj, x0

    attempts = 0
    while attempts < max_attempts:
        x0 = rand_x0()
        traj = odeint(func_foodchain, x0, t, args=(k,))
        if np.min(traj[:, 2]) > z_threshold:
            return t, traj, x0
        attempts += 1
        print(attempts)
    
    return t, traj, x0  # last attempt if none pass

def save_foodchain(k: float, **kwargs):
    """Simulate and save one run to 'foodchain_k{value}.pkl'."""
    t, traj, x0 = simulate_foodchain(k, **kwargs)
    
    traj = traj[::10, :]
    print(np.shape(traj))
    
    plot_length = 10000
    
    fig, ax = plt.subplots(3,1)
    ax[0].plot(traj[:plot_length, 0])
    ax[1].plot(traj[:plot_length, 1])
    ax[2].plot(traj[:plot_length, 2])
    plt.show()
    
    save_path = f"./save_data/foodchain_k{round(k, 4)}.pkl"  # 4 decimals for safety
    bundle = {"t": t, "traj": traj, "x0": x0, "k": k, "params": kwargs}
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    # Example: generate multiple files
    # ks = [0.97, 0.98, 0.99, 1.0, 1.05]
    # for k in ks:
    #     save_foodchain(k, dt=0.1, Tmax=2000.0, use_attempts=True) 
    
    T_max = 50000 * 10
    
    # save_foodchain(k=0.97, dt=0.1, Tmax=T_max, use_attempts=True)
    # save_foodchain(k=0.98, dt=0.1, Tmax=T_max, use_attempts=True)
    # save_foodchain(k=0.99, dt=0.1, Tmax=T_max, use_attempts=True)
    
    ks = np.arange(0.970, 0.996, 0.001)
    
    for k in ks:
        save_foodchain(k, dt=0.1, Tmax=T_max, use_attempts=True) 
    
    
    # T_max_c = 5000 * 10
    # save_foodchain(k=1.0, dt=0.1, Tmax=T_max_c, use_attempts=False)
    # save_foodchain(k=1.1, dt=0.1, Tmax=T_max_c, use_attempts=False)

























