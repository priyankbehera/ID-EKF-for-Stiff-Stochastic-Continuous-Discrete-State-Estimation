
import numpy as np
from typing import Tuple, Callable

def sample_dW_J10(dt: float, q: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    # For each of q independent Wiener channels:
    # Var(dW_i) = dt, Var(J10_i) = dt^3/3, Cov(dW_i, J10_i) = dt^2/2
    U = rng.normal(size=q)
    V = rng.normal(size=q)
    dW = np.sqrt(dt) * U
    J10 = 0.5 * dt * dW + (dt ** 1.5) * V / (2.0 * np.sqrt(3.0))
    return dW, J10

def ito_taylor_1p5_step_additive(x: np.ndarray, t: float, dt: float,
                                 f: Callable[[np.ndarray, float], np.ndarray],
                                 Jf: Callable[[np.ndarray, float], np.ndarray],
                                 G: np.ndarray,
                                 rng: np.random.Generator) -> np.ndarray:
    # Strong order 1.5 Ito-Taylor (additive noise) step:
    # x_{k+1} = x_k + f(x_k) dt + G dW + 0.5 * J_f(x_k) @ f(x_k) * dt^2 + J_f(x_k) @ (G @ J10)
    dx = f(x, t)
    J = Jf(x, t)
    q = G.shape[1] if G.ndim == 2 else 1
    dW, J10 = sample_dW_J10(dt, q, rng)
    term1 = x + dx * dt
    term2 = G @ dW
    term3 = 0.5 * (J @ dx) * (dt ** 2)
    term4 = J @ (G @ J10)
    return term1 + term2 + term3 + term4

def simulate_sde_it15(x0: np.ndarray, t0: float, tf: float, dt: float,
                      f, Jf, G: np.ndarray, rng: np.random.Generator,
                      refine: int = 1):
    n_steps = int(np.ceil((tf - t0) / dt))
    n = x0.shape[0]
    xs = np.zeros((n_steps + 1, n))
    ts = np.zeros(n_steps + 1)
    xs[0] = x0.copy()
    ts[0] = t0
    sub_dt = dt / refine
    for k in range(n_steps):
        x = xs[k].copy()
        t = t0 + k * dt
        for r in range(refine):
            x = ito_taylor_1p5_step_additive(x, t + r*sub_dt, sub_dt, f, Jf, G, rng)
        xs[k+1] = x
        ts[k+1] = t + dt
    return ts, xs
