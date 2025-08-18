
import os, numpy as np, matplotlib.pyplot as plt
from models import dahlquist_f, dahlquist_J, dahlquist_h, dahlquist_H
from discretization import simulate_sde_it15
from filters import DiscreteModel, EKF, UKF, CKF, IDEKF

os.makedirs("results", exist_ok=True)

def build_discrete_model(f, Jf, G, dt):
    def g(x):
        dx = f(x, 0.0)
        J = Jf(x, 0.0)
        return x + dx*dt + 0.5*(J @ dx)*(dt**2)
    def F(x):
        return np.eye(x.shape[0]) + Jf(x, 0.0)*dt
    Qd = G @ G.T * dt
    return DiscreteModel(g, F, Qd)

def grid_divergence(mu_vals, dt_vals, N_runs=20, seed=0):
    rng0 = np.random.default_rng(seed)
    j = 3
    G = np.array([[1.0]])
    x0 = np.array([1.0])
    t0, tf = 0.0, 4.0
    h, H = dahlquist_h(), dahlquist_H()
    R = np.array([[0.04]])

    names = ["EKF","UKF","CKF","IDEKF"]
    grids = {n: np.zeros((len(mu_vals), len(dt_vals))) for n in names}

    for i_mu, mu in enumerate(mu_vals):
        f, Jf = dahlquist_f(mu, j), dahlquist_J(mu, j)
        for j_dt, dt in enumerate(dt_vals):
            dm = build_discrete_model(f, Jf, G, dt)
            div_counts = dict((n, 0) for n in names)
            for r in range(N_runs):
                rng = np.random.default_rng(rng0.integers(1<<32))
                ts, xs = simulate_sde_it15(x0, t0, tf, dt, f, Jf, G, rng, refine=10)
                zs = np.zeros((xs.shape[0], R.shape[0]))
                for k in range(xs.shape[0]):
                    zs[k] = h(xs[k]) + np.linalg.cholesky(R) @ rng.normal(size=R.shape[0])

                P_init = np.eye(x0.shape[0]) * 1e-2
                x_init = x0 + 0.1*rng.normal(size=x0.shape[0])

                ekf  = EKF(dm, h, H, R)
                ukf  = UKF(dm, h, R)
                ckf  = CKF(dm, h, R)
                idek = IDEKF(dm, h, H, R)

                for name, ftr in zip(names, [ekf, ukf, ckf, idek]):
                    x, P = x_init.copy(), P_init.copy()
                    diverged = False
                    for k in range(1, xs.shape[0]):
                        try:
                            x_pred, P_pred = ftr.predict(x, P)
                            x, P, v, S = ftr.update(x_pred, P_pred, zs[k])
                        except Exception:
                            diverged = True
                            break
                    if diverged:
                        div_counts[name] += 1
                    else:
                        # PSD check at final time
                        Pk = 0.5*(P + P.T)
                        try:
                            np.linalg.cholesky(Pk + 1e-12*np.eye(Pk.shape[0]))
                        except np.linalg.LinAlgError:
                            div_counts[name] += 1
            for name in names:
                grids[name][i_mu, j_dt] = div_counts[name] / N_runs

    for name in names:
        plt.figure()
        plt.imshow(grids[name], origin="lower", aspect="auto",
                   extent=[dt_vals[0], dt_vals[-1], mu_vals[0], mu_vals[-1]])
        plt.colorbar(label="Divergence rate")
        plt.xlabel("dt")
        plt.ylabel("mu")
        plt.title(f"Dahlquist (j=3) — {name} divergence rate")
        plt.tight_layout()
        plt.savefig(f"results/dahlquist_divergence_{name}.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    mu_vals = [-1e0, -1e2, -1e4]
    dt_vals = [0.05, 0.1, 0.2, 0.5]
    grid_divergence(mu_vals, dt_vals, N_runs=20, seed=0)
    print("Stability heatmaps saved to ./results")
