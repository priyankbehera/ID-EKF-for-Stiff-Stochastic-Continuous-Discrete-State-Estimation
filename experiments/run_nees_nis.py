
import os, numpy as np, matplotlib.pyplot as plt
from models import dahlquist_f, dahlquist_J, dahlquist_h, dahlquist_H, vdp_f, vdp_J, vdp_h, vdp_H
from discretization import simulate_sde_it15
from filters import DiscreteModel, EKF, UKF, CKF, IDEKF
from common import chi2_bounds_mean_nees

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

def run_nees_nis(model_name: str, dt=0.2, N_runs=30, seed=0):
    rng0 = np.random.default_rng(seed)
    if model_name == "dahlquist":
        mu, j = -1e4, 3
        f, Jf = dahlquist_f(mu, j), dahlquist_J(mu, j)
        G = np.array([[1.0]])
        x0 = np.array([1.0])
        t0, tf = 0.0, 4.0
        h, H = dahlquist_h(), dahlquist_H()
        R = np.array([[0.04]])
        state_dim = 1
        m = 1
    else:
        mu = 1e4
        f, Jf = vdp_f(mu), vdp_J(mu)
        G = np.array([[0.0, 0.0],[0.0, 1.0]])
        x0 = np.array([2.0, 0.0])
        t0, tf = 0.0, 2.0
        h, H = vdp_h(), vdp_H()
        R = np.array([[0.04]])
        state_dim = 2
        m = 1

    dm = build_discrete_model(f, Jf, G, dt)
    T_steps = int(np.ceil((tf - t0)/dt)) + 1
    names = ["EKF","UKF","CKF","IDEKF"]

    all_mean_nees = {}
    all_mean_nis  = {}

    for name in names:
        all_mean_nees[name] = np.zeros(T_steps)
        all_mean_nis[name]  = np.zeros(T_steps)

    # Accumulate sums to compute means across runs
    counts = 0
    for i in range(N_runs):
        rng = np.random.default_rng(rng0.integers(1<<32))
        ts, xs = simulate_sde_it15(x0, t0, tf, dt, f, Jf, G, rng, refine=10)
        zs = np.zeros((xs.shape[0], R.shape[0]))
        for k in range(xs.shape[0]):
            zs[k] = h(xs[k]) + np.linalg.cholesky(R) @ rng.normal(size=R.shape[0])

        x_init = x0 + 0.1*rng.normal(size=x0.shape[0])
        P_init = np.eye(x0.shape[0]) * 1e-2

        from filters import EKF, UKF, CKF, IDEKF
        ekf  = EKF(dm, h, H, R)
        ukf  = UKF(dm, h, R)
        ckf  = CKF(dm, h, R)
        idek = IDEKF(dm, h, H, R)

        # storage
        X  = {n: np.zeros_like(xs) for n in names}
        P  = {n: np.zeros((xs.shape[0], x0.shape[0], x0.shape[0])) for n in names}
        V  = {n: np.zeros((xs.shape[0], R.shape[0])) for n in names}
        S  = {n: np.zeros((xs.shape[0], R.shape[0], R.shape[0])) for n in names}

        for name, ftr in zip(names, [ekf, ukf, ckf, idek]):
            x, Pk = x_init.copy(), P_init.copy()
            X[name][0], P[name][0] = x.copy(), Pk.copy()
            for k in range(1, xs.shape[0]):
                x_pred, P_pred = ftr.predict(x, Pk)
                x, Pk, v, Sk = ftr.update(x_pred, P_pred, zs[k])
                X[name][k], P[name][k] = x, Pk
                V[name][k], S[name][k] = v, Sk

        for name in names:
            e = X[name] - xs
            # NEES: e^T P^{-1} e
            nees_vals = np.array([e[k].T @ np.linalg.solve(P[name][k], e[k]) for k in range(xs.shape[0])])
            nis_vals  = np.array([V[name][k].T @ np.linalg.solve(S[name][k], V[name][k]) for k in range(xs.shape[0])])
            all_mean_nees[name] += nees_vals
            all_mean_nis[name]  += nis_vals
        counts += 1

    for name in names:
        all_mean_nees[name] /= counts
        all_mean_nis[name]  /= counts
        t = np.arange(T_steps)*dt
        lower, upper = chi2_bounds_mean_nees(0.05, state_dim, counts)
        plt.figure()
        plt.plot(t, all_mean_nees[name], label=f"{name} mean NEES")
        plt.axhline(lower, linestyle="--")
        plt.axhline(upper, linestyle="--")
        plt.xlabel("time")
        plt.ylabel("NEES")
        plt.title(f"{model_name.upper()} — {name} NEES (dt={dt})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{model_name}_{name}_NEES_dt{dt:.3g}.png", dpi=150)
        plt.close()

        plt.figure()
        plt.plot(t, all_mean_nis[name], label=f"{name} mean NIS")
        plt.xlabel("time")
        plt.ylabel("NIS")
        plt.title(f"{model_name.upper()} — {name} NIS (dt={dt})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{model_name}_{name}_NIS_dt{dt:.3g}.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    run_nees_nis("dahlquist", dt=0.2, N_runs=30, seed=0)
    run_nees_nis("vdp",        dt=0.2, N_runs=30, seed=1)
    print("NEES/NIS plots saved to ./results")
