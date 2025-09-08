# run_armse.py
# Benchmark driver for CD-EKF (matrix MDE) vs CD-UKF/CKF (SR time-update)
from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from scipy.integrate import solve_ivp

from filters import ContinuousModel, CDEKF, CDUKF, CDCKF, ensure_psd
from models import (
    dahlquist_f, dahlquist_J, dahlquist_h, dahlquist_H, dahlquist_G, dahlquist_Qc,
    vdp_f, vdp_J, vdp_h, vdp_H, vdp_G, vdp_Qc,
)

# ------------------------------- Truth simulation ---------------------------

def integrate_truth(f, t0: float, tf: float, x0: np.ndarray,
                    rtol=1e-12, atol=1e-12, max_step=1e-2):
    """Deterministic truth integration (drift only)."""
    def rhs(t, x):
        return f(t, x)
    sol = solve_ivp(rhs, (t0, tf), x0, method="Radau",
                    rtol=rtol, atol=atol, max_step=max_step)
    return sol.t, sol.y.T

def sample_solution(ts: np.ndarray, xs: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Linear interpolation of ODE solution xs(ts) at times t_grid."""
    n = xs.shape[1]
    X = np.zeros((t_grid.size, n))
    for j in range(n):
        X[:, j] = np.interp(t_grid, ts, xs[:, j])
    return X

# ------------------------------- Benchmark core -----------------------------

def run_cd(case: str, deltas: List[float], N_runs: int = 20, seed: int = 0, outdir: str = "results"):
    rng0 = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    results = {}

    for delta in deltas:
        if case == "dahlquist":
            # First-case ill-conditioning style: linear process, stiff mu
            mu, j = -1.0e4, 1
            f, J = dahlquist_f(mu, j), dahlquist_J(mu, j)
            G, Qc = dahlquist_G(), dahlquist_Qc()
            x0 = np.array([1.0], dtype=float)
            t0, tf = 0.0, 4.0
            h, H = dahlquist_h(), dahlquist_H()
            R = np.array([[0.04]], dtype=float)
        elif case == "vdp":
            mu = 1.0e4  # stiff Van der Pol
            f, J = vdp_f(mu), vdp_J(mu)
            G, Qc = vdp_G(), vdp_Qc()
            x0 = np.array([2.0, 0.0], dtype=float)
            t0, tf = 0.0, 2.0
            h, H = vdp_h(), vdp_H()
            R = np.array([[0.04]], dtype=float)
        else:
            raise ValueError("case must be 'dahlquist' or 'vdp'")

        cm = ContinuousModel(f=f, J=J, G=G, Qc=Qc)

        # EKF: matrix MDE time update; UKF/CKF: SR time update (different dynamics)
        ekf = CDEKF(cm, h, H, R)
        ukf = CDUKF(cm, h, R)
        ckf = CDCKF(cm, h, R)

        t_grid = np.arange(t0, tf + 1e-12, delta)
        err = {k: [] for k in ["EKF", "UKF", "CKF"]}

        for _ in range(N_runs):
            rng = np.random.default_rng(rng0.integers(1 << 32))

            # Truth (drift-only ODE) and sampling
            ts, xs = integrate_truth(f, t0, tf, x0, max_step=min(1e-2, delta/10))
            x_true = sample_solution(ts, xs, t_grid)

            # Measurements
            chol_R = np.linalg.cholesky(ensure_psd(R))
            zs = np.array([h(x_true[k]) + chol_R @ rng.normal(size=R.shape[0])
                           for k in range(len(t_grid))])

            # Run filters
            for name, flt in [("EKF", ekf), ("UKF", ukf), ("CKF", ckf)]:
                x, P = x0.copy(), np.eye(x0.size, dtype=float) * 1e-2
                x_est = [x.copy()]
                for k in range(1, len(t_grid)):
                    t_prev, t_curr = t_grid[k-1], t_grid[k]
                    x, P = flt.predict(t_prev, t_curr, x, P)
                    x, P, _, _ = flt.update(x, P, zs[k])
                    x_est.append(x.copy())
                x_est = np.vstack(x_est)
                armse = float(np.sqrt(np.mean(np.sum((x_est - x_true) ** 2, axis=1))))
                err[name].append(armse)

        results[delta] = {k: float(np.mean(err[k])) for k in err}

        # Write CSV row
        csv_path = os.path.join(outdir, f"{case}_cd_armse.csv")
        write_header = not os.path.exists(csv_path)
        import csv
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["delta", "EKF", "UKF", "CKF"])
            w.writerow([delta, results[delta]["EKF"], results[delta]["UKF"], results[delta]["CKF"]])

        # Per-delta bar (optional)
        labels = ["EKF", "UKF", "CKF"]
        vals = [results[delta][lab] for lab in labels]
        plt.figure()
        plt.bar(labels, vals)
        plt.ylabel("ARMSE")
        plt.title(f"{case.upper()} — ARMSE (delta={delta:g})")
        plt.tight_layout()
        png_path = os.path.join(outdir, f"{case}_cd_armse_delta{delta:g}.png")
        plt.savefig(png_path, dpi=150)
        plt.close()

    # Summary line plot over delta
    deltas_sorted = sorted(results.keys())
    plt.figure(figsize=(7, 5))
    for name in ["EKF", "UKF", "CKF"]:
        ys = [results[d][name] for d in deltas_sorted]
        plt.plot(deltas_sorted, ys, marker="o", label=name)
    plt.xlabel("sampling period δ")
    plt.ylabel("ARMSE")
    plt.title(f"CD benchmark — {case}")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{case}_cd_summary.png"), dpi=150)
    plt.close()

    return results

# ---------------------------------- CLI -------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["dahlquist", "vdp"], default="dahlquist")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--deltas", type=float, nargs="*", default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    args = parser.parse_args()

    res = run_cd(args.case, args.deltas, N_runs=args.runs, seed=args.seed, outdir=args.outdir)
    for d in sorted(res.keys()):
        print(f"delta={d:g}: EKF={res[d]['EKF']:.6g}, UKF={res[d]['UKF']:.6g}, CKF={res[d]['CKF']:.6g}")
