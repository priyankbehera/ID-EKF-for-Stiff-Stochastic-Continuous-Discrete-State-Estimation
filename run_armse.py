# run_armse.py
# Benchmark driver for conventional CD-EKF / CD-UKF / CD-CKF (no square-root).
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
    vdp_f, vdp_J, vdp_h, vdp_H, vdp_h_nonlinear, vdp_H_nonlinear, vdp_G, vdp_Qc,
)

# ------------------------------- Truth simulation ---------------------------

def integrate_truth(f, J, t0: float, tf: float, x0: np.ndarray, t_eval: np.ndarray,
                    rtol=1e-6, atol=1e-9, max_step=np.inf, method="BDF"):
    sol = solve_ivp(lambda t, x: f(t, x),
                    (t0, tf), x0, method=method,
                    jac=lambda t, x: J(t, x),
                    t_eval=t_eval,
                    rtol=rtol, atol=atol, max_step=max_step)
    return sol.t, sol.y.T

# ------------------------------- Benchmark core -----------------------------

def run_cd(case: str, deltas: List[float], N_runs: int, seed: int, outdir: str,
           profile: str = "fast", meas: str = "linear"):
    """
    profile: 'fast' (quick) or 'paper' (strict tolerances / step sizes)
    meas   : 'linear' (matches paper) or 'nonlinear' (accentuate UKF vs CKF)
    """
    rng0 = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    results = {}

    # profiles
    if profile == "paper":
        integ_method = "Radau"
        truth_rtol, truth_atol = 1e-12, 1e-12
        flt_rtol, flt_atol = 1e-12, 1e-12
        maxstep_factor = 0.1
        vdp_mu = 1.0e4
        vdp_tf = 2.0
    else:  # fast
        integ_method = "BDF"
        truth_rtol, truth_atol = 1e-6, 1e-9
        flt_rtol, flt_atol = 1e-6, 1e-9
        maxstep_factor = 0.5
        vdp_mu = 1.0e3
        vdp_tf = 1.0

    for delta in deltas:
        if case == "dahlquist":
            mu, j = -1.0e4, 1        # linear, ill-conditioned
            f, J = dahlquist_f(mu, j), dahlquist_J(mu, j)
            G, Qc = dahlquist_G(), dahlquist_Qc()
            x0 = np.array([1.0], dtype=float)
            t0, tf = 0.0, 4.0
            h, H = dahlquist_h(), dahlquist_H()
            R = np.array([[0.04]], dtype=float)
        elif case == "vdp":
            mu = vdp_mu
            f, J = vdp_f(mu), vdp_J(mu)
            G, Qc = vdp_G(), vdp_Qc()
            x0 = np.array([2.0, 0.0], dtype=float)
            t0, tf = 0.0, vdp_tf
            if meas == "nonlinear":
                h, H = vdp_h_nonlinear(), vdp_H_nonlinear()
            else:
                h, H = vdp_h(), vdp_H()
            R = np.array([[0.04]], dtype=float)
        else:
            raise ValueError("case must be 'dahlquist' or 'vdp'")

        cm = ContinuousModel(f=f, J=J, G=G, Qc=Qc)

        # Filters (conventional matrix form)
        ekf = CDEKF(cm, h, H, R, rtol=flt_rtol, atol=flt_atol, max_step=maxstep_factor*delta, method=integ_method)
        ukf = CDUKF(cm, h, R,      rtol=flt_rtol, atol=flt_atol, max_step=maxstep_factor*delta, method=integ_method)
        ckf = CDCKF(cm, h, R,      rtol=flt_rtol, atol=flt_atol, max_step=maxstep_factor*delta, method=integ_method)

        t_grid = np.arange(t0, tf + 1e-12, delta)
        err = {k: [] for k in ["EKF", "UKF", "CKF"]}

        for _ in range(N_runs):
            rng = np.random.default_rng(rng0.integers(1 << 32))

            # Truth (drift-only) at the same sample grid
            _, xs  = integrate_truth(f, J, t0, tf, x0, t_eval=t_grid,
                                     rtol=truth_rtol, atol=truth_atol, method=integ_method,
                                     max_step=maxstep_factor*delta)
            x_true = xs

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

        # CSV row
        csv_path = os.path.join(outdir, f"{case}_cd_armse_{profile}_{meas}.csv")
        write_header = not os.path.exists(csv_path)
        import csv
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["delta", "EKF", "UKF", "CKF"])
            w.writerow([delta, results[delta]["EKF"], results[delta]["UKF"], results[delta]["CKF"]])

        # Short bar per delta
        labels = ["EKF", "UKF", "CKF"]
        vals = [results[delta][lab] for lab in labels]
        plt.figure()
        plt.bar(labels, vals)
        plt.ylabel("ARMSE")
        plt.title(f"{case.upper()} — ARMSE (δ={delta:g}, {profile}, {meas})")
        plt.tight_layout()
        png_path = os.path.join(outdir, f"{case}_cd_armse_delta{delta:g}_{profile}_{meas}.png")
        plt.savefig(png_path, dpi=150)
        plt.close()

    # Summary line plot
    deltas_sorted = sorted(results.keys())
    plt.figure(figsize=(7, 5))
    for name in ["EKF", "UKF", "CKF"]:
        ys = [results[d][name] for d in deltas_sorted]
        plt.plot(deltas_sorted, ys, marker="o", label=name)
    plt.xlabel("sampling period δ")
    plt.ylabel("ARMSE")
    plt.title(f"CD benchmark — {case} ({profile}, {meas})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{case}_cd_summary_{profile}_{meas}.png"), dpi=150)
    plt.close()

    return results

# ---------------------------------- CLI -------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["dahlquist", "vdp"], default="vdp")
    parser.add_argument("--runs", type=int, default=1)                 # small by default (quick)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--deltas", type=float, nargs="*", default=[0.1,0.2,0.3,0.4,0.5])
    parser.add_argument("--profile", choices=["fast","paper"], default="fast")
    parser.add_argument("--meas", choices=["linear","nonlinear"], default="linear")
    args = parser.parse_args()

    res = run_cd(args.case, args.deltas, N_runs=args.runs, seed=args.seed,
                 outdir=args.outdir, profile=args.profile, meas=args.meas)
    for d in sorted(res.keys()):
        print(f"delta={d:g}: EKF={res[d]['EKF']:.6g}, UKF={res[d]['UKF']:.6g}, CKF={res[d]['CKF']:.6g}")

