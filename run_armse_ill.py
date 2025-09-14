# run_armse.py
# Paper-aligned benchmark for CD-EKF / CD-IDEKF (non–square-root only),
# with an optional ill-conditioned measurement model (per Kulikov & Kulikova, IET-CTA 2017).
from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from scipy.integrate import solve_ivp

from filters import ContinuousModel, CDEKF, CDIDEKF, ensure_psd
from models import (
    dahlquist_f, dahlquist_J, dahlquist_h, dahlquist_H, dahlquist_G, dahlquist_Qc,
    vdp_f, vdp_J, vdp_h, vdp_H, vdp_G, vdp_Qc,
)

# ------------------------------- Deterministic truth ------------------------

def integrate_truth(f, J, t0: float, tf: float, x0: np.ndarray, t_eval: np.ndarray,
                    rtol=1e-12, atol=1e-12, max_step=1e-1, method="BDF"):
    """
    Deterministic drift-only truth (no diffusion in truth), matching the paper spirit.
    """
    sol = solve_ivp(lambda t, x: f(t, x),
                    (t0, tf), x0, method=method,
                    jac=lambda t, x: J(t, x),
                    t_eval=t_eval,
                    rtol=rtol, atol=atol, max_step=max_step)
    if not sol.success:
        raise RuntimeError(f"Truth integration failed: {sol.message}")
    return sol.t, sol.y.T

# ------------------------------- Benchmark core -----------------------------

def run_cd(case: str, deltas: List[float], N_runs: int, seed: int, outdir: str,
           profile: str = "paper", sigma: float | None = None) -> Dict[float, Dict[str, float]]:
    """
    Profiles:
      • paper:  BDF, rtol/atol = 1e-6 (truth and filter), max_step = 0.1
      • harsh:  BDF, truth rtol/atol = 1e-8, filter rtol/atol = 1e-3, max_step = 0.5
    
    Ill-conditioned measurements (optional, VdP case):
      z_k = [[1, 1], [1, 1+sigma]] x_k + v_k,  R = diag(sigma^2, sigma^2)
    If sigma is None, use the default measurement model from models.* (e.g., z = x1 + x2 + v for VdP).
    """
    rng0 = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    results: Dict[float, Dict[str, float]] = {}

    # Profiles (truth vs filter tolerances)
    if profile == "paper":
        integ_method = "BDF"
        truth_rtol, truth_atol = 1e-6, 1e-6
        flt_rtol, flt_atol     = 1e-6, 1e-6
        max_step_abs = 1.0e-1
    elif profile == "harsh":
        integ_method = "BDF"
        truth_rtol, truth_atol = 1e-8, 1e-8   # keep truth tighter
        flt_rtol, flt_atol     = 1e-3, 1e-3   # loosen filter MDE tolerances
        max_step_abs = 5.0e-1                 # allow larger steps for filter ODEs
    else:
        raise ValueError("profile must be 'paper' or 'harsh'")

    # Problem setup (model dynamics and default measurements)
    for delta in deltas:
        if case == "dahlquist":
            # Nonlinear Dahlquist; stable negative mu and j=3
            mu, j = -1.0e4, 3
            f, J = dahlquist_f(mu, j), dahlquist_J(mu, j)
            G, Qc = dahlquist_G(), dahlquist_Qc()  # used only in filter covariance MDEs
            x0 = np.array([1.0], dtype=float)
            t0, tf = 0.0, 4.0
            # Default measurement (scalar): z = x + v
            h, H = dahlquist_h(), dahlquist_H()
            R = np.array([[0.04]], dtype=float)

        elif case == "vdp":
            mu = 1.0e5
            f, J = vdp_f(mu), vdp_J(mu)
            G, Qc = vdp_G(), vdp_Qc()
            x0 = np.array([2.0, 0.0], dtype=float)
            t0, tf = 0.0, 2.0

            if sigma is not None and sigma > 0:
                # Ill-conditioned 2D measurement (per IET-CTA 2017):
                def h(x: np.ndarray) -> np.ndarray:
                    return np.array([
                        [1.0, 1.0],
                        [1.0, 1.0 + sigma]
                    ], dtype=float) @ x
                def H(x: np.ndarray) -> np.ndarray:
                    return np.array([
                        [1.0, 1.0],
                        [1.0, 1.0 + sigma]
                    ], dtype=float)
                R = np.diag([sigma**2, sigma**2]).astype(float)
            else:
                # Paper’s well-conditioned linear measurement (single output): z = x1 + x2 + v
                h, H = vdp_h(), vdp_H()
                R = np.array([[0.04]], dtype=float)
        else:
            raise ValueError("case must be 'dahlquist' or 'vdp'")

        cm = ContinuousModel(f=f, J=J, G=G, Qc=Qc)

        # Uniform sampling grid
        t_grid = np.arange(t0, tf + 1e-12, delta)

        # High-accuracy truth (deterministic drift only)
        _, xs = integrate_truth(
            f, J, t0, tf, x0, t_eval=t_grid,
            rtol=truth_rtol, atol=truth_atol,
            max_step=max_step_abs, method=integ_method
        )
        x_true = xs

        # Prepare storage for ARMSEs across runs (EKF & IDEKF only)
        err = {k: [] for k in ["EKF", "IDEKF"]}

        # Cholesky of measurement noise
        R_psd = ensure_psd(R)
        chol_R = np.linalg.cholesky(R_psd)
        m = R.shape[0]

        # Pre-generate measurement noises for reproducibility across filters
        noises = rng0.normal(size=(N_runs, len(t_grid), m))
        # v = chol_R @ n, n~N(0,I)
        run_noises = np.einsum('ij,rtj->rti', chol_R, noises)

        for run_i in range(N_runs):
            # Measurements: z_k = h(x_true(t_k)) + v_k
            zs = np.array([h(x_true[k]) + run_noises[run_i, k] for k in range(len(t_grid))])

            # Filters (fresh per run) with chosen integrator settings
            ekf   = CDEKF(cm, h, H, R, rtol=flt_rtol, atol=flt_atol,
                          max_step=max_step_abs, method=integ_method)
            idekf = CDIDEKF(cm, h, H, R, rtol=flt_rtol, atol=flt_atol,
                            max_step=max_step_abs, method=integ_method)

            # Initial mean/cov (paper-like)
            x_init = x0.copy()
            P0 = np.eye(x0.size, dtype=float) * 1e-2

            # Run each filter on the same measurement sequence
            for name, flt in [("EKF", ekf), ("IDEKF", idekf)]:
                xk, Pk = x_init.copy(), P0.copy()
                x_est = [xk.copy()]
                for k in range(1, len(t_grid)):
                    t_prev, t_curr = t_grid[k-1], t_grid[k]
                    xk, Pk = flt.predict(t_prev, t_curr, xk, Pk)
                    xk, Pk, _, _ = flt.update(xk, Pk, zs[k])
                    x_est.append(xk.copy())
                x_est = np.vstack(x_est)
                # Accumulated RMSE over time (ARMSE in the paper style)
                armse = float(np.sqrt(np.mean(np.sum((x_est - x_true) ** 2, axis=1))))
                err[name].append(armse)

        # Aggregate results for this delta
        results[delta] = {k: float(np.mean(err[k])) for k in err}

        # Write/append CSV row per-delta
        suffix = "_illcond" if (case == "vdp" and sigma is not None and sigma > 0) else ""
        csv_path = os.path.join(outdir, f"{case}_cd_armse{suffix}.csv")
        write_header = not os.path.exists(csv_path)
        import csv
        with open(csv_path, "a", newline="") as fcsv:
            w = csv.writer(fcsv)
            if write_header:
                hdr = ["delta", "EKF", "IDEKF"]
                if case == "vdp" and sigma is not None and sigma > 0:
                    hdr.append("sigma")
                w.writerow(hdr)
            row = [delta, results[delta]["EKF"], results[delta]["IDEKF"]]
            if case == "vdp" and sigma is not None and sigma > 0:
                row.append(sigma)
            w.writerow(row)

        # Short bar per delta
        labels = ["EKF", "IDEKF"]
        vals = [results[delta][lab] for lab in labels]
        plt.figure()
        plt.bar(labels, vals)
        plt.ylabel("ARMSE")
        title = f"{case.upper()} — ARMSE (δ={delta:g}, {profile} profile"
        if case == "vdp" and sigma is not None and sigma > 0:
            title += f", σ={sigma:g}"
        title += ")"
        plt.title(title)
        plt.tight_layout()
        png_path = os.path.join(outdir, f"{case}_cd_armse_delta{delta:g}{suffix}.png")
        plt.savefig(png_path, dpi=150)
        plt.close()

    # Summary line plot across deltas
    deltas_sorted = sorted(results.keys())
    plt.figure(figsize=(8, 5))
    for name, marker in [("EKF","o"), ("IDEKF","D")]:
        ys = [results[d][name] for d in deltas_sorted]
        plt.plot(deltas_sorted, ys, marker=marker, label=name)
    plt.xlabel("sampling period δ")
    plt.ylabel("ARMSE")
    title = f"CD benchmark — {case} ({profile} profile)"
    if case == "vdp" and sigma is not None and sigma > 0:
        title += f" (ill-conditioned, σ={sigma:g})"
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    suffix = "_illcond" if (case == "vdp" and sigma is not None and sigma > 0) else ""
    png_path = os.path.join(outdir, f"{case}_cd_summary{suffix}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    return results

# ---------------------------------- Utility ---------------------------------

def export_armse_summary(results: Dict[float, Dict[str, float]], outdir: str, case: str,
                          profile: str, sigma: float | None = None):
    import csv
    deltas_sorted = sorted(results.keys())
    suffix = "_illcond" if (case == "vdp" and sigma is not None and sigma > 0) else ""
    csv_path = os.path.join(outdir, f"{case}_armse_vs_delta{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        headers = ["delta", "EKF", "IDEKF"]
        if case == "vdp" and sigma is not None and sigma > 0:
            headers.append("sigma")
        w.writerow(headers)
        for d in deltas_sorted:
            row = [d, results[d]["EKF"], results[d]["IDEKF"]]
            if case == "vdp" and sigma is not None and sigma > 0:
                row.append(sigma)
            w.writerow(row)
    print(f"Wrote {csv_path}")

    plt.figure(figsize=(8, 5))
    for name, marker in [("EKF","o"), ("IDEKF","D")]:
        ys = [results[d][name] for d in deltas_sorted]
        plt.plot(deltas_sorted, ys, marker=marker, label=name)
    plt.xlabel("sampling period δ")
    plt.ylabel("ARMSE")
    title = f"ARMSE vs δ — {case} ({profile} profile)"
    if case == "vdp" and sigma is not None and sigma > 0:
        title += f" (ill-conditioned, σ={sigma:g})"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(outdir, f"{case}_armse_vs_delta{suffix}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Wrote {png_path}")

# ---------------------------------- CLI -------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["dahlquist", "vdp"], default="vdp")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="results")
    # Example grids
    slow = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    parser.add_argument("--deltas", type=float, nargs="*", default=slow)
    parser.add_argument("--profile", choices=["paper", "harsh"], default="paper")
    sigma = 1e-6
    parser.add_argument("--sigma", type=float, default=sigma,
                        help="If set (>0) and case=vdp, use ill-conditioned 2x output: [[1,1],[1,1+sigma]] with R=diag(sigma^2).")
    args = parser.parse_args()

    res = run_cd(args.case, args.deltas, N_runs=args.runs, seed=args.seed,
                 outdir=args.outdir, profile=args.profile, sigma=args.sigma)
    for d in sorted(res.keys()):
        print(f"delta={d:g}: EKF={res[d]['EKF']:.6g}, IDEKF={res[d]['IDEKF']:.6g}")

    export_armse_summary(res, args.outdir, args.case, profile=args.profile, sigma=args.sigma)
