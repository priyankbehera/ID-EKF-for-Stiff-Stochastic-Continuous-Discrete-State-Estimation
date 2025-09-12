# run_armse.py
# Paper-aligned benchmark for CD-EKF / CD-UKF / CD-CKF / CD-IDEKF (non–square-root).
from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from scipy.integrate import solve_ivp

from filters import ContinuousModel, CDEKF, CDUKF, CDCKF, CDIDEKF, ensure_psd
from models import (
    dahlquist_f, dahlquist_J, dahlquist_h, dahlquist_H, dahlquist_G, dahlquist_Qc,
    vdp_f, vdp_J, vdp_h, vdp_H, vdp_G, vdp_Qc,
)

# ------------------------------- Deterministic truth ------------------------

def integrate_truth(f, J, t0: float, tf: float, x0: np.ndarray, t_eval: np.ndarray,
                    rtol=1e-12, atol=1e-12, max_step=1e-1, method="Radau"):
    """
    Deterministic drift-only truth, matching the paper (no diffusion in truth).
    """
    sol = solve_ivp(lambda t, x: f(t, x),
                    (t0, tf), x0, method=method,
                    jac=lambda t, x: J(t, x),
                    t_eval=t_eval,
                    rtol=rtol, atol=atol, max_step=max_step)
    return sol.t, sol.y.T

# ------------------------------- Benchmark core -----------------------------

def run_cd(case: str, deltas: List[float], N_runs: int, seed: int, outdir: str,
           profile: str = "paper"):
    """
    Paper settings:
      • Integrator: Radau
      • Tolerances: 1e-12 (abs/rel)
      • Max step: 0.1
    Van der Pol (stiff): mu=1e4, t in [0, 2], z = x1 + x2 + v, R=0.04
    Dahlquist: mu<0, j=3 (nonlinear), z = x + v, R=0.04
    """
    rng0 = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    results = {}

    # Paper profile (used for both truth ODE and filter MDEs)
    integ_method = "BDF"
    truth_rtol, truth_atol = 1e-6, 1e-6
    flt_rtol, flt_atol = 1e-6, 1e-6
    max_step_abs = 1.0e-1

    # Problem setup
    for delta in deltas:
        if case == "dahlquist":
            # Nonlinear Dahlquist; choose stable negative mu and j=3
            mu, j = -1.0e4, 1
            f, J = dahlquist_f(mu, j), dahlquist_J(mu, j)
            G, Qc = dahlquist_G(), dahlquist_Qc()  # used only in filter covariance MDE
            x0 = np.array([1.0], dtype=float)
            t0, tf = 0.0, 4.0
            h, H = dahlquist_h(), dahlquist_H()
            R = np.array([[0.04]], dtype=float)

        elif case == "vdp":
            # Stiff Van der Pol: paper settings
            mu = 1.0e4
            f, J = vdp_f(mu), vdp_J(mu)
            G, Qc = vdp_G(), vdp_Qc()             # process model in filters (not in truth)
            x0 = np.array([2.0, 0.0], dtype=float)
            t0, tf = 0.0, 2.0                     # paper horizon
            h, H = vdp_h(), vdp_H()               # paper measurement: z = x1 + x2 + v
            R = np.array([[0.04]], dtype=float)
        else:
            raise ValueError("case must be 'dahlquist' or 'vdp'")

        cm = ContinuousModel(f=f, J=J, G=G, Qc=Qc)

        # Uniform sampling grid
        t_grid = np.arange(t0, tf + 1e-12, delta)
        err = {k: [] for k in ["EKF", "UKF", "CKF", "IDEKF"]}

        for _ in range(N_runs):
            rng = np.random.default_rng(rng0.integers(1 << 32))

            # ----- Truth: deterministic drift-only ODE (matches paper) -----
            _, xs = integrate_truth(
                f, J, t0, tf, x0, t_eval=t_grid,
                rtol=truth_rtol, atol=truth_atol,
                max_step=max_step_abs, method=integ_method
            )
            x_true = xs

            # ----- Measurements: z_k = h(x_true(t_k)) + v_k -----
            chol_R = np.linalg.cholesky(ensure_psd(R))
            zs = np.array([h(x_true[k]) + chol_R @ rng.normal(size=R.shape[0])
                           for k in range(len(t_grid))])

            # ----- Filters (fresh per run) with paper integrator settings -----
            ekf   = CDEKF(cm, h, H, R, rtol=flt_rtol, atol=flt_atol,
                          max_step=max_step_abs, method=integ_method)
            ukf   = CDUKF(cm, h, R,      rtol=flt_rtol, atol=flt_atol,
                          max_step=max_step_abs, method=integ_method)
            ckf   = CDCKF(cm, h, R,      rtol=flt_rtol, atol=flt_atol,
                          max_step=max_step_abs, method=integ_method)
            idekf = CDIDEKF(cm, h, H, R, rtol=flt_rtol, atol=flt_atol,
                            max_step=max_step_abs, method=integ_method)

            # Initial mean/cov
            x, P0 = x0.copy(), np.eye(x0.size, dtype=float) * 1e-2

            for name, flt in [("EKF", ekf), ("UKF", ukf), ("CKF", ckf), ("IDEKF", idekf)]:
                xk, Pk = x.copy(), P0.copy()
                x_est = [xk.copy()]
                for k in range(1, len(t_grid)):
                    t_prev, t_curr = t_grid[k-1], t_grid[k]
                    xk, Pk = flt.predict(t_prev, t_curr, xk, Pk)
                    xk, Pk, _, _ = flt.update(xk, Pk, zs[k])
                    x_est.append(xk.copy())
                x_est = np.vstack(x_est)
                armse = float(np.sqrt(np.mean(np.sum((x_est - x_true) ** 2, axis=1))))
                err[name].append(armse)

        results[delta] = {k: float(np.mean(err[k])) for k in err}

        # CSV row (per-delta append)
        csv_path = os.path.join(outdir, f"{case}_cd_armse_paper_linear.csv")
        write_header = not os.path.exists(csv_path)
        import csv
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["delta", "EKF", "UKF", "CKF", "IDEKF"])
            w.writerow([delta, results[delta]["EKF"], results[delta]["UKF"], results[delta]["CKF"], results[delta]["IDEKF"]])

        # Short bar per delta
        labels = ["EKF", "UKF", "CKF", "IDEKF"]
        vals = [results[delta][lab] for lab in labels]
        plt.figure()
        plt.bar(labels, vals)
        plt.ylabel("ARMSE")
        plt.title(f"{case.upper()} — ARMSE (δ={delta:g}, paper profile)")
        plt.tight_layout()
        png_path = os.path.join(outdir, f"{case}_cd_armse_delta{delta:g}_paper_linear.png")
        plt.savefig(png_path, dpi=150)
        plt.close()

    # Summary line plot
    deltas_sorted = sorted(results.keys())
    plt.figure(figsize=(8, 5))
    for name, marker in [("EKF","o"), ("UKF","s"), ("CKF","^"), ("IDEKF","D")]:
        ys = [results[d][name] for d in deltas_sorted]
        plt.plot(deltas_sorted, ys, marker=marker, label=name)
    plt.xlabel("sampling period δ")
    plt.ylabel("ARMSE")
    plt.title(f"CD benchmark — {case} (paper profile, linear meas.)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    png_path = os.path.join(outdir, f"{case}_cd_summary_paper_linear.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    return results

# ---------------------------------- CLI -------------------------------------
def export_armse_summary(results, outdir, case):
    import csv
    import os
    import matplotlib.pyplot as plt
    deltas_sorted = sorted(results.keys())
    csv_path = os.path.join(outdir, f"{case}_armse_vs_delta_paper_linear.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        headers = ["delta", "EKF", "UKF", "CKF", "IDEKF"]
        w.writerow(headers)
        for d in deltas_sorted:
            row = [d, results[d]["EKF"], results[d]["UKF"], results[d]["CKF"], results[d]["IDEKF"]]
            w.writerow(row)
    print(f"Wrote {csv_path}")

    plt.figure(figsize=(8, 5))
    for name, marker in [("EKF","o"), ("UKF","s"), ("CKF","^"), ("IDEKF","D")]:
        ys = [results[d][name] for d in deltas_sorted]
        plt.plot(deltas_sorted, ys, marker=marker, label=name)
    plt.xlabel("sampling period δ")
    plt.ylabel("ARMSE")
    plt.title(f"ARMSE vs δ — {case} (paper profile)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(outdir, f"{case}_armse_vs_delta_paper_linear.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Wrote {png_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["dahlquist", "vdp"], default="vdp")
    parser.add_argument("--runs", type=int, default=10)          # Paper-style averaging
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--deltas", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    # We lock to paper profile to avoid drifting from the paper setup
    parser.add_argument("--profile", choices=["paper"], default="paper")
    args = parser.parse_args()

    res = run_cd(args.case, args.deltas, N_runs=args.runs, seed=args.seed,
                 outdir=args.outdir, profile=args.profile)
    for d in sorted(res.keys()):
        print(f"delta={d:g}: EKF={res[d]['EKF']:.6g}, UKF={res[d]['UKF']:.6g}, CKF={res[d]['CKF']:.6g}, IDEKF={res[d]['IDEKF']:.6g}")

    export_armse_summary(res, args.outdir, args.case)
