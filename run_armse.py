# run_armse.py
# Benchmark driver for conventional CD-EKF / CD-UKF / CD-CKF / CD-IDEKF (no square-root).
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
    vdp_f, vdp_J, vdp_h, vdp_H, vdp_h_nonlinear, vdp_H_nonlinear, vdp_G, vdp_Qc,
)

# ------------------------------- Truth simulation ---------------------------

def integrate_truth(f, J, t0: float, tf: float, x0: np.ndarray, t_eval: np.ndarray,
                    rtol=1e-6, atol=1e-9, max_step=np.inf, method="BDF"):
    """Deterministic ODE truth (not used by default; kept for reference)."""
    sol = solve_ivp(lambda t, x: f(t, x),
                    (t0, tf), x0, method=method,
                    jac=lambda t, x: J(t, x),
                    t_eval=t_eval,
                    rtol=rtol, atol=atol, max_step=max_step)
    return sol.t, sol.y.T

def simulate_truth_sde_em(f, G, Qc, t0: float, tf: float, x0: np.ndarray,
                          t_grid: np.ndarray, max_dt: float, rng: np.random.Generator):
    """
    Tamed Euler–Maruyama for dx = f(t,x) dt + G(t,x) dW, with Qd(t,x)=G Qc(t) G^T.
    - Adapts the substep to keep the **drift increment** bounded.
    - Uses 'taming': f_tamed = f / (1 + h * ||f||).
    - Works with G(t,x) or G(t), and Qc(t) or const.
    """
    def _eval_G(t, x):
        if callable(G):
            try:
                val = G(t, x)        # state-dependent
            except TypeError:
                val = G(t)           # time-only
        else:
            val = G
        return np.atleast_2d(np.asarray(val, dtype=float))

    def _eval_Qc(t):
        if callable(Qc):
            val = Qc(t)
        else:
            val = Qc
        return np.atleast_2d(np.asarray(val, dtype=float))

    # Drift increment cap per substep (tuneable). Smaller => safer, more substeps.
    DRIFT_CAP = 0.5
    EPS = 1e-30

    xs = [x0.astype(float).copy()]
    t_prev = t_grid[0]
    x = x0.astype(float).copy()

    for t_curr in t_grid[1:]:
        T = t_curr - t_prev
        remaining = T
        t = t_prev
        while remaining > 0:
            # start from max_dt, but shrink if drift too large
            h = min(max_dt, remaining)

            f_val = f(t, x)
            f_norm = float(np.linalg.norm(f_val))
            if f_norm * h > DRIFT_CAP:
                h = max(DRIFT_CAP / max(f_norm, EPS), 1e-12)
                if h > remaining:
                    h = remaining

            # evaluate diffusion at (t,x)
            Gtx = _eval_G(t, x)
            Qct = _eval_Qc(t)
            Qd  = ensure_psd(Gtx @ Qct @ Gtx.T)

            # Cholesky may still fail if near-semi-definite; fall back to eig
            try:
                L = np.linalg.cholesky(Qd)
            except np.linalg.LinAlgError:
                w, V = np.linalg.eigh(Qd)
                w = np.clip(w, 0.0, None)
                L = (V * np.sqrt(w)) @ V.T

            # Tamed Euler step
            sqrt_h = np.sqrt(h)
            xi = rng.normal(size=L.shape[1])
            f_tamed = f_val / (1.0 + h * max(f_norm, EPS))
            x = x + f_tamed * h + (L @ xi) * sqrt_h

            t += h
            remaining -= h

        xs.append(x.copy())
        t_prev = t_curr

    return t_grid, np.vstack(xs)


# ------------------------------- Benchmark core -----------------------------

def run_cd(case: str, deltas: List[float], N_runs: int, seed: int, outdir: str,
           profile: str = "fast", meas: str = "linear"):
    """
    profile: 'fast' (quick) or 'paper' (strict tolerances / step sizes)
    meas   : 'linear' (matches paper) or 'nonlinear' (accentuates UKF vs CKF)
    """
    rng0 = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    results = {}

    # Profiles
    if profile == "paper":
        integ_method = "Radau"          # stiff solver
        flt_rtol, flt_atol = 1e-12, 1e-12
        max_step_abs = 1.0e-1          
        vdp_mu = 1.0e4
        vdp_tf = 1.0
    else:  # fast
        integ_method = "BDF"
        flt_rtol, flt_atol = 1e-6, 1e-9
        max_step_abs = 1.0e-1
        vdp_mu = 1.0e4
        vdp_tf = 2.0

    for delta in deltas:
        if case == "dahlquist":
            mu, j = -1.0e4, 1
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

        t_grid = np.arange(t0, tf + 1e-12, delta)
        err = {k: [] for k in ["EKF", "UKF", "CKF", "IDEKF"]}

        for _ in range(N_runs):
            rng = np.random.default_rng(rng0.integers(1 << 32))

            # Stochastic truth (SDE) at the same sample grid
            _, xs = simulate_truth_sde_em(
                f=f, G=G, Qc=Qc,
                t0=t0, tf=tf, x0=x0, t_grid=t_grid,
                max_dt=max_step_abs if np.isfinite(max_step_abs) else delta,
                rng=rng
            )
            x_true = xs

            # Measurements
            chol_R = np.linalg.cholesky(ensure_psd(R))
            zs = np.array([h(x_true[k]) + chol_R @ rng.normal(size=R.shape[0])
                           for k in range(len(t_grid))])

            # Fresh filter objects each run
            ekf   = CDEKF(cm, h, H, R, rtol=flt_rtol, atol=flt_atol,
                          max_step=max_step_abs, method=integ_method)
            ukf   = CDUKF(cm, h, R,      rtol=flt_rtol, atol=flt_atol,
                          max_step=max_step_abs, method=integ_method)
            ckf   = CDCKF(cm, h, R,      rtol=flt_rtol, atol=flt_atol,
                          max_step=max_step_abs, method=integ_method)
            idekf = CDIDEKF(cm, h, H, R, rtol=flt_rtol, atol=flt_atol,
                            max_step=max_step_abs, method=integ_method)

            for name, flt in [("EKF", ekf), ("UKF", ukf), ("CKF", ckf), ("IDEKF", idekf)]:
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
                w.writerow(["delta", "EKF", "UKF", "CKF", "IDEKF"])
            w.writerow([delta, results[delta]["EKF"], results[delta]["UKF"], results[delta]["CKF"], results[delta]["IDEKF"]])

        # Short bar per delta
        labels = ["EKF", "UKF", "CKF", "IDEKF"]
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
    plt.figure(figsize=(8, 5))
    for name in ["EKF", "UKF", "CKF", "IDEKF"]:
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
def export_armse_summary(results, outdir, case, profile, meas):
    import csv
    import os
    import matplotlib.pyplot as plt

    deltas_sorted = sorted(results.keys())
    # Write one tidy CSV
    csv_path = os.path.join(outdir, f"{case}_armse_vs_delta_{profile}_{meas}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        headers = ["delta", "EKF", "UKF", "CKF", "IDEKF"]
        w.writerow(headers)
        for d in deltas_sorted:
            row = [d, results[d]["EKF"], results[d]["UKF"], results[d]["CKF"], results[d]["IDEKF"]]
            w.writerow(row)
    print(f"Wrote {csv_path}")

    # Make a single summary line plot
    plt.figure(figsize=(8, 5))
    for name, marker in [("EKF","o"), ("UKF","s"), ("CKF","^"), ("IDEKF","D")]:
        ys = [results[d][name] for d in deltas_sorted]
        plt.plot(deltas_sorted, ys, marker=marker, label=name)
    plt.xlabel("sampling period δ")
    plt.ylabel("ARMSE")
    plt.title(f"ARMSE vs δ — {case} ({profile}, {meas})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(outdir, f"{case}_armse_vs_delta_{profile}_{meas}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Wrote {png_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["dahlquist", "vdp"], default="vdp")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="results")
    fast = [0.1, 0.3, 0.5]
    slow = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    parser.add_argument("--deltas", type=float, nargs="*", default=slow)
    parser.add_argument("--profile", choices=["fast","paper"], default="fast")
    parser.add_argument("--meas", choices=["linear","nonlinear"], default="linear")
    args = parser.parse_args()

    res = run_cd(args.case, args.deltas, N_runs=args.runs, seed=args.seed,
                 outdir=args.outdir, profile=args.profile, meas=args.meas)
    for d in sorted(res.keys()):
        print(f"delta={d:g}: EKF={res[d]['EKF']:.6g}, UKF={res[d]['UKF']:.6g}, CKF={res[d]['CKF']:.6g}, IDEKF={res[d]['IDEKF']:.6g}")

    export_armse_summary(res, args.outdir, args.case, args.profile, args.meas)
