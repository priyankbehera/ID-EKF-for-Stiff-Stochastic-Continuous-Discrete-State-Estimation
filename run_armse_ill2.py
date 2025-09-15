"""
python run_idekf_advantage_fixed.py --case vdp --meas nonlin_cubic --sigma 1e-3 \
  --Rmode aniso --Rdiag 1e-4 1e-2 \
  --x0-perturb 1.2 -1.0 --deltas 0.4 0.6 0.8 \
  --truth-noise --truth-qscale 10 \
  --runs 100 --metric avg --idekf-iter-max 5 --idekf-iter-tol 1e-10
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from scipy.integrate import solve_ivp

# Expect these to exist in your repo
from filters import (
    ContinuousModel, CDEKF, CDIDEKF, ensure_psd,
    cov_to_inf, inf_to_cov, mupdate,
)
from models import (
    dahlquist_f, dahlquist_J, dahlquist_h, dahlquist_H, dahlquist_G, dahlquist_Qc,
    vdp_f, vdp_J, vdp_h, vdp_H, vdp_G, vdp_Qc,
)

# ------------------------------- Deterministic / stochastic truth ------------------------

def integrate_truth_path(
    f, J, G, Qc,
    t0: float, tf: float, x0: np.ndarray, t_grid: np.ndarray,
    rtol=1e-12, atol=1e-12, max_step=1e-1, method="BDF",
    truth_noise: bool = False, qscale: float = 1.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Integrate the truth along t_grid. If truth_noise=True, inject discrete-time process
    noise after each propagation step using Qd ≈ G(t_mid) Qc(t_mid) G(t_mid)^T * dt * qscale.
    Handles callable or constant G,Qc.
    """
    def _mat_at(M, t):
        A = M(t) if callable(M) else M
        return np.asarray(A, dtype=float)

    x = x0.copy().astype(float)
    xs = [x.copy()]
    for k in range(1, len(t_grid)):
        t_prev, t_curr = t_grid[k-1], t_grid[k]
        sol = solve_ivp(
            lambda t, x_: f(t, x_), (t_prev, t_curr), x, method=method,
            jac=lambda t, x_: J(t, x_), t_eval=[t_curr], rtol=rtol, atol=atol, max_step=max_step
        )
        if not sol.success:
            raise RuntimeError(f"Truth integration failed at step {k}: {sol.message}")
        x = sol.y[:, -1]

        if truth_noise:
            if rng is None:
                rng = np.random.default_rng()
            dt = float(t_curr - t_prev)
            t_mid = 0.5 * (t_prev + t_curr)
            Gt  = _mat_at(G,  t_mid)
            Qct = _mat_at(Qc, t_mid)
            Qd  = ensure_psd(Gt @ Qct @ Gt.T * dt * float(qscale))
            L = np.linalg.cholesky(Qd + 1e-18 * np.eye(Qd.shape[0]))
            x = x + L @ rng.normal(size=x.shape[0])
        xs.append(x.copy())
    return np.array(xs)

# ------------------------------- Measurement builders -----------------------------

def build_R(case: str, mode: str, sigma: float, diag_vals: List[float] | None, H_const: np.ndarray | None) -> np.ndarray:
    mode = mode.lower()
    if mode == "diag":
        if diag_vals is None:
            return np.array([[0.04]], dtype=float) if case == "dahlquist" else np.diag([0.04, 0.04]).astype(float)
        if len(diag_vals) == 1:
            return np.diag([diag_vals[0]] if case == "dahlquist" else [diag_vals[0], diag_vals[0]]).astype(float)
        return np.diag([diag_vals[0], diag_vals[1]]).astype(float)
    elif mode == "aniso":
        # Align R with near-nullspace of H (only for 2x meas)
        if H_const is None or H_const.shape[0] != 2:
            raise ValueError("Rmode=aniso requires a 2x measurement (vdp ill/nonlin).")
        u1 = np.array([1.0, 1.0]); u1 /= np.linalg.norm(u1)
        u2 = np.array([1.0, -(1.0 + sigma)]); u2 /= np.linalg.norm(u2)
        U = np.column_stack([u1, u2])
        if diag_vals is None:
            d1, d2 = 1e-4, 1e-2
        elif len(diag_vals) == 1:
            d1, d2 = diag_vals[0], diag_vals[0]
        else:
            d1, d2 = diag_vals[0], diag_vals[1]
        return U @ np.diag([d1, d2]) @ U.T
    else:
        raise ValueError("Rmode must be 'diag' or 'aniso'.")


def build_measurement(case: str, meas: str, sigma: float, Rmode: str, Rdiag: List[float] | None,
                      nl_kind: str, nl_eps: float, nl_alpha: float) -> Tuple:
    """Return (h, H, R, meas_tag) according to requested measurement model.
    meas ∈ {"paper", "ill", "nonlin_cubic", "nonlin_tanh"}.
    """
    meas = meas.lower()
    nl_kind = nl_kind.lower()

    if case == "dahlquist":
        if meas != "paper":
            raise ValueError("For dahlquist, only 'paper' measurement is supported.")
        def h(x: np.ndarray) -> np.ndarray:
            return np.array([x[0]], dtype=float)
        def H(x: np.ndarray) -> np.ndarray:
            return np.array([[1.0]], dtype=float)
        R = build_R(case, Rmode, sigma, Rdiag, None)
        return h, H, R, "paper"

    # Van der Pol cases (2x measurements except 'paper')
    if meas == "paper":
        h, H = vdp_h(), vdp_H()
        R = build_R(case, Rmode, sigma, Rdiag, None)
        return h, H, R, "paper"

    if meas == "ill":
        Hc = np.array([[1.0, 1.0],[1.0, 1.0 + sigma]], dtype=float)
        def h(x: np.ndarray) -> np.ndarray:
            return Hc @ x
        def H(x: np.ndarray) -> np.ndarray:
            return Hc
        R = build_R(case, Rmode, sigma, Rdiag, Hc)
        return h, H, R, "illcond"

    # Nonlinear near-collinear base
    if meas in {"nonlin_cubic", "nonlin_tanh"}:
        def s1s2(x: np.ndarray) -> Tuple[float, float]:
            s1 = x[0] + x[1]
            s2 = x[0] + (1.0 + sigma) * x[1]
            return s1, s2
        if meas == "nonlin_cubic" or nl_kind == "cubic":
            def h(x: np.ndarray) -> np.ndarray:
                s1, s2 = s1s2(x)
                return np.array([s1, s2 + nl_eps * (s2**3)], dtype=float)
            def H(x: np.ndarray) -> np.ndarray:
                s1, s2 = s1s2(x)
                d = 1.0 + 3.0 * nl_eps * (s2**2)
                return np.array([[1.0, 1.0], [d, d * (1.0 + sigma)]], dtype=float)
            tag = "nonlin_cubic"
        else:
            # Stable tanh variant: sech^2(u) = 1 - tanh^2(u)
            def h(x: np.ndarray) -> np.ndarray:
                s1, s2 = s1s2(x)
                return np.array([s1, s2 + nl_eps * np.tanh(nl_alpha * s2)], dtype=float)
            def H(x: np.ndarray) -> np.ndarray:
                s1, s2 = s1s2(x)
                t = np.tanh(nl_alpha * s2)
                sech2 = 1.0 - t*t
                fac = 1.0 + nl_eps * nl_alpha * sech2
                return np.array([[1.0, 1.0], [fac, fac * (1.0 + sigma)]], dtype=float)
            tag = "nonlin_tanh"
        R = build_R(case, Rmode, sigma, Rdiag, np.array([[1.0, 1.0],[1.0, 1.0 + sigma]], dtype=float))
        return h, H, R, tag

    raise ValueError("meas must be one of {'paper','ill','nonlin_cubic','nonlin_tanh'}")

# ------------------------------- Iterated IDEKF wrapper -----------------------------

class IDEKFIter(CDIDEKF):
    """A thin wrapper over your CDIDEKF to enforce Gauss–Newton iteration
    that re-linearizes around the updated mean within a single time step.
    """
    def __init__(self, *args, iter_max: int = 3, iter_tol: float = 1e-10, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter_max = iter_max
        self.iter_tol = iter_tol

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray):
        x = x_pred.copy()
        P = P_pred.copy()
        z_col = np.asarray(z).reshape(-1, 1)

        for _ in range(max(1, self.iter_max)):
            Hk = self.H(x)
            # prior in info/BN space
            B, V, _ = cov_to_inf(P, P.shape[0])
            u = x.reshape(-1, 1)
            def h_wrapped(u_vec):
                return np.asarray(self.h(np.asarray(u_vec).reshape(-1))).reshape(-1, 1)
            # BN/information-form measurement update
            u_post, V_post, B_post, _, _ = mupdate(1, z_col, u, B, V, self.R, Hk, h_wrapped)
            x_new = np.asarray(u_post).reshape(-1)
            P_new = ensure_psd(inf_to_cov(np.asarray(V_post).reshape(-1), np.asarray(B_post), x_new.size))
            # convergence check
            if np.linalg.norm(x_new - x) < self.iter_tol:
                x, P = x_new, P_new
                break
            x, P = x_new, P_new

        # Diagnostics (not used for update algebra)
        zhat = np.asarray(self.h(x_pred)).ravel()
        y = (np.asarray(z).ravel() - zhat)
        S = ensure_psd(self.H(x_pred) @ P_pred @ self.H(x_pred).T + self.R)
        return x, P, y, S

# ------------------------------- Benchmark core -----------------------------

def run_cd(case: str, deltas: List[float], N_runs: int, seed: int, outdir: str,
           profile: str = "paper", meas: str = "paper", sigma: float | None = None,
           Rmode: str = "diag", Rdiag: List[float] | None = None,
           metric: str = "avg", truth_noise: bool = False, truth_qscale: float = 1.0,
           nl_kind: str = "cubic", nl_eps: float = 0.4, nl_alpha: float = 1.0,
           x0_perturb: List[float] | None = None, idekf_iter_max: int = 5, idekf_iter_tol: float = 1e-10
           ) -> Dict[float, Dict[str, float]]:

    rng0 = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    results: Dict[float, Dict[str, float]] = {}

    # Profiles
    if profile == "paper":
        integ_method = "BDF"
        truth_rtol, truth_atol = 1e-6, 1e-6
        flt_rtol, flt_atol     = 1e-6, 1e-6
        max_step_abs = 1.0e-1
    elif profile == "harsh":
        integ_method = "BDF"
        truth_rtol, truth_atol = 1e-8, 1e-8
        flt_rtol, flt_atol     = 1e-3, 1e-3
        max_step_abs = 5.0e-1
    else:
        raise ValueError("profile must be 'paper' or 'harsh'")

    for delta in deltas:
        if case == "dahlquist":
            mu, j = -1.0e4, 3
            f, J = dahlquist_f(mu, j), dahlquist_J(mu, j)
            G, Qc = dahlquist_G(), dahlquist_Qc()
            x0 = np.array([1.0], dtype=float)
            t0, tf = 0.0, 4.0
        elif case == "vdp":
            mu = 1.0e5
            f, J = vdp_f(mu), vdp_J(mu)
            G, Qc = vdp_G(), vdp_Qc()
            x0 = np.array([2.0, 0.0], dtype=float)
            t0, tf = 0.0, 2.0
        else:
            raise ValueError("case must be 'dahlquist' or 'vdp'")

        if sigma is None:
            sigma = 1e-3
        h, H, R, meas_tag = build_measurement(case, meas, sigma, Rmode, Rdiag, nl_kind, nl_eps, nl_alpha)
        cm = ContinuousModel(f=f, J=J, G=G, Qc=Qc)

        # Grid
        t_grid = np.arange(t0, tf + 1e-12, delta)

        # Truth path
        rng_truth = np.random.default_rng(rng0.integers(1<<32))
        xs = integrate_truth_path(
            f, J, G, Qc, t0, tf, x0, t_grid,
            rtol=truth_rtol, atol=truth_atol, max_step=max_step_abs, method=integ_method,
            truth_noise=truth_noise, qscale=truth_qscale, rng=rng_truth
        )
        x_true = xs

        # Initial estimate
        x_init = x0.copy()
        if x0_perturb is not None and len(x0_perturb) > 0:
            add = np.zeros_like(x_init)
            for i in range(min(len(add), len(x0_perturb))):
                add[i] = float(x0_perturb[i])
            x_init = x_init + add
        P0 = np.eye(x0.size, dtype=float) * 1e-2

        err = {k: [] for k in ["EKF", "IDEKF"]}

        # Measurement noise generator
        R_psd = ensure_psd(R)
        chol_R = np.linalg.cholesky(R_psd + 1e-18 * np.eye(R_psd.shape[0]))
        m = R.shape[0]
        noises = rng0.normal(size=(N_runs, len(t_grid), m))
        run_noises = np.einsum('ij,rtj->rti', chol_R, noises)

        for run_i in range(N_runs):
            zs = np.array([h(x_true[k]) + run_noises[run_i, k] for k in range(len(t_grid))])

            ekf   = CDEKF(cm, h, H, R, rtol=flt_rtol, atol=flt_atol, max_step=max_step_abs, method=integ_method)
            idekf = IDEKFIter(cm, h, H, R, rtol=flt_rtol, atol=flt_atol, max_step=max_step_abs,
                               method=integ_method, iter_max=idekf_iter_max, iter_tol=idekf_iter_tol)

            for name, flt in [("EKF", ekf), ("IDEKF", idekf)]:
                xk, Pk = x_init.copy(), P0.copy()
                x_est = [xk.copy()]
                for k in range(1, len(t_grid)):
                    t_prev, t_curr = t_grid[k-1], t_grid[k]
                    xk, Pk = flt.predict(t_prev, t_curr, xk, Pk)
                    xk, Pk, _, _ = flt.update(xk, Pk, zs[k])
                    x_est.append(xk.copy())
                x_est = np.vstack(x_est)
                e2 = np.sum((x_est - x_true) ** 2, axis=1)
                armse = float(np.sqrt(np.mean(e2))) if metric == 'avg' else float(np.sqrt(np.sum(e2)))
                err[name].append(armse)

        results[delta] = {k: float(np.mean(err[k])) for k in err}

        # CSV row per-delta
        suffix = f"{case}_{meas_tag}_{Rmode}_{metric}"
        csv_path = os.path.join(outdir, f"{suffix}_cd_armse.csv")
        write_header = not os.path.exists(csv_path)
        import csv
        with open(csv_path, "a", newline="") as fcsv:
            w = csv.writer(fcsv)
            if write_header:
                hdr = ["delta", "EKF", "IDEKF", "sigma", "metric"]
                w.writerow(hdr)
            row = [delta, results[delta]["EKF"], results[delta]["IDEKF"], sigma, metric]
            w.writerow(row)

        

    return results

# ---------------------------------- CLI -------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["dahlquist", "vdp"], default="vdp")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="results")
    deltas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    parser.add_argument("--deltas", type=float, nargs="*", default=deltas)
    parser.add_argument("--profile", choices=["paper", "harsh"], default="paper")

    # Measurement controls
    parser.add_argument("--meas", choices=["paper", "ill", "nonlin_cubic", "nonlin_tanh"], default="ill",
                        help="paper=scalar linear; ill=2x linear ill-conditioned; nonlin_* = 2x nonlinear ill-conditioned")
    sigma = 1e-6
    parser.add_argument("--sigma", type=float, default=sigma,
                        help="Near-collinearity strength (ill/nonlin use [[1,1],[1,1+sigma]]).")
    parser.add_argument("--Rmode", choices=["diag", "aniso"], default="diag",
                        help="Measurement noise structure: diag (isotropic) or aniso (aligned with H nullspace).")
    parser.add_argument("--Rdiag", type=float, nargs="*", default=[0.04],
                        help="Diagonal values for R (diag mode) or eigenvalues (aniso mode).")

    # Nonlinear measurement parameters
    parser.add_argument("--nl-kind", choices=["cubic", "tanh"], default="cubic")
    parser.add_argument("--nl-eps", type=float, default=0.4)
    parser.add_argument("--nl-alpha", type=float, default=1.0)

    # Error metric
    parser.add_argument("--metric", choices=["avg", "cum"], default="avg")

    # Truth process noise injection
    parser.add_argument("--truth-noise", action="store_true")
    parser.add_argument("--truth-qscale", type=float, default=10.0)

    # Stress the update
    parser.add_argument("--x0-perturb", type=float, nargs="*", default=[1.2, -1.0])

    # IDEKF iteration knobs
    parser.add_argument("--idekf-iter-max", type=int, default=5)
    parser.add_argument("--idekf-iter-tol", type=float, default=1e-10)

    args = parser.parse_args()

    res = run_cd(
        case=args.case,
        deltas=args.deltas,
        N_runs=args.runs,
        seed=args.seed,
        outdir=args.outdir,
        profile=args.profile,
        meas=args.meas,
        sigma=args.sigma,
        Rmode=args.Rmode,
        Rdiag=args.Rdiag,
        metric=args.metric,
        truth_noise=args.truth_noise,
        truth_qscale=args.truth_qscale,
        nl_kind=args.nl_kind,
        nl_eps=args.nl_eps,
        nl_alpha=args.nl_alpha,
        x0_perturb=args.x0_perturb,
        idekf_iter_max=args.idekf_iter_max,
        idekf_iter_tol=args.idekf_iter_tol,
    )

    for d in sorted(res.keys()):
        print(f"delta={d:g}: EKF={res[d]['EKF']:.6g}, IDEKF={res[d]['IDEKF']:.6g}")

    # Final summary CSV
    import csv
    suffix = f"{args.case}_{args.meas}_{args.Rmode}_{args.metric}"
    csv_path = os.path.join(args.outdir, f"{suffix}_armse_vs_delta.csv")
    os.makedirs(args.outdir, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta", "EKF", "IDEKF"])
        for d in sorted(res.keys()):
            w.writerow([d, res[d]["EKF"], res[d]["IDEKF"]])
    print(f"Wrote {csv_path}")
