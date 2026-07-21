"""
Compare three continuous-discrete filters:

1. Standard EKF
2. Square-root EKF
3. Influence-diagram EKF

The influence-diagram EKF implementation is preserved exactly from
the original code. Only the square-root EKF is added.

Example:

python run_three_filter_comparison.py \
  --case vdp \
  --meas nonlin_cubic \
  --sigma 1e-3 \
  --Rmode aniso \
  --Rdiag 1e-4 1e-2 \
  --x0-perturb 1.5 -1.0 \
  --deltas 0.1 0.2 0.3 0.4 0.5 0.6 0.8 \
  --truth-noise \
  --truth-qscale 10 \
  --runs 100 \
  --metric avg \
  --idekf-iter-max 5 \
  --idekf-iter-tol 1e-10
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.integrate import solve_ivp

from filters import (
    ContinuousModel,
    CDEKF,
    CDIDEKF,
    ensure_psd,
    cov_to_inf,
    inf_to_cov,
    mupdate,
)

from models import (
    dahlquist_f,
    dahlquist_J,
    dahlquist_G,
    dahlquist_Qc,
    vdp_f,
    vdp_J,
    vdp_h,
    vdp_H,
    vdp_G,
    vdp_Qc,
)


# ============================================================
# Robust numerical helpers
# ============================================================

def force_symmetric(A: np.ndarray) -> np.ndarray:
    """
    Remove small numerical asymmetry.
    """
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


def project_positive_definite(
    A: np.ndarray,
    min_eigenvalue: float = 1e-10,
    max_condition: float = 1e12,
) -> np.ndarray:
    """
    Project a real symmetric matrix onto a numerically stable
    positive-definite matrix.

    This is stronger than simply adding a small diagonal jitter.
    """

    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(
            f"Expected a square matrix, received shape {A.shape}."
        )

    if not np.all(np.isfinite(A)):
        raise FloatingPointError(
            "Matrix contains NaN or Inf."
        )

    A = force_symmetric(A)

    eigenvalues, eigenvectors = np.linalg.eigh(A)

    largest_scale = max(
        1.0,
        float(np.max(np.abs(eigenvalues))),
    )

    eigenvalue_floor = max(
        min_eigenvalue,
        largest_scale / max_condition,
    )

    eigenvalues = np.maximum(
        eigenvalues,
        eigenvalue_floor,
    )

    A_pd = (
        eigenvectors
        @ np.diag(eigenvalues)
        @ eigenvectors.T
    )

    return force_symmetric(A_pd)


def stable_cholesky(
    A: np.ndarray,
    min_eigenvalue: float = 1e-10,
    max_condition: float = 1e12,
    max_tries: int = 12,
) -> np.ndarray:
    """
    Robust lower-triangular Cholesky factor.

    Steps:
    1. Symmetrize.
    2. Project eigenvalues to a safe positive range.
    3. Apply Cholesky.
    4. Add adaptive jitter if roundoff still causes failure.
    """

    A_pd = project_positive_definite(
        A,
        min_eigenvalue=min_eigenvalue,
        max_condition=max_condition,
    )

    eigenvalues = np.linalg.eigvalsh(A_pd)

    base_jitter = max(
        min_eigenvalue,
        float(np.max(eigenvalues)) / max_condition,
    )

    identity = np.eye(
        A_pd.shape[0],
        dtype=float,
    )

    jitter = 0.0

    for attempt in range(max_tries):
        try:
            return np.linalg.cholesky(
                A_pd + jitter * identity
            )

        except np.linalg.LinAlgError:
            if attempt == 0:
                jitter = base_jitter
            else:
                jitter *= 10.0

    raise np.linalg.LinAlgError(
        "Cholesky failed after positive-definite projection "
        "and adaptive jitter."
    )


def square_root_from_columns(
    A: np.ndarray,
) -> np.ndarray:
    """
    Given a matrix A, return a lower-triangular matrix S such that

        S S^T = A A^T.

    The factor is obtained through a QR decomposition of A^T.
    """

    A = np.asarray(A, dtype=float)

    if not np.all(np.isfinite(A)):
        raise FloatingPointError(
            "QR input contains NaN or Inf."
        )

    _, R_upper = np.linalg.qr(
        A.T,
        mode="reduced",
    )

    S_lower = R_upper.T

    diagonal = np.diag(S_lower)

    signs = np.where(
        diagonal < 0.0,
        -1.0,
        1.0,
    )

    S_lower = (
        S_lower
        @ np.diag(signs)
    )

    return S_lower


# ============================================================
# Truth simulation
# ============================================================

def integrate_truth_path(
    f,
    J,
    G,
    Qc,
    t0: float,
    tf: float,
    x0: np.ndarray,
    t_grid: np.ndarray,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    max_step: float = 1e-1,
    method: str = "BDF",
    truth_noise: bool = False,
    qscale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Integrate the true state along t_grid.

    If truth_noise=True, discrete process noise is injected after
    each propagation interval using

        Qd ≈ G(t_mid) Qc(t_mid) G(t_mid)^T dt qscale.
    """

    def matrix_at(M, t):
        value = M(t) if callable(M) else M
        return np.asarray(value, dtype=float)

    x = np.asarray(
        x0,
        dtype=float,
    ).copy()

    states = [x.copy()]

    for k in range(1, len(t_grid)):
        t_prev = float(t_grid[k - 1])
        t_curr = float(t_grid[k])

        solution = solve_ivp(
            fun=lambda t, state: f(t, state),
            t_span=(t_prev, t_curr),
            y0=x,
            method=method,
            jac=lambda t, state: J(t, state),
            t_eval=[t_curr],
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )

        if not solution.success:
            raise RuntimeError(
                f"Truth integration failed at step {k}: "
                f"{solution.message}"
            )

        x = solution.y[:, -1]

        if truth_noise:
            if rng is None:
                rng = np.random.default_rng()

            dt = t_curr - t_prev
            t_mid = 0.5 * (t_prev + t_curr)

            Gt = matrix_at(G, t_mid)
            Qct = matrix_at(Qc, t_mid)

            Qd = (
                Gt
                @ Qct
                @ Gt.T
                * dt
                * float(qscale)
            )

            Qd = project_positive_definite(
                Qd,
                min_eigenvalue=1e-18,
                max_condition=1e15,
            )

            Lq = stable_cholesky(
                Qd,
                min_eigenvalue=1e-18,
                max_condition=1e15,
            )

            x = (
                x
                + Lq
                @ rng.normal(size=x.size)
            )

        states.append(x.copy())

    return np.asarray(states)


# ============================================================
# Measurement covariance
# ============================================================

def build_R(
    case: str,
    mode: str,
    sigma: float,
    diag_vals: List[float] | None,
    H_const: np.ndarray | None,
) -> np.ndarray:
    """
    Construct the measurement-noise covariance R.
    """

    mode = mode.lower()

    if mode == "diag":
        if diag_vals is None:
            if case == "dahlquist":
                return np.array(
                    [[0.04]],
                    dtype=float,
                )

            return np.diag(
                [0.04, 0.04]
            ).astype(float)

        if len(diag_vals) == 1:
            if case == "dahlquist":
                return np.array(
                    [[diag_vals[0]]],
                    dtype=float,
                )

            return np.diag(
                [diag_vals[0], diag_vals[0]]
            ).astype(float)

        return np.diag(
            [diag_vals[0], diag_vals[1]]
        ).astype(float)

    if mode == "aniso":
        if H_const is None or H_const.shape[0] != 2:
            raise ValueError(
                "Rmode='aniso' requires a two-dimensional "
                "measurement."
            )

        u1 = np.array(
            [1.0, 1.0],
            dtype=float,
        )
        u1 /= np.linalg.norm(u1)

        u2 = np.array(
            [1.0, -(1.0 + sigma)],
            dtype=float,
        )
        u2 /= np.linalg.norm(u2)

        U = np.column_stack(
            [u1, u2]
        )

        if diag_vals is None:
            d1 = 1e-4
            d2 = 1e-2

        elif len(diag_vals) == 1:
            d1 = float(diag_vals[0])
            d2 = float(diag_vals[0])

        else:
            d1 = float(diag_vals[0])
            d2 = float(diag_vals[1])

        R = (
            U
            @ np.diag([d1, d2])
            @ U.T
        )

        return project_positive_definite(
            R,
            min_eigenvalue=1e-14,
            max_condition=1e14,
        )

    raise ValueError(
        "Rmode must be 'diag' or 'aniso'."
    )


# ============================================================
# Measurement models
# ============================================================

def build_measurement(
    case: str,
    meas: str,
    sigma: float,
    Rmode: str,
    Rdiag: List[float] | None,
    nl_kind: str,
    nl_eps: float,
    nl_alpha: float,
) -> Tuple:
    """
    Return h(x), H(x), R, and a measurement tag.
    """

    meas = meas.lower()
    nl_kind = nl_kind.lower()

    if case == "dahlquist":
        if meas != "paper":
            raise ValueError(
                "For Dahlquist, only meas='paper' is supported."
            )

        def h(x: np.ndarray) -> np.ndarray:
            return np.array(
                [x[0]],
                dtype=float,
            )

        def H(x: np.ndarray) -> np.ndarray:
            return np.array(
                [[1.0]],
                dtype=float,
            )

        R = build_R(
            case=case,
            mode=Rmode,
            sigma=sigma,
            diag_vals=Rdiag,
            H_const=None,
        )

        return h, H, R, "paper"

    if meas == "paper":
        h = vdp_h()
        H = vdp_H()

        R = build_R(
            case=case,
            mode=Rmode,
            sigma=sigma,
            diag_vals=Rdiag,
            H_const=None,
        )

        return h, H, R, "paper"

    if meas == "ill":
        H_const = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0 + sigma],
            ],
            dtype=float,
        )

        def h(x: np.ndarray) -> np.ndarray:
            return H_const @ x

        def H(x: np.ndarray) -> np.ndarray:
            return H_const

        R = build_R(
            case=case,
            mode=Rmode,
            sigma=sigma,
            diag_vals=Rdiag,
            H_const=H_const,
        )

        return h, H, R, "illcond"

    if meas in {
        "nonlin_cubic",
        "nonlin_tanh",
    }:

        def s1s2(
            x: np.ndarray,
        ) -> Tuple[float, float]:
            s1 = x[0] + x[1]

            s2 = (
                x[0]
                + (1.0 + sigma) * x[1]
            )

            return s1, s2

        if (
            meas == "nonlin_cubic"
            or nl_kind == "cubic"
        ):

            def h(
                x: np.ndarray,
            ) -> np.ndarray:
                s1, s2 = s1s2(x)

                return np.array(
                    [
                        s1,
                        s2 + nl_eps * s2**3,
                    ],
                    dtype=float,
                )

            def H(
                x: np.ndarray,
            ) -> np.ndarray:
                _, s2 = s1s2(x)

                derivative = (
                    1.0
                    + 3.0
                    * nl_eps
                    * s2**2
                )

                return np.array(
                    [
                        [1.0, 1.0],
                        [
                            derivative,
                            derivative * (1.0 + sigma),
                        ],
                    ],
                    dtype=float,
                )

            tag = "nonlin_cubic"

        else:

            def h(
                x: np.ndarray,
            ) -> np.ndarray:
                s1, s2 = s1s2(x)

                return np.array(
                    [
                        s1,
                        s2
                        + nl_eps
                        * np.tanh(
                            nl_alpha * s2
                        ),
                    ],
                    dtype=float,
                )

            def H(
                x: np.ndarray,
            ) -> np.ndarray:
                _, s2 = s1s2(x)

                tanh_value = np.tanh(
                    nl_alpha * s2
                )

                sech_squared = (
                    1.0
                    - tanh_value**2
                )

                derivative = (
                    1.0
                    + nl_eps
                    * nl_alpha
                    * sech_squared
                )

                return np.array(
                    [
                        [1.0, 1.0],
                        [
                            derivative,
                            derivative * (1.0 + sigma),
                        ],
                    ],
                    dtype=float,
                )

            tag = "nonlin_tanh"

        H_reference = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0 + sigma],
            ],
            dtype=float,
        )

        R = build_R(
            case=case,
            mode=Rmode,
            sigma=sigma,
            diag_vals=Rdiag,
            H_const=H_reference,
        )

        return h, H, R, tag

    raise ValueError(
        "meas must be one of "
        "{'paper', 'ill', 'nonlin_cubic', 'nonlin_tanh'}."
    )


# ============================================================
# Original influence-diagram EKF
#
# This class is preserved from the user's original code.
# ============================================================

class IDEKFIter(CDIDEKF):
    """
    Original influence-diagram-based EKF implementation.
    """

    def __init__(
        self,
        *args,
        iter_max: int = 3,
        iter_tol: float = 1e-10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.iter_max = iter_max
        self.iter_tol = iter_tol

    def update(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        z: np.ndarray,
    ):
        x = x_pred.copy()
        P = P_pred.copy()

        z_col = np.asarray(
            z
        ).reshape(-1, 1)

        for _ in range(
            max(1, self.iter_max)
        ):
            Hk = self.H(x)

            B, V, _ = cov_to_inf(
                P,
                P.shape[0],
            )

            u = x.reshape(-1, 1)

            def h_wrapped(u_vec):
                return np.asarray(
                    self.h(
                        np.asarray(
                            u_vec
                        ).reshape(-1)
                    )
                ).reshape(-1, 1)

            (
                u_post,
                V_post,
                B_post,
                _,
                _,
            ) = mupdate(
                1,
                z_col,
                u,
                B,
                V,
                self.R,
                Hk,
                h_wrapped,
            )

            x_new = np.asarray(
                u_post
            ).reshape(-1)

            P_new = ensure_psd(
                inf_to_cov(
                    np.asarray(
                        V_post
                    ).reshape(-1),
                    np.asarray(B_post),
                    x_new.size,
                )
            )

            if (
                np.linalg.norm(
                    x_new - x
                )
                < self.iter_tol
            ):
                x = x_new
                P = P_new
                break

            x = x_new
            P = P_new

        zhat = np.asarray(
            self.h(x_pred)
        ).ravel()

        innovation = (
            np.asarray(z).ravel()
            - zhat
        )

        innovation_covariance = ensure_psd(
            self.H(x_pred)
            @ P_pred
            @ self.H(x_pred).T
            + self.R
        )

        return (
            x,
            P,
            innovation,
            innovation_covariance,
        )


# ============================================================
# Square-root EKF
# ============================================================

class CDSREKF(CDEKF):
    """
    Continuous-discrete square-root EKF measurement update.

    The existing CDEKF.predict() method is used for continuous
    propagation. The predicted covariance is then projected and
    factorized before the square-root measurement update.

    This class does not change the influence-diagram EKF.
    """

    def update(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        z: np.ndarray,
    ):
        x_pred = np.asarray(
            x_pred,
            dtype=float,
        ).reshape(-1)

        z = np.asarray(
            z,
            dtype=float,
        ).reshape(-1)

        if not np.all(np.isfinite(x_pred)):
            raise FloatingPointError(
                "SR-EKF predicted state contains NaN or Inf."
            )

        if not np.all(np.isfinite(P_pred)):
            raise FloatingPointError(
                "SR-EKF predicted covariance contains NaN or Inf."
            )

        P_pred = project_positive_definite(
            P_pred,
            min_eigenvalue=1e-10,
            max_condition=1e12,
        )

        R = project_positive_definite(
            self.R,
            min_eigenvalue=1e-14,
            max_condition=1e14,
        )

        Hk = np.asarray(
            self.H(x_pred),
            dtype=float,
        )

        predicted_measurement = np.asarray(
            self.h(x_pred),
            dtype=float,
        ).reshape(-1)

        innovation = (
            z
            - predicted_measurement
        )

        Sx = stable_cholesky(
            P_pred,
            min_eigenvalue=1e-10,
            max_condition=1e12,
        )

        Sr = stable_cholesky(
            R,
            min_eigenvalue=1e-14,
            max_condition=1e14,
        )

        innovation_covariance = (
            Hk
            @ P_pred
            @ Hk.T
            + R
        )

        innovation_covariance = project_positive_definite(
            innovation_covariance,
            min_eigenvalue=1e-12,
            max_condition=1e14,
        )

        innovation_factor = stable_cholesky(
            innovation_covariance,
            min_eigenvalue=1e-12,
            max_condition=1e14,
        )

        PH_transpose = (
            P_pred
            @ Hk.T
        )

        temporary = linalg.solve_triangular(
            innovation_factor,
            PH_transpose.T,
            lower=True,
            check_finite=False,
        )

        kalman_gain = linalg.solve_triangular(
            innovation_factor.T,
            temporary,
            lower=False,
            check_finite=False,
        ).T

        x_post = (
            x_pred
            + kalman_gain
            @ innovation
        )

        identity = np.eye(
            x_pred.size,
            dtype=float,
        )

        joseph_columns = np.hstack(
            [
                (
                    identity
                    - kalman_gain @ Hk
                )
                @ Sx,
                kalman_gain @ Sr,
            ]
        )

        S_post = square_root_from_columns(
            joseph_columns
        )

        P_post = (
            S_post
            @ S_post.T
        )

        P_post = project_positive_definite(
            P_post,
            min_eigenvalue=1e-10,
            max_condition=1e12,
        )

        # Re-factor once so the returned covariance is guaranteed
        # to correspond to a valid square-root factor.
        S_post = stable_cholesky(
            P_post,
            min_eigenvalue=1e-10,
            max_condition=1e12,
        )

        P_post = (
            S_post
            @ S_post.T
        )

        return (
            x_post,
            P_post,
            innovation,
            innovation_covariance,
        )


# ============================================================
# Benchmark
# ============================================================

def run_cd(
    case: str,
    deltas: List[float],
    N_runs: int,
    seed: int,
    outdir: str,
    profile: str = "paper",
    meas: str = "paper",
    sigma: float | None = None,
    Rmode: str = "diag",
    Rdiag: List[float] | None = None,
    metric: str = "avg",
    truth_noise: bool = False,
    truth_qscale: float = 1.0,
    nl_kind: str = "cubic",
    nl_eps: float = 0.4,
    nl_alpha: float = 1.0,
    x0_perturb: List[float] | None = None,
    idekf_iter_max: int = 5,
    idekf_iter_tol: float = 1e-10,
) -> Dict[float, Dict[str, float]]:
    """
    Compare EKF, SR-EKF, and influence-diagram EKF.
    """

    rng = np.random.default_rng(seed)

    os.makedirs(
        outdir,
        exist_ok=True,
    )

    results: Dict[
        float,
        Dict[str, float],
    ] = {}

    if profile == "paper":
        integration_method = "BDF"

        truth_rtol = 1e-6
        truth_atol = 1e-6

        filter_rtol = 1e-6
        filter_atol = 1e-6

        max_step = 1e-1

    elif profile == "harsh":
        integration_method = "BDF"

        truth_rtol = 1e-8
        truth_atol = 1e-8

        filter_rtol = 1e-3
        filter_atol = 1e-3

        max_step = 5e-1

    else:
        raise ValueError(
            "profile must be 'paper' or 'harsh'."
        )

    for delta in deltas:
        if case == "dahlquist":
            mu = -1e4
            j = 3

            f = dahlquist_f(mu, j)
            J = dahlquist_J(mu, j)

            G = dahlquist_G()
            Qc = dahlquist_Qc()

            x0 = np.array(
                [1.0],
                dtype=float,
            )

            t0 = 0.0
            tf = 4.0

        elif case == "vdp":
            mu = 1e5

            f = vdp_f(mu)
            J = vdp_J(mu)

            G = vdp_G()
            Qc = vdp_Qc()

            x0 = np.array(
                [2.0, 0.0],
                dtype=float,
            )

            t0 = 0.0
            tf = 2.0

        else:
            raise ValueError(
                "case must be 'dahlquist' or 'vdp'."
            )

        if sigma is None:
            sigma = 1e-3

        (
            h,
            H,
            R,
            measurement_tag,
        ) = build_measurement(
            case=case,
            meas=meas,
            sigma=sigma,
            Rmode=Rmode,
            Rdiag=Rdiag,
            nl_kind=nl_kind,
            nl_eps=nl_eps,
            nl_alpha=nl_alpha,
        )

        continuous_model = ContinuousModel(
            f=f,
            J=J,
            G=G,
            Qc=Qc,
        )

        t_grid = np.arange(
            t0,
            tf + 1e-12,
            delta,
        )

        truth_rng = np.random.default_rng(
            rng.integers(1 << 32)
        )

        x_true = integrate_truth_path(
            f=f,
            J=J,
            G=G,
            Qc=Qc,
            t0=t0,
            tf=tf,
            x0=x0,
            t_grid=t_grid,
            rtol=truth_rtol,
            atol=truth_atol,
            max_step=max_step,
            method=integration_method,
            truth_noise=truth_noise,
            qscale=truth_qscale,
            rng=truth_rng,
        )

        x_init = x0.copy()

        if (
            x0_perturb is not None
            and len(x0_perturb) > 0
        ):
            perturbation = np.zeros_like(
                x_init
            )

            count = min(
                len(perturbation),
                len(x0_perturb),
            )

            for index in range(count):
                perturbation[index] = float(
                    x0_perturb[index]
                )

            x_init = (
                x_init
                + perturbation
            )

        P0 = (
            np.eye(
                x0.size,
                dtype=float,
            )
            * 1e-2
        )

        errors = {
            "EKF": [],
            "SR-EKF": [],
            "IDEKF": [],
        }

        failure_counts = {
            "EKF": 0,
            "SR-EKF": 0,
            "IDEKF": 0,
        }

        R_for_noise = project_positive_definite(
            R,
            min_eigenvalue=1e-14,
            max_condition=1e14,
        )

        chol_R = stable_cholesky(
            R_for_noise,
            min_eigenvalue=1e-14,
            max_condition=1e14,
        )

        measurement_dimension = (
            R.shape[0]
        )

        standard_noises = rng.normal(
            size=(
                N_runs,
                len(t_grid),
                measurement_dimension,
            )
        )

        run_noises = np.einsum(
            "ij,rtj->rti",
            chol_R,
            standard_noises,
        )

        for run_index in range(N_runs):
            measurements = np.asarray(
                [
                    h(x_true[k])
                    + run_noises[
                        run_index,
                        k,
                    ]
                    for k in range(
                        len(t_grid)
                    )
                ]
            )

            ekf = CDEKF(
                continuous_model,
                h,
                H,
                R,
                rtol=filter_rtol,
                atol=filter_atol,
                max_step=max_step,
                method=integration_method,
            )

            sr_ekf = CDSREKF(
                continuous_model,
                h,
                H,
                R,
                rtol=filter_rtol,
                atol=filter_atol,
                max_step=max_step,
                method=integration_method,
            )

            idekf = IDEKFIter(
                continuous_model,
                h,
                H,
                R,
                rtol=filter_rtol,
                atol=filter_atol,
                max_step=max_step,
                method=integration_method,
                iter_max=idekf_iter_max,
                iter_tol=idekf_iter_tol,
            )

            filters_to_run = [
                ("EKF", ekf),
                ("SR-EKF", sr_ekf),
                ("IDEKF", idekf),
            ]

            for name, filter_object in filters_to_run:
                xk = x_init.copy()
                Pk = P0.copy()

                state_estimates = [
                    xk.copy()
                ]

                filter_failed = False

                try:
                    for k in range(
                        1,
                        len(t_grid),
                    ):
                        t_prev = float(
                            t_grid[k - 1]
                        )

                        t_curr = float(
                            t_grid[k]
                        )

                        xk, Pk = filter_object.predict(
                            t_prev,
                            t_curr,
                            xk,
                            Pk,
                        )

                        if not np.all(
                            np.isfinite(xk)
                        ):
                            raise FloatingPointError(
                                f"{name} predicted state became non-finite."
                            )

                        if not np.all(
                            np.isfinite(Pk)
                        ):
                            raise FloatingPointError(
                                f"{name} predicted covariance became non-finite."
                            )

                        (
                            xk,
                            Pk,
                            _,
                            _,
                        ) = filter_object.update(
                            xk,
                            Pk,
                            measurements[k],
                        )

                        if not np.all(
                            np.isfinite(xk)
                        ):
                            raise FloatingPointError(
                                f"{name} posterior state became non-finite."
                            )

                        if not np.all(
                            np.isfinite(Pk)
                        ):
                            raise FloatingPointError(
                                f"{name} posterior covariance became non-finite."
                            )

                        state_estimates.append(
                            xk.copy()
                        )

                except (
                    np.linalg.LinAlgError,
                    FloatingPointError,
                    RuntimeError,
                    ValueError,
                ) as error:
                    filter_failed = True

                    failure_counts[name] += 1

                    print(
                        f"Warning: {name} failed for "
                        f"delta={delta:g}, run={run_index}: {error}"
                    )

                if filter_failed:
                    errors[name].append(
                        np.nan
                    )
                    continue

                state_estimates = np.vstack(
                    state_estimates
                )

                squared_error = np.sum(
                    (
                        state_estimates
                        - x_true
                    ) ** 2,
                    axis=1,
                )

                if metric == "avg":
                    armse = float(
                        np.sqrt(
                            np.mean(
                                squared_error
                            )
                        )
                    )

                else:
                    armse = float(
                        np.sqrt(
                            np.sum(
                                squared_error
                            )
                        )
                    )

                errors[name].append(
                    armse
                )

        results[delta] = {}

        for name in errors:
            valid_errors = np.asarray(
                errors[name],
                dtype=float,
            )

            valid_errors = valid_errors[
                np.isfinite(valid_errors)
            ]

            if valid_errors.size == 0:
                results[delta][name] = np.nan
            else:
                results[delta][name] = float(
                    np.mean(valid_errors)
                )

            results[delta][
                f"{name}_failures"
            ] = int(
                failure_counts[name]
            )

        print(
            f"delta={delta:g}: "
            f"EKF={results[delta]['EKF']:.6g}, "
            f"SR-EKF={results[delta]['SR-EKF']:.6g}, "
            f"IDEKF={results[delta]['IDEKF']:.6g} "
            f"| failures: "
            f"EKF={failure_counts['EKF']}, "
            f"SR-EKF={failure_counts['SR-EKF']}, "
            f"IDEKF={failure_counts['IDEKF']}"
        )

        suffix = (
            f"{case}_"
            f"{measurement_tag}_"
            f"{Rmode}_"
            f"{metric}"
        )

        per_delta_csv = os.path.join(
            outdir,
            f"{suffix}_cd_armse.csv",
        )

        write_header = not os.path.exists(
            per_delta_csv
        )

        with open(
            per_delta_csv,
            "a",
            newline="",
        ) as output_file:
            writer = csv.writer(
                output_file
            )

            if write_header:
                writer.writerow(
                    [
                        "delta",
                        "EKF",
                        "SR-EKF",
                        "IDEKF",
                        "EKF_failures",
                        "SR-EKF_failures",
                        "IDEKF_failures",
                        "sigma",
                        "metric",
                    ]
                )

            writer.writerow(
                [
                    delta,
                    results[delta]["EKF"],
                    results[delta]["SR-EKF"],
                    results[delta]["IDEKF"],
                    failure_counts["EKF"],
                    failure_counts["SR-EKF"],
                    failure_counts["IDEKF"],
                    sigma,
                    metric,
                ]
            )

    sorted_deltas = sorted(
        results.keys()
    )

    plt.figure(
        figsize=(8, 5)
    )

    plot_configuration = [
        ("EKF", "o"),
        ("SR-EKF", "s"),
        ("IDEKF", "D"),
    ]

    for name, marker in plot_configuration:
        values = np.asarray(
            [
                results[delta][name]
                for delta in sorted_deltas
            ],
            dtype=float,
        )

        plt.plot(
            sorted_deltas,
            values,
            marker=marker,
            label=name,
        )

    plt.xlabel(
        "Sampling period δ"
    )

    if metric == "avg":
        plt.ylabel("ARMSE")
    else:
        plt.ylabel(
            "Cumulative error (sqrt sum e²)"
        )

    title = (
        f"CD benchmark — "
        f"{case} — "
        f"{measurement_tag} — "
        f"R:{Rmode} "
        f"({profile}, {metric})"
    )

    if truth_noise:
        title += (
            f" (+ truth noise q={truth_qscale:g})"
        )

    plt.title(title)
    plt.legend()

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    suffix = (
        f"{case}_"
        f"{measurement_tag}_"
        f"{Rmode}_"
        f"{metric}"
    )

    plot_path = os.path.join(
        outdir,
        f"{suffix}_three_filter_summary.png",
    )

    plt.savefig(
        plot_path,
        dpi=180,
    )

    plt.close()

    return results


# ============================================================
# Command-line interface
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        choices=[
            "dahlquist",
            "vdp",
        ],
        default="vdp",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
    )

    parser.add_argument(
        "--deltas",
        type=float,
        nargs="*",
        default=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ],
    )

    parser.add_argument(
        "--profile",
        choices=[
            "paper",
            "harsh",
        ],
        default="paper",
    )

    parser.add_argument(
        "--meas",
        choices=[
            "paper",
            "ill",
            "nonlin_cubic",
            "nonlin_tanh",
        ],
        default="ill",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1e-10,
    )

    parser.add_argument(
        "--Rmode",
        choices=[
            "diag",
            "aniso",
        ],
        default="diag",
    )

    parser.add_argument(
        "--Rdiag",
        type=float,
        nargs="*",
        default=[0.04],
    )

    parser.add_argument(
        "--nl-kind",
        choices=[
            "cubic",
            "tanh",
        ],
        default="cubic",
    )

    parser.add_argument(
        "--nl-eps",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--nl-alpha",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--metric",
        choices=[
            "avg",
            "cum",
        ],
        default="avg",
    )

    parser.add_argument(
        "--truth-noise",
        action="store_true",
    )

    parser.add_argument(
        "--truth-qscale",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--x0-perturb",
        type=float,
        nargs="*",
        default=[0.0, 0.0],
    )

    parser.add_argument(
        "--idekf-iter-max",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--idekf-iter-tol",
        type=float,
        default=1e-10,
    )

    args = parser.parse_args()

    results = run_cd(
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

    print("\nFinal results:")

    for delta in sorted(
        results.keys()
    ):
        print(
            f"delta={delta:g}: "
            f"EKF={results[delta]['EKF']:.6g}, "
            f"SR-EKF={results[delta]['SR-EKF']:.6g}, "
            f"IDEKF={results[delta]['IDEKF']:.6g}, "
            f"EKF failures={results[delta]['EKF_failures']}, "
            f"SR-EKF failures={results[delta]['SR-EKF_failures']}, "
            f"IDEKF failures={results[delta]['IDEKF_failures']}"
        )

    suffix = (
        f"{args.case}_"
        f"{args.meas}_"
        f"{args.Rmode}_"
        f"{args.metric}"
    )

    summary_csv_path = os.path.join(
        args.outdir,
        f"{suffix}_armse_vs_delta.csv",
    )

    os.makedirs(
        args.outdir,
        exist_ok=True,
    )

    with open(
        summary_csv_path,
        "w",
        newline="",
    ) as output_file:
        writer = csv.writer(
            output_file
        )

        writer.writerow(
            [
                "delta",
                "EKF",
                "SR-EKF",
                "IDEKF",
                "EKF_failures",
                "SR-EKF_failures",
                "IDEKF_failures",
            ]
        )

        for delta in sorted(
            results.keys()
        ):
            writer.writerow(
                [
                    delta,
                    results[delta]["EKF"],
                    results[delta]["SR-EKF"],
                    results[delta]["IDEKF"],
                    results[delta]["EKF_failures"],
                    results[delta]["SR-EKF_failures"],
                    results[delta]["IDEKF_failures"],
                ]
            )

    print(
        f"Wrote {summary_csv_path}"
    )