"""
Validate a correction for repeated measurement assimilation in IDEKF.

The previous IDEKF wrapper repeatedly used the posterior state and covariance
as the prior for another update with the same measurement:

    x, P -> mupdate(z) -> x_new, P_new
    x_new, P_new -> mupdate(z) -> ...

That can count the same observation multiple times and produce severe
covariance overconfidence.

This file compares:

1. EKF
2. SR-EKF
3. Base CDIDEKF from filters.py
4. SingleAssimilationIDEKF:
       exactly one influence-diagram measurement update per observation
5. RepeatedAssimilationIDEKF:
       old behavior, retained only as a diagnostic baseline

The file performs:

1. Linear one-step equivalence tests
2. Information-matrix tests
3. Repeated-assimilation covariance tests
4. Paired Monte Carlo comparisons
5. ARMSE, NEES, NIS, covariance-trace, and win-rate summaries

Positive paired ARMSE differences favor the named IDEKF implementation.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

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
    vdp_G,
    vdp_Qc,
)

from run_three_filter_comparison import (
    CDSREKF,
    build_measurement,
    integrate_truth_path,
    project_positive_definite,
    stable_cholesky,
)


FILTER_NAMES = (
    "EKF",
    "SR-EKF",
    "Base-CDIDEKF",
    "Single-ID",
    "Repeated-ID",
)


# ============================================================
# Numerical helpers
# ============================================================

def symmetrize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def safe_positive_definite(
    matrix: np.ndarray,
    floor: float = 1e-12,
) -> np.ndarray:
    return project_positive_definite(
        symmetrize(matrix),
        min_eigenvalue=floor,
        max_condition=1e14,
    )


def quadratic_form(
    vector: np.ndarray,
    covariance: np.ndarray,
) -> float:
    vector = np.asarray(
        vector,
        dtype=float,
    ).reshape(-1)

    covariance = safe_positive_definite(
        covariance
    )

    try:
        solution = np.linalg.solve(
            covariance,
            vector,
        )
    except np.linalg.LinAlgError:
        solution = (
            np.linalg.pinv(covariance)
            @ vector
        )

    return float(
        vector @ solution
    )


def finite_values(
    values: Sequence[float],
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=float,
    )

    return array[
        np.isfinite(array)
    ]


def safe_mean(
    values: Sequence[float],
) -> float:
    array = finite_values(values)

    if array.size == 0:
        return np.nan

    return float(
        np.mean(array)
    )


def safe_median(
    values: Sequence[float],
) -> float:
    array = finite_values(values)

    if array.size == 0:
        return np.nan

    return float(
        np.median(array)
    )


def safe_std(
    values: Sequence[float],
) -> float:
    array = finite_values(values)

    if array.size < 2:
        return np.nan

    return float(
        np.std(
            array,
            ddof=1,
        )
    )


def matrix_summary(
    matrix: np.ndarray,
) -> Dict[str, float]:
    matrix = symmetrize(matrix)

    eigenvalues = np.linalg.eigvalsh(
        matrix
    )

    minimum = float(
        np.min(eigenvalues)
    )

    maximum = float(
        np.max(eigenvalues)
    )

    condition = float(
        abs(maximum)
        / max(
            abs(minimum),
            1e-300,
        )
    )

    return {
        "trace": float(
            np.trace(matrix)
        ),
        "minimum_eigenvalue": minimum,
        "maximum_eigenvalue": maximum,
        "condition_number": condition,
    }


def relative_matrix_error(
    actual: np.ndarray,
    reference: np.ndarray,
) -> float:
    numerator = float(
        np.linalg.norm(
            np.asarray(actual)
            - np.asarray(reference),
            ord="fro",
        )
    )

    denominator = max(
        float(
            np.linalg.norm(
                reference,
                ord="fro",
            )
        ),
        1e-15,
    )

    return numerator / denominator


def write_csv(
    path: str,
    rows: Sequence[Dict[str, object]],
) -> None:
    if not rows:
        print(
            f"Warning: no rows for {path}"
        )
        return

    fieldnames: List[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(
        path,
        "w",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ============================================================
# IDEKF implementations
# ============================================================

class SingleAssimilationIDEKF(CDIDEKF):
    """
    Corrected influence-diagram measurement wrapper.

    Each observation is assimilated exactly once.

    This preserves the same cov_to_inf, mupdate, and inf_to_cov operations
    used by the previous wrapper, but removes repeated reuse of the
    posterior as a new prior for the same observation.
    """

    def update(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        z: np.ndarray,
    ):
        x_prior = np.asarray(
            x_pred,
            dtype=float,
        ).copy()

        P_prior = np.asarray(
            P_pred,
            dtype=float,
        ).copy()

        z_vector = np.asarray(
            z,
            dtype=float,
        ).reshape(-1)

        z_column = z_vector.reshape(
            -1,
            1,
        )

        H_prior = np.asarray(
            self.H(x_prior),
            dtype=float,
        )

        B_prior, V_prior, _ = cov_to_inf(
            P_prior,
            P_prior.shape[0],
        )

        u_prior = x_prior.reshape(
            -1,
            1,
        )

        def h_wrapped(
            u_vector,
        ):
            state = np.asarray(
                u_vector,
                dtype=float,
            ).reshape(-1)

            return np.asarray(
                self.h(state),
                dtype=float,
            ).reshape(
                -1,
                1,
            )

        (
            u_posterior,
            V_posterior,
            B_posterior,
            _,
            _,
        ) = mupdate(
            1,
            z_column,
            u_prior,
            B_prior,
            V_prior,
            self.R,
            H_prior,
            h_wrapped,
        )

        x_posterior = np.asarray(
            u_posterior,
            dtype=float,
        ).reshape(-1)

        P_posterior = ensure_psd(
            inf_to_cov(
                np.asarray(
                    V_posterior
                ).reshape(-1),
                np.asarray(
                    B_posterior
                ),
                x_posterior.size,
            )
        )

        predicted_measurement = np.asarray(
            self.h(x_prior),
            dtype=float,
        ).reshape(-1)

        innovation = (
            z_vector
            - predicted_measurement
        )

        innovation_covariance = ensure_psd(
            H_prior
            @ P_prior
            @ H_prior.T
            + self.R
        )

        return (
            x_posterior,
            P_posterior,
            innovation,
            innovation_covariance,
        )


class RepeatedAssimilationIDEKF(CDIDEKF):
    """
    Previous wrapper behavior.

    This is retained only for comparison. It repeatedly assimilates the
    same observation while replacing the prior with the last posterior.
    """

    def __init__(
        self,
        *args,
        iter_max: int = 5,
        iter_tol: float = 1e-10,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.iter_max = int(
            iter_max
        )

        self.iter_tol = float(
            iter_tol
        )

        self.last_iterations_used = 0

    def update(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        z: np.ndarray,
    ):
        x = np.asarray(
            x_pred,
            dtype=float,
        ).copy()

        P = np.asarray(
            P_pred,
            dtype=float,
        ).copy()

        z_vector = np.asarray(
            z,
            dtype=float,
        ).reshape(-1)

        z_column = z_vector.reshape(
            -1,
            1,
        )

        self.last_iterations_used = 0

        for iteration in range(
            max(
                1,
                self.iter_max,
            )
        ):
            self.last_iterations_used = (
                iteration + 1
            )

            H_current = np.asarray(
                self.H(x),
                dtype=float,
            )

            B_current, V_current, _ = (
                cov_to_inf(
                    P,
                    P.shape[0],
                )
            )

            u_current = x.reshape(
                -1,
                1,
            )

            def h_wrapped(
                u_vector,
            ):
                state = np.asarray(
                    u_vector,
                    dtype=float,
                ).reshape(-1)

                return np.asarray(
                    self.h(state),
                    dtype=float,
                ).reshape(
                    -1,
                    1,
                )

            (
                u_new,
                V_new,
                B_new,
                _,
                _,
            ) = mupdate(
                1,
                z_column,
                u_current,
                B_current,
                V_current,
                self.R,
                H_current,
                h_wrapped,
            )

            x_new = np.asarray(
                u_new,
                dtype=float,
            ).reshape(-1)

            P_new = ensure_psd(
                inf_to_cov(
                    np.asarray(
                        V_new
                    ).reshape(-1),
                    np.asarray(B_new),
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

        H_prior = np.asarray(
            self.H(x_pred),
            dtype=float,
        )

        innovation = (
            z_vector
            - np.asarray(
                self.h(x_pred),
                dtype=float,
            ).reshape(-1)
        )

        innovation_covariance = ensure_psd(
            H_prior
            @ P_pred
            @ H_prior.T
            + self.R
        )

        return (
            x,
            P,
            innovation,
            innovation_covariance,
        )


# ============================================================
# Model and filter construction
# ============================================================

def construct_case(
    case: str,
):
    if case == "dahlquist":
        mu = -1.0e4
        j = 3

        f = dahlquist_f(
            mu,
            j,
        )

        J = dahlquist_J(
            mu,
            j,
        )

        G = dahlquist_G()
        Qc = dahlquist_Qc()

        x0 = np.array(
            [1.0],
            dtype=float,
        )

        t0 = 0.0
        tf = 4.0

        return (
            f,
            J,
            G,
            Qc,
            x0,
            t0,
            tf,
        )

    if case == "vdp":
        mu = 1.0e5

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

        return (
            f,
            J,
            G,
            Qc,
            x0,
            t0,
            tf,
        )

    raise ValueError(
        "case must be 'dahlquist' or 'vdp'."
    )


def integration_settings(
    profile: str,
) -> Dict[str, object]:
    if profile == "paper":
        return {
            "method": "BDF",
            "truth_rtol": 1e-6,
            "truth_atol": 1e-6,
            "filter_rtol": 1e-6,
            "filter_atol": 1e-6,
            "max_step": 1e-1,
        }

    if profile == "harsh":
        return {
            "method": "BDF",
            "truth_rtol": 1e-8,
            "truth_atol": 1e-8,
            "filter_rtol": 1e-3,
            "filter_atol": 1e-3,
            "max_step": 5e-1,
        }

    raise ValueError(
        "profile must be 'paper' or 'harsh'."
    )


def initial_estimate(
    x0: np.ndarray,
    perturbation: Sequence[float],
) -> np.ndarray:
    estimate = np.asarray(
        x0,
        dtype=float,
    ).copy()

    additive = np.zeros_like(
        estimate
    )

    for index in range(
        min(
            estimate.size,
            len(perturbation),
        )
    ):
        additive[index] = float(
            perturbation[index]
        )

    return estimate + additive


def make_filters(
    continuous_model: ContinuousModel,
    h,
    H,
    R: np.ndarray,
    settings: Dict[str, object],
    repeated_iterations: int,
    repeated_tolerance: float,
) -> Dict[str, object]:
    common = {
        "rtol": float(
            settings[
                "filter_rtol"
            ]
        ),
        "atol": float(
            settings[
                "filter_atol"
            ]
        ),
        "max_step": float(
            settings[
                "max_step"
            ]
        ),
        "method": str(
            settings[
                "method"
            ]
        ),
    }

    return {
        "EKF": CDEKF(
            continuous_model,
            h,
            H,
            R,
            **common,
        ),
        "SR-EKF": CDSREKF(
            continuous_model,
            h,
            H,
            R,
            **common,
        ),
        "Base-CDIDEKF": CDIDEKF(
            continuous_model,
            h,
            H,
            R,
            **common,
        ),
        "Single-ID": SingleAssimilationIDEKF(
            continuous_model,
            h,
            H,
            R,
            **common,
        ),
        "Repeated-ID": RepeatedAssimilationIDEKF(
            continuous_model,
            h,
            H,
            R,
            iter_max=repeated_iterations,
            iter_tol=repeated_tolerance,
            **common,
        ),
    }


# ============================================================
# Linear one-step validation
# ============================================================

def exact_linear_update(
    x_prior: np.ndarray,
    P_prior: np.ndarray,
    z: np.ndarray,
    H_matrix: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    innovation_covariance = (
        H_matrix
        @ P_prior
        @ H_matrix.T
        + R
    )

    gain = np.linalg.solve(
        innovation_covariance,
        H_matrix @ P_prior,
    ).T

    x_posterior = (
        x_prior
        + gain
        @ (
            z
            - H_matrix
            @ x_prior
        )
    )

    identity = np.eye(
        P_prior.shape[0]
    )

    posterior_covariance = (
        (
            identity
            - gain
            @ H_matrix
        )
        @ P_prior
        @ (
            identity
            - gain
            @ H_matrix
        ).T
        + gain
        @ R
        @ gain.T
    )

    return (
        x_posterior,
        ensure_psd(
            posterior_covariance
        ),
    )


def run_linear_validation(
    continuous_model: ContinuousModel,
    settings: Dict[str, object],
    repeated_iterations: int,
    repeated_tolerance: float,
) -> List[Dict[str, object]]:
    H_matrix = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.001],
        ],
        dtype=float,
    )

    R = np.diag(
        [
            0.04,
            0.04,
        ]
    )

    x_prior = np.array(
        [1.0, -0.5],
        dtype=float,
    )

    P_prior = np.array(
        [
            [0.5, 0.1],
            [0.1, 0.3],
        ],
        dtype=float,
    )

    z = np.array(
        [0.8, 0.805],
        dtype=float,
    )

    def h_linear(
        state: np.ndarray,
    ) -> np.ndarray:
        return H_matrix @ state

    def H_linear(
        state: np.ndarray,
    ) -> np.ndarray:
        del state
        return H_matrix

    (
        exact_state,
        exact_covariance,
    ) = exact_linear_update(
        x_prior=x_prior,
        P_prior=P_prior,
        z=z,
        H_matrix=H_matrix,
        R=R,
    )

    filters = make_filters(
        continuous_model=continuous_model,
        h=h_linear,
        H=H_linear,
        R=R,
        settings=settings,
        repeated_iterations=(
            repeated_iterations
        ),
        repeated_tolerance=(
            repeated_tolerance
        ),
    )

    prior_information = np.linalg.inv(
        P_prior
    )

    one_measurement_information = (
        prior_information
        + H_matrix.T
        @ np.linalg.inv(R)
        @ H_matrix
    )

    repeated_measurement_information = (
        prior_information
        + repeated_iterations
        * H_matrix.T
        @ np.linalg.inv(R)
        @ H_matrix
    )

    rows: List[
        Dict[str, object]
    ] = []

    print(
        "\n"
        "============================================================"
    )

    print(
        "LINEAR ONE-STEP FIX VALIDATION"
    )

    print(
        "============================================================"
    )

    print(
        "A correct implementation should match the exact "
        "linear Kalman posterior."
    )

    print(
        f"Exact posterior state:\n{exact_state}"
    )

    print(
        f"Exact posterior covariance:\n{exact_covariance}"
    )

    for name, filter_object in filters.items():
        try:
            (
                state,
                covariance,
                _,
                _,
            ) = filter_object.update(
                x_prior.copy(),
                P_prior.copy(),
                z.copy(),
            )

            state = np.asarray(
                state,
                dtype=float,
            )

            covariance = np.asarray(
                covariance,
                dtype=float,
            )

            information = np.linalg.inv(
                safe_positive_definite(
                    covariance
                )
            )

            covariance_summary = matrix_summary(
                covariance
            )

            state_error = float(
                np.linalg.norm(
                    state
                    - exact_state
                )
            )

            covariance_error = float(
                np.linalg.norm(
                    covariance
                    - exact_covariance,
                    ord="fro",
                )
            )

            relative_covariance_error = (
                relative_matrix_error(
                    covariance,
                    exact_covariance,
                )
            )

            information_error_one = (
                relative_matrix_error(
                    information,
                    one_measurement_information,
                )
            )

            information_error_repeated = (
                relative_matrix_error(
                    information,
                    repeated_measurement_information,
                )
            )

            passed = bool(
                state_error < 1e-8
                and relative_covariance_error
                < 1e-8
            )

            row = {
                "filter": name,
                "state_error_from_exact": (
                    state_error
                ),
                "covariance_error_from_exact": (
                    covariance_error
                ),
                "relative_covariance_error": (
                    relative_covariance_error
                ),
                "information_error_one_measurement": (
                    information_error_one
                ),
                "information_error_repeated_measurement": (
                    information_error_repeated
                ),
                "posterior_trace": (
                    covariance_summary[
                        "trace"
                    ]
                ),
                "posterior_minimum_eigenvalue": (
                    covariance_summary[
                        "minimum_eigenvalue"
                    ]
                ),
                "posterior_condition_number": (
                    covariance_summary[
                        "condition_number"
                    ]
                ),
                "linear_equivalence_passed": int(
                    passed
                ),
                "failure": 0,
                "failure_message": "",
            }

        except Exception as error:
            row = {
                "filter": name,
                "state_error_from_exact": np.nan,
                "covariance_error_from_exact": np.nan,
                "relative_covariance_error": np.nan,
                "information_error_one_measurement": np.nan,
                "information_error_repeated_measurement": np.nan,
                "posterior_trace": np.nan,
                "posterior_minimum_eigenvalue": np.nan,
                "posterior_condition_number": np.nan,
                "linear_equivalence_passed": 0,
                "failure": 1,
                "failure_message": str(error),
            }

        rows.append(row)

        print(
            f"{name}: "
            f"state error={float(row['state_error_from_exact']):.6g}, "
            f"relative P error="
            f"{float(row['relative_covariance_error']):.6g}, "
            f"one-measurement information error="
            f"{float(row['information_error_one_measurement']):.6g}, "
            f"repeated-information error="
            f"{float(row['information_error_repeated_measurement']):.6g}, "
            f"trace(P+)={float(row['posterior_trace']):.6g}, "
            f"passed={bool(row['linear_equivalence_passed'])}"
        )

    return rows


# ============================================================
# Monte Carlo realization generation
# ============================================================

def generate_realization(
    seed: int,
    delta_index: int,
    t_grid: np.ndarray,
    f,
    J,
    G,
    Qc,
    x0: np.ndarray,
    h,
    R: np.ndarray,
    settings: Dict[str, object],
    truth_noise: bool,
    truth_qscale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    seed_sequence = np.random.SeedSequence(
        [
            int(seed),
            int(delta_index),
            20260722,
        ]
    )

    truth_seed, measurement_seed = (
        seed_sequence.spawn(2)
    )

    truth_rng = np.random.default_rng(
        truth_seed
    )

    measurement_rng = np.random.default_rng(
        measurement_seed
    )

    x_true = integrate_truth_path(
        f=f,
        J=J,
        G=G,
        Qc=Qc,
        t0=float(
            t_grid[0]
        ),
        tf=float(
            t_grid[-1]
        ),
        x0=x0,
        t_grid=t_grid,
        rtol=float(
            settings[
                "truth_rtol"
            ]
        ),
        atol=float(
            settings[
                "truth_atol"
            ]
        ),
        max_step=float(
            settings[
                "max_step"
            ]
        ),
        method=str(
            settings[
                "method"
            ]
        ),
        truth_noise=truth_noise,
        qscale=truth_qscale,
        rng=truth_rng,
    )

    R_positive = safe_positive_definite(
        R,
        floor=1e-14,
    )

    measurement_root = stable_cholesky(
        R_positive,
        min_eigenvalue=1e-14,
        max_condition=1e14,
    )

    standard_noise = measurement_rng.normal(
        size=(
            len(t_grid),
            R.shape[0],
        )
    )

    measurement_noise = (
        standard_noise
        @ measurement_root.T
    )

    measurements = np.asarray(
        [
            np.asarray(
                h(
                    x_true[index]
                ),
                dtype=float,
            ).reshape(-1)
            + measurement_noise[index]
            for index in range(
                len(t_grid)
            )
        ]
    )

    return (
        x_true,
        measurements,
    )


# ============================================================
# Filter trajectory
# ============================================================

def run_filter(
    filter_name: str,
    filter_object,
    x_initial: np.ndarray,
    P_initial: np.ndarray,
    t_grid: np.ndarray,
    x_true: np.ndarray,
    measurements: np.ndarray,
) -> Dict[str, object]:
    x = np.asarray(
        x_initial,
        dtype=float,
    ).copy()

    P = np.asarray(
        P_initial,
        dtype=float,
    ).copy()

    squared_errors: List[float] = []
    nees_values: List[float] = []
    nis_values: List[float] = []
    traces: List[float] = []
    minimum_eigenvalues: List[float] = []
    condition_numbers: List[float] = []

    try:
        for index in range(
            1,
            len(t_grid),
        ):
            previous_time = float(
                t_grid[
                    index - 1
                ]
            )

            current_time = float(
                t_grid[index]
            )

            x_pred, P_pred = (
                filter_object.predict(
                    previous_time,
                    current_time,
                    x,
                    P,
                )
            )

            x_pred = np.asarray(
                x_pred,
                dtype=float,
            )

            P_pred = np.asarray(
                P_pred,
                dtype=float,
            )

            H_pred = np.asarray(
                filter_object.H(
                    x_pred
                ),
                dtype=float,
            )

            innovation = (
                measurements[index]
                - np.asarray(
                    filter_object.h(
                        x_pred
                    ),
                    dtype=float,
                ).reshape(-1)
            )

            innovation_covariance = (
                H_pred
                @ P_pred
                @ H_pred.T
                + filter_object.R
            )

            nis = quadratic_form(
                innovation,
                innovation_covariance,
            )

            (
                x,
                P,
                _,
                _,
            ) = filter_object.update(
                x_pred,
                P_pred,
                measurements[index],
            )

            x = np.asarray(
                x,
                dtype=float,
            )

            P = np.asarray(
                P,
                dtype=float,
            )

            if not np.all(
                np.isfinite(x)
            ):
                raise FloatingPointError(
                    "Posterior state contains NaN or Inf."
                )

            if not np.all(
                np.isfinite(P)
            ):
                raise FloatingPointError(
                    "Posterior covariance contains NaN or Inf."
                )

            error = (
                x_true[index]
                - x
            )

            squared_error = float(
                error @ error
            )

            nees = quadratic_form(
                error,
                P,
            )

            covariance_summary = matrix_summary(
                P
            )

            squared_errors.append(
                squared_error
            )

            nees_values.append(
                nees
            )

            nis_values.append(
                nis
            )

            traces.append(
                covariance_summary[
                    "trace"
                ]
            )

            minimum_eigenvalues.append(
                covariance_summary[
                    "minimum_eigenvalue"
                ]
            )

            condition_numbers.append(
                covariance_summary[
                    "condition_number"
                ]
            )

        return {
            "filter": filter_name,
            "failed": 0,
            "failure_message": "",
            "armse": float(
                np.sqrt(
                    np.mean(
                        squared_errors
                    )
                )
            ),
            "mean_nees": safe_mean(
                nees_values
            ),
            "median_nees": safe_median(
                nees_values
            ),
            "mean_nis": safe_mean(
                nis_values
            ),
            "median_nis": safe_median(
                nis_values
            ),
            "mean_covariance_trace": (
                safe_mean(traces)
            ),
            "median_covariance_trace": (
                safe_median(traces)
            ),
            "minimum_covariance_eigenvalue": (
                min(
                    minimum_eigenvalues
                )
            ),
            "mean_covariance_condition": (
                safe_mean(
                    condition_numbers
                )
            ),
        }

    except Exception as error:
        return {
            "filter": filter_name,
            "failed": 1,
            "failure_message": str(error),
            "armse": np.nan,
            "mean_nees": np.nan,
            "median_nees": np.nan,
            "mean_nis": np.nan,
            "median_nis": np.nan,
            "mean_covariance_trace": np.nan,
            "median_covariance_trace": np.nan,
            "minimum_covariance_eigenvalue": np.nan,
            "mean_covariance_condition": np.nan,
        }


# ============================================================
# Paired Monte Carlo experiment
# ============================================================

def run_monte_carlo(
    args: argparse.Namespace,
    continuous_model: ContinuousModel,
    settings: Dict[str, object],
    f,
    J,
    G,
    Qc,
    x0: np.ndarray,
    t0: float,
    tf: float,
    h,
    H,
    R: np.ndarray,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    x_initial = initial_estimate(
        x0,
        args.x0_perturb,
    )

    P_initial = (
        np.eye(
            x0.size,
            dtype=float,
        )
        * float(
            args.P0_scale
        )
    )

    per_run_rows: List[
        Dict[str, object]
    ] = []

    for delta_index, delta in enumerate(
        args.deltas
    ):
        delta = float(delta)

        t_grid = np.arange(
            t0,
            tf + 1e-12,
            delta,
        )

        print(
            f"\nRunning delta={delta:g}, "
            f"updates={len(t_grid) - 1}"
        )

        for seed_position, seed in enumerate(
            range(
                args.seed_start,
                args.seed_start
                + args.num_seeds,
            ),
            start=1,
        ):
            (
                x_true,
                measurements,
            ) = generate_realization(
                seed=seed,
                delta_index=delta_index,
                t_grid=t_grid,
                f=f,
                J=J,
                G=G,
                Qc=Qc,
                x0=x0,
                h=h,
                R=R,
                settings=settings,
                truth_noise=(
                    args.truth_noise
                ),
                truth_qscale=(
                    args.truth_qscale
                ),
            )

            filters = make_filters(
                continuous_model=(
                    continuous_model
                ),
                h=h,
                H=H,
                R=R,
                settings=settings,
                repeated_iterations=(
                    args.repeated_iterations
                ),
                repeated_tolerance=(
                    args.repeated_tolerance
                ),
            )

            for (
                filter_name,
                filter_object,
            ) in filters.items():
                result = run_filter(
                    filter_name=filter_name,
                    filter_object=filter_object,
                    x_initial=x_initial,
                    P_initial=P_initial,
                    t_grid=t_grid,
                    x_true=x_true,
                    measurements=measurements,
                )

                result.update(
                    {
                        "seed": seed,
                        "delta": delta,
                        "number_of_updates": (
                            len(t_grid) - 1
                        ),
                    }
                )

                per_run_rows.append(
                    result
                )

            if (
                seed_position
                % args.progress_every
                == 0
                or seed_position
                == args.num_seeds
            ):
                print(
                    f"  completed "
                    f"{seed_position}/"
                    f"{args.num_seeds} seeds"
                )

    summary_rows = summarize_monte_carlo(
        per_run_rows
    )

    return (
        per_run_rows,
        summary_rows,
    )


def summarize_monte_carlo(
    per_run_rows: Sequence[
        Dict[str, object]
    ],
) -> List[Dict[str, object]]:
    grouped: Dict[
        Tuple[float, str],
        List[Dict[str, object]],
    ] = defaultdict(list)

    for row in per_run_rows:
        grouped[
            (
                float(
                    row["delta"]
                ),
                str(
                    row["filter"]
                ),
            )
        ].append(row)

    summary_rows: List[
        Dict[str, object]
    ] = []

    for (
        delta,
        filter_name,
    ), rows in sorted(
        grouped.items()
    ):
        successful = [
            row
            for row in rows
            if int(
                row["failed"]
            )
            == 0
        ]

        summary_rows.append(
            {
                "delta": delta,
                "filter": filter_name,
                "runs": len(rows),
                "failures": (
                    len(rows)
                    - len(successful)
                ),
                "mean_ARMSE": safe_mean(
                    [
                        row["armse"]
                        for row in successful
                    ]
                ),
                "median_ARMSE": safe_median(
                    [
                        row["armse"]
                        for row in successful
                    ]
                ),
                "std_ARMSE": safe_std(
                    [
                        row["armse"]
                        for row in successful
                    ]
                ),
                "mean_NEES": safe_mean(
                    [
                        row["mean_nees"]
                        for row in successful
                    ]
                ),
                "median_NEES": safe_median(
                    [
                        row["median_nees"]
                        for row in successful
                    ]
                ),
                "mean_NIS": safe_mean(
                    [
                        row["mean_nis"]
                        for row in successful
                    ]
                ),
                "mean_covariance_trace": (
                    safe_mean(
                        [
                            row[
                                "mean_covariance_trace"
                            ]
                            for row
                            in successful
                        ]
                    )
                ),
                "minimum_covariance_eigenvalue": (
                    min(
                        [
                            row[
                                "minimum_covariance_eigenvalue"
                            ]
                            for row
                            in successful
                            if np.isfinite(
                                row[
                                    "minimum_covariance_eigenvalue"
                                ]
                            )
                        ],
                        default=np.nan,
                    )
                ),
                "mean_covariance_condition": (
                    safe_mean(
                        [
                            row[
                                "mean_covariance_condition"
                            ]
                            for row
                            in successful
                        ]
                    )
                ),
            }
        )

    return summary_rows


def paired_comparisons(
    per_run_rows: Sequence[
        Dict[str, object]
    ],
) -> List[Dict[str, object]]:
    realization_map: Dict[
        Tuple[float, int],
        Dict[str, Dict[str, object]],
    ] = defaultdict(dict)

    for row in per_run_rows:
        key = (
            float(
                row["delta"]
            ),
            int(
                row["seed"]
            ),
        )

        realization_map[key][
            str(
                row["filter"]
            )
        ] = row

    comparisons = [
        (
            "Single-ID",
            "EKF",
        ),
        (
            "Single-ID",
            "SR-EKF",
        ),
        (
            "Single-ID",
            "Base-CDIDEKF",
        ),
        (
            "Repeated-ID",
            "Single-ID",
        ),
    ]

    grouped_differences: Dict[
        Tuple[float, str, str],
        List[float],
    ] = defaultdict(list)

    for (
        delta,
        seed,
    ), filters in realization_map.items():
        del seed

        for target, baseline in comparisons:
            if (
                target not in filters
                or baseline not in filters
            ):
                continue

            target_row = filters[target]
            baseline_row = filters[baseline]

            if (
                int(target_row["failed"])
                or int(
                    baseline_row["failed"]
                )
            ):
                continue

            target_error = float(
                target_row["armse"]
            )

            baseline_error = float(
                baseline_row["armse"]
            )

            difference = (
                baseline_error
                - target_error
            )

            grouped_differences[
                (
                    delta,
                    target,
                    baseline,
                )
            ].append(difference)

    rows: List[
        Dict[str, object]
    ] = []

    for (
        delta,
        target,
        baseline,
    ), differences in sorted(
        grouped_differences.items()
    ):
        differences_array = np.asarray(
            differences,
            dtype=float,
        )

        rows.append(
            {
                "delta": delta,
                "target": target,
                "baseline": baseline,
                "paired_runs": len(
                    differences_array
                ),
                "mean_baseline_minus_target": float(
                    np.mean(
                        differences_array
                    )
                ),
                "median_baseline_minus_target": float(
                    np.median(
                        differences_array
                    )
                ),
                "target_win_rate": float(
                    np.mean(
                        differences_array > 0
                    )
                ),
            }
        )

    return rows


# ============================================================
# Reporting
# ============================================================

def print_summary(
    linear_rows: Sequence[
        Dict[str, object]
    ],
    summary_rows: Sequence[
        Dict[str, object]
    ],
    comparison_rows: Sequence[
        Dict[str, object]
    ],
) -> None:
    print(
        "\n"
        "============================================================"
    )

    print(
        "MONTE CARLO FILTER SUMMARY"
    )

    print(
        "============================================================"
    )

    for row in sorted(
        summary_rows,
        key=lambda item: (
            float(item["delta"]),
            str(item["filter"]),
        ),
    ):
        print(
            f"delta={float(row['delta']):g}, "
            f"{row['filter']}: "
            f"mean ARMSE="
            f"{float(row['mean_ARMSE']):.6g}, "
            f"median ARMSE="
            f"{float(row['median_ARMSE']):.6g}, "
            f"mean NEES="
            f"{float(row['mean_NEES']):.6g}, "
            f"median NEES="
            f"{float(row['median_NEES']):.6g}, "
            f"mean trace(P)="
            f"{float(row['mean_covariance_trace']):.6g}, "
            f"failures={int(row['failures'])}"
        )

    print(
        "\n"
        "============================================================"
    )

    print(
        "PAIRED ARMSE COMPARISONS"
    )

    print(
        "Positive mean differences favor the target filter."
    )

    print(
        "============================================================"
    )

    for row in comparison_rows:
        print(
            f"delta={float(row['delta']):g}: "
            f"target={row['target']}, "
            f"baseline={row['baseline']}, "
            f"mean difference="
            f"{float(row['mean_baseline_minus_target']):.6g}, "
            f"median difference="
            f"{float(row['median_baseline_minus_target']):.6g}, "
            f"target win rate="
            f"{float(row['target_win_rate']):.3f}"
        )

    linear_map = {
        str(row["filter"]): row
        for row in linear_rows
    }

    print(
        "\n"
        "============================================================"
    )

    print(
        "AUTOMATIC INTERPRETATION"
    )

    print(
        "============================================================"
    )

    single_passed = bool(
        linear_map.get(
            "Single-ID",
            {},
        ).get(
            "linear_equivalence_passed",
            0,
        )
    )

    base_passed = bool(
        linear_map.get(
            "Base-CDIDEKF",
            {},
        ).get(
            "linear_equivalence_passed",
            0,
        )
    )

    repeated_row = linear_map.get(
        "Repeated-ID",
        {},
    )

    repeated_one_error = float(
        repeated_row.get(
            "information_error_one_measurement",
            np.nan,
        )
    )

    repeated_many_error = float(
        repeated_row.get(
            "information_error_repeated_measurement",
            np.nan,
        )
    )

    if single_passed:
        print(
            "PASS: Single-ID matches the exact linear "
            "Kalman measurement update."
        )

        print(
            "Use SingleAssimilationIDEKF instead of the "
            "repeated-assimilation wrapper."
        )

    else:
        print(
            "WARNING: Single-ID does not match the exact "
            "linear Kalman update."
        )

        print(
            "Do not treat the correction as fully validated yet. "
            "The mupdate calling convention or the base influence-"
            "diagram update must be inspected next."
        )

    if base_passed:
        print(
            "PASS: Base CDIDEKF matches the exact linear update."
        )

        print(
            "The repository's base CDIDEKF is a valid candidate "
            "for the corrected production implementation."
        )

    else:
        print(
            "WARNING: Base CDIDEKF does not match the exact "
            "linear update under this test."
        )

    if (
        np.isfinite(repeated_one_error)
        and np.isfinite(
            repeated_many_error
        )
    ):
        if (
            repeated_many_error
            < repeated_one_error
        ):
            print(
                "CONFIRMED: Repeated-ID is closer to an "
                f"{5}-measurement information update than to "
                "a one-measurement update."
            )

            print(
                "The previous wrapper was effectively counting "
                "the same observation multiple times."
            )
        else:
            print(
                "Repeated-ID was not closer to the simple repeated-"
                "information formula. Its nonlinear/relinearized "
                "behavior requires further inspection."
            )


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a single-assimilation correction for "
            "the influence-diagram EKF."
        )
    )

    parser.add_argument(
        "--case",
        choices=[
            "dahlquist",
            "vdp",
        ],
        default="vdp",
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
        default="nonlin_cubic",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--Rmode",
        choices=[
            "diag",
            "aniso",
        ],
        default="aniso",
    )

    parser.add_argument(
        "--Rdiag",
        type=float,
        nargs="*",
        default=[
            1e-4,
            1e-2,
        ],
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
        "--x0-perturb",
        type=float,
        nargs="*",
        default=[
            1.5,
            -1.0,
        ],
    )

    parser.add_argument(
        "--P0-scale",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[
            0.1,
            0.2,
            0.7,
            0.8,
        ],
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
        "--num-seeds",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=2001,
    )

    parser.add_argument(
        "--repeated-iterations",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--repeated-tolerance",
        type=float,
        default=1e-10,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="results_idekf_fix",
    )

    return parser.parse_args()


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    if args.num_seeds < 1:
        raise ValueError(
            "num-seeds must be at least 1."
        )

    if args.P0_scale <= 0.0:
        raise ValueError(
            "P0-scale must be positive."
        )

    if args.repeated_iterations < 1:
        raise ValueError(
            "repeated-iterations must be at least 1."
        )

    if any(
        delta <= 0.0
        for delta in args.deltas
    ):
        raise ValueError(
            "All delta values must be positive."
        )


if __name__ == "__main__":
    arguments = parse_arguments()

    validate_arguments(
        arguments
    )

    os.makedirs(
        arguments.outdir,
        exist_ok=True,
    )

    settings = integration_settings(
        arguments.profile
    )

    (
        f,
        J,
        G,
        Qc,
        x0,
        t0,
        tf,
    ) = construct_case(
        arguments.case
    )

    continuous_model = ContinuousModel(
        f=f,
        J=J,
        G=G,
        Qc=Qc,
    )

    (
        h,
        H,
        R,
        measurement_tag,
    ) = build_measurement(
        case=arguments.case,
        meas=arguments.meas,
        sigma=arguments.sigma,
        Rmode=arguments.Rmode,
        Rdiag=arguments.Rdiag,
        nl_kind=arguments.nl_kind,
        nl_eps=arguments.nl_eps,
        nl_alpha=arguments.nl_alpha,
    )

    print(
        f"Measurement model: {measurement_tag}"
    )

    linear_rows = run_linear_validation(
        continuous_model=continuous_model,
        settings=settings,
        repeated_iterations=(
            arguments.repeated_iterations
        ),
        repeated_tolerance=(
            arguments.repeated_tolerance
        ),
    )

    (
        per_run_rows,
        summary_rows,
    ) = run_monte_carlo(
        args=arguments,
        continuous_model=continuous_model,
        settings=settings,
        f=f,
        J=J,
        G=G,
        Qc=Qc,
        x0=x0,
        t0=t0,
        tf=tf,
        h=h,
        H=H,
        R=R,
    )

    comparison_rows = paired_comparisons(
        per_run_rows
    )

    linear_path = os.path.join(
        arguments.outdir,
        "linear_fix_validation.csv",
    )

    per_run_path = os.path.join(
        arguments.outdir,
        "per_run_fix_validation.csv",
    )

    summary_path = os.path.join(
        arguments.outdir,
        "filter_fix_summary.csv",
    )

    comparison_path = os.path.join(
        arguments.outdir,
        "paired_fix_comparisons.csv",
    )

    write_csv(
        linear_path,
        linear_rows,
    )

    write_csv(
        per_run_path,
        per_run_rows,
    )

    write_csv(
        summary_path,
        summary_rows,
    )

    write_csv(
        comparison_path,
        comparison_rows,
    )

    print_summary(
        linear_rows=linear_rows,
        summary_rows=summary_rows,
        comparison_rows=comparison_rows,
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "FILES WRITTEN"
    )

    print(
        "============================================================"
    )

    for path in [
        linear_path,
        per_run_path,
        summary_path,
        comparison_path,
    ]:
        print(path)