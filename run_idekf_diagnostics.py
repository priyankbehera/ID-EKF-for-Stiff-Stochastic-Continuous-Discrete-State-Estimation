"""
IDEKF diagnostic suite.

This script investigates why the influence-diagram EKF may perform
poorly when the measurement interval delta is small.

It runs the following diagnostics:

1. Actual number of IDEKF inner iterations used.
2. Linear one-step equivalence test:
       EKF versus SR-EKF versus IDEKF.
3. Covariance round-trip test:
       P -> (B, V) -> P_reconstructed.
4. Filter-consistency diagnostics:
       ARMSE
       NEES
       NIS
       trace(P)
       minimum eigenvalue(P)
       maximum eigenvalue(P)
       condition number(P)
5. Fixed-horizon versus fixed-update-count experiments.
6. Paired multi-seed comparisons.

The influence-diagram update is copied directly from the user's
existing IDEKF implementation. Its update algebra is not modified.

Expected repository files:

    filters.py
    models.py
    run_three_filter_comparison.py
    run_idekf_diagnostics.py

Example:

python run_idekf_diagnostics.py \
  --case vdp \
  --meas nonlin_cubic \
  --sigma 1e-3 \
  --Rmode aniso \
  --Rdiag 1e-4 1e-2 \
  --x0-perturb 1.5 -1.0 \
  --P0-scale 1e-2 \
  --deltas 0.1 0.2 0.7 0.8 \
  --truth-noise \
  --truth-qscale 10 \
  --num-seeds 50 \
  --seed-start 1001 \
  --idekf-iter-max 5 \
  --idekf-iter-tol 1e-10 \
  --fixed-update-count 5
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
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
    "IDEKF",
)


# ============================================================
# General numerical helpers
# ============================================================

def symmetric_matrix(
    matrix: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    return 0.5 * (
        matrix + matrix.T
    )


def safe_inverse_quadratic_form(
    vector: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """
    Calculate v^T C^{-1} v without explicitly forming C^{-1}.
    """

    vector = np.asarray(
        vector,
        dtype=float,
    ).reshape(-1)

    covariance = project_positive_definite(
        covariance,
        min_eigenvalue=1e-12,
        max_condition=1e14,
    )

    try:
        solution = np.linalg.solve(
            covariance,
            vector,
        )

        return float(
            vector @ solution
        )

    except np.linalg.LinAlgError:
        pseudo_inverse = np.linalg.pinv(
            covariance
        )

        return float(
            vector
            @ pseudo_inverse
            @ vector
        )


def matrix_diagnostics(
    covariance: np.ndarray,
) -> Dict[str, float]:
    """
    Return basic covariance diagnostics.
    """

    covariance = symmetric_matrix(
        covariance
    )

    eigenvalues = np.linalg.eigvalsh(
        covariance
    )

    minimum_eigenvalue = float(
        np.min(eigenvalues)
    )

    maximum_eigenvalue = float(
        np.max(eigenvalues)
    )

    absolute_minimum = max(
        abs(minimum_eigenvalue),
        1e-300,
    )

    condition_number = float(
        abs(maximum_eigenvalue)
        / absolute_minimum
    )

    return {
        "trace": float(
            np.trace(covariance)
        ),
        "min_eig": minimum_eigenvalue,
        "max_eig": maximum_eigenvalue,
        "condition": condition_number,
    }


def mean_finite(
    values: Sequence[float],
) -> float:
    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return np.nan

    return float(
        np.mean(array)
    )


def median_finite(
    values: Sequence[float],
) -> float:
    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        return np.nan

    return float(
        np.median(array)
    )


def standard_deviation_finite(
    values: Sequence[float],
) -> float:
    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size < 2:
        return np.nan

    return float(
        np.std(
            array,
            ddof=1,
        )
    )


def write_csv(
    path: str,
    rows: Sequence[Dict[str, object]],
) -> None:
    """
    Write a list of dictionaries to CSV.
    """

    if not rows:
        print(
            f"Warning: no rows available for {path}"
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
# Original influence-diagram update with diagnostics
# ============================================================

class DiagnosticIDEKF(CDIDEKF):
    """
    Influence-diagram EKF with diagnostic logging.

    The state and covariance update algebra is preserved from the
    original IDEKF wrapper.

    Added diagnostics:

    - actual iterations used;
    - P -> influence diagram -> P round-trip error;
    - residual norm by iteration;
    - state-correction norm by iteration.
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
        self.last_roundtrip_relative_error = np.nan
        self.last_roundtrip_absolute_error = np.nan
        self.last_iteration_records: List[
            Dict[str, float]
        ] = []

    def covariance_roundtrip(
        self,
        covariance: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        float,
        float,
    ]:
        """
        Test:

            P -> cov_to_inf -> inf_to_cov -> P_reconstructed
        """

        covariance = np.asarray(
            covariance,
            dtype=float,
        )

        B, V, _ = cov_to_inf(
            covariance,
            covariance.shape[0],
        )

        reconstructed = inf_to_cov(
            np.asarray(
                V
            ).reshape(-1),
            np.asarray(B),
            covariance.shape[0],
        )

        reconstructed = np.asarray(
            reconstructed,
            dtype=float,
        )

        absolute_error = float(
            np.linalg.norm(
                reconstructed
                - covariance,
                ord="fro",
            )
        )

        denominator = max(
            float(
                np.linalg.norm(
                    covariance,
                    ord="fro",
                )
            ),
            1e-15,
        )

        relative_error = (
            absolute_error
            / denominator
        )

        return (
            reconstructed,
            absolute_error,
            relative_error,
        )

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

        z_array = np.asarray(
            z,
            dtype=float,
        ).reshape(-1)

        z_col = z_array.reshape(
            -1,
            1,
        )

        (
            _,
            roundtrip_absolute_error,
            roundtrip_relative_error,
        ) = self.covariance_roundtrip(
            P
        )

        self.last_roundtrip_absolute_error = (
            roundtrip_absolute_error
        )

        self.last_roundtrip_relative_error = (
            roundtrip_relative_error
        )

        self.last_iterations_used = 0
        self.last_iteration_records = []

        for iteration_index in range(
            max(
                1,
                self.iter_max,
            )
        ):
            self.last_iterations_used = (
                iteration_index + 1
            )

            Hk = self.H(x)

            B, V, _ = cov_to_inf(
                P,
                P.shape[0],
            )

            u = x.reshape(
                -1,
                1,
            )

            residual_before = (
                z_array
                - np.asarray(
                    self.h(x),
                    dtype=float,
                ).reshape(-1)
            )

            residual_before_norm = float(
                np.linalg.norm(
                    residual_before
                )
            )

            def h_wrapped(u_vec):
                return np.asarray(
                    self.h(
                        np.asarray(
                            u_vec
                        ).reshape(-1)
                    )
                ).reshape(
                    -1,
                    1,
                )

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
                u_post,
                dtype=float,
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

            residual_after = (
                z_array
                - np.asarray(
                    self.h(x_new),
                    dtype=float,
                ).reshape(-1)
            )

            residual_after_norm = float(
                np.linalg.norm(
                    residual_after
                )
            )

            correction_norm = float(
                np.linalg.norm(
                    x_new - x
                )
            )

            covariance_change_norm = float(
                np.linalg.norm(
                    P_new - P,
                    ord="fro",
                )
            )

            self.last_iteration_records.append(
                {
                    "iteration": float(
                        iteration_index + 1
                    ),
                    "residual_before": (
                        residual_before_norm
                    ),
                    "residual_after": (
                        residual_after_norm
                    ),
                    "correction_norm": (
                        correction_norm
                    ),
                    "covariance_change_norm": (
                        covariance_change_norm
                    ),
                }
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

        predicted_measurement = np.asarray(
            self.h(x_pred),
            dtype=float,
        ).reshape(-1)

        innovation = (
            z_array
            - predicted_measurement
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
# Model setup
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
        default_tf = 4.0

        return (
            f,
            J,
            G,
            Qc,
            x0,
            t0,
            default_tf,
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
        default_tf = 2.0

        return (
            f,
            J,
            G,
            Qc,
            x0,
            t0,
            default_tf,
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


def make_initial_estimate(
    x0: np.ndarray,
    perturbation_values: Sequence[float],
) -> np.ndarray:
    estimate = np.asarray(
        x0,
        dtype=float,
    ).copy()

    perturbation = np.zeros_like(
        estimate
    )

    for index in range(
        min(
            estimate.size,
            len(perturbation_values),
        )
    ):
        perturbation[index] = float(
            perturbation_values[index]
        )

    return estimate + perturbation


def make_time_grid(
    t0: float,
    default_tf: float,
    delta: float,
    experiment_mode: str,
    fixed_update_count: int,
) -> np.ndarray:
    """
    fixed_horizon:
        final time is constant and number of updates changes.

    fixed_updates:
        number of updates is constant and final time changes.
    """

    if experiment_mode == "fixed_horizon":
        return np.arange(
            t0,
            default_tf + 1e-12,
            delta,
        )

    if experiment_mode == "fixed_updates":
        return (
            t0
            + np.arange(
                fixed_update_count + 1,
                dtype=float,
            )
            * delta
        )

    raise ValueError(
        "Unknown experiment mode."
    )


# ============================================================
# Measurement generation
# ============================================================

def generate_realization(
    seed: int,
    delta_index: int,
    experiment_mode_index: int,
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
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Generate a truth path and one measurement sequence.

    All filters receive exactly the same realization.
    """

    seed_sequence = np.random.SeedSequence(
        [
            int(seed),
            int(delta_index),
            int(experiment_mode_index),
            8675309,
        ]
    )

    (
        truth_seed,
        measurement_seed,
    ) = seed_sequence.spawn(2)

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

    R_positive = project_positive_definite(
        R,
        min_eigenvalue=1e-14,
        max_condition=1e14,
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
                    x_true[
                        time_index
                    ]
                ),
                dtype=float,
            ).reshape(-1)
            + measurement_noise[
                time_index
            ]
            for time_index in range(
                len(t_grid)
            )
        ]
    )

    return (
        x_true,
        measurements,
    )


# ============================================================
# Filter trajectory diagnostics
# ============================================================

def run_filter_diagnostics(
    filter_name: str,
    filter_object,
    x_init: np.ndarray,
    P0: np.ndarray,
    t_grid: np.ndarray,
    x_true: np.ndarray,
    measurements: np.ndarray,
    seed: int,
    delta: float,
    experiment_mode: str,
) -> Tuple[
    Dict[str, object],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    """
    Run one filter and return:

    - one trajectory-level summary;
    - one row per measurement update;
    - one row per IDEKF internal iteration.
    """

    xk = np.asarray(
        x_init,
        dtype=float,
    ).copy()

    Pk = np.asarray(
        P0,
        dtype=float,
    ).copy()

    state_estimates = [
        xk.copy()
    ]

    step_rows: List[
        Dict[str, object]
    ] = []

    iteration_rows: List[
        Dict[str, object]
    ] = []

    failed = False
    failure_message = ""

    for time_index in range(
        1,
        len(t_grid),
    ):
        t_previous = float(
            t_grid[
                time_index - 1
            ]
        )

        t_current = float(
            t_grid[
                time_index
            ]
        )

        try:
            x_pred, P_pred = filter_object.predict(
                t_previous,
                t_current,
                xk,
                Pk,
            )

            x_pred = np.asarray(
                x_pred,
                dtype=float,
            )

            P_pred = np.asarray(
                P_pred,
                dtype=float,
            )

            if not np.all(
                np.isfinite(x_pred)
            ):
                raise FloatingPointError(
                    "Predicted state contains NaN or Inf."
                )

            if not np.all(
                np.isfinite(P_pred)
            ):
                raise FloatingPointError(
                    "Predicted covariance contains NaN or Inf."
                )

            predicted_measurement = np.asarray(
                filter_object.h(
                    x_pred
                ),
                dtype=float,
            ).reshape(-1)

            innovation_before = (
                measurements[
                    time_index
                ]
                - predicted_measurement
            )

            H_pred = np.asarray(
                filter_object.H(
                    x_pred
                ),
                dtype=float,
            )

            innovation_covariance_before = (
                H_pred
                @ P_pred
                @ H_pred.T
                + filter_object.R
            )

            nis_before = (
                safe_inverse_quadratic_form(
                    innovation_before,
                    innovation_covariance_before,
                )
            )

            x_post, P_post, _, _ = (
                filter_object.update(
                    x_pred,
                    P_pred,
                    measurements[
                        time_index
                    ],
                )
            )

            x_post = np.asarray(
                x_post,
                dtype=float,
            )

            P_post = np.asarray(
                P_post,
                dtype=float,
            )

            if not np.all(
                np.isfinite(x_post)
            ):
                raise FloatingPointError(
                    "Posterior state contains NaN or Inf."
                )

            if not np.all(
                np.isfinite(P_post)
            ):
                raise FloatingPointError(
                    "Posterior covariance contains NaN or Inf."
                )

            state_error = (
                x_true[
                    time_index
                ]
                - x_post
            )

            nees = safe_inverse_quadratic_form(
                state_error,
                P_post,
            )

            posterior_measurement_residual = (
                measurements[
                    time_index
                ]
                - np.asarray(
                    filter_object.h(
                        x_post
                    ),
                    dtype=float,
                ).reshape(-1)
            )

            diagnostics = matrix_diagnostics(
                P_post
            )

            iterations_used = np.nan
            roundtrip_relative_error = np.nan
            roundtrip_absolute_error = np.nan

            if isinstance(
                filter_object,
                DiagnosticIDEKF,
            ):
                iterations_used = (
                    filter_object.last_iterations_used
                )

                roundtrip_relative_error = (
                    filter_object
                    .last_roundtrip_relative_error
                )

                roundtrip_absolute_error = (
                    filter_object
                    .last_roundtrip_absolute_error
                )

                for iteration_record in (
                    filter_object
                    .last_iteration_records
                ):
                    iteration_rows.append(
                        {
                            "experiment_mode": (
                                experiment_mode
                            ),
                            "seed": seed,
                            "delta": delta,
                            "filter": filter_name,
                            "time_index": (
                                time_index
                            ),
                            "time": t_current,
                            **iteration_record,
                        }
                    )

            step_rows.append(
                {
                    "experiment_mode": (
                        experiment_mode
                    ),
                    "seed": seed,
                    "delta": delta,
                    "filter": filter_name,
                    "time_index": time_index,
                    "time": t_current,
                    "squared_state_error": float(
                        state_error
                        @ state_error
                    ),
                    "state_error_norm": float(
                        np.linalg.norm(
                            state_error
                        )
                    ),
                    "nees": nees,
                    "nis_before_update": (
                        nis_before
                    ),
                    "posterior_measurement_residual_norm": float(
                        np.linalg.norm(
                            posterior_measurement_residual
                        )
                    ),
                    "covariance_trace": (
                        diagnostics[
                            "trace"
                        ]
                    ),
                    "covariance_min_eig": (
                        diagnostics[
                            "min_eig"
                        ]
                    ),
                    "covariance_max_eig": (
                        diagnostics[
                            "max_eig"
                        ]
                    ),
                    "covariance_condition": (
                        diagnostics[
                            "condition"
                        ]
                    ),
                    "iterations_used": (
                        iterations_used
                    ),
                    "roundtrip_relative_error": (
                        roundtrip_relative_error
                    ),
                    "roundtrip_absolute_error": (
                        roundtrip_absolute_error
                    ),
                }
            )

            xk = x_post
            Pk = P_post

            state_estimates.append(
                xk.copy()
            )

        except (
            FloatingPointError,
            np.linalg.LinAlgError,
            RuntimeError,
            ValueError,
        ) as error:
            failed = True
            failure_message = str(
                error
            )
            break

    if failed:
        trajectory_summary = {
            "experiment_mode": experiment_mode,
            "seed": seed,
            "delta": delta,
            "filter": filter_name,
            "failed": 1,
            "failure_message": failure_message,
            "armse_post_update": np.nan,
            "mean_nees": np.nan,
            "median_nees": np.nan,
            "mean_nis": np.nan,
            "mean_covariance_trace": np.nan,
            "mean_covariance_condition": np.nan,
            "mean_iterations_used": np.nan,
            "max_roundtrip_relative_error": np.nan,
            "mean_roundtrip_relative_error": np.nan,
        }

        return (
            trajectory_summary,
            step_rows,
            iteration_rows,
        )

    estimates = np.vstack(
        state_estimates
    )

    # Exclude the shared pre-update initial point.
    errors = (
        estimates[1:]
        - x_true[1:]
    )

    squared_norms = np.sum(
        errors**2,
        axis=1,
    )

    armse_value = float(
        np.sqrt(
            np.mean(
                squared_norms
            )
        )
    )

    nees_values = [
        row["nees"]
        for row in step_rows
    ]

    nis_values = [
        row[
            "nis_before_update"
        ]
        for row in step_rows
    ]

    trace_values = [
        row[
            "covariance_trace"
        ]
        for row in step_rows
    ]

    condition_values = [
        row[
            "covariance_condition"
        ]
        for row in step_rows
    ]

    iteration_values = [
        row[
            "iterations_used"
        ]
        for row in step_rows
        if np.isfinite(
            row[
                "iterations_used"
            ]
        )
    ]

    roundtrip_values = [
        row[
            "roundtrip_relative_error"
        ]
        for row in step_rows
        if np.isfinite(
            row[
                "roundtrip_relative_error"
            ]
        )
    ]

    trajectory_summary = {
        "experiment_mode": experiment_mode,
        "seed": seed,
        "delta": delta,
        "filter": filter_name,
        "failed": 0,
        "failure_message": "",
        "number_of_updates": (
            len(t_grid) - 1
        ),
        "final_time": float(
            t_grid[-1]
        ),
        "armse_post_update": (
            armse_value
        ),
        "mean_nees": mean_finite(
            nees_values
        ),
        "median_nees": median_finite(
            nees_values
        ),
        "mean_nis": mean_finite(
            nis_values
        ),
        "median_nis": median_finite(
            nis_values
        ),
        "mean_covariance_trace": (
            mean_finite(
                trace_values
            )
        ),
        "median_covariance_trace": (
            median_finite(
                trace_values
            )
        ),
        "mean_covariance_condition": (
            mean_finite(
                condition_values
            )
        ),
        "median_covariance_condition": (
            median_finite(
                condition_values
            )
        ),
        "mean_iterations_used": (
            mean_finite(
                iteration_values
            )
        ),
        "max_iterations_used": (
            max(
                iteration_values
            )
            if iteration_values
            else np.nan
        ),
        "mean_roundtrip_relative_error": (
            mean_finite(
                roundtrip_values
            )
        ),
        "max_roundtrip_relative_error": (
            max(
                roundtrip_values
            )
            if roundtrip_values
            else np.nan
        ),
    }

    return (
        trajectory_summary,
        step_rows,
        iteration_rows,
    )


# ============================================================
# Linear one-step equivalence test
# ============================================================

def run_linear_one_step_test(
    continuous_model: ContinuousModel,
    settings: Dict[str, object],
    state_dimension: int,
    outdir: str,
    idekf_iter_max: int,
    idekf_iter_tol: float,
) -> List[Dict[str, object]]:
    """
    Compare a single linear-Gaussian update with identical prior
    state, covariance, and measurement.
    """

    if state_dimension == 1:
        x_pred = np.array(
            [0.75],
            dtype=float,
        )

        P_pred = np.array(
            [[0.4]],
            dtype=float,
        )

        H_linear = np.array(
            [[1.0]],
            dtype=float,
        )

        R_linear = np.array(
            [[0.05]],
            dtype=float,
        )

        z = np.array(
            [0.9],
            dtype=float,
        )

    else:
        x_pred = np.array(
            [1.0, -0.5],
            dtype=float,
        )

        P_pred = np.array(
            [
                [0.5, 0.1],
                [0.1, 0.3],
            ],
            dtype=float,
        )

        H_linear = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.001],
            ],
            dtype=float,
        )

        R_linear = np.diag(
            [
                0.04,
                0.04,
            ]
        )

        z = np.array(
            [
                0.8,
                0.805,
            ],
            dtype=float,
        )

    def h_linear(
        state: np.ndarray,
    ) -> np.ndarray:
        return (
            H_linear
            @ state
        )

    def H_function(
        state: np.ndarray,
    ) -> np.ndarray:
        return H_linear

    ekf = CDEKF(
        continuous_model,
        h_linear,
        H_function,
        R_linear,
        rtol=float(
            settings[
                "filter_rtol"
            ]
        ),
        atol=float(
            settings[
                "filter_atol"
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
    )

    sr_ekf = CDSREKF(
        continuous_model,
        h_linear,
        H_function,
        R_linear,
        rtol=float(
            settings[
                "filter_rtol"
            ]
        ),
        atol=float(
            settings[
                "filter_atol"
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
    )

    idekf = DiagnosticIDEKF(
        continuous_model,
        h_linear,
        H_function,
        R_linear,
        rtol=float(
            settings[
                "filter_rtol"
            ]
        ),
        atol=float(
            settings[
                "filter_atol"
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
        iter_max=idekf_iter_max,
        iter_tol=idekf_iter_tol,
    )

    outputs = {}

    for name, filter_object in [
        ("EKF", ekf),
        ("SR-EKF", sr_ekf),
        ("IDEKF", idekf),
    ]:
        (
            posterior_state,
            posterior_covariance,
            _,
            _,
        ) = filter_object.update(
            x_pred.copy(),
            P_pred.copy(),
            z.copy(),
        )

        outputs[name] = {
            "state": np.asarray(
                posterior_state,
                dtype=float,
            ),
            "covariance": np.asarray(
                posterior_covariance,
                dtype=float,
            ),
        }

    rows: List[
        Dict[str, object]
    ] = []

    reference_state = outputs[
        "EKF"
    ][
        "state"
    ]

    reference_covariance = outputs[
        "EKF"
    ][
        "covariance"
    ]

    for name in FILTER_NAMES:
        state_difference = float(
            np.linalg.norm(
                outputs[name]["state"]
                - reference_state
            )
        )

        covariance_difference = float(
            np.linalg.norm(
                outputs[name][
                    "covariance"
                ]
                - reference_covariance,
                ord="fro",
            )
        )

        rows.append(
            {
                "filter": name,
                "posterior_state": (
                    np.array2string(
                        outputs[name][
                            "state"
                        ],
                        precision=12,
                    )
                ),
                "posterior_covariance": (
                    np.array2string(
                        outputs[name][
                            "covariance"
                        ],
                        precision=12,
                    )
                ),
                "state_difference_from_EKF": (
                    state_difference
                ),
                "covariance_difference_from_EKF": (
                    covariance_difference
                ),
                "IDEKF_iterations_used": (
                    idekf.last_iterations_used
                    if name == "IDEKF"
                    else np.nan
                ),
                "IDEKF_roundtrip_relative_error": (
                    idekf.last_roundtrip_relative_error
                    if name == "IDEKF"
                    else np.nan
                ),
            }
        )

    output_path = os.path.join(
        outdir,
        "linear_one_step_equivalence.csv",
    )

    write_csv(
        output_path,
        rows,
    )

    return rows


# ============================================================
# Experiment runner
# ============================================================

def run_all_diagnostics(
    args: argparse.Namespace,
):
    os.makedirs(
        args.outdir,
        exist_ok=True,
    )

    settings = integration_settings(
        args.profile
    )

    (
        f,
        J,
        G,
        Qc,
        x0,
        t0,
        default_tf,
    ) = construct_case(
        args.case
    )

    (
        h,
        H,
        R,
        measurement_tag,
    ) = build_measurement(
        case=args.case,
        meas=args.meas,
        sigma=args.sigma,
        Rmode=args.Rmode,
        Rdiag=args.Rdiag,
        nl_kind=args.nl_kind,
        nl_eps=args.nl_eps,
        nl_alpha=args.nl_alpha,
    )

    continuous_model = ContinuousModel(
        f=f,
        J=J,
        G=G,
        Qc=Qc,
    )

    x_init = make_initial_estimate(
        x0,
        args.x0_perturb,
    )

    P0 = (
        np.eye(
            x0.size,
            dtype=float,
        )
        * args.P0_scale
    )

    seeds = list(
        range(
            args.seed_start,
            args.seed_start
            + args.num_seeds,
        )
    )

    linear_test_rows = (
        run_linear_one_step_test(
            continuous_model=continuous_model,
            settings=settings,
            state_dimension=x0.size,
            outdir=args.outdir,
            idekf_iter_max=args.idekf_iter_max,
            idekf_iter_tol=args.idekf_iter_tol,
        )
    )

    trajectory_rows: List[
        Dict[str, object]
    ] = []

    step_rows: List[
        Dict[str, object]
    ] = []

    iteration_rows: List[
        Dict[str, object]
    ] = []

    experiment_modes = [
        "fixed_horizon",
        "fixed_updates",
    ]

    for experiment_mode_index, experiment_mode in enumerate(
        experiment_modes
    ):
        print(
            "\n"
            "============================================================"
        )

        print(
            f"Running mode: {experiment_mode}"
        )

        print(
            "============================================================"
        )

        for delta_index, delta in enumerate(
            args.deltas
        ):
            delta = float(
                delta
            )

            t_grid = make_time_grid(
                t0=t0,
                default_tf=default_tf,
                delta=delta,
                experiment_mode=experiment_mode,
                fixed_update_count=args.fixed_update_count,
            )

            print(
                f"\ndelta={delta:g}, "
                f"updates={len(t_grid) - 1}, "
                f"final time={t_grid[-1]:g}"
            )

            for seed_position, seed in enumerate(
                seeds,
                start=1,
            ):
                (
                    x_true,
                    measurements,
                ) = generate_realization(
                    seed=seed,
                    delta_index=delta_index,
                    experiment_mode_index=(
                        experiment_mode_index
                    ),
                    t_grid=t_grid,
                    f=f,
                    J=J,
                    G=G,
                    Qc=Qc,
                    x0=x0,
                    h=h,
                    R=R,
                    settings=settings,
                    truth_noise=args.truth_noise,
                    truth_qscale=args.truth_qscale,
                )

                ekf = CDEKF(
                    continuous_model,
                    h,
                    H,
                    R,
                    rtol=float(
                        settings[
                            "filter_rtol"
                        ]
                    ),
                    atol=float(
                        settings[
                            "filter_atol"
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
                )

                sr_ekf = CDSREKF(
                    continuous_model,
                    h,
                    H,
                    R,
                    rtol=float(
                        settings[
                            "filter_rtol"
                        ]
                    ),
                    atol=float(
                        settings[
                            "filter_atol"
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
                )

                idekf = DiagnosticIDEKF(
                    continuous_model,
                    h,
                    H,
                    R,
                    rtol=float(
                        settings[
                            "filter_rtol"
                        ]
                    ),
                    atol=float(
                        settings[
                            "filter_atol"
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
                    iter_max=args.idekf_iter_max,
                    iter_tol=args.idekf_iter_tol,
                )

                filter_objects = [
                    ("EKF", ekf),
                    ("SR-EKF", sr_ekf),
                    ("IDEKF", idekf),
                ]

                for (
                    filter_name,
                    filter_object,
                ) in filter_objects:
                    (
                        trajectory_summary,
                        filter_step_rows,
                        filter_iteration_rows,
                    ) = run_filter_diagnostics(
                        filter_name=filter_name,
                        filter_object=filter_object,
                        x_init=x_init,
                        P0=P0,
                        t_grid=t_grid,
                        x_true=x_true,
                        measurements=measurements,
                        seed=seed,
                        delta=delta,
                        experiment_mode=experiment_mode,
                    )

                    trajectory_summary[
                        "measurement_model"
                    ] = measurement_tag

                    trajectory_summary[
                        "state_dimension"
                    ] = x0.size

                    trajectory_summary[
                        "measurement_dimension"
                    ] = R.shape[0]

                    trajectory_rows.append(
                        trajectory_summary
                    )

                    step_rows.extend(
                        filter_step_rows
                    )

                    iteration_rows.extend(
                        filter_iteration_rows
                    )

                if (
                    seed_position
                    % args.progress_every
                    == 0
                    or seed_position
                    == len(seeds)
                ):
                    print(
                        f"  completed "
                        f"{seed_position}/"
                        f"{len(seeds)} seeds"
                    )

    trajectory_path = os.path.join(
        args.outdir,
        "trajectory_diagnostics.csv",
    )

    step_path = os.path.join(
        args.outdir,
        "step_diagnostics.csv",
    )

    iteration_path = os.path.join(
        args.outdir,
        "idekf_iteration_diagnostics.csv",
    )

    write_csv(
        trajectory_path,
        trajectory_rows,
    )

    write_csv(
        step_path,
        step_rows,
    )

    write_csv(
        iteration_path,
        iteration_rows,
    )

    return (
        linear_test_rows,
        trajectory_rows,
        step_rows,
        iteration_rows,
    )


# ============================================================
# Aggregate summaries
# ============================================================

def aggregate_trajectory_results(
    trajectory_rows: Sequence[
        Dict[str, object]
    ],
) -> List[Dict[str, object]]:
    grouped: Dict[
        Tuple[str, float, str],
        List[Dict[str, object]],
    ] = defaultdict(list)

    for row in trajectory_rows:
        key = (
            str(
                row[
                    "experiment_mode"
                ]
            ),
            float(
                row[
                    "delta"
                ]
            ),
            str(
                row[
                    "filter"
                ]
            ),
        )

        grouped[key].append(
            row
        )

    summaries: List[
        Dict[str, object]
    ] = []

    for (
        experiment_mode,
        delta,
        filter_name,
    ), rows in sorted(
        grouped.items()
    ):
        successful_rows = [
            row
            for row in rows
            if int(
                row[
                    "failed"
                ]
            )
            == 0
        ]

        summaries.append(
            {
                "experiment_mode": (
                    experiment_mode
                ),
                "delta": delta,
                "filter": filter_name,
                "number_of_runs": len(
                    rows
                ),
                "failures": (
                    len(rows)
                    - len(
                        successful_rows
                    )
                ),
                "mean_ARMSE": mean_finite(
                    [
                        row[
                            "armse_post_update"
                        ]
                        for row
                        in successful_rows
                    ]
                ),
                "median_ARMSE": median_finite(
                    [
                        row[
                            "armse_post_update"
                        ]
                        for row
                        in successful_rows
                    ]
                ),
                "std_ARMSE": standard_deviation_finite(
                    [
                        row[
                            "armse_post_update"
                        ]
                        for row
                        in successful_rows
                    ]
                ),
                "mean_NEES": mean_finite(
                    [
                        row[
                            "mean_nees"
                        ]
                        for row
                        in successful_rows
                    ]
                ),
                "median_NEES": median_finite(
                    [
                        row[
                            "median_nees"
                        ]
                        for row
                        in successful_rows
                    ]
                ),
                "mean_NIS": mean_finite(
                    [
                        row[
                            "mean_nis"
                        ]
                        for row
                        in successful_rows
                    ]
                ),
                "mean_covariance_trace": mean_finite(
                    [
                        row[
                            "mean_covariance_trace"
                        ]
                        for row
                        in successful_rows
                    ]
                ),
                "mean_covariance_condition": mean_finite(
                    [
                        row[
                            "mean_covariance_condition"
                        ]
                        for row
                        in successful_rows
                    ]
                ),
                "mean_iterations_used": mean_finite(
                    [
                        row[
                            "mean_iterations_used"
                        ]
                        for row
                        in successful_rows
                    ]
                ),
                "max_iterations_used": max(
                    [
                        row[
                            "max_iterations_used"
                        ]
                        for row
                        in successful_rows
                        if np.isfinite(
                            row[
                                "max_iterations_used"
                            ]
                        )
                    ],
                    default=np.nan,
                ),
                "mean_roundtrip_relative_error": mean_finite(
                    [
                        row[
                            "mean_roundtrip_relative_error"
                        ]
                        for row
                        in successful_rows
                    ]
                ),
                "max_roundtrip_relative_error": max(
                    [
                        row[
                            "max_roundtrip_relative_error"
                        ]
                        for row
                        in successful_rows
                        if np.isfinite(
                            row[
                                "max_roundtrip_relative_error"
                            ]
                        )
                    ],
                    default=np.nan,
                ),
            }
        )

    return summaries


def create_comparison_rows(
    trajectory_rows: Sequence[
        Dict[str, object]
    ],
) -> List[Dict[str, object]]:
    """
    Create paired IDEKF-EKF and IDEKF-SR-EKF comparisons.
    """

    by_realization: Dict[
        Tuple[str, float, int],
        Dict[str, Dict[str, object]],
    ] = defaultdict(dict)

    for row in trajectory_rows:
        key = (
            str(
                row[
                    "experiment_mode"
                ]
            ),
            float(
                row[
                    "delta"
                ]
            ),
            int(
                row[
                    "seed"
                ]
            ),
        )

        by_realization[key][
            str(
                row[
                    "filter"
                ]
            )
        ] = row

    comparison_rows: List[
        Dict[str, object]
    ] = []

    for (
        experiment_mode,
        delta,
        seed,
    ), filter_rows in by_realization.items():
        if not all(
            name in filter_rows
            for name in FILTER_NAMES
        ):
            continue

        if any(
            int(
                filter_rows[name][
                    "failed"
                ]
            )
            != 0
            for name in FILTER_NAMES
        ):
            continue

        ekf_error = float(
            filter_rows[
                "EKF"
            ][
                "armse_post_update"
            ]
        )

        sr_error = float(
            filter_rows[
                "SR-EKF"
            ][
                "armse_post_update"
            ]
        )

        idekf_error = float(
            filter_rows[
                "IDEKF"
            ][
                "armse_post_update"
            ]
        )

        comparison_rows.append(
            {
                "experiment_mode": (
                    experiment_mode
                ),
                "delta": delta,
                "seed": seed,
                "EKF_ARMSE": ekf_error,
                "SR_EKF_ARMSE": sr_error,
                "IDEKF_ARMSE": idekf_error,
                "EKF_minus_IDEKF": (
                    ekf_error
                    - idekf_error
                ),
                "SR_EKF_minus_IDEKF": (
                    sr_error
                    - idekf_error
                ),
                "IDEKF_beats_EKF": int(
                    idekf_error
                    < ekf_error
                ),
                "IDEKF_beats_SR_EKF": int(
                    idekf_error
                    < sr_error
                ),
            }
        )

    return comparison_rows


def aggregate_comparisons(
    comparison_rows: Sequence[
        Dict[str, object]
    ],
) -> List[Dict[str, object]]:
    grouped: Dict[
        Tuple[str, float],
        List[Dict[str, object]],
    ] = defaultdict(list)

    for row in comparison_rows:
        key = (
            str(
                row[
                    "experiment_mode"
                ]
            ),
            float(
                row[
                    "delta"
                ]
            ),
        )

        grouped[key].append(
            row
        )

    summaries: List[
        Dict[str, object]
    ] = []

    for (
        experiment_mode,
        delta,
    ), rows in sorted(
        grouped.items()
    ):
        ekf_differences = np.asarray(
            [
                row[
                    "EKF_minus_IDEKF"
                ]
                for row in rows
            ],
            dtype=float,
        )

        sr_differences = np.asarray(
            [
                row[
                    "SR_EKF_minus_IDEKF"
                ]
                for row in rows
            ],
            dtype=float,
        )

        summaries.append(
            {
                "experiment_mode": (
                    experiment_mode
                ),
                "delta": delta,
                "paired_runs": len(
                    rows
                ),
                "mean_EKF_minus_IDEKF": float(
                    np.mean(
                        ekf_differences
                    )
                ),
                "median_EKF_minus_IDEKF": float(
                    np.median(
                        ekf_differences
                    )
                ),
                "IDEKF_win_rate_vs_EKF": float(
                    np.mean(
                        ekf_differences > 0
                    )
                ),
                "mean_SR_EKF_minus_IDEKF": float(
                    np.mean(
                        sr_differences
                    )
                ),
                "median_SR_EKF_minus_IDEKF": float(
                    np.median(
                        sr_differences
                    )
                ),
                "IDEKF_win_rate_vs_SR_EKF": float(
                    np.mean(
                        sr_differences > 0
                    )
                ),
            }
        )

    return summaries


# ============================================================
# Plotting
# ============================================================

def create_summary_plots(
    aggregate_rows: Sequence[
        Dict[str, object]
    ],
    outdir: str,
) -> None:
    for experiment_mode in [
        "fixed_horizon",
        "fixed_updates",
    ]:
        mode_rows = [
            row
            for row in aggregate_rows
            if row[
                "experiment_mode"
            ]
            == experiment_mode
        ]

        if not mode_rows:
            continue

        deltas = sorted(
            {
                float(
                    row[
                        "delta"
                    ]
                )
                for row in mode_rows
            }
        )

        plt.figure(
            figsize=(9, 5.5)
        )

        for filter_name in FILTER_NAMES:
            values = []

            for delta in deltas:
                matching = [
                    row
                    for row in mode_rows
                    if (
                        float(
                            row[
                                "delta"
                            ]
                        )
                        == delta
                        and row[
                            "filter"
                        ]
                        == filter_name
                    )
                ]

                values.append(
                    matching[0][
                        "mean_ARMSE"
                    ]
                    if matching
                    else np.nan
                )

            plt.plot(
                deltas,
                values,
                marker="o",
                label=filter_name,
            )

        plt.xlabel(
            "Measurement interval δ"
        )

        plt.ylabel(
            "Mean post-update ARMSE"
        )

        plt.title(
            f"Filter performance: "
            f"{experiment_mode}"
        )

        plt.grid(
            True,
            linestyle="--",
            alpha=0.4,
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                outdir,
                f"{experiment_mode}_armse.png",
            ),
            dpi=180,
        )

        plt.close()

        plt.figure(
            figsize=(9, 5.5)
        )

        for filter_name in FILTER_NAMES:
            values = []

            for delta in deltas:
                matching = [
                    row
                    for row in mode_rows
                    if (
                        float(
                            row[
                                "delta"
                            ]
                        )
                        == delta
                        and row[
                            "filter"
                        ]
                        == filter_name
                    )
                ]

                values.append(
                    matching[0][
                        "mean_NEES"
                    ]
                    if matching
                    else np.nan
                )

            plt.plot(
                deltas,
                values,
                marker="o",
                label=filter_name,
            )

        state_dimension = 2

        plt.axhline(
            state_dimension,
            linestyle="--",
            label=(
                "Expected NEES "
                f"≈ {state_dimension}"
            ),
        )

        plt.xlabel(
            "Measurement interval δ"
        )

        plt.ylabel(
            "Mean NEES"
        )

        plt.title(
            f"Filter consistency: "
            f"{experiment_mode}"
        )

        plt.grid(
            True,
            linestyle="--",
            alpha=0.4,
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                outdir,
                f"{experiment_mode}_nees.png",
            ),
            dpi=180,
        )

        plt.close()


# ============================================================
# Printed interpretation summary
# ============================================================

def print_diagnostic_summary(
    linear_test_rows: Sequence[
        Dict[str, object]
    ],
    aggregate_rows: Sequence[
        Dict[str, object]
    ],
    comparison_summary_rows: Sequence[
        Dict[str, object]
    ],
) -> None:
    print(
        "\n"
        "============================================================"
    )

    print(
        "LINEAR ONE-STEP EQUIVALENCE TEST"
    )

    print(
        "============================================================"
    )

    for row in linear_test_rows:
        print(
            f"{row['filter']}: "
            f"state difference from EKF="
            f"{float(row['state_difference_from_EKF']):.6g}, "
            f"covariance difference from EKF="
            f"{float(row['covariance_difference_from_EKF']):.6g}"
        )

    print(
        "\n"
        "============================================================"
    )

    print(
        "IDEKF ITERATION AND ROUND-TRIP SUMMARY"
    )

    print(
        "============================================================"
    )

    idekf_rows = [
        row
        for row in aggregate_rows
        if row[
            "filter"
        ]
        == "IDEKF"
    ]

    for row in idekf_rows:
        print(
            f"{row['experiment_mode']}, "
            f"delta={float(row['delta']):g}: "
            f"mean iterations="
            f"{float(row['mean_iterations_used']):.6g}, "
            f"max iterations="
            f"{float(row['max_iterations_used']):.6g}, "
            f"mean round-trip rel. error="
            f"{float(row['mean_roundtrip_relative_error']):.6g}, "
            f"max round-trip rel. error="
            f"{float(row['max_roundtrip_relative_error']):.6g}, "
            f"mean NEES="
            f"{float(row['mean_NEES']):.6g}, "
            f"mean covariance trace="
            f"{float(row['mean_covariance_trace']):.6g}"
        )

    print(
        "\n"
        "============================================================"
    )

    print(
        "PAIRED PERFORMANCE SUMMARY"
    )

    print(
        "Positive differences favor IDEKF."
    )

    print(
        "============================================================"
    )

    for row in comparison_summary_rows:
        print(
            f"{row['experiment_mode']}, "
            f"delta={float(row['delta']):g}: "
            f"EKF-ID mean="
            f"{float(row['mean_EKF_minus_IDEKF']):.6g}, "
            f"ID win rate vs EKF="
            f"{float(row['IDEKF_win_rate_vs_EKF']):.3f}, "
            f"SR-ID mean="
            f"{float(row['mean_SR_EKF_minus_IDEKF']):.6g}, "
            f"ID win rate vs SR="
            f"{float(row['IDEKF_win_rate_vs_SR_EKF']):.3f}"
        )


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run IDEKF numerical, consistency, and "
            "sampling-frequency diagnostics."
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
        "--truth-noise",
        action="store_true",
    )

    parser.add_argument(
        "--truth-qscale",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=1001,
    )

    parser.add_argument(
        "--num-seeds",
        type=int,
        default=50,
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

    parser.add_argument(
        "--fixed-update-count",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="results_diagnostics",
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

    if args.fixed_update_count < 1:
        raise ValueError(
            "fixed-update-count must be at least 1."
        )

    if args.idekf_iter_max < 1:
        raise ValueError(
            "idekf-iter-max must be at least 1."
        )

    if any(
        delta <= 0.0
        for delta in args.deltas
    ):
        raise ValueError(
            "Every delta must be positive."
        )


if __name__ == "__main__":
    arguments = parse_arguments()

    validate_arguments(
        arguments
    )

    (
        linear_rows,
        trajectory_rows,
        step_rows,
        iteration_rows,
    ) = run_all_diagnostics(
        arguments
    )

    aggregate_rows = (
        aggregate_trajectory_results(
            trajectory_rows
        )
    )

    comparison_rows = (
        create_comparison_rows(
            trajectory_rows
        )
    )

    comparison_summary_rows = (
        aggregate_comparisons(
            comparison_rows
        )
    )

    aggregate_path = os.path.join(
        arguments.outdir,
        "aggregate_diagnostics.csv",
    )

    comparison_path = os.path.join(
        arguments.outdir,
        "paired_filter_comparisons.csv",
    )

    comparison_summary_path = os.path.join(
        arguments.outdir,
        "paired_filter_comparison_summary.csv",
    )

    write_csv(
        aggregate_path,
        aggregate_rows,
    )

    write_csv(
        comparison_path,
        comparison_rows,
    )

    write_csv(
        comparison_summary_path,
        comparison_summary_rows,
    )

    create_summary_plots(
        aggregate_rows=aggregate_rows,
        outdir=arguments.outdir,
    )

    print_diagnostic_summary(
        linear_test_rows=linear_rows,
        aggregate_rows=aggregate_rows,
        comparison_summary_rows=(
            comparison_summary_rows
        ),
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

    output_files = [
        "linear_one_step_equivalence.csv",
        "trajectory_diagnostics.csv",
        "step_diagnostics.csv",
        "idekf_iteration_diagnostics.csv",
        "aggregate_diagnostics.csv",
        "paired_filter_comparisons.csv",
        "paired_filter_comparison_summary.csv",
        "fixed_horizon_armse.png",
        "fixed_horizon_nees.png",
        "fixed_updates_armse.png",
        "fixed_updates_nees.png",
    ]

    for filename in output_files:
        print(
            os.path.join(
                arguments.outdir,
                filename,
            )
        )