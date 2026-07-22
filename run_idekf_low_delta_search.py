"""
Diagnose and improve low-delta performance of the influence-diagram EKF.

This script is designed to be placed beside:

    run_three_filter_comparison.py
    filters.py
    models.py

It imports the model, truth simulator, measurement builder, and SR-EKF from
run_three_filter_comparison.py.  It DOES NOT use the old IDEKFIter wrapper,
because that wrapper repeatedly assimilates the same observation.

Instead, every IDEKF candidate performs exactly one influence-diagram
measurement update per observation.

The script runs two linked experiments:

1. Baseline mechanism diagnostics
   - SR-EKF versus corrected single-assimilation IDEKF
   - predicted and posterior state error
   - predicted and posterior NEES
   - NIS
   - Jacobian and whitened-Jacobian condition numbers
   - error and covariance in strongly and weakly observed directions
   - covariance contraction during the measurement update

2. Principled IDEKF candidate search
   - prior covariance scaling
   - measurement covariance tempering
   - weak-direction prior inflation
   - posterior covariance inflation

The purpose is not to select a winner from the same Monte Carlo samples.
The script separates tuning seeds and validation seeds.  A candidate is only
reported as promising if it improves IDEKF on the validation seeds and is
compared against the unchanged SR-EKF on identical truth and measurements.

Positive SR-EKF minus IDEKF ARMSE values favor IDEKF.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats

from filters import (
    ContinuousModel,
    CDIDEKF,
    ensure_psd,
    cov_to_inf,
    inf_to_cov,
    mupdate,
)

from models import (
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


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class IDEKFConfiguration:
    name: str
    prior_scale: float = 1.0
    r_scale: float = 1.0
    weak_inflation: float = 0.0
    posterior_scale: float = 1.0


# ============================================================
# Numerical helpers
# ============================================================


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def positive_definite(
    matrix: np.ndarray,
    minimum_eigenvalue: float = 1e-12,
    maximum_condition: float = 1e14,
) -> np.ndarray:
    return project_positive_definite(
        symmetrize(matrix),
        min_eigenvalue=minimum_eigenvalue,
        max_condition=maximum_condition,
    )


def solve_quadratic_form(vector: np.ndarray, covariance: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    covariance = positive_definite(covariance)

    try:
        solution = np.linalg.solve(covariance, vector)
    except np.linalg.LinAlgError:
        solution = np.linalg.pinv(covariance) @ vector

    return float(vector @ solution)


def finite_array(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    return values[np.isfinite(values)]


def safe_mean(values: Iterable[float]) -> float:
    values = finite_array(values)
    return float(np.mean(values)) if values.size else np.nan


def safe_median(values: Iterable[float]) -> float:
    values = finite_array(values)
    return float(np.median(values)) if values.size else np.nan


def safe_percentile(values: Iterable[float], percentile: float) -> float:
    values = finite_array(values)
    return float(np.percentile(values, percentile)) if values.size else np.nan


def safe_std(values: Iterable[float]) -> float:
    values = finite_array(values)
    if values.size < 2:
        return np.nan
    return float(np.std(values, ddof=1))


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        print(f"Warning: no rows available for {path}")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def paired_bootstrap_ci(
    differences: Sequence[float],
    rng: np.random.Generator,
    repetitions: int = 5000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    differences = finite_array(differences)
    if differences.size == 0:
        return np.nan, np.nan

    indices = rng.integers(
        0,
        differences.size,
        size=(repetitions, differences.size),
    )
    bootstrap_means = np.mean(differences[indices], axis=1)
    alpha = 1.0 - confidence
    return (
        float(np.quantile(bootstrap_means, alpha / 2.0)),
        float(np.quantile(bootstrap_means, 1.0 - alpha / 2.0)),
    )


def sign_flip_pvalue(
    differences: Sequence[float],
    rng: np.random.Generator,
    repetitions: int = 20000,
) -> float:
    differences = finite_array(differences)
    if differences.size == 0:
        return np.nan

    observed = float(np.mean(differences))
    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(repetitions, differences.size),
    )
    null_means = np.mean(signs * differences[None, :], axis=1)

    # One-sided alternative: mean difference > 0 favors IDEKF.
    return float((1 + np.sum(null_means >= observed)) / (repetitions + 1))


# ============================================================
# Observation geometry
# ============================================================


def whitened_observation_directions(
    H: np.ndarray,
    R: np.ndarray,
) -> Dict[str, object]:
    """
    Return strong/weak state directions from the SVD of R^{-1/2} H.

    The right singular vector associated with the largest singular value is
    the strongly observed state direction.  The right singular vector
    associated with the smallest singular value is the weakly observed state
    direction.
    """

    H = np.asarray(H, dtype=float)
    R = positive_definite(R, 1e-14, 1e14)
    L = stable_cholesky(R, min_eigenvalue=1e-14, max_condition=1e14)
    whitened_H = np.linalg.solve(L, H)

    _, singular_values, right_transpose = np.linalg.svd(
        whitened_H,
        full_matrices=False,
    )

    strong_direction = right_transpose[0].copy()
    weak_direction = right_transpose[-1].copy()

    strong_direction /= max(np.linalg.norm(strong_direction), 1e-15)
    weak_direction /= max(np.linalg.norm(weak_direction), 1e-15)

    smallest = max(float(singular_values[-1]), 1e-300)
    largest = float(singular_values[0])

    return {
        "strong_direction": strong_direction,
        "weak_direction": weak_direction,
        "singular_max": largest,
        "singular_min": smallest,
        "whitened_condition": largest / smallest,
    }


def directional_variance(P: np.ndarray, direction: np.ndarray) -> float:
    direction = np.asarray(direction, dtype=float).reshape(-1)
    return float(direction @ P @ direction)


# ============================================================
# Corrected IDEKF candidate
# ============================================================


class ConfiguredSingleAssimilationIDEKF(CDIDEKF):
    """
    One influence-diagram measurement update per observation.

    The candidate modifications are covariance calibration operations:

    prior_scale:
        Scalar multiplier applied to the predicted covariance before update.

    r_scale:
        Scalar multiplier applied to R inside the IDEKF update.

    weak_inflation:
        Adds covariance only in the weakest state direction of the whitened
        measurement Jacobian:

            P <- P + weak_inflation * (v_w^T P v_w) v_w v_w^T

    posterior_scale:
        Scalar multiplier applied to the returned posterior covariance.
        This does not alter the current posterior mean, but changes future
        gains through the next prediction/update cycles.
    """

    def __init__(
        self,
        *args,
        configuration: IDEKFConfiguration,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.configuration = configuration
        self.last_diagnostics: Dict[str, float] = {}

    def update(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        z: np.ndarray,
    ):
        configuration = self.configuration

        x_prior = np.asarray(x_pred, dtype=float).reshape(-1).copy()
        P_original = positive_definite(P_pred)
        z_vector = np.asarray(z, dtype=float).reshape(-1)

        H_prior = np.asarray(self.H(x_prior), dtype=float)
        geometry = whitened_observation_directions(H_prior, self.R)
        weak_direction = np.asarray(geometry["weak_direction"], dtype=float)

        P_working = configuration.prior_scale * P_original

        weak_variance_before = directional_variance(
            P_working,
            weak_direction,
        )

        if configuration.weak_inflation > 0.0:
            P_working = (
                P_working
                + configuration.weak_inflation
                * max(weak_variance_before, 1e-14)
                * np.outer(weak_direction, weak_direction)
            )

        P_working = positive_definite(P_working)
        R_working = positive_definite(
            configuration.r_scale * self.R,
            minimum_eigenvalue=1e-14,
            maximum_condition=1e14,
        )

        B_prior, V_prior, _ = cov_to_inf(
            P_working,
            P_working.shape[0],
        )

        u_prior = x_prior.reshape(-1, 1)
        z_column = z_vector.reshape(-1, 1)

        def h_wrapped(u_vector):
            state = np.asarray(u_vector, dtype=float).reshape(-1)
            return np.asarray(self.h(state), dtype=float).reshape(-1, 1)

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
            R_working,
            H_prior,
            h_wrapped,
        )

        x_posterior = np.asarray(u_posterior, dtype=float).reshape(-1)
        P_posterior = inf_to_cov(
            np.asarray(V_posterior).reshape(-1),
            np.asarray(B_posterior),
            x_posterior.size,
        )
        P_posterior = positive_definite(
            configuration.posterior_scale * P_posterior
        )

        innovation = z_vector - np.asarray(
            self.h(x_prior),
            dtype=float,
        ).reshape(-1)

        innovation_covariance = positive_definite(
            H_prior @ P_working @ H_prior.T + R_working,
            minimum_eigenvalue=1e-14,
            maximum_condition=1e14,
        )

        self.last_diagnostics = {
            "prior_scale": configuration.prior_scale,
            "r_scale": configuration.r_scale,
            "weak_inflation": configuration.weak_inflation,
            "posterior_scale": configuration.posterior_scale,
            "working_prior_trace": float(np.trace(P_working)),
            "posterior_trace": float(np.trace(P_posterior)),
            "weak_prior_variance": directional_variance(
                P_working,
                weak_direction,
            ),
            "weak_posterior_variance": directional_variance(
                P_posterior,
                weak_direction,
            ),
        }

        return (
            x_posterior,
            P_posterior,
            innovation,
            innovation_covariance,
        )


# ============================================================
# Model and simulation setup
# ============================================================


def scaled_process_covariance(Qc, scale: float):
    if callable(Qc):
        def scaled(time):
            return scale * np.asarray(Qc(time), dtype=float)
        return scaled

    constant = scale * np.asarray(Qc, dtype=float)

    def scaled(time):
        del time
        return constant

    return scaled


def make_time_grid(t0: float, tf: float, delta: float) -> np.ndarray:
    grid = np.arange(t0, tf + 1e-12, delta)
    if grid[-1] < tf - 1e-10:
        grid = np.append(grid, tf)
    return grid


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
    truth_qscale: float,
    truth_noise: bool,
    truth_rtol: float,
    truth_atol: float,
    max_step: float,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    seed_sequence = np.random.SeedSequence(
        [int(seed), int(delta_index), 20260722, 401]
    )
    truth_seed, measurement_seed = seed_sequence.spawn(2)
    truth_rng = np.random.default_rng(truth_seed)
    measurement_rng = np.random.default_rng(measurement_seed)

    truth = integrate_truth_path(
        f=f,
        J=J,
        G=G,
        Qc=Qc,
        t0=float(t_grid[0]),
        tf=float(t_grid[-1]),
        x0=x0,
        t_grid=t_grid,
        rtol=truth_rtol,
        atol=truth_atol,
        max_step=max_step,
        method=method,
        truth_noise=truth_noise,
        qscale=truth_qscale,
        rng=truth_rng,
    )

    R_positive = positive_definite(R, 1e-14, 1e14)
    root = stable_cholesky(
        R_positive,
        min_eigenvalue=1e-14,
        max_condition=1e14,
    )
    standard_noise = measurement_rng.normal(
        size=(len(t_grid), R.shape[0])
    )
    measurement_noise = standard_noise @ root.T

    measurements = np.asarray(
        [
            np.asarray(h(truth[index]), dtype=float).reshape(-1)
            + measurement_noise[index]
            for index in range(len(t_grid))
        ]
    )

    return truth, measurements


# ============================================================
# Filter execution and step diagnostics
# ============================================================


def run_one_filter(
    filter_name: str,
    filter_object,
    x_initial: np.ndarray,
    P_initial: np.ndarray,
    truth: np.ndarray,
    measurements: np.ndarray,
    t_grid: np.ndarray,
    delta: float,
    seed: int,
    split: str,
    collect_steps: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    x = np.asarray(x_initial, dtype=float).copy()
    P = np.asarray(P_initial, dtype=float).copy()

    posterior_squared_errors: List[float] = []
    predicted_squared_errors: List[float] = []
    posterior_nees_values: List[float] = []
    predicted_nees_values: List[float] = []
    nis_values: List[float] = []
    strong_error_squared_values: List[float] = []
    weak_error_squared_values: List[float] = []
    strong_variance_values: List[float] = []
    weak_variance_values: List[float] = []
    contraction_values: List[float] = []
    jacobian_conditions: List[float] = []
    information_conditions: List[float] = []
    correction_norms: List[float] = []

    step_rows: List[Dict[str, object]] = []

    try:
        for index in range(1, len(t_grid)):
            previous_time = float(t_grid[index - 1])
            current_time = float(t_grid[index])

            x_pred, P_pred = filter_object.predict(
                previous_time,
                current_time,
                x,
                P,
            )
            x_pred = np.asarray(x_pred, dtype=float).reshape(-1)
            P_pred = positive_definite(P_pred)

            H_pred = np.asarray(filter_object.H(x_pred), dtype=float)
            R_filter = np.asarray(filter_object.R, dtype=float)
            geometry = whitened_observation_directions(H_pred, R_filter)
            strong_direction = np.asarray(
                geometry["strong_direction"],
                dtype=float,
            )
            weak_direction = np.asarray(
                geometry["weak_direction"],
                dtype=float,
            )

            predicted_error = truth[index] - x_pred
            predicted_squared_error = float(predicted_error @ predicted_error)
            predicted_nees = solve_quadratic_form(predicted_error, P_pred)

            innovation = measurements[index] - np.asarray(
                filter_object.h(x_pred),
                dtype=float,
            ).reshape(-1)
            innovation_covariance = positive_definite(
                H_pred @ P_pred @ H_pred.T + R_filter,
                1e-14,
                1e14,
            )
            nis = solve_quadratic_form(innovation, innovation_covariance)

            information_matrix = positive_definite(
                H_pred.T
                @ np.linalg.solve(positive_definite(R_filter, 1e-14, 1e14), H_pred),
                1e-18,
                1e18,
            )
            information_eigenvalues = np.linalg.eigvalsh(information_matrix)
            information_condition = float(
                np.max(information_eigenvalues)
                / max(np.min(information_eigenvalues), 1e-300)
            )

            x_post, P_post, _, _ = filter_object.update(
                x_pred,
                P_pred,
                measurements[index],
            )
            x_post = np.asarray(x_post, dtype=float).reshape(-1)
            P_post = positive_definite(P_post)

            posterior_error = truth[index] - x_post
            posterior_squared_error = float(posterior_error @ posterior_error)
            posterior_nees = solve_quadratic_form(posterior_error, P_post)

            strong_error = float(strong_direction @ posterior_error)
            weak_error = float(weak_direction @ posterior_error)
            strong_variance = directional_variance(P_post, strong_direction)
            weak_variance = directional_variance(P_post, weak_direction)
            trace_contraction = float(
                np.trace(P_post) / max(np.trace(P_pred), 1e-300)
            )
            correction_norm = float(np.linalg.norm(x_post - x_pred))

            posterior_squared_errors.append(posterior_squared_error)
            predicted_squared_errors.append(predicted_squared_error)
            posterior_nees_values.append(posterior_nees)
            predicted_nees_values.append(predicted_nees)
            nis_values.append(nis)
            strong_error_squared_values.append(strong_error**2)
            weak_error_squared_values.append(weak_error**2)
            strong_variance_values.append(strong_variance)
            weak_variance_values.append(weak_variance)
            contraction_values.append(trace_contraction)
            jacobian_conditions.append(float(geometry["whitened_condition"]))
            information_conditions.append(information_condition)
            correction_norms.append(correction_norm)

            if collect_steps:
                step_rows.append(
                    {
                        "split": split,
                        "seed": seed,
                        "delta": delta,
                        "filter": filter_name,
                        "time_index": index,
                        "time": current_time,
                        "predicted_state_error_norm": math.sqrt(predicted_squared_error),
                        "posterior_state_error_norm": math.sqrt(posterior_squared_error),
                        "predicted_nees": predicted_nees,
                        "posterior_nees": posterior_nees,
                        "nis": nis,
                        "strong_error": strong_error,
                        "weak_error": weak_error,
                        "strong_variance": strong_variance,
                        "weak_variance": weak_variance,
                        "weak_normalized_error": weak_error**2 / max(weak_variance, 1e-300),
                        "strong_normalized_error": strong_error**2 / max(strong_variance, 1e-300),
                        "whitened_H_condition": float(geometry["whitened_condition"]),
                        "whitened_H_singular_max": float(geometry["singular_max"]),
                        "whitened_H_singular_min": float(geometry["singular_min"]),
                        "measurement_information_condition": information_condition,
                        "predicted_trace": float(np.trace(P_pred)),
                        "posterior_trace": float(np.trace(P_post)),
                        "trace_contraction_ratio": trace_contraction,
                        "correction_norm": correction_norm,
                    }
                )

            x = x_post
            P = P_post

        result = {
            "split": split,
            "seed": seed,
            "delta": delta,
            "filter": filter_name,
            "failed": 0,
            "failure_message": "",
            "posterior_ARMSE": float(np.sqrt(np.mean(posterior_squared_errors))),
            "predicted_ARMSE": float(np.sqrt(np.mean(predicted_squared_errors))),
            "mean_posterior_NEES": safe_mean(posterior_nees_values),
            "median_posterior_NEES": safe_median(posterior_nees_values),
            "mean_predicted_NEES": safe_mean(predicted_nees_values),
            "mean_NIS": safe_mean(nis_values),
            "strong_direction_RMSE": float(
                np.sqrt(np.mean(strong_error_squared_values))
            ),
            "weak_direction_RMSE": float(
                np.sqrt(np.mean(weak_error_squared_values))
            ),
            "mean_strong_variance": safe_mean(strong_variance_values),
            "mean_weak_variance": safe_mean(weak_variance_values),
            "mean_trace_contraction": safe_mean(contraction_values),
            "mean_whitened_H_condition": safe_mean(jacobian_conditions),
            "mean_information_condition": safe_mean(information_conditions),
            "mean_correction_norm": safe_mean(correction_norms),
        }

    except Exception as error:
        result = {
            "split": split,
            "seed": seed,
            "delta": delta,
            "filter": filter_name,
            "failed": 1,
            "failure_message": str(error),
            "posterior_ARMSE": np.nan,
            "predicted_ARMSE": np.nan,
            "mean_posterior_NEES": np.nan,
            "median_posterior_NEES": np.nan,
            "mean_predicted_NEES": np.nan,
            "mean_NIS": np.nan,
            "strong_direction_RMSE": np.nan,
            "weak_direction_RMSE": np.nan,
            "mean_strong_variance": np.nan,
            "mean_weak_variance": np.nan,
            "mean_trace_contraction": np.nan,
            "mean_whitened_H_condition": np.nan,
            "mean_information_condition": np.nan,
            "mean_correction_norm": np.nan,
        }

    return result, step_rows


# ============================================================
# Candidate grid
# ============================================================


def create_candidate_configurations(args: argparse.Namespace) -> List[IDEKFConfiguration]:
    configurations: List[IDEKFConfiguration] = []

    # Always include the mathematically corrected baseline.
    configurations.append(
        IDEKFConfiguration(name="ID_baseline")
    )

    for prior_scale in args.prior_scales:
        for r_scale in args.r_scales:
            for weak_inflation in args.weak_inflations:
                for posterior_scale in args.posterior_scales:
                    if (
                        prior_scale == 1.0
                        and r_scale == 1.0
                        and weak_inflation == 0.0
                        and posterior_scale == 1.0
                    ):
                        continue

                    name = (
                        f"ID_p{prior_scale:g}"
                        f"_r{r_scale:g}"
                        f"_w{weak_inflation:g}"
                        f"_post{posterior_scale:g}"
                    )
                    configurations.append(
                        IDEKFConfiguration(
                            name=name,
                            prior_scale=float(prior_scale),
                            r_scale=float(r_scale),
                            weak_inflation=float(weak_inflation),
                            posterior_scale=float(posterior_scale),
                        )
                    )

    # Deduplicate configurations while preserving order.
    unique: List[IDEKFConfiguration] = []
    seen = set()
    for configuration in configurations:
        key = (
            configuration.prior_scale,
            configuration.r_scale,
            configuration.weak_inflation,
            configuration.posterior_scale,
        )
        if key not in seen:
            seen.add(key)
            unique.append(configuration)

    return unique


# ============================================================
# Experiment
# ============================================================


def integration_settings(profile: str) -> Dict[str, object]:
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

    raise ValueError("profile must be 'paper' or 'harsh'.")


def run_experiment(args: argparse.Namespace):
    os.makedirs(args.outdir, exist_ok=True)
    settings = integration_settings(args.profile)

    mu = 1e5
    f = vdp_f(mu)
    J = vdp_J(mu)
    G = vdp_G()
    Qc_truth = vdp_Qc()
    Qc_filter = scaled_process_covariance(vdp_Qc(), args.filter_qscale)

    x0 = np.array([2.0, 0.0], dtype=float)
    t0 = 0.0
    tf = 2.0

    h, H, R, measurement_tag = build_measurement(
        case="vdp",
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
        Qc=Qc_filter,
    )

    x_initial = x0 + np.asarray(args.x0_perturb, dtype=float)
    P_initial = args.P0_scale * np.eye(x0.size)

    configurations = create_candidate_configurations(args)
    print(f"Candidate IDEKF configurations: {len(configurations)}")

    with open(os.path.join(args.outdir, "candidate_configurations.json"), "w") as file:
        json.dump([asdict(item) for item in configurations], file, indent=2)

    common_filter_arguments = {
        "rtol": float(settings["filter_rtol"]),
        "atol": float(settings["filter_atol"]),
        "max_step": float(settings["max_step"]),
        "method": str(settings["method"]),
    }

    tuning_seeds = list(
        range(args.tuning_seed_start, args.tuning_seed_start + args.tuning_seeds)
    )
    validation_seeds = list(
        range(
            args.validation_seed_start,
            args.validation_seed_start + args.validation_seeds,
        )
    )

    per_run_rows: List[Dict[str, object]] = []
    step_rows: List[Dict[str, object]] = []

    split_seed_pairs = [
        ("tuning", tuning_seeds),
        ("validation", validation_seeds),
    ]

    for split, seeds in split_seed_pairs:
        print("\n" + "=" * 70)
        print(f"Running {split} split with {len(seeds)} seeds")
        print("=" * 70)

        for delta_index, delta in enumerate(args.deltas):
            delta = float(delta)
            t_grid = make_time_grid(t0, tf, delta)
            print(
                f"\n{split}: delta={delta:g}, "
                f"updates={len(t_grid) - 1}"
            )

            for seed_position, seed in enumerate(seeds, start=1):
                truth, measurements = generate_realization(
                    seed=seed,
                    delta_index=delta_index,
                    t_grid=t_grid,
                    f=f,
                    J=J,
                    G=G,
                    Qc=Qc_truth,
                    x0=x0,
                    h=h,
                    R=R,
                    truth_qscale=args.truth_qscale,
                    truth_noise=args.truth_noise,
                    truth_rtol=float(settings["truth_rtol"]),
                    truth_atol=float(settings["truth_atol"]),
                    max_step=float(settings["max_step"]),
                    method=str(settings["method"]),
                )

                sr_filter = CDSREKF(
                    continuous_model,
                    h,
                    H,
                    R,
                    **common_filter_arguments,
                )
                sr_result, sr_steps = run_one_filter(
                    filter_name="SR-EKF",
                    filter_object=sr_filter,
                    x_initial=x_initial,
                    P_initial=P_initial,
                    truth=truth,
                    measurements=measurements,
                    t_grid=t_grid,
                    delta=delta,
                    seed=seed,
                    split=split,
                    collect_steps=(split == "validation" and args.collect_step_rows),
                )
                per_run_rows.append(sr_result)
                step_rows.extend(sr_steps)

                for configuration in configurations:
                    idekf = ConfiguredSingleAssimilationIDEKF(
                        continuous_model,
                        h,
                        H,
                        R,
                        configuration=configuration,
                        **common_filter_arguments,
                    )
                    result, candidate_steps = run_one_filter(
                        filter_name=configuration.name,
                        filter_object=idekf,
                        x_initial=x_initial,
                        P_initial=P_initial,
                        truth=truth,
                        measurements=measurements,
                        t_grid=t_grid,
                        delta=delta,
                        seed=seed,
                        split=split,
                        collect_steps=(
                            split == "validation"
                            and args.collect_step_rows
                            and configuration.name == "ID_baseline"
                        ),
                    )
                    result.update(asdict(configuration))
                    per_run_rows.append(result)
                    step_rows.extend(candidate_steps)

                if (
                    seed_position % args.progress_every == 0
                    or seed_position == len(seeds)
                ):
                    print(f"  completed {seed_position}/{len(seeds)} seeds")

    return (
        per_run_rows,
        step_rows,
        configurations,
        measurement_tag,
    )


# ============================================================
# Summaries and selection
# ============================================================


def summarize_per_filter(
    per_run_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, float, str], List[Dict[str, object]]] = defaultdict(list)

    for row in per_run_rows:
        grouped[
            (
                str(row["split"]),
                float(row["delta"]),
                str(row["filter"]),
            )
        ].append(row)

    summaries: List[Dict[str, object]] = []

    for (split, delta, filter_name), rows in sorted(grouped.items()):
        successful = [row for row in rows if int(row["failed"]) == 0]

        summaries.append(
            {
                "split": split,
                "delta": delta,
                "filter": filter_name,
                "runs": len(rows),
                "failures": len(rows) - len(successful),
                "mean_ARMSE": safe_mean(row["posterior_ARMSE"] for row in successful),
                "median_ARMSE": safe_median(row["posterior_ARMSE"] for row in successful),
                "ARMSE_90": safe_percentile(
                    (row["posterior_ARMSE"] for row in successful),
                    90.0,
                ),
                "ARMSE_95": safe_percentile(
                    (row["posterior_ARMSE"] for row in successful),
                    95.0,
                ),
                "mean_predicted_ARMSE": safe_mean(
                    row["predicted_ARMSE"] for row in successful
                ),
                "mean_posterior_NEES": safe_mean(
                    row["mean_posterior_NEES"] for row in successful
                ),
                "median_posterior_NEES": safe_median(
                    row["median_posterior_NEES"] for row in successful
                ),
                "mean_predicted_NEES": safe_mean(
                    row["mean_predicted_NEES"] for row in successful
                ),
                "mean_NIS": safe_mean(row["mean_NIS"] for row in successful),
                "strong_direction_RMSE": safe_mean(
                    row["strong_direction_RMSE"] for row in successful
                ),
                "weak_direction_RMSE": safe_mean(
                    row["weak_direction_RMSE"] for row in successful
                ),
                "mean_strong_variance": safe_mean(
                    row["mean_strong_variance"] for row in successful
                ),
                "mean_weak_variance": safe_mean(
                    row["mean_weak_variance"] for row in successful
                ),
                "mean_trace_contraction": safe_mean(
                    row["mean_trace_contraction"] for row in successful
                ),
                "mean_whitened_H_condition": safe_mean(
                    row["mean_whitened_H_condition"] for row in successful
                ),
                "mean_information_condition": safe_mean(
                    row["mean_information_condition"] for row in successful
                ),
                "mean_correction_norm": safe_mean(
                    row["mean_correction_norm"] for row in successful
                ),
            }
        )

    return summaries


def paired_candidate_summary(
    per_run_rows: Sequence[Dict[str, object]],
    rng_seed: int = 90210,
) -> List[Dict[str, object]]:
    by_realization: Dict[
        Tuple[str, float, int],
        Dict[str, Dict[str, object]],
    ] = defaultdict(dict)

    for row in per_run_rows:
        if int(row["failed"]) != 0:
            continue
        by_realization[
            (
                str(row["split"]),
                float(row["delta"]),
                int(row["seed"]),
            )
        ][str(row["filter"])] = row

    grouped: Dict[Tuple[str, float, str], List[float]] = defaultdict(list)

    for (split, delta, seed), filters in by_realization.items():
        del seed
        if "SR-EKF" not in filters:
            continue
        sr_error = float(filters["SR-EKF"]["posterior_ARMSE"])

        for filter_name, row in filters.items():
            if filter_name == "SR-EKF":
                continue
            candidate_error = float(row["posterior_ARMSE"])
            grouped[(split, delta, filter_name)].append(sr_error - candidate_error)

    rows: List[Dict[str, object]] = []

    for group_index, ((split, delta, filter_name), differences) in enumerate(
        sorted(grouped.items())
    ):
        values = finite_array(differences)
        rng = np.random.default_rng(rng_seed + group_index)
        ci_low, ci_high = paired_bootstrap_ci(values, rng)
        permutation_p = sign_flip_pvalue(values, rng)

        try:
            wilcoxon_p = float(
                stats.wilcoxon(
                    values,
                    alternative="greater",
                    zero_method="wilcox",
                ).pvalue
            )
        except ValueError:
            wilcoxon_p = np.nan

        rows.append(
            {
                "split": split,
                "delta": delta,
                "candidate": filter_name,
                "paired_runs": values.size,
                "mean_SR_minus_candidate": float(np.mean(values)),
                "median_SR_minus_candidate": float(np.median(values)),
                "bootstrap_95_low": ci_low,
                "bootstrap_95_high": ci_high,
                "candidate_win_rate": float(np.mean(values > 0.0)),
                "sign_flip_p": permutation_p,
                "wilcoxon_p": wilcoxon_p,
            }
        )

    return rows


def select_tuning_candidates(
    paired_rows: Sequence[Dict[str, object]],
    deltas: Sequence[float],
    top_k: int,
) -> List[str]:
    target_deltas = {float(delta) for delta in deltas}
    candidate_scores: Dict[str, List[float]] = defaultdict(list)

    for row in paired_rows:
        if row["split"] != "tuning":
            continue
        if float(row["delta"]) not in target_deltas:
            continue
        candidate_scores[str(row["candidate"])].append(
            float(row["mean_SR_minus_candidate"])
        )

    ranked = sorted(
        candidate_scores.items(),
        key=lambda item: safe_mean(item[1]),
        reverse=True,
    )

    return [name for name, _ in ranked[:top_k]]


def print_results(
    aggregate_rows: Sequence[Dict[str, object]],
    paired_rows: Sequence[Dict[str, object]],
    selected_candidates: Sequence[str],
    target_deltas: Sequence[float],
) -> None:
    print("\n" + "=" * 70)
    print("BASELINE MECHANISM DIAGNOSTICS — VALIDATION SPLIT")
    print("=" * 70)

    target_deltas = {float(delta) for delta in target_deltas}

    for delta in sorted(target_deltas):
        for filter_name in ("SR-EKF", "ID_baseline"):
            matches = [
                row
                for row in aggregate_rows
                if row["split"] == "validation"
                and float(row["delta"]) == delta
                and row["filter"] == filter_name
            ]
            if not matches:
                continue
            row = matches[0]
            print(
                f"delta={delta:g}, {filter_name}: "
                f"ARMSE={float(row['mean_ARMSE']):.6g}, "
                f"predicted ARMSE={float(row['mean_predicted_ARMSE']):.6g}, "
                f"post NEES={float(row['mean_posterior_NEES']):.6g}, "
                f"pred NEES={float(row['mean_predicted_NEES']):.6g}, "
                f"NIS={float(row['mean_NIS']):.6g}, "
                f"strong RMSE={float(row['strong_direction_RMSE']):.6g}, "
                f"weak RMSE={float(row['weak_direction_RMSE']):.6g}, "
                f"weak variance={float(row['mean_weak_variance']):.6g}, "
                f"trace contraction={float(row['mean_trace_contraction']):.6g}, "
                f"cond(R^-1/2 H)={float(row['mean_whitened_H_condition']):.6g}"
            )

    print("\n" + "=" * 70)
    print("TOP CANDIDATES CHOSEN ON TUNING SEEDS")
    print("=" * 70)
    for candidate in selected_candidates:
        print(candidate)

    print("\n" + "=" * 70)
    print("INDEPENDENT VALIDATION RESULTS FOR SELECTED CANDIDATES")
    print("Positive SR-EKF minus candidate values favor IDEKF.")
    print("=" * 70)

    for candidate in selected_candidates:
        for delta in sorted(target_deltas):
            matches = [
                row
                for row in paired_rows
                if row["split"] == "validation"
                and float(row["delta"]) == delta
                and row["candidate"] == candidate
            ]
            if not matches:
                continue
            row = matches[0]
            print(
                f"delta={delta:g}, {candidate}: "
                f"mean difference={float(row['mean_SR_minus_candidate']):.6g}, "
                f"95% CI=[{float(row['bootstrap_95_low']):.6g}, "
                f"{float(row['bootstrap_95_high']):.6g}], "
                f"win rate={float(row['candidate_win_rate']):.3f}, "
                f"permutation p={float(row['sign_flip_p']):.6g}"
            )


# ============================================================
# CLI
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose and search for lower-ARMSE IDEKF variants at low delta."
    )

    parser.add_argument("--profile", choices=["paper", "harsh"], default="paper")
    parser.add_argument(
        "--meas",
        choices=["ill", "nonlin_cubic", "nonlin_tanh"],
        default="nonlin_cubic",
    )
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--Rmode", choices=["diag", "aniso"], default="aniso")
    parser.add_argument("--Rdiag", type=float, nargs="+", default=[1e-4, 1e-2])
    parser.add_argument("--nl-kind", choices=["cubic", "tanh"], default="cubic")
    parser.add_argument("--nl-eps", type=float, default=0.4)
    parser.add_argument("--nl-alpha", type=float, default=1.0)
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--x0-perturb", type=float, nargs=2, default=[1.5, -1.0])
    parser.add_argument("--P0-scale", type=float, default=0.1)
    parser.add_argument("--truth-noise", action="store_true")
    parser.add_argument("--truth-qscale", type=float, default=10.0)
    parser.add_argument("--filter-qscale", type=float, default=10.0)

    parser.add_argument(
        "--prior-scales",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 10.0],
    )
    parser.add_argument(
        "--r-scales",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
    )
    parser.add_argument(
        "--weak-inflations",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 10.0],
    )
    parser.add_argument(
        "--posterior-scales",
        type=float,
        nargs="+",
        default=[1.0, 3.0],
    )

    parser.add_argument("--tuning-seeds", type=int, default=30)
    parser.add_argument("--tuning-seed-start", type=int, default=4001)
    parser.add_argument("--validation-seeds", type=int, default=50)
    parser.add_argument("--validation-seed-start", type=int, default=5001)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--collect-step-rows", action="store_true")
    parser.add_argument("--outdir", type=str, default="results_idekf_low_delta_search")

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.tuning_seeds < 1 or args.validation_seeds < 1:
        raise ValueError("Both tuning-seeds and validation-seeds must be positive.")
    if args.P0_scale <= 0.0:
        raise ValueError("P0-scale must be positive.")
    if args.truth_qscale <= 0.0 or args.filter_qscale <= 0.0:
        raise ValueError("Truth and filter Q scales must be positive.")
    if any(delta <= 0.0 for delta in args.deltas):
        raise ValueError("All deltas must be positive.")
    if any(value <= 0.0 for value in args.prior_scales):
        raise ValueError("All prior scales must be positive.")
    if any(value <= 0.0 for value in args.r_scales):
        raise ValueError("All R scales must be positive.")
    if any(value < 0.0 for value in args.weak_inflations):
        raise ValueError("Weak-direction inflations cannot be negative.")
    if any(value <= 0.0 for value in args.posterior_scales):
        raise ValueError("All posterior scales must be positive.")


if __name__ == "__main__":
    arguments = parse_arguments()
    validate_arguments(arguments)

    (
        per_run_rows,
        step_rows,
        configurations,
        measurement_tag,
    ) = run_experiment(arguments)

    aggregate_rows = summarize_per_filter(per_run_rows)
    paired_rows = paired_candidate_summary(per_run_rows)
    selected_candidates = select_tuning_candidates(
        paired_rows,
        deltas=arguments.deltas,
        top_k=arguments.top_k,
    )

    os.makedirs(arguments.outdir, exist_ok=True)

    per_run_path = os.path.join(arguments.outdir, "per_run_results.csv")
    aggregate_path = os.path.join(arguments.outdir, "aggregate_results.csv")
    paired_path = os.path.join(arguments.outdir, "paired_results.csv")
    step_path = os.path.join(arguments.outdir, "baseline_step_diagnostics.csv")
    selected_path = os.path.join(arguments.outdir, "selected_candidates.txt")

    write_csv(per_run_path, per_run_rows)
    write_csv(aggregate_path, aggregate_rows)
    write_csv(paired_path, paired_rows)
    if step_rows:
        write_csv(step_path, step_rows)

    with open(selected_path, "w") as file:
        for candidate in selected_candidates:
            file.write(candidate + "\n")

    print_results(
        aggregate_rows=aggregate_rows,
        paired_rows=paired_rows,
        selected_candidates=selected_candidates,
        target_deltas=arguments.deltas,
    )

    print("\n" + "=" * 70)
    print("FILES WRITTEN")
    print("=" * 70)
    print(per_run_path)
    print(aggregate_path)
    print(paired_path)
    if step_rows:
        print(step_path)
    print(selected_path)
    print(os.path.join(arguments.outdir, "candidate_configurations.json"))
    print(f"Measurement model: {measurement_tag}")
