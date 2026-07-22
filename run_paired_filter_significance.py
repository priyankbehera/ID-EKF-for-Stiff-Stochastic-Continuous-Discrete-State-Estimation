"""
Paired statistical comparison of:

1. EKF
2. Square-root EKF
3. Influence-diagram EKF

This file imports the filter implementations from:

    run_three_filter_comparison.py

The IDEKF implementation is not modified.

Each seed produces an independent truth trajectory and measurement
realization. All three filters receive the same realization for that seed,
making the comparisons paired.

Primary comparison:

    difference = ARMSE_SR-EKF - ARMSE_IDEKF

A positive difference means IDEKF performed better.

Outputs:

1. Per-seed results CSV
2. Per-delta statistical summary CSV
3. Overall primary-hypothesis summary CSV
4. Mean ARMSE plot with bootstrap confidence intervals
5. IDEKF improvement plot with bootstrap confidence intervals

Example:

python run_paired_filter_significance.py \
  --case vdp \
  --meas nonlin_cubic \
  --sigma 1e-3 \
  --Rmode aniso \
  --Rdiag 1e-4 1e-2 \
  --x0-perturb 1.5 -1.0 \
  --P0-scale 1e-2 \
  --deltas 0.01 0.02 0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --primary-deltas 0.4 0.5 0.7 0.8 1.0 \
  --truth-noise \
  --truth-qscale 10 \
  --num-seeds 100 \
  --seed-start 1 \
  --metric avg \
  --idekf-iter-max 10 \
  --idekf-iter-tol 1e-10 \
  --bootstrap-reps 10000 \
  --permutation-reps 100000
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from filters import (
    ContinuousModel,
    CDEKF,
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
    IDEKFIter,
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
# Utility functions
# ============================================================

def format_float(value: float) -> str:
    """
    Convert a numerical value to a CSV-friendly string.
    """

    if value is None:
        return ""

    value = float(value)

    if not np.isfinite(value):
        return "nan"

    return f"{value:.12g}"


def safe_mean(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    return float(np.mean(values))


def safe_median(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    return float(np.median(values))


def safe_std(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 2:
        return np.nan

    return float(np.std(values, ddof=1))


def armse(
    state_estimates: np.ndarray,
    state_truth: np.ndarray,
    metric: str,
    include_initial: bool,
) -> float:
    """
    Calculate the requested trajectory error.

    By default, the initial point is excluded because every filter starts
    from the same perturbed initial state before receiving any measurement.
    Including it can distort comparisons across sampling periods because
    different deltas produce different numbers of time points.
    """

    state_estimates = np.asarray(
        state_estimates,
        dtype=float,
    )

    state_truth = np.asarray(
        state_truth,
        dtype=float,
    )

    if state_estimates.shape != state_truth.shape:
        raise ValueError(
            "Estimate and truth arrays have different shapes: "
            f"{state_estimates.shape} versus {state_truth.shape}."
        )

    if not include_initial and len(state_estimates) > 1:
        state_estimates = state_estimates[1:]
        state_truth = state_truth[1:]

    squared_error = np.sum(
        (state_estimates - state_truth) ** 2,
        axis=1,
    )

    if metric == "avg":
        return float(
            np.sqrt(
                np.mean(squared_error)
            )
        )

    if metric == "cum":
        return float(
            np.sqrt(
                np.sum(squared_error)
            )
        )

    raise ValueError(
        "metric must be 'avg' or 'cum'."
    )


def bootstrap_mean_ci(
    values: Sequence[float],
    confidence: float,
    repetitions: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Percentile bootstrap confidence interval for a sample mean.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan

    if values.size == 1:
        value = float(values[0])
        return value, value

    repetitions = max(
        100,
        int(repetitions),
    )

    chunk_size = min(
        repetitions,
        2000,
    )

    bootstrap_means: List[np.ndarray] = []

    remaining = repetitions

    while remaining > 0:
        current = min(
            chunk_size,
            remaining,
        )

        indices = rng.integers(
            low=0,
            high=values.size,
            size=(current, values.size),
        )

        bootstrap_means.append(
            np.mean(
                values[indices],
                axis=1,
            )
        )

        remaining -= current

    bootstrap_means_array = np.concatenate(
        bootstrap_means
    )

    alpha = 1.0 - confidence

    lower = float(
        np.quantile(
            bootstrap_means_array,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_means_array,
            1.0 - alpha / 2.0,
        )
    )

    return lower, upper


def one_sided_paired_t_test(
    differences: Sequence[float],
) -> Tuple[float, float]:
    """
    One-sided paired t-test using paired differences.

    H0: mean difference <= 0
    H1: mean difference > 0

    A positive difference favors IDEKF.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if differences.size < 2:
        return np.nan, np.nan

    sample_std = float(
        np.std(
            differences,
            ddof=1,
        )
    )

    if sample_std == 0.0:
        mean_difference = float(
            np.mean(differences)
        )

        if mean_difference > 0.0:
            return np.inf, 0.0

        if mean_difference < 0.0:
            return -np.inf, 1.0

        return 0.0, 1.0

    t_statistic = float(
        np.mean(differences)
        / (
            sample_std
            / np.sqrt(differences.size)
        )
    )

    p_value = float(
        stats.t.sf(
            t_statistic,
            df=differences.size - 1,
        )
    )

    return t_statistic, p_value


def one_sided_wilcoxon(
    differences: Sequence[float],
) -> Tuple[float, float]:
    """
    One-sided Wilcoxon signed-rank test.

    H0: paired differences are centered at zero or below
    H1: paired differences are greater than zero
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if differences.size == 0:
        return np.nan, np.nan

    nonzero = differences[
        differences != 0.0
    ]

    if nonzero.size == 0:
        return 0.0, 1.0

    try:
        result = stats.wilcoxon(
            nonzero,
            alternative="greater",
            zero_method="wilcox",
            correction=False,
            method="auto",
        )

        return (
            float(result.statistic),
            float(result.pvalue),
        )

    except ValueError:
        return np.nan, np.nan


def sign_flip_permutation_test(
    differences: Sequence[float],
    repetitions: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Paired randomization test using sign flips.

    Test statistic:
        mean(SR-EKF ARMSE - IDEKF ARMSE)

    H1:
        the mean difference is greater than zero.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    n = differences.size

    if n == 0:
        return np.nan, np.nan

    observed = float(
        np.mean(differences)
    )

    if n <= 20:
        total_patterns = 1 << n

        exceedances = 0

        for pattern in range(total_patterns):
            signs = np.ones(
                n,
                dtype=float,
            )

            for index in range(n):
                if (
                    pattern
                    >> index
                ) & 1:
                    signs[index] = -1.0

            permuted_statistic = float(
                np.mean(
                    differences * signs
                )
            )

            if permuted_statistic >= observed:
                exceedances += 1

        p_value = (
            exceedances
            / total_patterns
        )

        return observed, float(p_value)

    repetitions = max(
        1000,
        int(repetitions),
    )

    chunk_size = min(
        repetitions,
        5000,
    )

    exceedances = 0
    completed = 0

    while completed < repetitions:
        current = min(
            chunk_size,
            repetitions - completed,
        )

        signs = rng.choice(
            np.array(
                [-1.0, 1.0]
            ),
            size=(
                current,
                n,
            ),
        )

        permuted_statistics = np.mean(
            signs * differences,
            axis=1,
        )

        exceedances += int(
            np.sum(
                permuted_statistics
                >= observed
            )
        )

        completed += current

    p_value = (
        exceedances + 1
    ) / (
        repetitions + 1
    )

    return observed, float(p_value)


def paired_cohen_dz(
    differences: Sequence[float],
) -> float:
    """
    Paired-sample Cohen's dz.

    Positive values favor IDEKF.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if differences.size < 2:
        return np.nan

    standard_deviation = float(
        np.std(
            differences,
            ddof=1,
        )
    )

    if standard_deviation == 0.0:
        mean_difference = float(
            np.mean(differences)
        )

        if mean_difference > 0:
            return np.inf

        if mean_difference < 0:
            return -np.inf

        return 0.0

    return float(
        np.mean(differences)
        / standard_deviation
    )


def idekf_win_rate(
    sr_errors: Sequence[float],
    idekf_errors: Sequence[float],
) -> Tuple[float, int, int, int]:
    """
    Fraction of paired trials in which IDEKF has lower ARMSE.

    Ties are tracked separately and are not counted as wins.
    """

    sr_errors = np.asarray(
        sr_errors,
        dtype=float,
    )

    idekf_errors = np.asarray(
        idekf_errors,
        dtype=float,
    )

    valid = (
        np.isfinite(sr_errors)
        & np.isfinite(idekf_errors)
    )

    sr_errors = sr_errors[valid]
    idekf_errors = idekf_errors[valid]

    wins = int(
        np.sum(
            idekf_errors < sr_errors
        )
    )

    losses = int(
        np.sum(
            idekf_errors > sr_errors
        )
    )

    ties = int(
        np.sum(
            idekf_errors == sr_errors
        )
    )

    non_ties = wins + losses

    if non_ties == 0:
        return np.nan, wins, losses, ties

    return (
        wins / non_ties,
        wins,
        losses,
        ties,
    )


def one_sided_binomial_test(
    wins: int,
    losses: int,
) -> float:
    """
    Test whether IDEKF wins more than 50% of non-tied paired trials.
    """

    non_ties = wins + losses

    if non_ties == 0:
        return np.nan

    result = stats.binomtest(
        k=wins,
        n=non_ties,
        p=0.5,
        alternative="greater",
    )

    return float(result.pvalue)


def holm_adjust(
    p_values: Sequence[float],
) -> np.ndarray:
    """
    Holm-Bonferroni adjusted p-values.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        p_values.shape,
        np.nan,
        dtype=float,
    )

    valid_indices = np.where(
        np.isfinite(p_values)
    )[0]

    if valid_indices.size == 0:
        return adjusted

    valid_p_values = p_values[
        valid_indices
    ]

    ordering = np.argsort(
        valid_p_values
    )

    sorted_p_values = valid_p_values[
        ordering
    ]

    number_of_tests = len(
        sorted_p_values
    )

    sorted_adjusted = np.empty(
        number_of_tests,
        dtype=float,
    )

    running_maximum = 0.0

    for rank, p_value in enumerate(
        sorted_p_values
    ):
        multiplier = (
            number_of_tests - rank
        )

        candidate = min(
            1.0,
            multiplier * p_value,
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        sorted_adjusted[rank] = (
            running_maximum
        )

    reverse_order = np.empty_like(
        ordering
    )

    reverse_order[ordering] = np.arange(
        number_of_tests
    )

    valid_adjusted = sorted_adjusted[
        reverse_order
    ]

    adjusted[
        valid_indices
    ] = valid_adjusted

    return adjusted


# ============================================================
# Model and filter construction
# ============================================================

def construct_case(
    case: str,
):
    """
    Return the model functions and default experiment horizon.
    """

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


def profile_settings(
    profile: str,
) -> Dict[str, float | str]:
    """
    Numerical integration settings.
    """

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
    perturbation_values: Sequence[float],
) -> np.ndarray:
    """
    Apply the requested deterministic initial-state perturbation.
    """

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


def run_single_filter(
    filter_object,
    x_init: np.ndarray,
    P0: np.ndarray,
    t_grid: np.ndarray,
    measurements: np.ndarray,
    x_true: np.ndarray,
    metric: str,
    include_initial: bool,
) -> Tuple[float, float, bool, str]:
    """
    Run one filter on one paired realization.

    Returns:
        error
        elapsed seconds
        failure flag
        failure message
    """

    xk = np.asarray(
        x_init,
        dtype=float,
    ).copy()

    Pk = np.asarray(
        P0,
        dtype=float,
    ).copy()

    estimates = [
        xk.copy()
    ]

    start_time = time.perf_counter()

    try:
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

            xk, Pk = filter_object.predict(
                t_previous,
                t_current,
                xk,
                Pk,
            )

            if not np.all(
                np.isfinite(xk)
            ):
                raise FloatingPointError(
                    "Predicted state contains NaN or Inf."
                )

            if not np.all(
                np.isfinite(Pk)
            ):
                raise FloatingPointError(
                    "Predicted covariance contains NaN or Inf."
                )

            (
                xk,
                Pk,
                _,
                _,
            ) = filter_object.update(
                xk,
                Pk,
                measurements[
                    time_index
                ],
            )

            if not np.all(
                np.isfinite(xk)
            ):
                raise FloatingPointError(
                    "Posterior state contains NaN or Inf."
                )

            if not np.all(
                np.isfinite(Pk)
            ):
                raise FloatingPointError(
                    "Posterior covariance contains NaN or Inf."
                )

            estimates.append(
                np.asarray(
                    xk,
                    dtype=float,
                ).copy()
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        error_value = armse(
            state_estimates=np.vstack(
                estimates
            ),
            state_truth=x_true,
            metric=metric,
            include_initial=include_initial,
        )

        return (
            error_value,
            elapsed,
            False,
            "",
        )

    except (
        FloatingPointError,
        np.linalg.LinAlgError,
        RuntimeError,
        ValueError,
    ) as error:
        elapsed = (
            time.perf_counter()
            - start_time
        )

        return (
            np.nan,
            elapsed,
            True,
            str(error),
        )


# ============================================================
# Experiment
# ============================================================

def run_experiment(
    args: argparse.Namespace,
) -> Tuple[
    List[Dict[str, object]],
    Dict[float, List[Dict[str, object]]],
]:
    """
    Run all independent paired seed experiments.
    """

    os.makedirs(
        args.outdir,
        exist_ok=True,
    )

    settings = profile_settings(
        args.profile
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
        args.case
    )

    sigma = float(
        args.sigma
    )

    (
        h,
        H,
        R,
        measurement_tag,
    ) = build_measurement(
        case=args.case,
        meas=args.meas,
        sigma=sigma,
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

    x_init = initial_estimate(
        x0=x0,
        perturbation_values=args.x0_perturb,
    )

    P0 = (
        np.eye(
            x0.size,
            dtype=float,
        )
        * float(args.P0_scale)
    )

    if args.seeds:
        seeds = [
            int(seed)
            for seed in args.seeds
        ]
    else:
        seeds = list(
            range(
                int(args.seed_start),
                int(args.seed_start)
                + int(args.num_seeds),
            )
        )

    all_rows: List[
        Dict[str, object]
    ] = []

    rows_by_delta: Dict[
        float,
        List[Dict[str, object]],
    ] = defaultdict(list)

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
            f"\nRunning delta={delta:g} "
            f"with {len(seeds)} independent seeds..."
        )

        for position, seed in enumerate(
            seeds,
            start=1,
        ):
            seed_sequence = np.random.SeedSequence(
                [
                    int(seed),
                    int(delta_index),
                    271828,
                ]
            )

            (
                truth_sequence,
                measurement_sequence,
            ) = seed_sequence.spawn(2)

            truth_rng = np.random.default_rng(
                truth_sequence
            )

            measurement_rng = np.random.default_rng(
                measurement_sequence
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
                truth_noise=args.truth_noise,
                qscale=args.truth_qscale,
                rng=truth_rng,
            )

            R_for_noise = (
                project_positive_definite(
                    R,
                    min_eigenvalue=1e-14,
                    max_condition=1e14,
                )
            )

            measurement_root = stable_cholesky(
                R_for_noise,
                min_eigenvalue=1e-14,
                max_condition=1e14,
            )

            standard_noise = (
                measurement_rng.normal(
                    size=(
                        len(t_grid),
                        R.shape[0],
                    )
                )
            )

            measurement_noise = (
                standard_noise
                @ measurement_root.T
            )

            measurements = np.asarray(
                [
                    np.asarray(
                        h(x_true[time_index]),
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

            idekf = IDEKFIter(
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

            filter_objects = {
                "EKF": ekf,
                "SR-EKF": sr_ekf,
                "IDEKF": idekf,
            }

            row: Dict[
                str,
                object,
            ] = {
                "seed": seed,
                "delta": delta,
                "case": args.case,
                "measurement": measurement_tag,
                "sigma": sigma,
                "Rmode": args.Rmode,
                "metric": args.metric,
                "include_initial": int(
                    args.include_initial
                ),
                "truth_noise": int(
                    args.truth_noise
                ),
                "truth_qscale": float(
                    args.truth_qscale
                ),
                "P0_scale": float(
                    args.P0_scale
                ),
                "x0_perturb": " ".join(
                    str(value)
                    for value
                    in args.x0_perturb
                ),
            }

            for (
                filter_name,
                filter_object,
            ) in filter_objects.items():
                (
                    error_value,
                    elapsed,
                    failed,
                    failure_message,
                ) = run_single_filter(
                    filter_object=filter_object,
                    x_init=x_init,
                    P0=P0,
                    t_grid=t_grid,
                    measurements=measurements,
                    x_true=x_true,
                    metric=args.metric,
                    include_initial=args.include_initial,
                )

                row[
                    f"{filter_name}_armse"
                ] = error_value

                row[
                    f"{filter_name}_time"
                ] = elapsed

                row[
                    f"{filter_name}_failed"
                ] = int(failed)

                row[
                    f"{filter_name}_failure_message"
                ] = failure_message

            sr_error = float(
                row[
                    "SR-EKF_armse"
                ]
            )

            idekf_error = float(
                row[
                    "IDEKF_armse"
                ]
            )

            ekf_error = float(
                row[
                    "EKF_armse"
                ]
            )

            if (
                np.isfinite(sr_error)
                and np.isfinite(idekf_error)
            ):
                row[
                    "SR_minus_IDEKF"
                ] = (
                    sr_error
                    - idekf_error
                )

                row[
                    "IDEKF_percent_improvement_over_SR"
                ] = (
                    100.0
                    * (
                        sr_error
                        - idekf_error
                    )
                    / max(
                        abs(sr_error),
                        1e-15,
                    )
                )

                row[
                    "IDEKF_beats_SR"
                ] = int(
                    idekf_error
                    < sr_error
                )

            else:
                row[
                    "SR_minus_IDEKF"
                ] = np.nan

                row[
                    "IDEKF_percent_improvement_over_SR"
                ] = np.nan

                row[
                    "IDEKF_beats_SR"
                ] = 0

            if (
                np.isfinite(ekf_error)
                and np.isfinite(idekf_error)
            ):
                row[
                    "EKF_minus_IDEKF"
                ] = (
                    ekf_error
                    - idekf_error
                )
            else:
                row[
                    "EKF_minus_IDEKF"
                ] = np.nan

            all_rows.append(row)
            rows_by_delta[delta].append(
                row
            )

            if (
                position % args.progress_every == 0
                or position == len(seeds)
            ):
                print(
                    f"  completed "
                    f"{position}/{len(seeds)} seeds"
                )

    return (
        all_rows,
        rows_by_delta,
    )


# ============================================================
# Statistical summaries
# ============================================================

def summarize_delta(
    delta: float,
    rows: Sequence[Dict[str, object]],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """
    Calculate per-delta paired summaries and hypothesis tests.
    """

    ekf_errors = np.asarray(
        [
            row["EKF_armse"]
            for row in rows
        ],
        dtype=float,
    )

    sr_errors = np.asarray(
        [
            row["SR-EKF_armse"]
            for row in rows
        ],
        dtype=float,
    )

    idekf_errors = np.asarray(
        [
            row["IDEKF_armse"]
            for row in rows
        ],
        dtype=float,
    )

    paired_valid = (
        np.isfinite(sr_errors)
        & np.isfinite(idekf_errors)
    )

    paired_sr = sr_errors[
        paired_valid
    ]

    paired_idekf = idekf_errors[
        paired_valid
    ]

    differences = (
        paired_sr
        - paired_idekf
    )

    (
        mean_difference_ci_lower,
        mean_difference_ci_upper,
    ) = bootstrap_mean_ci(
        values=differences,
        confidence=args.confidence,
        repetitions=args.bootstrap_reps,
        rng=rng,
    )

    (
        ekf_ci_lower,
        ekf_ci_upper,
    ) = bootstrap_mean_ci(
        values=ekf_errors,
        confidence=args.confidence,
        repetitions=args.bootstrap_reps,
        rng=rng,
    )

    (
        sr_ci_lower,
        sr_ci_upper,
    ) = bootstrap_mean_ci(
        values=sr_errors,
        confidence=args.confidence,
        repetitions=args.bootstrap_reps,
        rng=rng,
    )

    (
        idekf_ci_lower,
        idekf_ci_upper,
    ) = bootstrap_mean_ci(
        values=idekf_errors,
        confidence=args.confidence,
        repetitions=args.bootstrap_reps,
        rng=rng,
    )

    (
        t_statistic,
        t_p_value,
    ) = one_sided_paired_t_test(
        differences
    )

    (
        wilcoxon_statistic,
        wilcoxon_p_value,
    ) = one_sided_wilcoxon(
        differences
    )

    (
        permutation_statistic,
        permutation_p_value,
    ) = sign_flip_permutation_test(
        differences=differences,
        repetitions=args.permutation_reps,
        rng=rng,
    )

    (
        win_rate,
        wins,
        losses,
        ties,
    ) = idekf_win_rate(
        sr_errors=paired_sr,
        idekf_errors=paired_idekf,
    )

    binomial_p_value = (
        one_sided_binomial_test(
            wins=wins,
            losses=losses,
        )
    )

    sr_failures = int(
        np.sum(
            ~np.isfinite(sr_errors)
        )
    )

    idekf_failures = int(
        np.sum(
            ~np.isfinite(idekf_errors)
        )
    )

    ekf_failures = int(
        np.sum(
            ~np.isfinite(ekf_errors)
        )
    )

    relative_improvements = (
        100.0
        * differences
        / np.maximum(
            np.abs(paired_sr),
            1e-15,
        )
    )

    return {
        "delta": delta,
        "paired_n": int(
            differences.size
        ),
        "EKF_mean": safe_mean(
            ekf_errors
        ),
        "EKF_median": safe_median(
            ekf_errors
        ),
        "EKF_std": safe_std(
            ekf_errors
        ),
        "EKF_ci_lower": ekf_ci_lower,
        "EKF_ci_upper": ekf_ci_upper,
        "EKF_failures": ekf_failures,
        "SR-EKF_mean": safe_mean(
            sr_errors
        ),
        "SR-EKF_median": safe_median(
            sr_errors
        ),
        "SR-EKF_std": safe_std(
            sr_errors
        ),
        "SR-EKF_ci_lower": sr_ci_lower,
        "SR-EKF_ci_upper": sr_ci_upper,
        "SR-EKF_failures": sr_failures,
        "IDEKF_mean": safe_mean(
            idekf_errors
        ),
        "IDEKF_median": safe_median(
            idekf_errors
        ),
        "IDEKF_std": safe_std(
            idekf_errors
        ),
        "IDEKF_ci_lower": idekf_ci_lower,
        "IDEKF_ci_upper": idekf_ci_upper,
        "IDEKF_failures": idekf_failures,
        "mean_SR_minus_IDEKF": safe_mean(
            differences
        ),
        "median_SR_minus_IDEKF": safe_median(
            differences
        ),
        "difference_ci_lower": (
            mean_difference_ci_lower
        ),
        "difference_ci_upper": (
            mean_difference_ci_upper
        ),
        "mean_percent_improvement": safe_mean(
            relative_improvements
        ),
        "median_percent_improvement": safe_median(
            relative_improvements
        ),
        "IDEKF_win_rate": win_rate,
        "IDEKF_wins": wins,
        "IDEKF_losses": losses,
        "ties": ties,
        "paired_cohen_dz": paired_cohen_dz(
            differences
        ),
        "paired_t_statistic": t_statistic,
        "paired_t_p_one_sided": t_p_value,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p_one_sided": wilcoxon_p_value,
        "permutation_statistic": (
            permutation_statistic
        ),
        "permutation_p_one_sided": (
            permutation_p_value
        ),
        "binomial_p_one_sided": (
            binomial_p_value
        ),
    }


def apply_multiple_comparison_corrections(
    summaries: List[Dict[str, object]],
) -> None:
    """
    Add Holm-adjusted p-values across deltas.
    """

    test_columns = [
        "paired_t_p_one_sided",
        "wilcoxon_p_one_sided",
        "permutation_p_one_sided",
        "binomial_p_one_sided",
    ]

    for column in test_columns:
        p_values = [
            float(
                summary[column]
            )
            for summary in summaries
        ]

        adjusted = holm_adjust(
            p_values
        )

        adjusted_column = (
            f"{column}_holm"
        )

        for summary, adjusted_value in zip(
            summaries,
            adjusted,
        ):
            summary[
                adjusted_column
            ] = float(
                adjusted_value
            )


def summarize_primary_hypothesis(
    rows: Sequence[Dict[str, object]],
    primary_deltas: Sequence[float],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """
    Test one pre-specified overall hypothesis.

    For each seed, calculate the mean paired difference across the
    pre-specified primary deltas:

        mean_delta(
            SR-EKF ARMSE - IDEKF ARMSE
        )

    These seed-level averages are then tested against zero.

    This avoids treating every delta as an unrelated opportunity to
    declare significance.
    """

    primary_delta_set = {
        float(value)
        for value in primary_deltas
    }

    differences_by_seed: Dict[
        int,
        List[float],
    ] = defaultdict(list)

    sr_values_by_seed: Dict[
        int,
        List[float],
    ] = defaultdict(list)

    idekf_values_by_seed: Dict[
        int,
        List[float],
    ] = defaultdict(list)

    for row in rows:
        delta = float(
            row["delta"]
        )

        if delta not in primary_delta_set:
            continue

        seed = int(
            row["seed"]
        )

        sr_value = float(
            row["SR-EKF_armse"]
        )

        idekf_value = float(
            row["IDEKF_armse"]
        )

        if (
            np.isfinite(sr_value)
            and np.isfinite(idekf_value)
        ):
            differences_by_seed[
                seed
            ].append(
                sr_value
                - idekf_value
            )

            sr_values_by_seed[
                seed
            ].append(
                sr_value
            )

            idekf_values_by_seed[
                seed
            ].append(
                idekf_value
            )

    seed_average_differences = []
    seed_average_sr = []
    seed_average_idekf = []
    complete_seeds = []

    required_count = len(
        primary_delta_set
    )

    for seed in sorted(
        differences_by_seed
    ):
        if (
            len(
                differences_by_seed[
                    seed
                ]
            )
            != required_count
        ):
            continue

        seed_average_differences.append(
            float(
                np.mean(
                    differences_by_seed[
                        seed
                    ]
                )
            )
        )

        seed_average_sr.append(
            float(
                np.mean(
                    sr_values_by_seed[
                        seed
                    ]
                )
            )
        )

        seed_average_idekf.append(
            float(
                np.mean(
                    idekf_values_by_seed[
                        seed
                    ]
                )
            )
        )

        complete_seeds.append(seed)

    differences = np.asarray(
        seed_average_differences,
        dtype=float,
    )

    (
        ci_lower,
        ci_upper,
    ) = bootstrap_mean_ci(
        values=differences,
        confidence=args.confidence,
        repetitions=args.bootstrap_reps,
        rng=rng,
    )

    (
        t_statistic,
        t_p_value,
    ) = one_sided_paired_t_test(
        differences
    )

    (
        wilcoxon_statistic,
        wilcoxon_p_value,
    ) = one_sided_wilcoxon(
        differences
    )

    (
        permutation_statistic,
        permutation_p_value,
    ) = sign_flip_permutation_test(
        differences=differences,
        repetitions=args.permutation_reps,
        rng=rng,
    )

    (
        win_rate,
        wins,
        losses,
        ties,
    ) = idekf_win_rate(
        sr_errors=seed_average_sr,
        idekf_errors=seed_average_idekf,
    )

    binomial_p_value = one_sided_binomial_test(
        wins=wins,
        losses=losses,
    )

    return {
        "primary_deltas": " ".join(
            f"{value:g}"
            for value in sorted(
                primary_delta_set
            )
        ),
        "complete_seed_count": len(
            complete_seeds
        ),
        "complete_seeds": " ".join(
            str(seed)
            for seed in complete_seeds
        ),
        "mean_SR_minus_IDEKF": safe_mean(
            differences
        ),
        "median_SR_minus_IDEKF": safe_median(
            differences
        ),
        "difference_ci_lower": ci_lower,
        "difference_ci_upper": ci_upper,
        "mean_SR_ARMSE_across_primary_deltas": safe_mean(
            seed_average_sr
        ),
        "mean_IDEKF_ARMSE_across_primary_deltas": safe_mean(
            seed_average_idekf
        ),
        "paired_cohen_dz": paired_cohen_dz(
            differences
        ),
        "IDEKF_win_rate_by_seed": win_rate,
        "IDEKF_wins": wins,
        "IDEKF_losses": losses,
        "ties": ties,
        "paired_t_statistic": t_statistic,
        "paired_t_p_one_sided": t_p_value,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p_one_sided": wilcoxon_p_value,
        "permutation_statistic": (
            permutation_statistic
        ),
        "permutation_p_one_sided": (
            permutation_p_value
        ),
        "binomial_p_one_sided": (
            binomial_p_value
        ),
    }


# ============================================================
# Output
# ============================================================

def write_dictionary_rows(
    path: str,
    rows: Sequence[Dict[str, object]],
) -> None:
    """
    Write dictionaries to a CSV file.
    """

    if not rows:
        raise ValueError(
            f"No rows available for {path}."
        )

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
            cleaned_row = {}

            for key in fieldnames:
                value = row.get(
                    key,
                    "",
                )

                if isinstance(
                    value,
                    (
                        float,
                        np.floating,
                    ),
                ):
                    cleaned_row[key] = format_float(
                        float(value)
                    )
                else:
                    cleaned_row[key] = value

            writer.writerow(
                cleaned_row
            )


def create_plots(
    summaries: Sequence[Dict[str, object]],
    outdir: str,
    prefix: str,
    confidence: float,
) -> None:
    """
    Create performance and paired-difference plots.
    """

    ordered = sorted(
        summaries,
        key=lambda item: float(
            item["delta"]
        ),
    )

    deltas = np.asarray(
        [
            summary["delta"]
            for summary in ordered
        ],
        dtype=float,
    )

    plt.figure(
        figsize=(9, 5.5)
    )

    for filter_name in FILTER_NAMES:
        means = np.asarray(
            [
                summary[
                    f"{filter_name}_mean"
                ]
                for summary in ordered
            ],
            dtype=float,
        )

        lower = np.asarray(
            [
                summary[
                    f"{filter_name}_ci_lower"
                ]
                for summary in ordered
            ],
            dtype=float,
        )

        upper = np.asarray(
            [
                summary[
                    f"{filter_name}_ci_upper"
                ]
                for summary in ordered
            ],
            dtype=float,
        )

        lower_error = (
            means - lower
        )

        upper_error = (
            upper - means
        )

        plt.errorbar(
            deltas,
            means,
            yerr=np.vstack(
                [
                    lower_error,
                    upper_error,
                ]
            ),
            marker="o",
            capsize=3,
            label=filter_name,
        )

    plt.xlabel(
        "Sampling period δ"
    )

    plt.ylabel(
        "Mean ARMSE"
    )

    plt.title(
        f"Paired filter comparison with "
        f"{confidence * 100:.0f}% bootstrap intervals"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    plt.legend()
    plt.tight_layout()

    performance_path = os.path.join(
        outdir,
        f"{prefix}_mean_armse_ci.png",
    )

    plt.savefig(
        performance_path,
        dpi=180,
    )

    plt.close()

    mean_differences = np.asarray(
        [
            summary[
                "mean_SR_minus_IDEKF"
            ]
            for summary in ordered
        ],
        dtype=float,
    )

    lower_differences = np.asarray(
        [
            summary[
                "difference_ci_lower"
            ]
            for summary in ordered
        ],
        dtype=float,
    )

    upper_differences = np.asarray(
        [
            summary[
                "difference_ci_upper"
            ]
            for summary in ordered
        ],
        dtype=float,
    )

    plt.figure(
        figsize=(9, 5.5)
    )

    plt.errorbar(
        deltas,
        mean_differences,
        yerr=np.vstack(
            [
                mean_differences
                - lower_differences,
                upper_differences
                - mean_differences,
            ]
        ),
        marker="o",
        capsize=3,
    )

    plt.axhline(
        y=0.0,
        linestyle="--",
    )

    plt.xlabel(
        "Sampling period δ"
    )

    plt.ylabel(
        "Mean paired difference: SR-EKF − IDEKF"
    )

    plt.title(
        "Positive values favor IDEKF"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    difference_path = os.path.join(
        outdir,
        f"{prefix}_paired_improvement_ci.png",
    )

    plt.savefig(
        difference_path,
        dpi=180,
    )

    plt.close()


def print_summary(
    summaries: Sequence[Dict[str, object]],
    primary_summary: Dict[str, object],
    alpha: float,
) -> None:
    """
    Print the most important results.
    """

    print(
        "\n"
        "============================================================"
    )

    print(
        "PER-DELTA PAIRED RESULTS"
    )

    print(
        "Positive SR-EKF - IDEKF values favor IDEKF."
    )

    print(
        "============================================================"
    )

    for summary in sorted(
        summaries,
        key=lambda item: float(
            item["delta"]
        ),
    ):
        permutation_p = float(
            summary[
                "permutation_p_one_sided"
            ]
        )

        permutation_holm = float(
            summary[
                "permutation_p_one_sided_holm"
            ]
        )

        significant = (
            np.isfinite(
                permutation_holm
            )
            and permutation_holm
            < alpha
        )

        print(
            f"delta={float(summary['delta']):g}: "
            f"SR mean={float(summary['SR-EKF_mean']):.6g}, "
            f"IDEKF mean={float(summary['IDEKF_mean']):.6g}, "
            f"mean difference="
            f"{float(summary['mean_SR_minus_IDEKF']):.6g}, "
            f"{100.0 * (1.0 - alpha):.0f}% CI=["
            f"{float(summary['difference_ci_lower']):.6g}, "
            f"{float(summary['difference_ci_upper']):.6g}], "
            f"win rate="
            f"{float(summary['IDEKF_win_rate']):.3f}, "
            f"permutation p={permutation_p:.6g}, "
            f"Holm p={permutation_holm:.6g}, "
            f"significant={significant}"
        )

    print(
        "\n"
        "============================================================"
    )

    print(
        "PRE-SPECIFIED PRIMARY HYPOTHESIS"
    )

    print(
        "============================================================"
    )

    primary_p = float(
        primary_summary[
            "permutation_p_one_sided"
        ]
    )

    primary_ci_lower = float(
        primary_summary[
            "difference_ci_lower"
        ]
    )

    primary_ci_upper = float(
        primary_summary[
            "difference_ci_upper"
        ]
    )

    supports_idekf = (
        np.isfinite(primary_p)
        and primary_p < alpha
        and primary_ci_lower > 0.0
    )

    print(
        "Primary deltas: "
        f"{primary_summary['primary_deltas']}"
    )

    print(
        "Complete independent seeds: "
        f"{primary_summary['complete_seed_count']}"
    )

    print(
        "Mean seed-level paired difference "
        "(SR-EKF - IDEKF): "
        f"{float(primary_summary['mean_SR_minus_IDEKF']):.6g}"
    )

    print(
        f"{100.0 * (1.0 - alpha):.0f}% bootstrap CI: "
        f"[{primary_ci_lower:.6g}, "
        f"{primary_ci_upper:.6g}]"
    )

    print(
        "Paired Cohen dz: "
        f"{float(primary_summary['paired_cohen_dz']):.6g}"
    )

    print(
        "IDEKF seed-level win rate: "
        f"{float(primary_summary['IDEKF_win_rate_by_seed']):.3f}"
    )

    print(
        "One-sided paired t-test p: "
        f"{float(primary_summary['paired_t_p_one_sided']):.6g}"
    )

    print(
        "One-sided Wilcoxon p: "
        f"{float(primary_summary['wilcoxon_p_one_sided']):.6g}"
    )

    print(
        "One-sided sign-flip permutation p: "
        f"{primary_p:.6g}"
    )

    print(
        "One-sided binomial win-rate p: "
        f"{float(primary_summary['binomial_p_one_sided']):.6g}"
    )

    print(
        "Primary hypothesis supports IDEKF superiority: "
        f"{supports_idekf}"
    )


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired, multi-seed statistical comparisons of "
            "EKF, SR-EKF, and influence-diagram EKF."
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
            0.01, 
            0.02,
            0.05, 
            0.1,
            0.12,
            0.15,
            0.18,
            0.2,
            0.3,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ],
    )

    parser.add_argument(
        "--primary-deltas",
        type=float,
        nargs="+",
        default=[
            0.4,
            0.5,
            0.7,
            0.8,
            1.0,
        ],
        help=(
            "Pre-specified deltas used for the single overall "
            "primary hypothesis. Choose these before viewing the "
            "new multi-seed results."
        ),
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
        help=(
            "Initial covariance scale, with P0 = scale * I."
        ),
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
        "--include-initial",
        action="store_true",
        help=(
            "Include the common pre-measurement initial error in "
            "the trajectory metric. The default excludes it."
        ),
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
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Explicit list of seeds. If omitted, seed-start and "
            "num-seeds are used."
        ),
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num-seeds",
        type=int,
        default=100,
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
        "--confidence",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--permutation-reps",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="results_significance",
    )

    return parser.parse_args()


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    if args.num_seeds < 2 and not args.seeds:
        raise ValueError(
            "Use at least two independent seeds."
        )

    if not (
        0.0
        < args.confidence
        < 1.0
    ):
        raise ValueError(
            "confidence must lie between 0 and 1."
        )

    if not (
        0.0
        < args.alpha
        < 1.0
    ):
        raise ValueError(
            "alpha must lie between 0 and 1."
        )

    requested_deltas = {
        float(value)
        for value in args.deltas
    }

    missing_primary = [
        value
        for value in args.primary_deltas
        if float(value)
        not in requested_deltas
    ]

    if missing_primary:
        raise ValueError(
            "Every primary delta must also appear in --deltas. "
            f"Missing: {missing_primary}"
        )

    if args.P0_scale <= 0.0:
        raise ValueError(
            "P0-scale must be positive."
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

    (
        experiment_rows,
        rows_grouped_by_delta,
    ) = run_experiment(
        arguments
    )

    analysis_seed = np.random.SeedSequence(
        [
            314159,
            int(arguments.seed_start),
            int(arguments.num_seeds),
        ]
    )

    analysis_rng = np.random.default_rng(
        analysis_seed
    )

    delta_summaries: List[
        Dict[str, object]
    ] = []

    for delta in sorted(
        rows_grouped_by_delta
    ):
        delta_summary = summarize_delta(
            delta=delta,
            rows=rows_grouped_by_delta[
                delta
            ],
            args=arguments,
            rng=analysis_rng,
        )

        delta_summaries.append(
            delta_summary
        )

    apply_multiple_comparison_corrections(
        delta_summaries
    )

    primary_summary = (
        summarize_primary_hypothesis(
            rows=experiment_rows,
            primary_deltas=arguments.primary_deltas,
            args=arguments,
            rng=analysis_rng,
        )
    )

    prefix = (
        f"{arguments.case}_"
        f"{arguments.meas}_"
        f"{arguments.Rmode}_"
        f"{arguments.metric}"
    )

    per_seed_path = os.path.join(
        arguments.outdir,
        f"{prefix}_per_seed_results.csv",
    )

    delta_summary_path = os.path.join(
        arguments.outdir,
        f"{prefix}_per_delta_statistics.csv",
    )

    primary_summary_path = os.path.join(
        arguments.outdir,
        f"{prefix}_primary_hypothesis.csv",
    )

    write_dictionary_rows(
        path=per_seed_path,
        rows=experiment_rows,
    )

    write_dictionary_rows(
        path=delta_summary_path,
        rows=delta_summaries,
    )

    write_dictionary_rows(
        path=primary_summary_path,
        rows=[
            primary_summary
        ],
    )

    create_plots(
        summaries=delta_summaries,
        outdir=arguments.outdir,
        prefix=prefix,
        confidence=arguments.confidence,
    )

    print_summary(
        summaries=delta_summaries,
        primary_summary=primary_summary,
        alpha=arguments.alpha,
    )

    print(
        "\nWrote:"
    )

    print(
        f"  {per_seed_path}"
    )

    print(
        f"  {delta_summary_path}"
    )

    print(
        f"  {primary_summary_path}"
    )