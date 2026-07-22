"""
Filter consistency and calibration sweep.

This script investigates why EKF, SR-EKF, and the corrected IDEKF
produce extremely large NEES values.

It sweeps:

1. Truth process-noise scale
2. Filter process-noise scale
3. Initial covariance scale
4. Measurement covariance scale

The corrected IDEKF is the repository's original CDIDEKF. No repeated
measurement assimilation is used.

Required files:

    filters.py
    models.py
    run_three_filter_comparison.py
    run_idekf_fix_validation.py
    run_filter_calibration_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

from filters import (
    ContinuousModel,
    CDEKF,
    CDIDEKF,
)

from run_three_filter_comparison import (
    CDSREKF,
    build_measurement,
)

from run_idekf_fix_validation import (
    construct_case,
    integration_settings,
    initial_estimate,
    generate_realization,
    run_filter,
)


FILTER_NAMES = (
    "EKF",
    "SR-EKF",
    "IDEKF",
)


def finite_array(
    values: Sequence[float],
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )

    return values[
        np.isfinite(values)
    ]


def safe_mean(
    values: Sequence[float],
) -> float:
    values = finite_array(values)

    if values.size == 0:
        return np.nan

    return float(
        np.mean(values)
    )


def safe_median(
    values: Sequence[float],
) -> float:
    values = finite_array(values)

    if values.size == 0:
        return np.nan

    return float(
        np.median(values)
    )


def safe_percentile(
    values: Sequence[float],
    percentile: float,
) -> float:
    values = finite_array(values)

    if values.size == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            percentile,
        )
    )


def write_csv(
    path: str,
    rows: Sequence[Dict[str, object]],
) -> None:
    if not rows:
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
        writer.writerows(rows)


def scale_noise_model(
    G,
    Qc,
    filter_qscale: float,
):
    """
    Scale the process covariance assumed by the filter.

    Qc_scaled(t) = filter_qscale * Qc(t)
    """

    if callable(Qc):
        def scaled_Qc(time):
            return (
                filter_qscale
                * np.asarray(
                    Qc(time),
                    dtype=float,
                )
            )
    else:
        scaled_constant = (
            filter_qscale
            * np.asarray(
                Qc,
                dtype=float,
            )
        )

        def scaled_Qc(time):
            del time
            return scaled_constant

    return G, scaled_Qc


def create_filters(
    continuous_model: ContinuousModel,
    h,
    H,
    R: np.ndarray,
    settings: Dict[str, object],
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
        "IDEKF": CDIDEKF(
            continuous_model,
            h,
            H,
            R,
            **common,
        ),
    }


def anees_bounds(
    state_dimension: int,
    number_of_values: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Chi-square consistency bounds for average NEES.

    If NEES values were independent and correctly calibrated:

        sum(NEES) ~ chi-square(n_x * N)

    Therefore:

        ANEES bounds =
        chi-square quantiles / N
    """

    if number_of_values < 1:
        return np.nan, np.nan

    alpha = 1.0 - confidence

    degrees_of_freedom = (
        state_dimension
        * number_of_values
    )

    lower = float(
        stats.chi2.ppf(
            alpha / 2.0,
            degrees_of_freedom,
        )
        / number_of_values
    )

    upper = float(
        stats.chi2.ppf(
            1.0 - alpha / 2.0,
            degrees_of_freedom,
        )
        / number_of_values
    )

    return lower, upper


def run_sweep(
    args: argparse.Namespace,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
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
        tf,
    ) = construct_case(
        args.case
    )

    (
        h,
        H,
        R_base,
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

    x_initial = initial_estimate(
        x0,
        args.x0_perturb,
    )

    per_run_rows: List[
        Dict[str, object]
    ] = []

    configuration_index = 0

    for delta in args.deltas:
        delta = float(delta)

        t_grid = np.arange(
            t0,
            tf + 1e-12,
            delta,
        )

        for truth_qscale in args.truth_qscales:
            truth_qscale = float(
                truth_qscale
            )

            for filter_qscale in args.filter_qscales:
                filter_qscale = float(
                    filter_qscale
                )

                G_filter, Qc_filter = (
                    scale_noise_model(
                        G,
                        Qc,
                        filter_qscale,
                    )
                )

                continuous_model = (
                    ContinuousModel(
                        f=f,
                        J=J,
                        G=G_filter,
                        Qc=Qc_filter,
                    )
                )

                for P0_scale in args.P0_scales:
                    P0_scale = float(
                        P0_scale
                    )

                    P_initial = (
                        np.eye(
                            x0.size,
                            dtype=float,
                        )
                        * P0_scale
                    )

                    for R_scale in args.R_scales:
                        R_scale = float(
                            R_scale
                        )

                        R = (
                            R_scale
                            * np.asarray(
                                R_base,
                                dtype=float,
                            )
                        )

                        configuration_index += 1

                        print(
                            "\n"
                            f"Configuration "
                            f"{configuration_index}: "
                            f"delta={delta:g}, "
                            f"truth Q scale="
                            f"{truth_qscale:g}, "
                            f"filter Q scale="
                            f"{filter_qscale:g}, "
                            f"P0 scale="
                            f"{P0_scale:g}, "
                            f"R scale="
                            f"{R_scale:g}"
                        )

                        for seed_position, seed in enumerate(
                            range(
                                args.seed_start,
                                args.seed_start
                                + args.num_seeds,
                            ),
                            start=1,
                        ):
                            realization_seed = (
                                int(seed)
                                + 100000
                                * configuration_index
                            )

                            (
                                x_true,
                                measurements,
                            ) = generate_realization(
                                seed=realization_seed,
                                delta_index=(
                                    configuration_index
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
                                truth_noise=(
                                    args.truth_noise
                                ),
                                truth_qscale=(
                                    truth_qscale
                                ),
                            )

                            filters = create_filters(
                                continuous_model=(
                                    continuous_model
                                ),
                                h=h,
                                H=H,
                                R=R,
                                settings=settings,
                            )

                            for (
                                filter_name,
                                filter_object,
                            ) in filters.items():
                                result = run_filter(
                                    filter_name=(
                                        filter_name
                                    ),
                                    filter_object=(
                                        filter_object
                                    ),
                                    x_initial=(
                                        x_initial
                                    ),
                                    P_initial=(
                                        P_initial
                                    ),
                                    t_grid=t_grid,
                                    x_true=x_true,
                                    measurements=(
                                        measurements
                                    ),
                                )

                                result.update(
                                    {
                                        "seed": seed,
                                        "delta": delta,
                                        "truth_qscale": (
                                            truth_qscale
                                        ),
                                        "filter_qscale": (
                                            filter_qscale
                                        ),
                                        "P0_scale": (
                                            P0_scale
                                        ),
                                        "R_scale": (
                                            R_scale
                                        ),
                                        "measurement": (
                                            measurement_tag
                                        ),
                                        "number_of_updates": (
                                            len(
                                                t_grid
                                            )
                                            - 1
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
                                    f"{args.num_seeds}"
                                )

    summary_rows = summarize_results(
        per_run_rows=per_run_rows,
        state_dimension=x0.size,
    )

    return (
        per_run_rows,
        summary_rows,
    )


def summarize_results(
    per_run_rows: Sequence[
        Dict[str, object]
    ],
    state_dimension: int,
) -> List[Dict[str, object]]:
    grouped: Dict[
        Tuple[
            float,
            float,
            float,
            float,
            float,
            str,
        ],
        List[Dict[str, object]],
    ] = defaultdict(list)

    for row in per_run_rows:
        key = (
            float(row["delta"]),
            float(row["truth_qscale"]),
            float(row["filter_qscale"]),
            float(row["P0_scale"]),
            float(row["R_scale"]),
            str(row["filter"]),
        )

        grouped[key].append(row)

    summaries: List[
        Dict[str, object]
    ] = []

    for key, rows in sorted(
        grouped.items()
    ):
        (
            delta,
            truth_qscale,
            filter_qscale,
            P0_scale,
            R_scale,
            filter_name,
        ) = key

        successful = [
            row
            for row in rows
            if int(row["failed"]) == 0
        ]

        armse_values = [
            float(row["armse"])
            for row in successful
        ]

        nees_values = [
            float(row["mean_nees"])
            for row in successful
        ]

        median_nees_values = [
            float(row["median_nees"])
            for row in successful
        ]

        number_of_updates = int(
            successful[0][
                "number_of_updates"
            ]
        ) if successful else 0

        approximate_nees_count = (
            len(successful)
            * number_of_updates
        )

        (
            nees_lower,
            nees_upper,
        ) = anees_bounds(
            state_dimension=(
                state_dimension
            ),
            number_of_values=(
                approximate_nees_count
            ),
        )

        mean_nees = safe_mean(
            nees_values
        )

        summaries.append(
            {
                "delta": delta,
                "truth_qscale": (
                    truth_qscale
                ),
                "filter_qscale": (
                    filter_qscale
                ),
                "Q_scale_ratio_truth_over_filter": (
                    truth_qscale
                    / filter_qscale
                ),
                "P0_scale": P0_scale,
                "R_scale": R_scale,
                "filter": filter_name,
                "runs": len(rows),
                "failures": (
                    len(rows)
                    - len(successful)
                ),
                "mean_ARMSE": safe_mean(
                    armse_values
                ),
                "median_ARMSE": safe_median(
                    armse_values
                ),
                "ARMSE_90th_percentile": (
                    safe_percentile(
                        armse_values,
                        90.0,
                    )
                ),
                "ARMSE_95th_percentile": (
                    safe_percentile(
                        armse_values,
                        95.0,
                    )
                ),
                "maximum_ARMSE": (
                    max(
                        armse_values
                    )
                    if armse_values
                    else np.nan
                ),
                "mean_NEES": mean_nees,
                "median_of_run_mean_NEES": (
                    safe_median(
                        nees_values
                    )
                ),
                "median_of_run_median_NEES": (
                    safe_median(
                        median_nees_values
                    )
                ),
                "ANEES_95_lower": (
                    nees_lower
                ),
                "ANEES_95_upper": (
                    nees_upper
                ),
                "mean_NEES_inside_bounds": int(
                    np.isfinite(
                        mean_nees
                    )
                    and nees_lower
                    <= mean_nees
                    <= nees_upper
                ),
                "mean_covariance_trace": (
                    safe_mean(
                        [
                            float(
                                row[
                                    "mean_covariance_trace"
                                ]
                            )
                            for row
                            in successful
                        ]
                    )
                ),
            }
        )

    return summaries


def print_best_configurations(
    summary_rows: Sequence[
        Dict[str, object]
    ],
) -> None:
    print(
        "\n"
        "============================================================"
    )

    print(
        "BEST CONFIGURATIONS BY FILTER AND DELTA"
    )

    print(
        "Sorted primarily by median ARMSE, then by NEES "
        "closeness to the state dimension."
    )

    print(
        "============================================================"
    )

    grouped: Dict[
        Tuple[float, str],
        List[Dict[str, object]],
    ] = defaultdict(list)

    for row in summary_rows:
        grouped[
            (
                float(row["delta"]),
                str(row["filter"]),
            )
        ].append(row)

    for (
        delta,
        filter_name,
    ), rows in sorted(
        grouped.items()
    ):
        ranked = sorted(
            rows,
            key=lambda row: (
                float(
                    row[
                        "median_ARMSE"
                    ]
                ),
                abs(
                    np.log10(
                        max(
                            float(
                                row[
                                    "mean_NEES"
                                ]
                            ),
                            1e-300,
                        )
                        / 2.0
                    )
                ),
            ),
        )

        best = ranked[0]

        print(
            f"delta={delta:g}, "
            f"{filter_name}: "
            f"truthQ={float(best['truth_qscale']):g}, "
            f"filterQ={float(best['filter_qscale']):g}, "
            f"P0={float(best['P0_scale']):g}, "
            f"Rscale={float(best['R_scale']):g}, "
            f"mean ARMSE="
            f"{float(best['mean_ARMSE']):.6g}, "
            f"median ARMSE="
            f"{float(best['median_ARMSE']):.6g}, "
            f"95th percentile="
            f"{float(best['ARMSE_95th_percentile']):.6g}, "
            f"mean NEES="
            f"{float(best['mean_NEES']):.6g}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep truth Q, filter Q, P0, and R scales "
            "to diagnose filter inconsistency."
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
        "--truth-noise",
        action="store_true",
    )

    parser.add_argument(
        "--truth-qscales",
        type=float,
        nargs="+",
        default=[
            1.0,
            10.0,
        ],
    )

    parser.add_argument(
        "--filter-qscales",
        type=float,
        nargs="+",
        default=[
            1.0,
            10.0,
            100.0,
        ],
    )

    parser.add_argument(
        "--P0-scales",
        type=float,
        nargs="+",
        default=[
            0.01,
            0.1,
            1.0,
            10.0,
        ],
    )

    parser.add_argument(
        "--R-scales",
        type=float,
        nargs="+",
        default=[
            1.0,
            10.0,
        ],
    )

    parser.add_argument(
        "--num-seeds",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=3001,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="results_calibration_sweep",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    os.makedirs(
        arguments.outdir,
        exist_ok=True,
    )

    (
        per_run_rows,
        summary_rows,
    ) = run_sweep(arguments)

    per_run_path = os.path.join(
        arguments.outdir,
        "calibration_per_run.csv",
    )

    summary_path = os.path.join(
        arguments.outdir,
        "calibration_summary.csv",
    )

    write_csv(
        per_run_path,
        per_run_rows,
    )

    write_csv(
        summary_path,
        summary_rows,
    )

    print_best_configurations(
        summary_rows
    )

    print(
        "\nFiles written:"
    )

    print(per_run_path)
    print(summary_path)