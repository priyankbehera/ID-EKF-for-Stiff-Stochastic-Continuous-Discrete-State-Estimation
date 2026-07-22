"""
Confirmatory low-delta validation for corrected influence-diagram EKF.

This script freezes the two IDEKF variants selected before this run:

1. ID_baseline: one influence-diagram measurement assimilation, R scale = 1.
2. ID_Rhalf:    one influence-diagram measurement assimilation, R scale = 0.5.

It compares them with the existing SR-EKF using fresh paired Monte Carlo
realizations. No candidate search or tuning is performed in this file.

The script evaluates:
- a finer low-delta grid;
- several measurement-conditioning values sigma;
- fixed or randomized initial errors;
- matched truth/filter process-noise scales;
- paired bootstrap confidence intervals;
- sign-flip permutation tests;
- Wilcoxon signed-rank tests;
- Holm correction across every delta/sigma/candidate comparison;
- mean, median, 90th and 95th percentile ARMSE;
- catastrophic-error rates;
- strong- and weak-direction RMSE;
- predicted and posterior NEES/NIS diagnostics.

Required repository files:
    filters.py
    models.py
    run_three_filter_comparison.py
    run_idekf_low_delta_search.py
    run_idekf_confirmatory_validation.py
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

from filters import ContinuousModel
from models import vdp_f, vdp_J, vdp_G, vdp_Qc
from run_three_filter_comparison import CDSREKF, build_measurement
from run_idekf_low_delta_search import (
    IDEKFConfiguration,
    ConfiguredSingleAssimilationIDEKF,
    generate_realization,
    integration_settings,
    make_time_grid,
    positive_definite,
    run_one_filter,
    scaled_process_covariance,
)


CANDIDATES = (
    IDEKFConfiguration(
        name="ID_baseline",
        prior_scale=1.0,
        r_scale=1.0,
        weak_inflation=0.0,
        posterior_scale=1.0,
    ),
    IDEKFConfiguration(
        name="ID_Rhalf",
        prior_scale=1.0,
        r_scale=0.5,
        weak_inflation=0.0,
        posterior_scale=1.0,
    ),
)


def finite_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def safe_mean(values: Sequence[float]) -> float:
    array = finite_array(values)
    return float(np.mean(array)) if array.size else np.nan


def safe_median(values: Sequence[float]) -> float:
    array = finite_array(values)
    return float(np.median(array)) if array.size else np.nan


def safe_percentile(values: Sequence[float], q: float) -> float:
    array = finite_array(values)
    return float(np.percentile(array, q)) if array.size else np.nan


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(
    differences: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
) -> Tuple[float, float]:
    differences = finite_array(differences)
    if differences.size == 0:
        return np.nan, np.nan
    indices = rng.integers(
        0,
        differences.size,
        size=(repetitions, differences.size),
    )
    means = np.mean(differences[indices], axis=1)
    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def sign_flip_pvalue(
    differences: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
) -> float:
    differences = finite_array(differences)
    if differences.size == 0:
        return np.nan
    observed = float(np.mean(differences))
    extreme = 0
    batch = 2000
    completed = 0
    while completed < repetitions:
        count = min(batch, repetitions - completed)
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(count, differences.size),
        )
        null_means = np.mean(signs * differences, axis=1)
        extreme += int(np.sum(null_means >= observed))
        completed += count
    return float((extreme + 1) / (repetitions + 1))


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(pvalues, np.nan)
    valid_indices = np.where(np.isfinite(pvalues))[0]
    if valid_indices.size == 0:
        return adjusted
    order = valid_indices[np.argsort(pvalues[valid_indices])]
    running = 0.0
    m = order.size
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def make_initial_condition(
    x0: np.ndarray,
    fixed_perturbation: Sequence[float],
    P0: np.ndarray,
    seed: int,
    mode: str,
) -> np.ndarray:
    if mode == "fixed":
        perturbation = np.zeros_like(x0, dtype=float)
        for index in range(min(len(perturbation), len(fixed_perturbation))):
            perturbation[index] = float(fixed_perturbation[index])
        return np.asarray(x0, dtype=float) + perturbation

    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 20260722, 902])
    )
    root = np.linalg.cholesky(positive_definite(P0))
    return np.asarray(x0, dtype=float) + root @ rng.normal(size=x0.size)


def construct_filters(
    continuous_model: ContinuousModel,
    h,
    H,
    R: np.ndarray,
    settings: Dict[str, object],
) -> Dict[str, object]:
    common = {
        "rtol": float(settings["filter_rtol"]),
        "atol": float(settings["filter_atol"]),
        "max_step": float(settings["max_step"]),
        "method": str(settings["method"]),
    }
    filters: Dict[str, object] = {
        "SR-EKF": CDSREKF(
            continuous_model,
            h,
            H,
            R,
            **common,
        )
    }
    for configuration in CANDIDATES:
        filters[configuration.name] = ConfiguredSingleAssimilationIDEKF(
            continuous_model,
            h,
            H,
            R,
            configuration=configuration,
            **common,
        )
    return filters


def run_validation(args: argparse.Namespace):
    settings = integration_settings(args.profile)

    mu = 1.0e5
    f = vdp_f(mu)
    J = vdp_J(mu)
    G = vdp_G()
    Qc_truth = vdp_Qc()
    Qc_filter = scaled_process_covariance(Qc_truth, args.filter_qscale)

    x0 = np.array([2.0, 0.0], dtype=float)
    t0 = 0.0
    tf = 2.0
    P0 = args.P0_scale * np.eye(x0.size, dtype=float)

    continuous_model = ContinuousModel(
        f=f,
        J=J,
        G=G,
        Qc=Qc_filter,
    )

    per_run_rows: List[Dict[str, object]] = []
    step_rows: List[Dict[str, object]] = []

    scenario_index = 0
    for sigma in args.sigmas:
        h, H, R, measurement_tag = build_measurement(
            case="vdp",
            meas=args.meas,
            sigma=float(sigma),
            Rmode=args.Rmode,
            Rdiag=args.Rdiag,
            nl_kind=args.nl_kind,
            nl_eps=args.nl_eps,
            nl_alpha=args.nl_alpha,
        )

        for delta_index, delta in enumerate(args.deltas):
            scenario_index += 1
            delta = float(delta)
            t_grid = make_time_grid(t0, tf, delta)
            print(
                f"\nsigma={sigma:g}, delta={delta:g}, "
                f"updates={len(t_grid)-1}, initial={args.initial_mode}"
            )

            for seed_position, seed in enumerate(
                range(args.seed_start, args.seed_start + args.num_seeds),
                start=1,
            ):
                truth, measurements = generate_realization(
                    seed=seed,
                    delta_index=scenario_index,
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

                x_initial = make_initial_condition(
                    x0=x0,
                    fixed_perturbation=args.x0_perturb,
                    P0=P0,
                    seed=seed,
                    mode=args.initial_mode,
                )

                filters = construct_filters(
                    continuous_model=continuous_model,
                    h=h,
                    H=H,
                    R=R,
                    settings=settings,
                )

                for filter_name, filter_object in filters.items():
                    result, rows = run_one_filter(
                        split="confirmatory",
                        seed=seed,
                        delta=delta,
                        filter_name=filter_name,
                        filter_object=filter_object,
                        x_initial=x_initial,
                        P_initial=P0,
                        t_grid=t_grid,
                        truth=truth,
                        measurements=measurements,
                        collect_step_rows=args.collect_step_rows,
                    )
                    result.update(
                        {
                            "sigma": float(sigma),
                            "measurement": measurement_tag,
                            "initial_mode": args.initial_mode,
                            "truth_qscale": args.truth_qscale,
                            "filter_qscale": args.filter_qscale,
                            "P0_scale": args.P0_scale,
                        }
                    )
                    per_run_rows.append(result)
                    for row in rows:
                        row.update(
                            {
                                "sigma": float(sigma),
                                "initial_mode": args.initial_mode,
                            }
                        )
                    step_rows.extend(rows)

                if seed_position % args.progress_every == 0 or seed_position == args.num_seeds:
                    print(f"  completed {seed_position}/{args.num_seeds} seeds")

    return per_run_rows, step_rows


def aggregate(per_run_rows: Sequence[Dict[str, object]]):
    grouped: Dict[Tuple[float, float, str], List[Dict[str, object]]] = defaultdict(list)
    for row in per_run_rows:
        grouped[(float(row["sigma"]), float(row["delta"]), str(row["filter"]))].append(row)

    rows: List[Dict[str, object]] = []
    for (sigma, delta, filter_name), values in sorted(grouped.items()):
        valid = [row for row in values if int(row.get("failed", 0)) == 0]
        armse = [float(row["armse"]) for row in valid]
        rows.append(
            {
                "sigma": sigma,
                "delta": delta,
                "filter": filter_name,
                "runs": len(values),
                "failures": len(values) - len(valid),
                "mean_ARMSE": safe_mean(armse),
                "median_ARMSE": safe_median(armse),
                "ARMSE_90": safe_percentile(armse, 90.0),
                "ARMSE_95": safe_percentile(armse, 95.0),
                "catastrophic_rate_ARMSE_gt_5": safe_mean([value > 5.0 for value in armse]),
                "catastrophic_rate_ARMSE_gt_10": safe_mean([value > 10.0 for value in armse]),
                "mean_predicted_ARMSE": safe_mean([row["predicted_armse"] for row in valid]),
                "mean_posterior_NEES": safe_mean([row["mean_posterior_nees"] for row in valid]),
                "median_posterior_NEES": safe_median([row["median_posterior_nees"] for row in valid]),
                "mean_NIS": safe_mean([row["mean_nis"] for row in valid]),
                "strong_direction_RMSE": float(np.sqrt(safe_mean([row["mean_strong_error_squared"] for row in valid]))),
                "weak_direction_RMSE": float(np.sqrt(safe_mean([row["mean_weak_error_squared"] for row in valid]))),
                "mean_correction_norm": safe_mean([row["mean_correction_norm"] for row in valid]),
                "mean_trace_contraction": safe_mean([row["mean_trace_contraction"] for row in valid]),
            }
        )
    return rows


def paired_results(
    per_run_rows: Sequence[Dict[str, object]],
    bootstrap_repetitions: int,
    permutation_repetitions: int,
    random_seed: int,
):
    realization_map: Dict[Tuple[float, float, int], Dict[str, Dict[str, object]]] = defaultdict(dict)
    for row in per_run_rows:
        realization_map[(float(row["sigma"]), float(row["delta"]), int(row["seed"]))][str(row["filter"])] = row

    grouped: Dict[Tuple[float, float, str], List[float]] = defaultdict(list)
    for (sigma, delta, _seed), filters in realization_map.items():
        if "SR-EKF" not in filters or int(filters["SR-EKF"].get("failed", 0)):
            continue
        sr_error = float(filters["SR-EKF"]["armse"])
        for configuration in CANDIDATES:
            name = configuration.name
            if name not in filters or int(filters[name].get("failed", 0)):
                continue
            grouped[(sigma, delta, name)].append(sr_error - float(filters[name]["armse"]))

    rng = np.random.default_rng(random_seed)
    rows: List[Dict[str, object]] = []
    for (sigma, delta, candidate), differences in sorted(grouped.items()):
        array = np.asarray(differences, dtype=float)
        low, high = bootstrap_mean_ci(array, rng, bootstrap_repetitions)
        try:
            wilcoxon_p = float(stats.wilcoxon(array, alternative="greater", zero_method="wilcox").pvalue)
        except ValueError:
            wilcoxon_p = np.nan
        rows.append(
            {
                "sigma": sigma,
                "delta": delta,
                "candidate": candidate,
                "paired_runs": array.size,
                "mean_SR_minus_candidate": float(np.mean(array)),
                "median_SR_minus_candidate": float(np.median(array)),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "candidate_win_rate": float(np.mean(array > 0.0)),
                "sign_flip_p": sign_flip_pvalue(array, rng, permutation_repetitions),
                "wilcoxon_p": wilcoxon_p,
            }
        )

    adjusted = holm_adjust([row["sign_flip_p"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_sign_flip_p"] = float(value)
        row["significant_after_Holm"] = int(
            np.isfinite(value)
            and value < 0.05
            and float(row["bootstrap_95_low"]) > 0.0
        )
    return rows


def print_report(aggregate_rows, paired_rows):
    print("\n" + "=" * 78)
    print("CONFIRMATORY FILTER SUMMARY")
    print("=" * 78)
    for row in aggregate_rows:
        print(
            f"sigma={row['sigma']:g}, delta={row['delta']:g}, {row['filter']}: "
            f"mean ARMSE={row['mean_ARMSE']:.6g}, median={row['median_ARMSE']:.6g}, "
            f"95th={row['ARMSE_95']:.6g}, P(ARMSE>5)={row['catastrophic_rate_ARMSE_gt_5']:.3f}, "
            f"strong RMSE={row['strong_direction_RMSE']:.6g}, weak RMSE={row['weak_direction_RMSE']:.6g}, "
            f"correction={row['mean_correction_norm']:.6g}"
        )

    print("\n" + "=" * 78)
    print("PAIRED CONFIRMATORY RESULTS")
    print("Positive SR-EKF minus candidate values favor IDEKF.")
    print("=" * 78)
    for row in paired_rows:
        print(
            f"sigma={row['sigma']:g}, delta={row['delta']:g}, {row['candidate']}: "
            f"mean difference={row['mean_SR_minus_candidate']:.6g}, "
            f"95% CI=[{row['bootstrap_95_low']:.6g}, {row['bootstrap_95_high']:.6g}], "
            f"win rate={row['candidate_win_rate']:.3f}, "
            f"permutation p={row['sign_flip_p']:.6g}, Holm p={row['holm_sign_flip_p']:.6g}, "
            f"significant={bool(row['significant_after_Holm'])}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["paper", "harsh"], default="paper")
    parser.add_argument("--meas", choices=["nonlin_cubic", "nonlin_tanh", "ill", "paper"], default="nonlin_cubic")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[1e-3])
    parser.add_argument("--Rmode", choices=["diag", "aniso"], default="aniso")
    parser.add_argument("--Rdiag", type=float, nargs="*", default=[1e-4, 1e-2])
    parser.add_argument("--nl-kind", choices=["cubic", "tanh"], default="cubic")
    parser.add_argument("--nl-eps", type=float, default=0.4)
    parser.add_argument("--nl-alpha", type=float, default=1.0)
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    parser.add_argument("--truth-noise", action="store_true")
    parser.add_argument("--truth-qscale", type=float, default=10.0)
    parser.add_argument("--filter-qscale", type=float, default=10.0)
    parser.add_argument("--P0-scale", type=float, default=0.1)
    parser.add_argument("--x0-perturb", type=float, nargs="*", default=[1.5, -1.0])
    parser.add_argument("--initial-mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--num-seeds", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=6001)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--permutation-repetitions", type=int, default=20000)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--collect-step-rows", action="store_true")
    parser.add_argument("--outdir", type=str, default="results_idekf_confirmatory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(args.outdir, exist_ok=True)

    per_run_rows, step_rows = run_validation(args)
    aggregate_rows = aggregate(per_run_rows)
    paired_rows = paired_results(
        per_run_rows=per_run_rows,
        bootstrap_repetitions=args.bootstrap_repetitions,
        permutation_repetitions=args.permutation_repetitions,
        random_seed=args.seed_start + 991,
    )

    write_csv(os.path.join(args.outdir, "confirmatory_per_run.csv"), per_run_rows)
    write_csv(os.path.join(args.outdir, "confirmatory_aggregate.csv"), aggregate_rows)
    write_csv(os.path.join(args.outdir, "confirmatory_paired.csv"), paired_rows)
    if args.collect_step_rows:
        write_csv(os.path.join(args.outdir, "confirmatory_step_rows.csv"), step_rows)

    with open(os.path.join(args.outdir, "frozen_candidates.txt"), "w") as handle:
        for configuration in CANDIDATES:
            handle.write(str(asdict(configuration)) + "\n")

    print_report(aggregate_rows, paired_rows)

    print("\nFILES WRITTEN")
    print(os.path.join(args.outdir, "confirmatory_per_run.csv"))
    print(os.path.join(args.outdir, "confirmatory_aggregate.csv"))
    print(os.path.join(args.outdir, "confirmatory_paired.csv"))
    if args.collect_step_rows:
        print(os.path.join(args.outdir, "confirmatory_step_rows.csv"))
