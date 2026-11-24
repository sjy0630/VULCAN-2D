"""Strict validation of terminal-refined RESET transition parameters.

Cycles 1--3 remain excluded by experimental confirmation.  For each target in
cycles 4--53, all baseline laws and the two existing RESET transition
parameters are fitted using only the other 49 cycles.  The target current is
read only after prediction.  A cycles 4--35 to 36--53 chronological test is
reported separately.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vulcan2d-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import validate_full_cycle as VF
from . import validate_postconditioning as PC
from . import validate_unified as UV
from . import validate_v12 as V12
from . import validate_v13 as A
from . import validate_v5 as V5
from .reset_terminal_refinement import (
    fit_terminal_reset_transition,
    simulate_with_terminal_refinement,
)


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_NAME = "reset_platform_terminal_refinement.json"
CYCLES_NAME = "reset_platform_terminal_refinement_cycles.csv"
CURVES_NAME = "reset_platform_terminal_refinement_curves.npz"
FIGURE_NAME = "26_reset_platform_terminal_refinement.png"
LAST_TRAINING_CYCLE = 35


def _training_inputs(named_set, accepted, reset_by_id, raw_by_id,
                     target_cycle=None, last_training_cycle=None):
    training = A._select_training(
        named_set, accepted, reset_by_id,
        target_cycle=target_cycle,
        last_training_cycle=last_training_cycle)
    provenance = training[3]
    rows = [raw_by_id[index]
            for index in provenance["training_RESET_cycle_ids"]]
    return training, rows


def _predict_record(row, bundle, refinement, spatial, base_params,
                    reset_mapping, split, provenance, eligible_ids):
    prediction = simulate_with_terminal_refinement(
        row, bundle, refinement, base_params, reset_mapping, spatial)
    record = A._prediction_record(
        row, prediction, split, "same_protocol", provenance,
        eligible_ids, {})
    record["terminal_refinement"] = asdict(refinement)
    return record, prediction


def _run_loo(raw_pairs, raw_by_id, named_set, accepted, reset_by_id,
             spatial, base_params, reset_mapping):
    records, predictions, refinements = [], {}, []
    eligible_ids = {row["cycle"] for row in accepted}
    for position, row in enumerate(raw_pairs):
        training, training_rows = _training_inputs(
            named_set, accepted, reset_by_id, raw_by_id,
            target_cycle=row["cycle"])
        bundle = V12._fit_bundle(*training[:3], base_params)
        refinement = fit_terminal_reset_transition(
            bundle, training_rows, base_params, reset_mapping,
            spatial=spatial)
        provenance = {
            **training[3],
            "terminal_refinement_training_cycle_ids": (
                training[3]["training_RESET_cycle_ids"]),
            "terminal_refinement_target_current_used": False,
        }
        record, prediction = _predict_record(
            row, bundle, refinement, spatial, base_params, reset_mapping,
            "leave_one_cycle_out", provenance, eligible_ids)
        records.append(record)
        predictions[("leave_one_cycle_out", row["cycle"],
                     "same_protocol")] = prediction
        refinements.append({"target_cycle_id": row["cycle"],
                            **asdict(refinement)})
        if (position + 1) % 5 == 0 or position + 1 == len(raw_pairs):
            print(f"RESET terminal-refined LOO {position + 1}/{len(raw_pairs)}",
                  flush=True)
    return records, predictions, refinements


def _run_chronological(raw_pairs, raw_by_id, named_set, accepted,
                       reset_by_id, spatial, base_params, reset_mapping):
    training, training_rows = _training_inputs(
        named_set, accepted, reset_by_id, raw_by_id,
        last_training_cycle=LAST_TRAINING_CYCLE)
    bundle = V12._fit_bundle(*training[:3], base_params)
    refinement = fit_terminal_reset_transition(
        bundle, training_rows, base_params, reset_mapping, spatial=spatial)
    provenance = {
        **training[3],
        "reference_calibration_status": "ok",
        "reference_calibration_reason": None,
        "terminal_refinement_training_cycle_ids": (
            training[3]["training_RESET_cycle_ids"]),
        "terminal_refinement_target_current_used": False,
    }
    eligible_ids = {row["cycle"] for row in accepted}
    records, predictions = [], {}
    targets = [row for row in raw_pairs
               if row["cycle"] > LAST_TRAINING_CYCLE]
    for position, row in enumerate(targets):
        record, prediction = _predict_record(
            row, bundle, refinement, spatial, base_params, reset_mapping,
            "chronological_first35_to_later", provenance, eligible_ids)
        records.append(record)
        predictions[("chronological_first35_to_later", row["cycle"],
                     "same_protocol")] = prediction
        if (position + 1) % 6 == 0 or position + 1 == len(targets):
            print(f"RESET terminal-refined chronological "
                  f"{position + 1}/{len(targets)}", flush=True)
    return records, predictions, refinement, provenance


def _region_scores(raw_pairs, predictions, split="leave_one_cycle_out"):
    regions = {
        "outward_low_0p10_to_0p60V": ("outward", 0.10, 0.60),
        "outward_plateau_0p70_to_1p40V": ("outward", 0.70, 1.40),
        "outward_high_1p50_to_1p70V": ("outward", 1.50, 1.70),
        "return_0p10_to_1p70V": ("return", 0.10, 1.70),
    }
    output = {}
    for name, (branch, lo, hi) in regions.items():
        residuals = []
        per_cycle = []
        for row in raw_pairs:
            key = (split, row["cycle"], "same_protocol")
            if key not in predictions:
                continue
            voltage = np.asarray(row["reset_voltage"], dtype=float)
            observed = np.abs(np.asarray(row["reset_current"], dtype=float))
            predicted = np.abs(np.asarray(
                predictions[key]["reset"]["Is"], dtype=float))
            apex = int(np.argmax(np.abs(voltage)))
            index = np.arange(len(voltage))
            side = index <= apex if branch == "outward" else index >= apex
            mask = (
                side & (np.abs(voltage) >= lo - 1e-10)
                & (np.abs(voltage) <= hi + 1e-10)
                & np.isfinite(observed) & np.isfinite(predicted)
                & (observed >= 1e-12) & (predicted > 0.0)
            )
            local = np.log10(predicted[mask]) - np.log10(observed[mask])
            residuals.extend(local.tolist())
            per_cycle.append(float(np.median(local)))
        residuals = np.asarray(residuals)
        per_cycle = np.asarray(per_cycle)
        output[name] = {
            "n_points": int(len(residuals)),
            "rmse_decade": float(np.sqrt(np.mean(residuals ** 2))),
            "median_signed_decade": float(np.median(residuals)),
            "median_absolute_decade": float(np.median(np.abs(residuals))),
            "median_predicted_over_observed": float(10.0 ** np.median(residuals)),
            "cycles_with_negative_median_residual": int(np.count_nonzero(
                per_cycle < 0.0)),
            "n_cycles": int(len(per_cycle)),
        }
    return output


def _median_curves(raw_pairs, predictions):
    observed, predicted = [], []
    for row in raw_pairs:
        result = predictions[("leave_one_cycle_out", row["cycle"],
                              "same_protocol")]["reset"]
        observed.append((row["reset_voltage"], row["reset_current"]))
        predicted.append((result["Vs"], result["Is"]))
    return observed, predicted


def _plot(path, raw_pairs, predictions, report):
    observed, predicted = _median_curves(raw_pairs, predictions)
    baseline_archive = np.load(
        ROOT / "analysis" / PC.CURVES_NAME)
    baseline = []
    for row in raw_pairs:
        prefix = (
            f"leave_one_cycle_out_same_protocol_cycle_{row['cycle']}_reset")
        baseline.append((baseline_archive[f"{prefix}_voltage"],
                         baseline_archive[f"{prefix}_current"]))
    grid = np.arange(0.10, 1.7001, 0.02)
    measured, low, high = V5._median_branch(observed, grid, True)
    original, _, _ = V5._median_branch(baseline, grid, True)
    refined, _, _ = V5._median_branch(predicted, grid, True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Unicode MS", "Hiragino Sans GB", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    axis = axes[0]
    axis.fill_between(grid, low, high, color="#9aa0a6", alpha=0.18,
                      label="实测10–90%")
    axis.plot(grid, measured, color="#3f4a59", lw=2.2, label="实测中位线")
    axis.plot(grid, original, color="#d55e00", lw=1.8, ls="--",
              label="原标定")
    axis.plot(grid, refined, color="#1769aa", lw=2.3,
              label="端口电流重标定")
    axis.axvspan(0.70, 1.40, color="#7a5195", alpha=0.08)
    axis.set_yscale("log")
    axis.set_xlabel(r"$|V_{app}|$ (V)")
    axis.set_ylabel(r"$|I|$ (A)")
    axis.set_title("a  RESET去程平台", loc="left", fontweight="semibold")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=9)

    axis = axes[1]
    base_regions = report["baseline_postconditioning_region_scores"]
    new_regions = report["refined_region_scores"]
    keys = list(new_regions)
    labels = ["低压\n0.10–0.60", "平台\n0.70–1.40",
              "高压\n1.50–1.70", "回程\n0.10–1.70"]
    x = np.arange(len(keys))
    axis.bar(x - 0.18, [base_regions[k]["rmse_decade"] for k in keys],
             width=0.36, color="#d55e00", label="原标定")
    axis.bar(x + 0.18, [new_regions[k]["rmse_decade"] for k in keys],
             width=0.36, color="#1769aa", label="端口电流重标定")
    axis.set_xticks(x, labels)
    axis.set_ylabel("RMSE (decade)")
    axis.set_title("b  分区严格留一误差", loc="left", fontweight="semibold")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    figure.suptitle(
        "RESET平台优化：保持模型结构与参数数量不变",
        fontsize=14, fontweight="semibold")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    spatial, cafm_summary = UV._spatial_configuration(args.k, args.seed)
    base_params, reset_mapping = VF._fit_reset_parameters()
    accepted, _ = VF._load_paired_cycles(base_params, reset_mapping)
    accepted = [row for row in accepted if row["cycle"] >= 4]
    raw_pairs = [row for row in A._load_raw_pairs() if row["cycle"] >= 4]
    raw_by_id = {row["cycle"]: row for row in raw_pairs}
    named_set, _ = V12._load_set_cycles()
    named_set = [item for item in named_set if PC._cycle_id(item) >= 4]
    reset_by_id = dict(zip(
        [row["cycle"] for row in accepted],
        V12._reconstruct_resets(accepted, reset_mapping)))
    expected = list(range(4, 54))
    if [row["cycle"] for row in accepted] != expected:
        raise ValueError("Expected every cycle 4--53 to be training-eligible")

    loo, loo_predictions, refinements = _run_loo(
        raw_pairs, raw_by_id, named_set, accepted, reset_by_id,
        spatial, base_params, reset_mapping)
    chronological, chronological_predictions, chronological_refinement, provenance = (
        _run_chronological(
            raw_pairs, raw_by_id, named_set, accepted, reset_by_id,
            spatial, base_params, reset_mapping))
    records = loo + chronological
    predictions = {**loo_predictions, **chronological_predictions}
    groups = {"cycles_4_to_53": expected}

    baseline_summary = json.loads((
        ROOT / "analysis" / PC.SUMMARY_NAME).read_text(encoding="utf-8"))
    baseline_predictions = {}
    baseline_archive = np.load(ROOT / "analysis" / PC.CURVES_NAME)
    for row in raw_pairs:
        cycle = row["cycle"]
        prefix = f"leave_one_cycle_out_same_protocol_cycle_{cycle}_reset"
        baseline_predictions[("leave_one_cycle_out", cycle,
                              "same_protocol")] = {
            "reset": {
                "Vs": baseline_archive[f"{prefix}_voltage"],
                "Is": baseline_archive[f"{prefix}_current"],
            }
        }

    report = {
        "model": "VULCAN-2D unified local-connectivity model",
        "refinement": (
            "training-only direct terminal-current calibration of the existing "
            "RESET E50 and width; no new state, transport term, or parameter"),
        "validation_date": "2026-09-22",
        "data_contract": {
            "valid_cycle_ids": expected,
            "excluded_cycle_ids": [1, 2, 3],
            "exclusion_reason": "experimentally confirmed unsuccessful SET",
            "target_currents_enter_refinement": False,
            "new_independent_test_data": False,
        },
        "spatial_prior": {"K": spatial.k, "seed": args.seed,
                          "cafm_summary": cafm_summary},
        "leave_one_cycle_out": {
            name: A._summarize_group(
                ids, "leave_one_cycle_out", records, predictions, raw_by_id)
            for name, ids in groups.items()
        },
        "chronological_cycles4to35_to_36to53": {
            "last_training_cycle": LAST_TRAINING_CYCLE,
            **provenance,
            "terminal_refinement": asdict(chronological_refinement),
            "groups": {
                name: A._summarize_group(
                    [i for i in ids if i > LAST_TRAINING_CYCLE],
                    "chronological_first35_to_later", records,
                    predictions, raw_by_id)
                for name, ids in groups.items()
            },
        },
        "baseline_postconditioning_scores": baseline_summary[
            "leave_one_cycle_out"]["cycles_4_to_53"]["modes"]
            ["same_protocol"]["scores"],
        "baseline_postconditioning_region_scores": _region_scores(
            raw_pairs, baseline_predictions),
        "refined_region_scores": _region_scores(raw_pairs, predictions),
        "leave_one_out_refinements": refinements,
        "cycle_predictions": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(
        report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    diagnostics = [{
        "cycle_id": row["cycle"], "status": "not_run",
        "reason": "Not needed for terminal-refinement score",
    } for row in raw_pairs]
    csv_rows = list(A._csv_rows(records, diagnostics))
    fieldnames = list(dict.fromkeys(key for row in csv_rows for key in row))
    csv_path = args.output_dir / CYCLES_NAME
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    curves_path = args.output_dir / CURVES_NAME
    A._save_curves(curves_path, raw_pairs, predictions)
    figure_path = args.output_dir / "figures" / FIGURE_NAME
    _plot(figure_path, raw_pairs, predictions, report)
    print(json.dumps({
        "summary": str(summary_path),
        "cycles": str(csv_path),
        "curves": str(curves_path),
        "figure": str(figure_path),
        "refined_scores": report["leave_one_cycle_out"]
            ["cycles_4_to_53"]["modes"]["same_protocol"]["scores"],
        "baseline_scores": report["baseline_postconditioning_scores"],
        "refined_regions": report["refined_region_scores"],
    }, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
