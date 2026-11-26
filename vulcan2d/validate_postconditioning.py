"""Refit the unified VULCAN-2D model on experimentally valid cycles 4--53.

Cycles 1--3 are retained in the source workbooks but excluded from calibration
and scoring because the experimental team confirmed that SET did not succeed.
Every leave-one-cycle-out target is predicted from a fresh fit to the other 49
valid cycles.  A chronological fit on cycles 4--35 is also tested on 36--53.
"""
from __future__ import annotations

import argparse
import csv
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

from . import model_unified as U
from . import validate_full_cycle as VF
from . import validate_unified as UV
from . import validate_v12 as V12
from . import validate_v13 as A
from . import validate_v5 as V5


ROOT = Path(__file__).resolve().parents[1]
FIRST_VALID_CYCLE = 4
LAST_TRAINING_CYCLE = 35
SUMMARY_NAME = "postconditioning_refit_cycle4plus.json"
CYCLES_NAME = "postconditioning_refit_cycle4plus_cycles.csv"
CURVES_NAME = "postconditioning_refit_cycle4plus_curves.npz"
FIGURE_NAME = "25_postconditioning_refit_cycle4plus.png"


def _cycle_id(named_cycle):
    return int(named_cycle[0].upper().replace("S", ""))


def _calibration_summary(bundle):
    return {
        "SET_pre_beta_per_sqrt_V": (
            bundle["set_transport"].beta_pre_per_sqrt_v),
        "SET_connected_exponent": (
            bundle["set_transport"].connected_exponent),
        "SET_pre_shape_RMSE_decade": (
            bundle["set_transport"].pre_shape_rmse_decade),
        "SET_connected_shape_RMSE_decade": (
            bundle["set_transport"].connected_shape_rmse_decade),
        "SET_E50_V2": bundle["set_transition"].e50_v2,
        "SET_width_V2": bundle["set_transition"].width_v2,
        "SET_state_identification_RMS": (
            bundle["set_transition"].residual_rms_state),
        "SET_RESET_polarity_amplitude_factor": (
            bundle["link"].retention_factor),
        "SET_RESET_link_sigma_decade": (
            bundle["link"].residual_sigma_decade),
        "RESET_PF_beta_per_sqrt_V": (
            bundle["reset_transport"].beta_per_sqrt_v),
        "RESET_E50_MV_cm": bundle["reset_transition"].e50_mvc,
        "RESET_width_MV_cm": bundle["reset_transition"].width_mvc,
        "RESET_state_identification_RMS": (
            bundle["reset_transition"].residual_rms_state),
        "RESET_endpoint_sigma_decade": (
            bundle["endpoint"].residual_sigma_decade),
    }


def _old_scores():
    path = ROOT / "analysis" / "unified_local_connectivity_validation.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    return report["leave_one_cycle_out"]["all_pairs"]["modes"]


def _plot_median(axis, observed, predicted, vmax, title):
    grid = np.arange(0.1, vmax + 1e-9, 0.02)
    for forward, linestyle in ((True, "-"), (False, "--")):
        measured, low, high = V5._median_branch(observed, grid, forward)
        modeled, _, _ = V5._median_branch(predicted, grid, forward)
        axis.fill_between(
            grid, low, high, color="#9aa0a6", alpha=0.18,
            label="实测10–90%" if forward else None)
        axis.plot(
            grid, measured, color="#4b5563", linestyle=linestyle,
            linewidth=1.8, label="实测中位线" if forward else None)
        axis.plot(
            grid, modeled, color="#d55e00", linestyle=linestyle,
            linewidth=2.0, label="留一预测中位线" if forward else None)
    axis.set_yscale("log")
    axis.set_xlabel(r"$|V_{app}|$ (V)")
    axis.set_ylabel(r"$|I|$ (A)")
    axis.set_title(title, loc="left", fontweight="semibold")
    axis.grid(alpha=0.22)


def _plot(path, raw_pairs, predictions, report):
    observed_set, predicted_set = [], []
    observed_reset, predicted_reset = [], []
    for row in raw_pairs:
        prediction = predictions[
            ("leave_one_cycle_out", row["cycle"], "same_protocol")]
        observed_set.append((row["set_voltage"], row["set_current"]))
        predicted_set.append((prediction["set"]["Vs"], prediction["set"]["Is"]))
        observed_reset.append((row["reset_voltage"], row["reset_current"]))
        predicted_reset.append((
            prediction["reset"]["Vs"], prediction["reset"]["Is"]))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Unicode MS", "Hiragino Sans GB", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.1))
    _plot_median(axes[0, 0], observed_set, predicted_set, 5.0,
                 "a  SET：第4–53圈留一预测")
    _plot_median(axes[0, 1], observed_reset, predicted_reset, 1.7,
                 "b  RESET：第4–53圈留一预测")
    axes[0, 0].legend(frameon=False, fontsize=9)

    new_scores = report["leave_one_cycle_out"]["cycles_4_to_53"][
        "modes"]["same_protocol"]["scores"]
    old = report["comparison_to_previous_all53"]
    axis = axes[1, 0]
    labels = ["SET\n逐点", "RESET\n逐点", "SET\n中位回线", "RESET\n中位回线"]
    new_values = [
        new_scores["SET_pointwise"]["pooled_rmse_decade"],
        new_scores["RESET_pointwise"]["pooled_rmse_decade"],
        np.mean(list(new_scores["SET_median_loop_rmse_decade"].values())),
        np.mean(list(new_scores["RESET_median_loop_rmse_decade"].values())),
    ]
    x = np.arange(len(labels))
    if old is not None:
        old_same = old["same_protocol"]["scores"]
        old_values = [
            old_same["SET_pointwise"]["pooled_rmse_decade"],
            old_same["RESET_pointwise"]["pooled_rmse_decade"],
            np.mean(list(old_same["SET_median_loop_rmse_decade"].values())),
            np.mean(list(old_same["RESET_median_loop_rmse_decade"].values())),
        ]
        axis.bar(x - 0.18, old_values, width=0.36, color="#9aa0a6",
                 label="旧口径：全部53圈")
        axis.bar(x + 0.18, new_values, width=0.36, color="#1769aa",
                 label="新口径：有效50圈")
    else:
        axis.bar(x, new_values, width=0.55, color="#1769aa",
                 label="有效50圈")
    axis.set_xticks(x, labels)
    axis.set_ylabel("RMSE (decade)")
    axis.set_title("c  排除失败SET后的误差", loc="left", fontweight="semibold")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, fontsize=9)

    axes[1, 1].axis("off")
    chronology = report["chronological_cycles4to35_to_36to53"][
        "groups"]["cycles_4_to_53"]["modes"]["same_protocol"]["scores"]
    message = (
        "数据口径\n"
        "• 第1–3圈：实验确认SET未成功，不参与拟合与评分\n"
        "• 第4–53圈：50个有效完整循环\n"
        "• 留一验证：每个目标均由其他49圈重新拟合\n\n"
        "留一预测（第4–53圈）\n"
        f"SET逐点RMSE：{new_values[0]:.3f} decade\n"
        f"RESET逐点RMSE：{new_values[1]:.3f} decade\n\n"
        "顺序预测（第4–35圈 → 第36–53圈）\n"
        f"SET逐点RMSE：{chronology['SET_pointwise']['pooled_rmse_decade']:.3f} decade\n"
        f"RESET逐点RMSE：{chronology['RESET_pointwise']['pooled_rmse_decade']:.3f} decade\n\n"
        "这些是电压输入下的严格目标排除误差，\n"
        "不是用目标电流反演状态后的回顾性拟合误差。"
    )
    axes[1, 1].text(0.02, 0.98, message, va="top", fontsize=10.5,
                    linespacing=1.45)
    figure.suptitle(
        "VULCAN-2D 后成形有效循环重新拟合（第4–53圈）",
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
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Regenerate the figure from saved JSON/NPZ without refitting")
    args = parser.parse_args(argv)

    if args.plot_only:
        summary_path = args.output_dir / SUMMARY_NAME
        curves_path = args.output_dir / CURVES_NAME
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        raw_pairs = [
            row for row in A._load_raw_pairs()
            if row["cycle"] >= FIRST_VALID_CYCLE]
        archive = np.load(curves_path)
        predictions = {}
        for row in raw_pairs:
            index = row["cycle"]
            prediction = {}
            for phase in ("set", "reset"):
                prefix = (
                    f"leave_one_cycle_out_same_protocol_cycle_{index}_{phase}")
                prediction[phase] = {
                    "Vs": archive[f"{prefix}_voltage"],
                    "Is": archive[f"{prefix}_current"],
                }
            predictions[("leave_one_cycle_out", index, "same_protocol")] = prediction
        figure_path = args.output_dir / "figures" / FIGURE_NAME
        _plot(figure_path, raw_pairs, predictions, report)
        print(f"wrote {figure_path}")
        return report

    spatial, cafm_summary = UV._spatial_configuration(args.k, args.seed)
    base_params, reset_mapping = VF._fit_reset_parameters()
    accepted_all, rejected_all = VF._load_paired_cycles(
        base_params, reset_mapping)
    raw_all = A._load_raw_pairs()
    raw_pairs = [row for row in raw_all if row["cycle"] >= FIRST_VALID_CYCLE]
    accepted = [
        row for row in accepted_all if row["cycle"] >= FIRST_VALID_CYCLE]
    named_set_all, _ = V12._load_set_cycles()
    named_set = [
        item for item in named_set_all if _cycle_id(item) >= FIRST_VALID_CYCLE]
    reset_by_id = dict(zip(
        [row["cycle"] for row in accepted],
        V12._reconstruct_resets(accepted, reset_mapping)))

    expected = list(range(FIRST_VALID_CYCLE, 54))
    actual = [row["cycle"] for row in raw_pairs]
    if actual != expected:
        raise ValueError(f"Expected consecutive cycles 4--53, received {actual}")
    if [row["cycle"] for row in accepted] != expected:
        raise ValueError("Every cycle 4--53 must pass the existing training rules")
    if [int(name.upper().replace("S", "")) for name, _ in named_set] != expected:
        raise ValueError("SET cycle labels 4--53 are incomplete")

    full_bundle = V12._fit_bundle(
        named_set, accepted,
        [reset_by_id[index] for index in expected], base_params)

    original_simulator = A._simulate
    A._simulate = UV._model_adapter(spatial)
    try:
        loo, loo_predictions, _, _ = A._evaluate(
            raw_pairs, named_set, accepted, reset_by_id,
            base_params, reset_mapping, {})
        chronological, chronological_predictions, reference, provenance = A._evaluate(
            raw_pairs, named_set, accepted, reset_by_id,
            base_params, reset_mapping, {},
            last_training_cycle=LAST_TRAINING_CYCLE)
    finally:
        A._simulate = original_simulator

    records = loo + chronological
    predictions = {**loo_predictions, **chronological_predictions}
    raw_by_id = {row["cycle"]: row for row in raw_pairs}
    groups = {"cycles_4_to_53": expected}
    diagnostics = A._retrospective_diagnostics(
        raw_pairs, reference, base_params, reset_mapping,
        reference_failure_reason=provenance.get(
            "reference_calibration_reason"))
    old = _old_scores()
    report = {
        "model": "VULCAN-2D unified local-connectivity quasi-static model",
        "validation_date": "2026-09-22",
        "data_contract": {
            "source_workbooks_preserved": True,
            "excluded_cycle_ids": [1, 2, 3],
            "exclusion_reason": (
                "Experimental team and advisor confirmed unsuccessful SET; "
                "the following RESET traces are not valid nominal SET-RESET cycles"),
            "first_valid_cycle": FIRST_VALID_CYCLE,
            "valid_cycle_ids": expected,
            "n_valid_cycles": len(expected),
            "gate_voltage_V": VF.GATE_VOLTAGE,
            "protocol": "+5-V SET and -1.7-V RESET DC sweeps",
            "legacy_rejections_before_scope_filter": rejected_all,
        },
        "prediction_contract": {
            "leave_one_cycle_out": (
                "Each target cycle is predicted by refitting all shared laws "
                "to the other 49 valid cycles; target currents are scoring-only"),
            "chronological": (
                "Cycles 4--35 fit all laws; cycles 36--53 are predicted "
                "without using their currents"),
            "new_independent_test_data": False,
            "runtime_inputs": (
                "SET voltage waveform, RESET voltage waveform and gate voltage"),
        },
        "spatial_prior": {
            "K": spatial.k,
            "weights": spatial.weights.tolist(),
            "seed": args.seed,
            "cafm_summary": cafm_summary,
        },
        "full_50_cycle_calibration": _calibration_summary(full_bundle),
        "leave_one_cycle_out": {
            name: A._summarize_group(
                ids, "leave_one_cycle_out", records, predictions, raw_by_id)
            for name, ids in groups.items()
        },
        "chronological_cycles4to35_to_36to53": {
            "last_training_cycle": LAST_TRAINING_CYCLE,
            **provenance,
            "groups": {
                name: A._summarize_group(
                    [index for index in ids if index > LAST_TRAINING_CYCLE],
                    "chronological_first35_to_later", records,
                    predictions, raw_by_id)
                for name, ids in groups.items()
            },
        },
        "comparison_to_previous_all53": old,
        "cycle_predictions": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(
        report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    rows = list(A._csv_rows(records, diagnostics))
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    csv_path = args.output_dir / CYCLES_NAME
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    curves_path = args.output_dir / CURVES_NAME
    A._save_curves(curves_path, raw_pairs, predictions)
    figure_path = args.output_dir / "figures" / FIGURE_NAME
    _plot(figure_path, raw_pairs, predictions, report)
    print(json.dumps({
        "summary": str(summary_path),
        "cycles": str(csv_path),
        "curves": str(curves_path),
        "figure": str(figure_path),
        "leave_one_out_same_protocol": report[
            "leave_one_cycle_out"]["cycles_4_to_53"]["modes"]
            ["same_protocol"]["scores"],
        "chronological_same_protocol": report[
            "chronological_cycles4to35_to_36to53"]["groups"]
            ["cycles_4_to_53"]["modes"]["same_protocol"]["scores"],
        "full_50_cycle_calibration": report["full_50_cycle_calibration"],
    }, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
