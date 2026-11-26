"""Target-excluding audit of the unified local-connectivity model.

The electrical calibration and scoring contract are intentionally identical
to :mod:`vulcan2d.validate_v13`.  Only the simulator is replaced by the
single-state ``z_k`` implementation.  This tests whether the physical
unification preserves the established terminal prediction, while also
auditing exact SET-to-RESET state continuity and the K>1 spatial reduction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
import pandas as pd

from . import model_unified as U
from . import validate_full_cycle as VF
from . import validate_v12 as V12
from . import validate_v13 as A
from .spatial_prior import SpotAreaDistribution


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_NAME = "unified_local_connectivity_validation.json"
CYCLES_NAME = "unified_local_connectivity_validation_cycles.csv"
CURVES_NAME = "unified_local_connectivity_validation_curves.npz"
FIGURE_NAME = "24_unified_local_connectivity_validation.png"
CAFM_TABLE = ROOT / "analysis" / "afm_spot_distribution.csv"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _spatial_configuration(k, seed):
    frame = pd.read_csv(CAFM_TABLE)
    distribution = SpotAreaDistribution(
        area_m2=frame["projected_area_m2"].to_numpy(float),
        counts=frame["count"].to_numpy(int),
    )
    weights = distribution.coarse_area_weights(k)
    spatial = U.SpatialConfiguration.from_weights(
        weights, seed=seed,
        source=(
            "K equal-count CAFM size strata weighted by projected area; "
            "local electrical order is unmeasured and seeded only for display"
        ),
    )
    return spatial, distribution.summary(k=k)


def _model_adapter(spatial):
    def simulate(*args, **kwargs):
        return U.run_unified_full_cycle(*args, spatial=spatial, **kwargs)
    return simulate


def _representative_reduction_audit(spatial):
    set_forward = np.linspace(0.0, 5.0, 61)
    reset_forward = np.linspace(0.0, -1.7, 61)
    set_voltage = np.r_[set_forward, set_forward[-2::-1]]
    reset_voltage = np.r_[reset_forward, reset_forward[-2::-1]]
    distributed = U.simulate_calibrated_cycle(
        set_voltage, reset_voltage, spatial=spatial)
    mean_field = U.simulate_calibrated_cycle(
        set_voltage, reset_voltage,
        spatial=U.SpatialConfiguration.mean_field())
    scale = np.maximum(np.abs(mean_field["Is"]), 1e-15)
    return {
        "max_terminal_current_absolute_difference_A": float(np.max(
            np.abs(distributed["Is"] - mean_field["Is"]))),
        "max_terminal_current_relative_difference": float(np.max(
            np.abs(distributed["Is"] - mean_field["Is"]) / scale)),
        "max_weighted_state_identity_error": float(np.max(np.abs(
            distributed["zs"]
            - np.sum(distributed["z_local"] * spatial.weights[None, :],
                     axis=1)))),
        "max_set_reset_local_state_handoff_error": float(np.max(np.abs(
            distributed["set"]["z_local"][-1]
            - distributed["reset"]["z_local"][0]))),
        "set_state_monotone_on_outward_branch": bool(np.all(np.diff(
            distributed["set"]["zs"][:len(set_forward)]) >= -2e-12)),
        "reset_state_monotone_on_outward_branch": bool(np.all(np.diff(
            distributed["reset"]["zs"][:len(reset_forward)]) <= 2e-12)),
    }


def _plot(path, raw_by_id, predictions, report):
    candidates = [key for key in predictions
                  if key[0] == "leave_one_cycle_out"
                  and key[2] == "same_protocol"]
    if not candidates:
        return
    key = sorted(candidates, key=lambda item: item[1])[len(candidates) // 2]
    cycle_id = key[1]
    prediction = predictions[key]
    observed = raw_by_id[cycle_id]
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    colors = {"measured": "#777777", "predicted": "#1769aa"}
    for axis, phase, title in (
            (axes[0, 0], "set", "SET: voltage-only prediction"),
            (axes[0, 1], "reset", "RESET: same persistent state")):
        axis.semilogy(
            observed[f"{phase}_voltage"],
            np.maximum(np.abs(observed[f"{phase}_current"]), 1e-13),
            color=colors["measured"], linewidth=1.5, label="measured")
        axis.semilogy(
            prediction[phase]["Vs"],
            np.maximum(np.abs(prediction[phase]["Is"]), 1e-13),
            color=colors["predicted"], linewidth=2.0, label="predicted")
        axis.set_xlabel("Applied voltage (V)")
        axis.set_ylabel("|Current| (A)")
        axis.set_title(title)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)

    x_set = np.linspace(0.0, 1.0, len(prediction["set"]["zs"]))
    x_reset = 1.0 + np.linspace(0.0, 1.0, len(prediction["reset"]["zs"]))
    axis = axes[1, 0]
    for index in range(prediction["set"]["z_local"].shape[1]):
        axis.plot(x_set, prediction["set"]["z_local"][:, index],
                  color="#8bc5c0", alpha=0.45, linewidth=0.9)
        axis.plot(x_reset, prediction["reset"]["z_local"][:, index],
                  color="#ef9a9a", alpha=0.45, linewidth=0.9)
    axis.plot(x_set, prediction["set"]["zs"], color="#00796b",
              linewidth=2.5, label=r"SET $\bar z$")
    axis.plot(x_reset, prediction["reset"]["zs"], color="#c62828",
              linewidth=2.5, label=r"RESET $\bar z$")
    axis.axvline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    axis.set_xlim(0.0, 2.0)
    axis.set_ylim(-0.04, 1.04)
    axis.set_xticks([0.0, 1.0, 2.0], ["SET start", "handoff", "RESET end"])
    axis.set_ylabel("Local connectivity state")
    axis.set_title("One state vector generates both polarities")
    axis.legend(frameon=False, ncol=2)
    axis.grid(alpha=0.2)

    scores = report["leave_one_cycle_out"]["all_pairs"]["modes"]
    same = scores["same_protocol"]["scores"]
    prior = scores["cross_gate_prior"]["scores"]
    audit = report["exact_spatial_reduction_audit"]
    axes[1, 1].axis("off")
    message = (
        "Strict target-excluding audit (all 53 pairs)\n"
        "Target currents are scoring data only\n\n"
        f"Same-protocol SET median-loop RMSE (out/back): "
        f"{same['SET_median_loop_rmse_decade']['forward_decade']:.3f} / "
        f"{same['SET_median_loop_rmse_decade']['return_decade']:.3f} decade\n"
        f"Same-protocol RESET median-loop RMSE (out/back): "
        f"{same['RESET_median_loop_rmse_decade']['forward_decade']:.3f} / "
        f"{same['RESET_median_loop_rmse_decade']['return_decade']:.3f} decade\n"
        f"Cross-gate RESET median-loop RMSE (out/back): "
        f"{prior['RESET_median_loop_rmse_decade']['forward_decade']:.3f} / "
        f"{prior['RESET_median_loop_rmse_decade']['return_decade']:.3f} decade\n\n"
        f"K={report['spatial_prior']['K']} terminal reduction error: "
        f"{audit['max_terminal_current_relative_difference']:.2e}\n"
        f"SET/RESET local-state continuity error: "
        f"{audit['max_set_reset_local_state_handoff_error']:.2e}\n\n"
        "Interpretation: coarse local conductive connectivity;\n"
        "atomic species and local switching order are not identified."
    )
    axes[1, 1].text(0.02, 0.98, message, va="top", fontsize=10.5,
                    linespacing=1.4)
    figure.suptitle(
        f"Unified local-connectivity model — representative cycle {cycle_id}",
        fontsize=14, fontweight="bold")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    spatial, cafm_summary = _spatial_configuration(args.k, args.seed)
    base_params, reset_mapping = VF._fit_reset_parameters()
    accepted, rejected = VF._load_paired_cycles(base_params, reset_mapping)
    raw_pairs = A._load_raw_pairs()
    raw_by_id = {row["cycle"]: row for row in raw_pairs}
    eligible_ids = {row["cycle"] for row in accepted}
    reasons = {row["cycle"]: row["reason"] for row in rejected}
    named_set, _ = V12._load_set_cycles()
    reset_by_id = dict(zip(
        [row["cycle"] for row in accepted],
        V12._reconstruct_resets(accepted, reset_mapping)))

    original_simulator = A._simulate
    A._simulate = _model_adapter(spatial)
    try:
        loo, loo_predictions, _, _ = A._evaluate(
            raw_pairs, named_set, accepted, reset_by_id,
            base_params, reset_mapping, reasons)
        chronological, chronological_predictions, reference, provenance = (
            A._evaluate(
                raw_pairs, named_set, accepted, reset_by_id,
                base_params, reset_mapping, reasons,
                last_training_cycle=35))
    finally:
        A._simulate = original_simulator

    records = loo + chronological
    predictions = {**loo_predictions, **chronological_predictions}
    groups = A._group_ids(raw_pairs, eligible_ids)
    diagnostics = A._retrospective_diagnostics(
        raw_pairs, reference, base_params, reset_mapping,
        reference_failure_reason=provenance.get(
            "reference_calibration_reason"))
    report = {
        "model": "VULCAN-2D unified local-connectivity quasi-static model",
        "validation_date": "2026-09-22",
        "provenance": A._source_metadata(),
        "scope": "1.1-V gate; recorded +5-V SET and -1.7-V RESET DC protocols",
        "prediction_contract": {
            "runtime_inputs": (
                "SET voltage waveform, RESET voltage waveform and gate voltage"),
            "target_currents_enter_simulator": False,
            "training_and_scoring": (
                "same target-excluding rules as the v0.13 all-pairs audit"),
            "state_model": (
                "one persistent z_k vector generates current and evolves "
                "through SET and RESET"),
            "time_model": (
                "quasi-static voltage-domain exposure/internal field; no "
                "pulse-time or dwell-time prediction"),
        },
        "spatial_prior": {
            "K": spatial.k,
            "weights": spatial.weights.tolist(),
            "effective_patch_count": float(
                1.0 / np.sum(spatial.weights ** 2)),
            "set_order": spatial.set_order.tolist(),
            "reset_order": spatial.reset_order.tolist(),
            "order_identified": False,
            "seed": args.seed,
            "source_table": str(CAFM_TABLE.relative_to(ROOT)),
            "source_table_sha256": _sha256(CAFM_TABLE),
            "cafm_summary": cafm_summary,
            "claim_scope": (
                "CAFM constrains projected-area weights only; it does not "
                "identify local conductance, threshold, or defect chemistry"),
        },
        "exact_spatial_reduction_audit": _representative_reduction_audit(
            spatial),
        "data": {
            "all_pairs": len(raw_pairs),
            "legacy_training_eligible_pairs": len(accepted),
            "legacy_rejected_pairs": rejected,
            "groups": groups,
        },
        "leave_one_cycle_out": {
            name: A._summarize_group(
                indices, "leave_one_cycle_out", records,
                predictions, raw_by_id)
            for name, indices in groups.items()
        },
        "chronological_first35_to_later": {
            "last_training_cycle": 35,
            **provenance,
            "groups": {
                name: A._summarize_group(
                    [index for index in indices if index > 35],
                    "chronological_first35_to_later", records,
                    predictions, raw_by_id)
                for name, indices in groups.items()
            },
        },
        "interpretation_boundary": U.claim_scope(),
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
    A._save_curves(args.output_dir / CURVES_NAME, raw_pairs, predictions)
    figure_path = args.output_dir / "figures" / FIGURE_NAME
    _plot(figure_path, raw_by_id, predictions, report)
    print(json.dumps({
        "summary": str(summary_path),
        "cycles": str(csv_path),
        "curves": str(args.output_dir / CURVES_NAME),
        "figure": str(figure_path),
        "all_pairs_same_protocol_scores": report[
            "leave_one_cycle_out"]["all_pairs"]["modes"]
            ["same_protocol"]["scores"],
        "exact_spatial_reduction_audit": report[
            "exact_spatial_reduction_audit"],
    }, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
