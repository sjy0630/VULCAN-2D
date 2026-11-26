"""Compare saved conditional SET curves; no ODE, KCL, inversion or fitting."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, MultipleLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SINH = ROOT / "analysis/soft_drive_structure_2026-11-03"
M1 = ROOT / "analysis/reversible_family_2026-11-03"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-authorized", action="store_true")
    args = parser.parse_args()
    if not args.run_authorized:
        raise SystemExit("Root authorization is required to render the terminal validated comparison.")
    outputs = {ext: HERE / f"cooperative_set_comparison.{ext}"
               for ext in ("png", "pdf", "metadata.json")}
    if any(p.exists() for p in outputs.values()):
        raise FileExistsError("Do not overwrite a completed or partial comparison")
    paths = [SINH / "sinh_field_probe.json", SINH / "sinh_field_probe.npz",
             M1 / "finite_family_fit.json", M1 / "final_refined.json", M1 / "final_refined.npz",
             HERE / "finite_family_fit.json", HERE / "final_refined.json", HERE / "final_refined.npz",
             Path(__file__)]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError("All three complete refined model artifacts are required")
    source_hashes = {str(p): sha(p) for p in paths}
    old_report = json.loads(paths[0].read_text())
    old_refined = next(r for r in old_report["runs"] if r["precision"]["name"] == "refined")
    reports, summaries = {}, {}
    for name, base in (("m1_linear_recovery", M1), ("m2_fixed_cooperative_recovery", HERE)):
        reports[name] = json.loads((base / "finite_family_fit.json").read_text())
        summaries[name] = json.loads((base / "final_refined.json").read_text())
        r, s = reports[name], summaries[name]
        if r.get("status") != "completed_bounded_attempt_no_global_or_full_model_claim" or r.get("numerical_validation_passed") is not True:
            raise ValueError(f"{name}: terminal numerical validation is required")
        if not r.get("source_hashes_still_match") or not all(r["source_hashes_still_match"].values()):
            raise ValueError(f"{name}: fit report did not certify intact sources")
        if s.get("status") != "completed" or s.get("precision", {}).get("name") != "refined":
            raise ValueError(f"{name}: incomplete refined trajectory")
    # Independently rehash declared calculation sources, without importing a kernel.
    declared_source_checks = {}
    hash_cache = dict(source_hashes)
    for name, report in (("prior_sinh", old_report), *reports.items()):
        mapping = report.get("source_sha256", {})
        if not mapping:
            raise ValueError(f"{name}: missing source hash mapping")
        declared_source_checks[name] = {}
        for raw, expected in mapping.items():
            actual = hash_cache.setdefault(raw, sha(Path(raw))) if raw not in hash_cache else hash_cache[raw]
            passed = actual == expected
            declared_source_checks[name][raw] = passed
            if not passed:
                raise ValueError(f"{name}: changed calculation source {raw}")
    with np.load(paths[1], allow_pickle=False) as archive:
        voltage = archive["command_voltage_v"].copy()
        observed_signed = archive["observed_current_a"].copy()
        mask = archive["original_scoring_mask"].copy()
        predictions = {"prior_sinh": archive["refined_current_a"].copy()}
    saved_errors = {}
    for name, base in (("m1_linear_recovery", M1), ("m2_fixed_cooperative_recovery", HERE)):
        with np.load(base / "final_refined.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(voltage, archive["command_voltage_v"])
            np.testing.assert_array_equal(observed_signed, archive["observed_current_a"])
            np.testing.assert_array_equal(mask, archive["original_scoring_mask"])
            predictions[name] = archive["current_a"].copy()
            saved_errors[name] = archive["log10_current_residual"].copy()
    observed = abs(observed_signed)
    outward = np.arange(len(voltage)) <= np.argmax(voltage)
    counts = (len(voltage), int(mask.sum()), int(np.sum(mask & outward)), int(np.sum(mask & ~outward)))
    if counts != (503, 500, 250, 250) or mask.dtype != np.dtype(bool):
        raise ValueError("Changed original 503 native / 500 scored / 250+250 branch mask")
    for name, values in (("observed", observed), *predictions.items()):
        if not np.all(np.isfinite(values[mask])) or not np.all(values[mask] > 0):
            raise ValueError(f"{name}: log plotting would require forbidden point removal or clipping")
    if not np.all((voltage[mask] >= 0) & (voltage[mask] <= 5)):
        raise ValueError("Declared voltage limits would omit an original scored sample")
    groups = {"all500": mask, "outward250": mask & outward, "return250": mask & ~outward}
    metrics = {}
    for name, current in predictions.items():
        metrics[name] = {group: float(np.sqrt(np.mean(np.log10(current[use] / observed[use]) ** 2)))
                         for group, use in groups.items()}
        if name != "prior_sinh":
            np.testing.assert_allclose(np.log10(current[mask] / observed[mask]), saved_errors[name][mask], rtol=0, atol=1e-12)
            np.testing.assert_allclose(metrics[name]["all500"], reports[name]["verified_conditional_RMSE_decade"], rtol=0, atol=1e-12)
            for group in groups:
                np.testing.assert_allclose(metrics[name][group], summaries[name]["metrics"][group]["RMSE_decade"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(metrics["prior_sinh"]["all500"], reports[name]["plan"]["prior_conditional_500point_RMSE_decade"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(metrics["prior_sinh"]["all500"], old_refined["metrics"]["all500"]["RMSE_decade"], rtol=0, atol=1e-12)
    m2 = predictions["m2_fixed_cooperative_recovery"]
    outward_peak_index = int(np.flatnonzero(outward & np.isclose(voltage, 5, rtol=0, atol=1e-7))[0])
    high_diagnostic = {
        "outward_5v_observed_current_a": float(observed[outward_peak_index]),
        "outward_5v_m2_current_a": float(m2[outward_peak_index]),
        "m2_outward_peak_command_v": float(voltage[np.flatnonzero(outward)[np.argmax(m2[outward])]]),
        "scope": "Saved-curve descriptive values only; no new physical calculation",
    }
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 10.5, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "-",
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "text.usetex": False,
    })
    all_values = [observed, *predictions.values()]
    ymin = 0.7 * min(float(np.min(v[mask])) for v in all_values)
    ymax = 1.4 * max(float(np.max(v[mask])) for v in all_values)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, branch, title, group in zip(
            axes, (outward, ~outward), ("a  Outward SET: 0 to 5 V", "b  Return SET: 5 to 0 V"),
            ("outward250", "return250")):
        use = mask & branch
        ax.plot(voltage[use], predictions["m1_linear_recovery"][use], color="#9AA5B1",
                linestyle="-.", linewidth=0.95, alpha=0.75, zorder=1,
                label=f'Linear recovery (m=1): {metrics["m1_linear_recovery"]["all500"]:.3f} dec')
        ax.plot(voltage[use], predictions["prior_sinh"][use], color="#0072B2",
                linestyle="--", linewidth=1.35, zorder=2,
                label=f'Prior sinh probe: {metrics["prior_sinh"]["all500"]:.3f} dec')
        ax.plot(voltage[use], m2[use], color="#D55E00", linestyle="-", linewidth=2.0, zorder=3,
                label=f'Fixed m=2 conditional fit: {metrics["m2_fixed_cooperative_recovery"]["all500"]:.3f} dec')
        ax.scatter(voltage[use], observed[use], s=10, facecolors="none", edgecolors="#222222",
                   linewidths=0.6, label="Observed cycle 4 (all 500 scored points)", zorder=4)
        scores = [metrics[name][group] for name in ("prior_sinh", "m1_linear_recovery", "m2_fixed_cooperative_recovery")]
        panel_title = title + " (250 points)\nRMSE (sinh / m=1 / m=2): " + " / ".join(f"{x:.3f}" for x in scores) + " dec"
        ax.set(xlabel="Applied command voltage (V)", title=panel_title,
               xlim=(0, 5), yscale="log", ylim=(ymin, ymax))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=20))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="both", which="both", direction="out")
    axes[0].set_ylabel("Current magnitude (A; logarithmic scale)")
    axes[1].invert_xaxis()
    # Point out the remaining high-field defect, in empty space below the curves.
    annotation = ("m=2 high-V downturn remains\n"
                  f'5 V: {m2[outward_peak_index]*1e6:.1f} vs {observed[outward_peak_index]*1e6:.1f} microampere (observed)')
    axes[0].annotate(annotation, xy=(5.0, m2[outward_peak_index]), xytext=(2.15, 5e-7),
                     fontsize=8, color="#814000", ha="left", va="center",
                     arrowprops={"arrowstyle": "->", "color": "#814000", "lw": 0.85,
                                 "connectionstyle": "arc3,rad=0.08"})
    fig.suptitle("Cycle 4 SET: conditional fit with a fixed m=2 recovery hypothesis", fontsize=14, y=0.99)
    handles, labels = axes[0].get_legend_handles_labels()
    order = (3, 1, 0, 2)
    fig.legend([handles[i] for i in order], [labels[i] for i in order], ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 0.94), fontsize=9,
               handlelength=2.7, columnspacing=2.0)
    fig.text(0.5, 0.015,
             "All 500 original scored samples, including the low-current ends, are retained; initial states are conditioned at outward 0.2 V.\n"
             "m=1 and m=2 each fit 3 kinetic parameters; selecting m=2 is an additional discrete model-structure choice.\n"
             "These are same-cycle training errors, not holdout or full unified 1R/1T1R validation; high-V and return-shape errors remain.",
             ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.15, 1, 0.83), w_pad=2.0)
    if any(sha(p) != source_hashes[str(p)] for p in paths):
        raise ValueError("An input changed during preparation; refuse mixed-state figure")
    fig.savefig(outputs["png"], dpi=300, bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "status": "saved_verified_m1_m2_conditional_comparison",
        "scope": "same-cycle 4 SET conditional training comparison; m=2 is a discrete structure choice, not a measured microscopic exponent",
        "source_sha256": source_hashes, "declared_calculation_sources_rehashed": declared_source_checks,
        "source_verification": "All report-declared calculation source hashes independently matched at rendering",
        "figure_style": "academic-plotting numeric-data workflow; Okabe-Ito blue/vermilion, pale thin m1, all observations, vector PDF and 300 dpi PNG",
        "counts": {"native": 503, "original_scored": 500, "outward": 250, "return": 250, "newly_removed_scored_points": 0},
        "scoring_mask": "Exact prior 500-point saved mask; no new current threshold, clipping, smoothing or point removal",
        "score_unit": "decade of log10 current ratio", "RMSE": metrics,
        "new_high_voltage_mismatch": high_diagnostic,
        "axis_limits": {"command_v": [0, 5], "current_a": [ymin, ymax], "both_panels_log_current": True},
        "m2_outperforms_sinh_all500": metrics["m2_fixed_cooperative_recovery"]["all500"] < metrics["prior_sinh"]["all500"],
        "m2_outperforms_sinh_return250": metrics["m2_fixed_cooperative_recovery"]["return250"] < metrics["prior_sinh"]["return250"],
        "kinetic_parameters_per_reversible_fit": 3, "m2_discrete_structure_selection": True,
        "parameters": {name: s["parameters"] for name, s in summaries.items()},
        "limitations": "Same-data model selection/calibration; no holdout, RESET, bare-device or complete chronological unified-model validation",
        "outputs": {ext: {"path": str(outputs[ext]), "sha256": sha(outputs[ext])} for ext in ("png", "pdf")},
    }
    outputs["metadata.json"].write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": metadata["status"], "RMSE": metrics,
                      "high_voltage_mismatch": high_diagnostic,
                      "outputs": {ext: str(p) for ext, p in outputs.items()}}, indent=2))


if __name__ == "__main__":
    main()
