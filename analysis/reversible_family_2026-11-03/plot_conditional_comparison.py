"""Plot saved, numerically verified conditional fits; never rerun a model.

Prepared before fit completion. Invoke only after root confirms terminal fit
and numerical validation, with --run-authorized. Every original scored point
is retained. No new observation/current threshold is introduced.
"""
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
OLD = ROOT / "analysis/soft_drive_structure_2026-11-03"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-authorized", action="store_true")
    args = parser.parse_args()
    if not args.run_authorized:
        raise SystemExit("Prepared only: root must confirm terminal validated fit before plotting.")
    outputs = {ext: HERE / f"conditional_set_comparison.{ext}"
               for ext in ("png", "pdf", "metadata.json")}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("Preserve previous figure artifacts; no automatic overwrite")
    paths = [HERE / "finite_family_fit.json", HERE / "final_refined.json",
             HERE / "final_refined.npz", OLD / "sinh_field_probe.json",
             OLD / "sinh_field_probe.npz", Path(__file__)]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("Complete final_refined and terminal fit artifacts are required")
    hashes = {str(path): sha(path) for path in paths}
    result = json.loads(paths[0].read_text())
    if result.get("status") != "completed_bounded_attempt_no_global_or_full_model_claim":
        raise ValueError("Fit is not in the declared successful terminal state")
    if result.get("numerical_validation_passed") is not True:
        raise ValueError("No new-model figure without complete numerical validation")
    if not result.get("source_hashes_still_match") or not all(result["source_hashes_still_match"].values()):
        raise ValueError("Terminal report does not confirm intact calculation sources")
    refined = json.loads(paths[1].read_text())
    if refined.get("status") != "completed" or refined.get("precision", {}).get("name") != "refined":
        raise ValueError("Final refined trajectory is incomplete or mislabeled")
    old_report = json.loads(paths[3].read_text())
    old_refined = next(r for r in old_report["runs"] if r["precision"]["name"] == "refined")
    with np.load(paths[4], allow_pickle=False) as archive:
        voltage = archive["command_voltage_v"].copy()
        observed_signed = archive["observed_current_a"].copy()
        mask = archive["original_scoring_mask"].copy()
        baseline = archive["refined_current_a"].copy()
    with np.load(paths[2], allow_pickle=False) as archive:
        np.testing.assert_array_equal(voltage, archive["command_voltage_v"])
        np.testing.assert_array_equal(observed_signed, archive["observed_current_a"])
        np.testing.assert_array_equal(mask, archive["original_scoring_mask"])
        candidate = archive["current_a"].copy()
        saved_error = archive["log10_current_residual"].copy()
    outward = np.arange(len(voltage)) <= np.argmax(voltage)
    counts = (len(voltage), int(mask.sum()), int(np.sum(mask & outward)), int(np.sum(mask & ~outward)))
    if counts != (503, 500, 250, 250) or mask.dtype != np.dtype(bool):
        raise ValueError("The original native/scoring/branch sample counts changed")
    observed = abs(observed_signed)
    for name, current in (("observed", observed), ("baseline", baseline), ("candidate", candidate)):
        if not np.all(np.isfinite(current[mask])) or not np.all(current[mask] > 0):
            raise ValueError(f"Cannot log-plot all original scored {name} points without clipping")
    if not np.all((voltage[mask] >= 0) & (voltage[mask] <= 5)):
        raise ValueError("Fixed voltage range would clip scored data")
    groups = {"all500": mask, "outward250": mask & outward, "return250": mask & ~outward}
    metrics = {}
    for name, current in (("prior_sinh", baseline), ("reversible_conditional_fit", candidate)):
        error = np.log10(current[mask] / observed[mask])
        if name == "reversible_conditional_fit":
            np.testing.assert_allclose(error, saved_error[mask], rtol=0, atol=1e-12)
        metrics[name] = {group: float(np.sqrt(np.mean(np.log10(current[use] / observed[use]) ** 2)))
                         for group, use in groups.items()}
    np.testing.assert_allclose(metrics["prior_sinh"]["all500"],
                               old_refined["metrics"]["all500"]["RMSE_decade"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(metrics["prior_sinh"]["all500"],
                               result["plan"]["prior_conditional_500point_RMSE_decade"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(metrics["reversible_conditional_fit"]["all500"],
                               result["verified_conditional_RMSE_decade"], rtol=0, atol=1e-12)
    for group in groups:
        np.testing.assert_allclose(metrics["reversible_conditional_fit"][group],
                                   refined["metrics"][group]["RMSE_decade"], rtol=0, atol=1e-12)

    # Okabe-Ito palette plus different line styles; all data remain vector in PDF.
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "-",
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "text.usetex": False,
    })
    ymin = 0.7 * min(float(np.min(v[mask])) for v in (observed, baseline, candidate))
    ymax = 1.4 * max(float(np.max(v[mask])) for v in (observed, baseline, candidate))
    if not 0 < ymin < ymax:
        raise ValueError("Invalid full-data logarithmic axis limits")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.25), sharey=True)
    old_score = metrics["prior_sinh"]["all500"]
    new_score = metrics["reversible_conditional_fit"]["all500"]
    comparison_note = ("This bounded reversible fit has higher training error than the prior sinh probe."
                       if new_score >= old_score else
                       "This bounded reversible fit has lower training error; this is not independent validation.")
    for ax, branch, title, group in zip(
            axes, (outward, ~outward), ("a  Outward SET: 0 to 5 V", "b  Return SET: 5 to 0 V"),
            ("outward250", "return250")):
        use = mask & branch
        ax.plot(voltage[use], baseline[use], color="#0072B2", linestyle="--", linewidth=1.65,
                label=f"Prior sinh probe: {old_score:.3f} dec", zorder=2)
        ax.plot(voltage[use], candidate[use], color="#D55E00", linestyle="-", linewidth=1.8,
                label=f"Reversible conditional fit: {new_score:.3f} dec", zorder=3)
        ax.scatter(voltage[use], observed[use], s=10, facecolors="none", edgecolors="#222222",
                   linewidths=0.6, label="Observed cycle 4 (all 500 scored points)", zorder=4)
        panel_title = (title + f' (250 points)\nRMSE: {metrics["prior_sinh"][group]:.3f} / '
                       f'{metrics["reversible_conditional_fit"][group]:.3f} dec (prior / reversible)')
        ax.set(xlabel="Applied command voltage (V)", title=panel_title, xlim=(0, 5),
               yscale="log", ylim=(ymin, ymax))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=20))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="both", which="both", direction="out")
    axes[0].set_ylabel("Current magnitude (A; logarithmic scale)")
    axes[1].invert_xaxis()
    fig.suptitle("Cycle 4 SET: conditional single-cycle comparison", fontsize=14, y=0.99)
    handles, labels = axes[0].get_legend_handles_labels()
    order = (2, 0, 1)
    fig.legend([handles[i] for i in order], [labels[i] for i in order], ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 0.925), fontsize=8.5,
               handlelength=2.5, columnspacing=1.3)
    fig.text(0.5, 0.015,
             "Both initial states are conditioned on the measured outward 0.2 V point; all original 500 scored samples are retained.\n"
             "Reversible curve: 3 fitted kinetic parameters. No full 1R/1T1R or holdout validation.\n" + comparison_note,
             ha="center", va="bottom", fontsize=8.3)
    fig.tight_layout(rect=(0, 0.15, 1, 0.86), w_pad=2.0)
    if any(sha(path) != hashes[str(path)] for path in paths):
        raise ValueError("An input changed during figure preparation; do not publish a mixed-state figure")
    fig.savefig(outputs["png"], dpi=300, bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "status": "saved_verified_conditional_comparison",
        "scope": "Same cycle4 SET; observed-conditioned initial states; fitted training curves, not full unified/holdout validation",
        "source_sha256": hashes,
        "figure_style": "academic-plotting data workflow; Okabe-Ito colors; line styles/markers; vector PDF and 300 dpi PNG",
        "scoring_mask": "Exact original saved 500-point mask; no new threshold or truncation",
        "counts": {"native": 503, "original_scored": 500, "outward": 250, "return": 250,
                   "newly_removed_scored_points": 0},
        "score_unit": "decade of log10 current ratio", "RMSE": metrics,
        "improves_prior_training_score": bool(new_score < old_score),
        "training_score_comparison": comparison_note,
        "axis_limits": {"command_v": [0, 5], "current_a": [ymin, ymax], "both_panels_log_current": True},
        "all_original_scored_points_visible": True,
        "new_model_parameters": refined["parameters"],
        "numerical_validation_passed": True,
        "limitations": "Prior probe versus a three-parameter calibrated candidate; lower training error alone is not independent mechanism evidence",
        "outputs": {ext: {"path": str(outputs[ext]), "sha256": sha(outputs[ext])} for ext in ("png", "pdf")},
    }
    outputs["metadata.json"].write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": metadata["status"], "RMSE": metrics,
                      "outputs": {ext: str(path) for ext, path in outputs.items()}}, indent=2))


if __name__ == "__main__":
    main()
