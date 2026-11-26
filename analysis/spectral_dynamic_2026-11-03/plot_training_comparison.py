"""Plot frozen full32 training predictions; no simulation or holdout access."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from evaluate_initiation import load_training_predictions, phase_metrics
from joint_problem import SpectralJointProblem, ROOT, HERE, TRAIN


def main():
    report_path = HERE/"initiation_full32_evaluation.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "completed_frozen_full32_training_evaluation_no_holdout_no_refit":
        raise ValueError("Require a completed frozen full32 evaluation")
    if report.get("holdout_arrays_loaded") is not False:
        raise ValueError("Unexpected evaluation scope")
    problem = SpectralJointProblem(lambda q, ron: None, include_reset_initiation=True)
    candidate_path = Path(report["full_native_training_predictions"])
    if hashlib.sha256(candidate_path.read_bytes()).hexdigest() != report["training_predictions_sha256"]:
        raise ValueError("Frozen prediction digest changed")
    baseline_path = ROOT/"analysis/shared_gap_refinement_2026-11-03/block_evaluation.npz"
    sources = {"GapLaw baseline": baseline_path, "Spectral trial": candidate_path}
    predictions = {name: load_training_predictions(path, problem) for name, path in sources.items()}
    colors = {"GapLaw baseline": "#1976A3", "Spectral trial": "#CB573D"}
    metrics = {name: phase_metrics(problem, curves) for name, curves in predictions.items()}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), layout="constrained")
    plot_checks = []
    for ax, (top, phase), letter in zip(axes.flat,
            [("1r", "set"), ("1r", "reset"), ("1t1r", "set"), ("1t1r", "reset")], "abcd"):
        voltage = abs(problem.waveforms[top, phase])
        observed = problem.raw_training[top, phase]
        eligible = (voltage[None, :] > 0) & np.isfinite(observed) & (observed > 0)
        xx = np.broadcast_to(voltage, observed.shape)
        ax.scatter(xx[eligible], observed[eligible], s=2.5, color="#606870", alpha=.17,
                   edgecolors="none", rasterized=True, zorder=1)
        apex = int(np.argmax(voltage))
        branches = ((np.arange(apex+1), "-"), (np.arange(apex, len(voltage)), "--"))
        for name, curves in predictions.items():
            currents = abs(np.array([curves[top, cid, phase]["current_a"] for cid in TRAIN[top]]))
            median = np.median(currents, axis=0)
            for indices, style in branches:
                for row in currents:
                    use = indices[(voltage[indices] > 0) & (row[indices] > 0)]
                    ax.plot(voltage[use], row[use], color=colors[name], lw=.55, alpha=.13,
                            ls=style, zorder=2)
                use = indices[(voltage[indices] > 0) & (median[indices] > 0)]
                ax.plot(voltage[use], median[use], color=colors[name], lw=1.8, ls=style, zorder=3)
            score = metrics[name][f"{top}_{phase}"]["groups"]["all_eligible"]["plain_RMSE_decade"]
            reference = (report["phase_plain_log_metrics"] if name == "Spectral trial" else
                         report["fitted_GapLaw_baselines"]["block"]["phase_plain_log_metrics"])
            if not np.isclose(score, reference[f"{top}_{phase}"]["groups"]["all_eligible"]["plain_RMSE_decade"],
                              atol=1e-12, rtol=1e-12):
                raise ValueError("Plotted trace RMSE disagrees with frozen evaluation")
            plot_checks.append({"topology": top, "phase": phase, "model": name,
                                "n_records": len(TRAIN[top]), "RMSE_decade": score})
        old = metrics["GapLaw baseline"][f"{top}_{phase}"]["groups"]["all_eligible"]["plain_RMSE_decade"]
        new = metrics["Spectral trial"][f"{top}_{phase}"]["groups"]["all_eligible"]["plain_RMSE_decade"]
        ax.set_title(f"({letter}) {'Bare 1R' if top == '1r' else '1T1R'} {phase.upper()}", loc="left", weight="bold")
        ax.text(.03, .04, f"Raw-record RMSE: {old:.3f} / {new:.3f} decade\nGapLaw / spectral",
                transform=ax.transAxes, fontsize=8.7, va="bottom",
                bbox={"facecolor": "white", "alpha": .88, "edgecolor": "none", "pad": 3})
        ax.set_yscale("log")
        ax.set_xlim(0, float(max(voltage))*1.015)
        ax.set_ylim(3e-12 if phase == "set" else 3e-13, .03 if top == "1r" else 3e-4)
        ax.set_xlabel("Applied voltage magnitude (V)")
        ax.set_ylabel("Current magnitude (A)")
        ax.grid(True, which="major", lw=.45, alpha=.22)
    handles = [Line2D([], [], ls="none", marker=".", color="#606870", label="Training observations"),
               *[Line2D([], [], color=colors[name], lw=2, label=name) for name in sources],
               Line2D([], [], color="#444444", ls="-", label="Outward sweep"),
               Line2D([], [], color="#444444", ls="--", label="Return sweep")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=9)
    fig.supxlabel("Training only: bare 14 conditional pairs; series cycles 4–35 propagated continuously.\n"
                  "Thin model traces show every record; bold traces show medians. Scores use the original masks.", fontsize=9)
    output = HERE/"training_curve_comparison"
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    manifest = {"status": "plotted_frozen_training_predictions_no_refit",
                "holdout_arrays_loaded": False, "score_replay_checks": plot_checks,
                "display": "All positive raw observations plotted, including below-score-floor points; score masks unchanged. Model lines omitted only at nonpositive voltage/current for log plotting. Axes limits are explicit.",
                "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (report_path, candidate_path, baseline_path, Path(__file__))},
                "outputs": [str(output.with_suffix(ext)) for ext in (".png", ".pdf")]}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"outputs": manifest["outputs"], "score_replay_checks": len(plot_checks)}))


if __name__ == "__main__":
    main()
