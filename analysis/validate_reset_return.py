"""Calibrate a two-channel erased-state I–V and score gate-only IV41 cycles."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from analysis import refit_iv41 as refit, validate_paper_claim as paper
from analysis import validate_programming_bridge as bridge
from analysis.fit_reset_population import load_data, paired_block_interval
from vulcan2d import gate_only


def return_targets(rows):
    voltage = np.abs(rows[0].voltage)
    returning = np.arange(len(voltage)) > voltage.argmax()
    eligible = np.stack([row.masks["use_unclamped"] for row in rows])
    eligible &= returning & (voltage >= .05)
    weight = eligible / np.maximum(eligible.sum(axis=1), 1)[:, None]
    support = weight.sum(axis=0)
    use = support > 0
    log_current = np.stack([refit.log_i(row.current) for row in rows])
    target = (weight * log_current).sum(axis=0)[use] / support[use]
    return voltage[use], target, np.sqrt(support[use] / len(rows))


def fit_transport(data, gates):
    cohorts = [(refit.circuits(g)[1], *return_targets(data[g]["RESET"][0])) for g in gates]

    def residual(parameters):
        beta = parameters[-1]
        pairs = parameters[:-1].reshape(-1, 2)
        return np.concatenate([
            weight * (refit.log_i(gate_only.erased_state_current(v, circuit, terms, beta)) - target)
            for (circuit, v, target, weight), terms in zip(cohorts, pairs)
        ])

    n = len(cohorts)
    fit = least_squares(residual, [-11., -8.] * n + [6.],
                        bounds=([-19., -19.] * n + [0.], [-4., -4.] * n + [20.]),
                        diff_step=5e-5, max_nfev=160)
    if not fit.success:
        raise RuntimeError(fit.message)
    return {"beta_sqrt_v": float(fit.x[-1]),
            "log_conductances": dict(zip(gates, fit.x[:-1].reshape(-1, 2).tolist()))}


def score_gate(data, calibration, transport, gate):
    baseline = gate_only.predict(calibration, gate)
    candidate = gate_only.predict({**calibration, "reset_return_transport": transport}, gate)
    phase = "RESET"
    rows = data[gate][phase][1]
    voltage, original = baseline[phase]
    _, updated = candidate[phase]
    median, use = paper.phase_median(rows)
    returning = np.arange(len(voltage)) > np.abs(voltage).argmax()
    measured = paper.observed_measures(rows, phase)
    observed = {name: paper.summarize(values, len(rows)) for name, values in measured.items()}
    scores = {}
    for name, prediction in (("baseline", original), ("candidate", updated)):
        measured_error = (refit.log_i(prediction) - median)**2
        scores[name] = {
            **refit.error_scores(rows, phase, prediction),
            "median_reset_rmse": float(np.sqrt(np.mean(measured_error[use]))),
            "median_return_rmse": float(np.sqrt(np.mean(measured_error[use & returning]))),
            "key_metrics": paper.compare_point_measures(
                observed, paper.measures(voltage, prediction, phase, use)),
        }
    scores["SET"] = refit.error_scores(data[gate]["SET"][1], "SET", baseline["SET"][1])
    scores["paired_block_comparison"] = paired_block_interval(
        rows, refit.log_i(updated), refit.log_i(original))
    return scores, (voltage, median, original, updated)


def pooled(scores):
    def rmse(items, term):
        return float(np.sqrt(np.average([item[term]**2 for item in items],
                                        weights=[item["n"] for item in items])))

    set_rmse = rmse([r["SET"] for r in scores.values()], "valid")
    result = {}
    for model in ("baseline", "candidate"):
        cases = [r[model] for r in scores.values()]
        reset_rmse = rmse(cases, "valid")
        result[model] = {"SET": set_rmse, "RESET": reset_rmse,
                         "RESET_return": rmse(cases, "return"),
                         "full_cycle": float(np.hypot(set_rmse, reset_rmse) / np.sqrt(2))}
    return result


def figure(path, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.1), sharex=True, sharey=True)
    colors = ("#24496c", "#747b7e", "#057454")
    labels = ("Measured median", "Previous model", "Erased-state I–V")
    for ax, (gate, (voltage, median, original, updated)) in zip(axes.flat, curves.items()):
        v = np.abs(voltage)
        apex = v.argmax()
        k = np.arange(len(v))
        for use, style in (((k <= apex) & (v > 0), "-"), ((k >= apex) & (v > 0), "--")):
            for color, current in zip(colors, (median, refit.log_i(original), refit.log_i(updated))):
                ax.plot(v[use], current[use]+6, color=color, ls=style, lw=1.55)
        ax.set_title(f"{gate:g} V gate  |  RESET", loc="left", fontweight="bold")
        ax.set_xlabel("Terminal voltage magnitude (V)")
        ax.set_ylabel(r"$\log_{10}(|I| / \mu\mathrm{A})$")
        ax.grid(alpha=.22)
    handles = [Line2D([], [], color=c, label=t) for c, t in zip(colors, labels)]
    handles.append(Line2D([], [], color="#444", ls="--", label="Return sweep"))
    fig.legend(handles=handles, loc="lower center", ncols=4, frameon=False)
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args(argv)
    calibration = json.loads(args.baseline.read_text())
    data = load_data(refit.origin_cache(args.workbook, args.cache), args.cache)
    args.output.mkdir(parents=True, exist_ok=True)
    anchors = bridge.PROGRAM_GATES

    development = {}
    for held in anchors:
        train = tuple(g for g in anchors if g != held)
        config = {**calibration, "anchors": {g: profile for g, profile in calibration["anchors"].items()
                                            if float(g) in train}}
        development[held], _ = score_gate(data, config, fit_transport(data, train), held)

    transport = fit_transport(data, anchors)
    metrics, curves = {}, {}
    for gate in bridge.ASSESS_GATES:
        metrics[gate], curves[gate] = score_gate(data, calibration, transport, gate)

    report = {"workbook_sha256": refit.sha_file(args.workbook),
              "baseline_sha256": refit.sha_file(args.baseline),
              "training_gates_v": anchors, "transport": transport,
              "anchor_leave_one_gate_out": development,
              "historical_held_out_gates": metrics, "historical_pooled": pooled(metrics)}
    (args.output / "reset_return_results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    candidate = {**calibration, "reset_return_transport": transport}
    (args.output / "reset_return_calibration.json").write_text(json.dumps(candidate, allow_nan=False)+"\n")
    figure(args.output / "Vulcan_RESET_return.png", curves)
    print(json.dumps(report["historical_pooled"], indent=2))


if __name__ == "__main__":
    main()
