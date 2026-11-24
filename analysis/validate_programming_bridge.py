"""Compare circuit-driven hBN state laws with the old topology calibration."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from analysis import refit_iv41 as refit
from vulcan2d import model_shared_device as model
from vulcan2d import predict_gate
from vulcan2d import validate_shared_device as shared
from vulcan2d import validate_transition_regions as audit
from vulcan2d.transistor import load_default_lookup, load_reverse_lookup

SET, RESET = refit.PHASES
PROGRAM_GATES = refit.ANCHORS
ASSESS_GATES = (1., 1.2, 1.4, 1.5)
BARE_TRAINING_SCANS = 14
READ_WINDOW_V = (.10, .20)
MIN_LOCAL_V = .025
BARE_COMPLIANCE = 1e-3


def fit_preparation(groups, device, prep, fit_growth, soft_amplitude_upper_a=1e-3):
    """Fit one IV41 preparation across programming gates."""
    fields = refit.PHASE_FIELD[SET]
    fit_data = {g: shared._training_data(rows[SET]) for g, rows in groups.items()}

    def set_parts(x):
        candidate = replace(device, soft_growth_current_a=10 ** x[2]) if fit_growth else device
        return candidate, replace(prep, **dict(zip(fields, 10 ** x[:2])))

    def set_residual(x):
        candidate, preparation = set_parts(x)
        result = []
        for g, (v, keep, target, weights) in fit_data.items():
            calc = model.simulate_sweep(v, SET, candidate, preparation, refit.circuits(g)[0])
            result.append(weights * (refit.log_i(calc["current_A"][keep]) - target))
        return np.concatenate(result)

    start = [np.log10(getattr(prep, k)) for k in fields]
    low, high = [-17., -10.], [-6., np.log10(soft_amplitude_upper_a)]
    if fit_growth:
        start += [-4.5]
        low += [-8.]
        high += [-2.5]
    solved = shared._least_squares(set_residual, start, (low, high))
    device, prep = set_parts(solved)
    fit_data = {g: shared._training_data(rows[RESET]) for g, rows in groups.items()}
    initials = {g: model.simulate_sweep(rows[SET][0].voltage, SET, device, prep,
                                        refit.circuits(g)[0])["final"] for g, rows in groups.items()}
    fields = refit.PHASE_FIELD[RESET]

    def reset_parts(x):
        return replace(prep, **dict(zip(fields, 10 ** x)))

    def reset_residual(x):
        preparation = reset_parts(x)
        result = []
        for g, (v, keep, target, weights) in fit_data.items():
            calc = model.simulate_sweep(v, RESET, device, preparation, refit.circuits(g)[1], initials[g])
            result.append(weights * (refit.log_i(calc["current_A"][keep]) - target))
        return np.concatenate(result)

    solved = shared._least_squares(reset_residual, [np.log10(getattr(prep, k)) for k in fields],
                                  ([-18., -12.], [-5., -3.]))
    return device, reset_parts(solved)


def fixed_read_conductance(device, prep, state):
    """Intrinsic RESET-polarity secant conductance at 0.1 V, before RESET."""
    voltage = .1
    return float(model.device_current(voltage, state.soft, state.hard, device, prep, RESET) / voltage)


def compare_gate_models(index, cache, calibration, device):
    train, assessment = {}, {}
    for gate in (*PROGRAM_GATES, *ASSESS_GATES):
        sheet = round((gate - .9) * 10) + 5
        rows, _ = refit.load_cohort(index, cache, sheet, gate)
        if gate in PROGRAM_GATES:
            train[gate] = {phase: refit.screen(rows[phase], gate, phase)[0] for phase in refit.PHASES}
        else:
            assessment[gate] = rows
    seed = calibration.preparation(1.1)
    fixed = fit_preparation(train, device, seed, False)
    current_limited = fit_preparation(train, device, seed, True)
    profile = {str(bound): (fit_preparation(train, device, seed, True, bound)
                           if bound != 1e-3 else current_limited)
               for bound in (3e-4, 1e-3, 3e-3)}
    results = {}
    for gate, rows in assessment.items():
        edge = min(gate, PROGRAM_GATES[-1])
        separate = (calibration.device_at(edge), calibration.preparation(edge))
        variants = {"per_gate_reference": separate, "common_preparation": fixed,
                    "common_with_current_growth": current_limited}
        scores = {}
        for name, (d, p) in variants.items():
            written, erased = refit.simulate(gate, rows, d, p)
            scores[name] = {
                phase: refit.error_scores(rows[phase], phase, predicted["current_A"])
                for phase, predicted in zip(refit.PHASES, (written, erased))
            }
            scores[name]["prepared_conductance_at_device_0p1V_S"] = fixed_read_conductance(
                d, p, written["final"])
            scores[name]["SET_end"] = asdict(written["final"])
        for phase in refit.PHASES:
            scores[f"{phase}_standard_grid_n"] = len(rows[phase])
        results[gate] = scores
    sensitivity = {
        bound: {"soft_set_amplitude_a": p.soft_set_a,
                "soft_growth_current_a": d.soft_growth_current_a,
                "anchor_SET_rmse": {
                    g: refit.error_scores(
                        rows[SET], SET, refit.simulate(g, rows, d, p)[0]["current_A"])["valid"]
                    for g, rows in train.items()}}
        for bound, (d, p) in profile.items()
    }
    return results, fixed, current_limited, assessment, sensitivity


def fit_bare_programming(rows, circuit, device, prep):
    """Estimate the low-current transport from uncensored SET before the SMU clips."""
    cutoff = np.array([shared._clip_voltage(r.voltage, r.current) for r in rows])
    before = np.stack([r.masks["branch_outward"] & (r.voltage <= clip - .15 + 1e-8)
                       for r, clip in zip(rows, cutoff)])
    wave, keep, target, weights = shared._training_data(rows, use_transition=False, additional_mask=before)
    amps = refit.PHASE_FIELD[SET]

    def prepare(p):
        return replace(prep, soft_available=1., **dict(zip(amps, 10 ** p)))

    def fit_error(p):
        calc = model.simulate_sweep(wave, SET, device, prepare(p), circuit)
        return weights * (refit.log_i(calc["current_A"][keep]) - target)

    start = [np.log10(prep.off_set_a), -7.]
    solved = least_squares(fit_error, start, bounds=([-18., -12.], [-6., -3.]),
                           diff_step=3e-5, max_nfev=150)
    if not solved.success:
        raise RuntimeError(solved.message)
    return prepare(solved.x)


def bare_scores(groups, circuits, d, p):
    result = {}
    output = model.simulate_cycle(groups[SET][0].voltage, groups[RESET][0].voltage,
                                  d, p, *circuits)
    for phase, pred in zip(refit.PHASES, output):
        rows = groups[phase]
        result[phase] = refit.error_scores(rows, phase, pred["current_A"])
        if phase == SET:
            before = []
            clips = []
            for row in rows:
                clip = shared._clip_voltage(row.voltage, row.current)
                use = row.masks["use_unclamped"] & row.masks["branch_outward"] & (row.voltage <= clip - .15 + 1e-8)
                before.append((abs(row.current[use]), abs(pred["current_A"][use])))
                clips.append(clip)
            rmse = audit._summarize_residuals(before)["pooled_rmse_decade"]
            clip = shared._clip_voltage(pred["voltage_V"], pred["current_A"])
            score = result[phase]
            score.update({"pre_compliance_rmse_decade": rmse, "clip_pred_V": clip,
                          "clip_mae_V": None if clip is None else float(np.mean(abs(np.array(clips) - clip)))})
    result["SET_end"] = asdict(output[0]["final"])
    return result


def fit_bare_comparison(device, initial_prep, preprocess=None, score_on_raw=False):
    groups, originals, circuits = {}, {}, {}
    for phase in refit.PHASES:
        source = next(file for topology, _, p, file in audit.WORKBOOKS if topology == "1M" and p == phase)
        circuits[phase] = model.Circuit(compliance_a=BARE_COMPLIANCE if phase == SET else None)
        limit = circuits[phase].compliance_a
        originals[phase] = [
            shared.Sweep(label, v, i, *audit.prepare_curve(v, i, phase, smu_limit=limit))
            for label, _, (v, i) in audit.read_sweeps(audit.ROOT / source)
        ]
        groups[phase] = (preprocess(originals[phase], phase)
                         if preprocess is not None else originals[phase])
    train = {phase: rows[:BARE_TRAINING_SCANS] for phase, rows in groups.items()}
    held = {phase: rows[BARE_TRAINING_SCANS:] for phase, rows in groups.items()}
    if score_on_raw:
        held = {phase: [replace(filtered, current=raw.current)
                        for raw, filtered in zip(originals[phase][BARE_TRAINING_SCANS:], rows)]
                for phase, rows in held.items()}
    set_circuit, reset_circuit = (circuits[p] for p in refit.PHASES)
    prior = replace(device, hard_access_mode="available", soft_growth_current_a=None)
    prior, bare_off = shared.fit_bare_set(train[SET], set_circuit, prior, initial_prep)
    legacy, legacy_prep = shared.fit_bare_reset(train[RESET], reset_circuit, prior, bare_off,
                                               model.State(hard=1.))
    # Bare data identify the voltage interval and fully on conductance. Keep the
    # old current scale as a provisional threshold and report a profile below.
    d = replace(device, hard_set_v=prior.hard_set_v, hard_conductance_s=legacy.hard_conductance_s)
    transferred = replace(initial_prep, off_set_a=bare_off.off_set_a, soft_available=1.)
    fitted = fit_bare_programming(train[SET], set_circuit, d, transferred)
    results = {}
    fitted_cal = None
    for name, law, prep in (("legacy", legacy, legacy_prep), ("transferred_soft", d, transferred),
                            ("calibrated_soft", d, fitted)):
        if name != "legacy":
            initial = model.simulate_sweep(train[SET][0].voltage, SET, law, prep, set_circuit)["final"]
            if initial.hard <= 0.:
                raise RuntimeError("The calibration sweep could not reach the strong state")
            law, prep = shared.fit_bare_reset(train[RESET], reset_circuit, law, prep, initial)
        results[name] = bare_scores(held, (set_circuit, reset_circuit), law, prep)
        results[name]["preparation"] = asdict(prep)
        if name == "calibrated_soft":
            fitted_cal = (law, prep)
    return fitted_cal, results, groups


def summaries(index, cache, bare):
    """Summarize early RESET readout without using gates as inferred labels."""
    reverse = load_reverse_lookup()
    results = {}
    for sheet, gate in [(j, (j+4)/10) for j in range(5, 12)] + [(12, 5.), (13, 5.)]:
        sheet_name = index["names"][sheet - 1]
        phases = {str(row[4]): row[6] for row in index["curves"][1:] if row[13] == sheet_name}
        grid, curve = audit.transistor_curve(reverse, gate)
        levels, indices = np.unique(curve, return_index=True)
        raw, internal, peak_set = [], [], []
        n_points = [0, 0]
        seen = set()
        with np.load(cache / f"s{sheet}.npz") as data:
            for key in data.files:
                voltage, current = data[key][:, 1:].T
                phase = phases[key]
                fingerprint = (phase, voltage.tobytes(), current.tobytes())
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                if phase == "正":
                    peak_set.append(float(np.max(abs(current))))
                    continue
                top = int(np.argmax(abs(voltage)))
                v, i = abs(voltage[:top + 1]), abs(current[:top + 1])
                use = (v >= READ_WINDOW_V[0] - 1e-9) & (v <= READ_WINDOW_V[1] + 1e-9) & (i > 0)
                if not np.any(use):
                    continue
                n_points[0] += int(np.count_nonzero(use))
                raw.append(np.median(i[use] / v[use]))
                supported = i <= np.interp(v, grid, curve)
                local = v - np.interp(i, levels, grid[indices])
                usable = use & supported & (local >= MIN_LOCAL_V)
                n_points[1] += int(np.count_nonzero(usable))
                if np.count_nonzero(usable) >= 3:
                    internal.append(np.median(i[usable] / local[usable]))
        results[sheet_name] = {"program_gate_V": gate, "SET_peak_A": quantiles(peak_set),
                               "SET_sweeps": len(peak_set), "RESET_series_G_S": quantiles(raw),
                               "RESET_deembedded_G_S": quantiles(internal), "RESET_sweeps": len(raw),
                               "RESET_deembedded_sweeps": len(internal), "read_points_usable_total": n_points[::-1]}
    raw = []
    for row in bare[RESET]:
        v, i = np.abs(row.voltage), np.abs(row.current)
        use = row.masks["branch_outward"] & (v >= .10-1e-9) & (v <= .20+1e-9)
        raw.append(np.median(i[use] / v[use]))
    results["bare"] = {"program_gate_V": None, "RESET_series_G_S": quantiles(raw),
                       "RESET_sweeps": len(raw)}
    return results


def quantiles(values):
    return np.quantile(values, [.1, .5, .9]).tolist() if len(values) else None


def current_threshold_audit(bare_cal, iv_device):
    """Compute local eligibility and candidate-current ranges without extrapolation."""
    forward = load_default_lookup()
    cases = {}
    gates = (*PROGRAM_GATES, *ASSESS_GATES, 2., 2.5, 3.)
    for gate in (*sorted(set(gates)), None):
        prep = bare_cal[1] if gate is None else iv_device[1]
        device = bare_cal[0] if gate is None else iv_device[0]
        circuit = (model.Circuit(compliance_a=BARE_COMPLIANCE) if gate is None else
                   model.Circuit(transistor=audit.transistor_curve(forward, gate)))
        voltage = predict_gate.triangle(5., .01)
        outward = voltage[:np.argmax(voltage) + 1]
        _, pre_voltage, _ = model.solve_soft_set_outward(outward, model.State(), device, prep, circuit)
        possible = pre_voltage >= device.hard_set_v
        current, _ = model.strong_operating_point(outward[possible], device, circuit)
        example_thresholds = (.0002, .0005, .0008)
        outcomes = {str(cutoff): bool(np.any(current >= cutoff)) for cutoff in example_thresholds}
        predicted = model.simulate_sweep(voltage, SET, device, prep, circuit)
        cases[str(gate)] = {"hard_state_end": predicted["final"].hard,
                            "high_path_candidate_max_A": float(current.max()) if len(current) else None,
                            "source_short_circuit_bound_A": float(circuit.supply_bound(outward).max()),
                            "threshold_sensitivity": outcomes}
    return cases


def early_read_comparison(summaries_by_sheet, fixed, current, assessment):
    """Test the prepared-state prediction against a readout-only null."""
    d, p = fixed
    nominal_set_wave = predict_gate.triangle(5., .01)
    nominal = model.simulate_sweep(nominal_set_wave, SET, d, p,
                                   refit.circuits(1.1)[0])["final"]
    voltage = np.linspace(*READ_WINDOW_V, num=11)
    observed = {x["program_gate_V"]: x["RESET_series_G_S"][1]
                for x in summaries_by_sheet.values() if x["program_gate_V"] in ASSESS_GATES}
    scores = {}
    for gate, rows in assessment.items():
        reset_c = refit.circuits(gate)[1]
        states = {"readout_only": (d, p, nominal)}
        for name, (device, prep) in (("common_preparation", fixed), ("current_growth", current)):
            state = refit.simulate(gate, rows, device, prep)[0]["final"]
            states[name] = device, prep, state
        values = {}
        for name, (device, prep, state) in states.items():
            measured, _ = reset_c.operating_point(voltage, lambda vh: model.device_current(
                vh, state.soft, state.hard, device, prep, RESET))
            values[name] = float(np.median(measured / voltage))
        scores[gate] = {"observed_series_G_S": observed[gate], "predicted_series_G_S": values}
    return scores


def plot_bridge(path, observations, early_predictions, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = list(observations.values())
    labels = [f"{x['program_gate_V']:g}" for x in entries[:-3]] + ["5 IV41", "5 IV42", "Bare"]
    values = np.asarray([x["RESET_series_G_S"] for x in entries]) * 1e6
    pos = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(9., 7.8), height_ratios=(2., 1.5))
    ax = axes[0]
    ax.errorbar(pos, values[:, 1], yerr=[values[:, 1]-values[:, 0], values[:, 2]-values[:, 1]],
                fmt="o", capsize=3, color="#244d71", label="Measured terminal G, median and 10–90%")
    styles = [("readout_only", "Readout-only null", "#8b8b8b", "x"),
              ("common_preparation", "Common field-driven model", "#b8772a", "s"),
              ("current_growth", "Current-assisted soft growth", "#14816b", "^")]
    for key, label, color, marker in styles:
        xs = [labels.index(f"{gate:g}") for gate in early_predictions]
        ys = [r["predicted_series_G_S"][key]*1e6 for r in early_predictions.values()]
        ax.scatter(xs, ys, color=color, marker=marker, label=label, zorder=4)
    ax.set_xticks(pos, labels)
    ax.set_xlabel("Programming gate (V) and topology")
    ax.set_ylabel("Early RESET terminal conductance (µS)")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=.22, which="both")
    ax.set_title("Early RESET readout across programming conditions", loc="left", weight="bold")
    ax = axes[1]
    gates = [float(g) for g in summary if g != "None"]
    hypothetical = np.array([summary[str(g)]["high_path_candidate_max_A"]
                             for g in gates], dtype=float) * 1e3
    external = [summary[str(g)]["source_short_circuit_bound_A"]*1e3 for g in gates]
    ax.plot(gates, external, "o--", color="#8b8b8b", label="Transistor short-circuit bound")
    ax.plot(gates, hypothetical, "s-", color="#14816b", label="Strong-path load-line current after field criterion")
    ax.axhline(.5, linestyle=":", color="#b8772a", label="Provisional sustaining current")
    ax.set_xlabel("Gate voltage within measured forward lookup (V)")
    ax.set_ylabel("Current (mA)")
    ax.set_title("Strong-state access at a 5 V terminal sweep", loc="left", weight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("iv41_calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    index = refit.origin_cache(args.workbook, args.cache)
    inherited = predict_gate.GateCalibration.from_file(args.iv41_calibration)
    device = replace(inherited.device, hard_access_mode="sustained")
    bare_cal, bare_scores_by_model, bare_all = fit_bare_comparison(device, inherited.preparation(1.1))
    print("Bare validation", {name: [r[SET]["pre_compliance_rmse_decade"], r[RESET]["valid"]]
                              for name, r in bare_scores_by_model.items()}, flush=True)
    # The strong-state conductance and formation voltage come only from the bare training data.
    bare_d = bare_cal[0]
    device = replace(device, hard_set_v=bare_d.hard_set_v, hard_conductance_s=bare_d.hard_conductance_s,
                     hard_reset_centers_v=bare_d.hard_reset_centers_v,
                     hard_reset_widths_v=bare_d.hard_reset_widths_v,
                     hard_reset_late_fraction=bare_d.hard_reset_late_fraction)
    scores, fixed, current, assessment, profile = compare_gate_models(index, args.cache, inherited, device)
    for g, results in scores.items():
        print("IV41", g, {name: [round(m[SET]["valid"], 4), round(m[RESET]["valid"], 4)]
                          for name, m in results.items() if isinstance(m, dict)}, flush=True)
    readout = summaries(index, args.cache, bare_all)
    early_predictions = early_read_comparison(readout, fixed, current, assessment)
    access = current_threshold_audit(bare_cal, fixed)
    audit_results = {
        "input_sha256": {"workbook": refit.sha_file(args.workbook),
                          "calibration": refit.sha_file(args.iv41_calibration),
                          "transistor_lookups": {name: refit.sha_file(audit.ROOT / "vulcan2d/data" / name)
                                                 for name in audit.TRANSISTOR_FILES},
                          "bare_workbooks": {name: refit.sha_file(audit.ROOT / name)
                                             for t, _, _, name in audit.WORKBOOKS if t == "1M"}},
        "gate_fits": scores, "bare_holdout": bare_scores_by_model,
        "early_reset_readout": readout,
        "early_read_prediction": early_predictions,
        "strong_access": access,
        "soft_growth_parameter_sensitivity": profile,
        "calibrations": {name: {"device": asdict(d), "preparation": asdict(p)}
                         for name, (d, p) in {"IV41_common": fixed, "IV41_current_growth": current,
                                             "bare": bare_cal}.items()},
    }
    (args.output / "programming_bridge_results.json").write_text(
        json.dumps(audit_results, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for name, (d, p) in (("common_iv41", fixed), ("current_growth_iv41", current)):
        calibration = predict_gate.GateCalibration(PROGRAM_GATES, d, (p,) * len(PROGRAM_GATES))
        calibration.to_file(args.output / f"{name}.json")
    plot_bridge(args.output / "Vulcan_programming_bridge.png", readout, early_predictions, access)


if __name__ == "__main__":
    main()
