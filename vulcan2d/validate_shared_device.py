"""Fit the shared h-BN model with cycle and gate holdouts.

Run from the repository root:
    python -m vulcan2d.validate_shared_device --output-dir PATH

Outputs JSON with calibration and errors, and NPZ with exact prediction arrays.
"""
from dataclasses import asdict, dataclass, replace
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import model_shared_device as M
from . import validate_transition_regions as audit
from .transistor import load_default_lookup, load_reverse_lookup


REFERENCE = M.Device(6.5, 1.33, 1.54, 0.035, 6.2, 2.0, 0.50, 0.11,
                     2.675, 0.5e-3, 0.0104, (0.59, 1.0), (0.075, 0.095), 0.1)
PREPARATION = M.Preparation(2.6e-10, 1.1e-5, 6.5e-10, 6.0e-7)
BASELINE = audit.ROOT / "analysis" / "reset_platform_terminal_refinement_curves.npz"
TRANSITION_WEIGHT = 0.5


@dataclass
class Sweep:
    label: str
    voltage: np.ndarray
    current: np.ndarray
    masks: dict
    event: dict


def load_groups(*, include_abnormal_pairs=False):
    groups, circuits = {}, {}
    lookups = {"SET": load_default_lookup(), "RESET": load_reverse_lookup()}
    for topology, gate, phase, filename in audit.WORKBOOKS:
        key = (topology, gate, phase)
        circuit = M.Circuit(
            transistor=None if gate is None else audit.transistor_curve(lookups[phase], gate),
            compliance_a=1e-3 if topology == "1M" and phase == "SET" else None)
        circuits[key] = circuit
        group = groups.setdefault(key, [])
        for sheet, _, (voltage, current) in audit.read_sweeps(audit.ROOT / filename):
            if not include_abnormal_pairs and gate == 1.1 and int(sheet[1:]) < 4:
                continue
            masks, event = audit.prepare_curve(voltage, current, phase,
                                              circuit.compliance_a, circuit.transistor)
            group.append(Sweep(sheet, voltage, current, masks, event))
    return groups, circuits


def _training_data(rows, use_transition=True, additional_mask=None, cycle_weights=None):
    v = rows[0].voltage
    if any(not np.array_equal(row.voltage, v) for row in rows):
        raise ValueError("Training sweeps do not share a voltage grid")
    raw = np.abs([row.current for row in rows])
    keep = np.asarray([row.masks["use_unclamped"] for row in rows])
    if additional_mask is not None:
        keep &= additional_mask
    event = np.asarray([row.masks["use_transition"] for row in rows]) & keep
    if not np.any(keep):
        raise ValueError("No eligible observations for calibration")
    if cycle_weights is None:
        cycle_weights = np.ones(len(rows))
    cycle_weights = np.asarray(cycle_weights, dtype=float)
    if (cycle_weights.shape != (len(rows),) or not np.all(np.isfinite(cycle_weights))
            or np.any(cycle_weights <= 0)):
        raise ValueError("Cycle weights must be positive and match the training sweeps")
    weight = cycle_weights[:, None] * keep
    if use_transition and np.any(event):
        transition_weight = cycle_weights[:, None] * event
        weight = ((1 - TRANSITION_WEIGHT) * weight / weight.sum()
                  + TRANSITION_WEIGHT * transition_weight / transition_weight.sum())
    else:
        weight /= weight.sum()
    # One deterministic prediction per voltage grid. Grouped sufficient
    # statistics give exactly the same minimizer as weighted log residuals.
    log_i = np.zeros_like(raw)
    log_i[keep] = np.log10(raw[keep])
    w = weight.sum(axis=0)
    valid = w > 0
    y = (weight * log_i).sum(axis=0)[valid] / w[valid]
    return v, valid, y, np.sqrt(w[valid])


def _least_squares(function, start, bounds):
    result = least_squares(function, start, bounds=bounds, x_scale="jac",
                           max_nfev=120, ftol=2e-8, xtol=2e-8, gtol=2e-8,
                           diff_step=3e-5)
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(result.message)
    return result.x


def fit_soft_set(rows, circuit, device=REFERENCE, preparation=PREPARATION,
                 fit_transitions=True, cycle_weights=None):
    v, valid, log_i, weight = _training_data(rows, cycle_weights=cycle_weights)
    x0 = [np.log10(preparation.off_set_a), np.log10(preparation.soft_set_a)]
    lo, hi = [-13.5, -6.7], [-7.5, -3.5]
    if fit_transitions:
        x0.extend([device.beta_set, device.soft_exponent, device.set_center_v,
                   np.log(device.set_width_v)])
        lo.extend([2.0, 0.6, 0.45, np.log(0.008)])
        hi.extend([10.0, 2.4, 2.10, np.log(0.30)])

    def models(p):
        prepared = replace(preparation, off_set_a=10 ** p[0], soft_set_a=10 ** p[1])
        params = device
        if fit_transitions:
            params = replace(device, beta_set=p[2], soft_exponent=p[3],
                             set_center_v=p[4], set_width_v=np.exp(p[5]))
        return params, prepared

    def residual(p):
        params, prepared = models(p)
        simulated = M.simulate_sweep(v, "SET", params, prepared, circuit)
        return weight * (np.log10(np.abs(simulated["current_A"][valid])) - log_i)

    result = _least_squares(residual, x0, (lo, hi))
    return models(result)


def fit_set_conditioning(rows, circuit):
    """Fit a drift in SET threshold from training cycle labels and currents."""
    device, prep = fit_soft_set(rows, circuit)
    ids = np.array([int(r.label[1:]) for r in rows])
    scale = max(ids.max() - ids.min(), 1)
    u = (ids - ids.min()) / scale
    voltage = rows[0].voltage
    apex = np.argmax(voltage)
    outward = np.broadcast_to(voltage[:apex + 1], (len(rows), apex + 1))
    returning = np.broadcast_to(voltage[apex + 1:], (len(rows), len(voltage) - apex - 1))
    raw = np.abs([r.current for r in rows])
    keep = np.array([r.masks["use_unclamped"] for r in rows])
    events = np.array([r.masks["use_transition"] for r in rows])
    weight = np.sqrt((1 - TRANSITION_WEIGHT) * keep / np.sum(keep)
                     + TRANSITION_WEIGHT * events / np.sum(events))[keep]
    target = np.log10(raw[keep])

    def residual(p):
        fitted = replace(device, set_center_v=p[0] + p[1] * u[:, None],
                         set_width_v=np.exp(p[2]))
        model, _, soft = M.solve_soft_set_outward(
            outward, M.State(), fitted, prep, circuit)
        back, _ = circuit.operating_point(returning, lambda vh: M.device_current(
            vh, soft[:, -1:], 0.0, fitted, prep, "SET"))
        observed = np.concatenate((model, back), axis=1)[keep]
        return weight * (np.log10(observed) - target)

    x = _least_squares(residual, [device.set_center_v, .08, np.log(device.set_width_v)],
                       ([.55, 0, np.log(.008)], [1.95, .70, np.log(.30)]))
    # Beyond the training endpoint use the last learned center. No held-out
    # measurements or unbounded extrapolation in cycle count enter prediction.
    terminal = replace(device, set_center_v=float(x[0] + x[1]),
                       set_width_v=float(np.exp(x[2])))
    return terminal, prep, float(x[1])


def fit_soft_reset(rows, circuit, device, preparation, initial,
                   fit_transitions=True):
    v, valid, log_i, weight = _training_data(rows)
    x0 = [np.log10(preparation.off_reset_a), np.log10(preparation.soft_reset_a)]
    lo, hi = [-15.0, -10.0], [-8.3, -4.3]
    if fit_transitions:
        x0.extend([device.beta_reset, np.log(device.reset_soft_alpha),
                   device.reset_center_v, np.log(device.reset_width_v)])
        lo.extend([0.3, np.log(0.35), 0.15, np.log(0.025)])
        hi.extend([12, np.log(5.0), 1.60, np.log(0.55)])

    def models(p):
        prep = replace(preparation, off_reset_a=10 ** p[0], soft_reset_a=10 ** p[1])
        fitted = device
        if fit_transitions:
            fitted = replace(device, beta_reset=p[2], reset_soft_alpha=np.exp(p[3]),
                             reset_center_v=p[4], reset_width_v=np.exp(p[5]))
        return fitted, prep

    def residual(p):
        params, prep = models(p)
        simulated = M.simulate_sweep(v, "RESET", params, prep, circuit, initial)
        return weight * (np.log10(np.abs(simulated["current_A"][valid])) - log_i)

    result = _least_squares(residual, x0, (lo, hi))
    return models(result)


def fit_bare_set(rows, circuit, device, preparation):
    bounds = []
    before = []
    for row in rows:
        first = int(np.flatnonzero(row.masks["smu_compliance"] & row.masks["branch_outward"])[0])
        bounds.append((row.voltage[first - 1], row.voltage[first]))
        before.append(row.masks["branch_outward"] & (row.voltage < row.voltage[first]))
    v, valid, log_i, weight = _training_data(rows, additional_mask=np.asarray(before))
    # The uncensored data identify a pre-SET amplitude at the shared SET shape.
    unit = M.device_current(v[valid], 0, 0, device, replace(preparation, off_set_a=1.0), "SET")
    off = 10 ** np.average(log_i - np.log10(unit), weights=weight ** 2)
    threshold = float(np.median(np.mean(bounds, axis=1)))
    return (replace(device, hard_set_v=threshold),
            replace(preparation, off_set_a=off, soft_available=0.0))


def fit_bare_reset(rows, circuit, device, preparation, initial):
    v, valid, log_i, weight = _training_data(rows)
    # Identify small-signal strong-path conductance before any measurable
    # rupture, rather than trading it off against the late drop during fitting.
    initial_s = []
    for row in rows:
        early = row.masks["branch_outward"] & (np.abs(row.voltage) >= .075 - 1e-8) & (np.abs(row.voltage) <= .30 + 1e-8)
        initial_s.extend(np.abs(row.current[early] / row.voltage[early]))
    device = replace(device, hard_conductance_s=float(np.median(initial_s)))

    def models(p):
        prep = replace(preparation, off_reset_a=10 ** p[0])
        fitted = replace(device, hard_reset_centers_v=(p[1], p[1] + p[2]),
                         hard_reset_widths_v=(np.exp(p[3]), np.exp(p[4])),
                         hard_reset_late_fraction=10 ** p[5])
        return fitted, prep

    def residual(p):
        params, prep = models(p)
        simulated = M.simulate_sweep(v, "RESET", params, prep, circuit, initial)
        return weight * (np.log10(np.abs(simulated["current_A"][valid])) - log_i)

    x = _least_squares(residual, [-14.0, .61, .68, np.log(.085), np.log(.058), -1.0],
                       ([-19, .25, .04, np.log(.018), np.log(.007), -4.0],
                        [-10, 1.03, 1.80, np.log(.25), np.log(.30), -.02]))
    return models(x)


def simulation(row, phase, device, preparation, circuit, initial=M.State()):
    return M.simulate_sweep(row.voltage, phase, device, preparation, circuit, initial)


def _clip_voltage(voltage, current):
    eligible = audit.persistent((voltage > 0) & (np.abs(current) >= 0.99e-3))
    choices = np.flatnonzero(eligible)
    return float(voltage[choices[0]]) if len(choices) else None


def scores(rows, predictions, phase, circuit):
    result = {}
    current = [p["current_A"] if isinstance(p, dict) else p for p in predictions]
    for key in ("eligible", "use_unclamped", "use_transition"):
        pairs = [(np.abs(r.current[r.masks[key]]), np.abs(i[r.masks[key]]))
                 for r, i in zip(rows, current)]
        result[key] = audit._summarize_residuals(pairs)
    shifts = []
    clip_predictions = 0
    for row, predicted in zip(rows, current):
        if circuit.compliance_a and phase == "SET":
            actual_v = _clip_voltage(row.voltage, row.current)
            predicted_v = _clip_voltage(row.voltage, predicted)
            clip_predictions += predicted_v is not None
        else:
            actual_v = row.event.get("event_abs_voltage_V")
            _, event = audit.prepare_curve(row.voltage, predicted, phase)
            predicted_v = event.get("event_abs_voltage_V")
        if actual_v is not None and predicted_v is not None:
            shifts.append(predicted_v - actual_v)
    if circuit.compliance_a and phase == "SET":
        result["compliance_onset"] = {
            "predicted": clip_predictions, "n_compared": len(shifts),
            "mae_V": float(np.mean(np.abs(shifts))) if shifts else None}
    else:
        result["event"] = {"n_compared": len(shifts),
                           "mae_V": float(np.mean(np.abs(shifts))) if shifts else None,
                           "bias_V": float(np.mean(shifts)) if shifts else None}
    return result


def fit_set_strategy(rows, circuit, strategy):
    if strategy == "conditioning":
        return fit_set_conditioning(rows, circuit)
    count = int(strategy) if strategy != "static" else None
    device, prepared = fit_soft_set(rows[-count:] if count else rows, circuit)
    return device, prepared, None


def choose_set_strategy(rows, circuit):
    # Only the early training block enters model selection. The choice is
    # applied to the outer time split regardless of the resulting test score.
    evaluations = {}
    for strategy in ("static", "12", "8", "conditioning"):
        tests, preds = [], []
        for stop in (19, 25):
            training, test = rows[:stop], rows[stop:stop + 6]
            params, prepared, _ = fit_set_strategy(training, circuit, strategy)
            preds.extend(_batch(test, "SET", params, prepared, circuit))
            tests.extend(test)
        evaluations[strategy] = scores(tests, preds, "SET", circuit)
    best = min(evaluations, key=lambda k: evaluations[k]["use_transition"]["pooled_rmse_decade"])
    return best, evaluations


def common_curve_floor(rows):
    """In-sample lower bound on the RMSE of an unrestricted common trace."""
    result = {}
    raw = np.abs([r.current for r in rows])
    for subset in ("use_unclamped", "use_transition"):
        keep = np.array([r.masks[subset] for r in rows])
        logs = np.zeros_like(raw)
        logs[keep] = np.log10(raw[keep])
        center = logs.sum(axis=0) / np.maximum(1, keep.sum(axis=0))
        result[subset] = float(np.sqrt(np.mean(np.square((logs - center)[keep]))))
    return result


def _aligned_baseline(set_rows, reset_rows, split):
    output = []
    with np.load(BASELINE, allow_pickle=False) as baseline:
        for rows, side in ((set_rows, "set"), (reset_rows, "reset")):
            expected = []
            for row in rows:
                cycle = int(row.label[1:])
                prefix = f"{split}_same_protocol_cycle_{cycle}_{side}"
                if not np.array_equal(row.voltage, baseline[prefix + "_voltage"]):
                    raise ValueError(f"Baseline grid does not match {row.label}")
                if not np.array_equal(row.current, baseline[f"observed_cycle_{cycle}_{side}_current"]):
                    raise ValueError(f"Baseline data does not match {row.label}")
                expected.append(baseline[prefix + "_current"])
            output.append(expected)
    return output


def score_ensemble(rows, samples, phase, circuit):
    """Predict a population band using calibrated training sweeps only."""
    distribution = np.abs(np.stack([s["current_A"] for s in samples]))
    log_distribution = np.log10(np.maximum(distribution, 1e-30))
    low, median, high = 10 ** np.quantile(log_distribution, [.10, .50, .90], axis=0)
    sign = np.sign(rows[0].voltage)
    result = scores(rows, [sign * median] * len(rows), phase, circuit)
    for subset in ("use_unclamped", "use_transition"):
        n_covered = n_eligible = 0
        for row in rows:
            keep = row.masks[subset]
            observation = np.abs(row.current[keep])
            n_covered += np.count_nonzero((observation >= low[keep]) & (observation <= high[keep]))
            n_eligible += len(observation)
        result[subset]["coverage_10_to_90"] = n_covered / n_eligible if n_eligible else None
    if phase == "SET" and circuit.compliance_a:
        onsets = np.array([_clip_voltage(rows[0].voltage, sample["current_A"]) for sample in samples])
        limits = np.quantile(onsets, [.10, .90])
        actual = np.array([_clip_voltage(row.voltage, row.current) for row in rows])
        # Discrete crossing voltages: compare against observed grids, allowing
        # numerical roundoff at endpoints and no extra voltage tolerance.
        result["compliance_onset"]["band_10_to_90_V"] = limits.tolist()
        result["compliance_onset"]["band_covered"] = int(np.count_nonzero(
            (actual >= limits[0] - 1e-8) & (actual <= limits[1] + 1e-8)))
    return result, np.stack((low, median, high))


def _record_curves(target, prefix, rows, predictions):
    target[prefix + "_voltage"] = np.stack([r.voltage for r in rows])
    target[prefix + "_measured"] = np.stack([r.current for r in rows])
    for field in ("current_A", "device_voltage_V", "soft_state", "hard_state"):
        target[prefix + "_" + field] = np.stack([p[field] for p in predictions])
    target[prefix + "_names"] = np.asarray([r.label for r in rows])


def _batch(rows, phase, device, preparation, circuit, initial=M.State()):
    return [simulation(row, phase, device, preparation, circuit, initial) for row in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    groups, circuits = load_groups()
    sk, rk = ("1T1M", 1.1, "SET"), ("1T1M", 1.1, "RESET")
    soft_s, soft_r = groups[sk], groups[rk]
    cs, cr = circuits[sk], circuits[rk]
    arrays = {}
    base_s, base_r = _aligned_baseline(soft_s[32:], soft_r[32:], "chronological_first35_to_later")
    strategy, inner = choose_set_strategy(soft_s[:32], cs)
    report = {"conditioned_1p1V": {"inner_train_only_window_selection": inner,
                                  "selected_set_strategy": strategy}}
    chronological = report["conditioned_1p1V"]["chronological_cycles_36_to_53"] = {}
    for label, method in (("static", "static"), ("selected", strategy)):
        device, prep, center_drift = fit_set_strategy(soft_s[:32], cs, method)
        initial = simulation(soft_s[0], "SET", device, prep, cs)["final"]
        device, prep = fit_soft_reset(soft_r[:32], cr, device, prep, initial)
        p_s = _batch(soft_s[32:], "SET", device, prep, cs)
        p_r = _batch(soft_r[32:], "RESET", device, prep, cr, initial)
        chronological[label] = {"SET": scores(soft_s[32:], p_s, "SET", cs),
                                "RESET": scores(soft_r[32:], p_r, "RESET", cr),
                                "parameters": asdict(device), "preparation": asdict(prep),
                                "training_center_drift_V": center_drift}
        _record_curves(arrays, label + "_1p1_SET", soft_s[32:], p_s)
        _record_curves(arrays, label + "_1p1_RESET", soft_r[32:], p_r)
    chronological["prior_model"] = {"SET": scores(soft_s[32:], base_s, "SET", cs),
                                    "RESET": scores(soft_r[32:], base_r, "RESET", cr)}
    arrays["prior_1p1_SET_current_A"] = np.stack(base_s)
    arrays["prior_1p1_RESET_current_A"] = np.stack(base_r)

    # Target excluding cycle scores are independent of the chronology tuning.
    ps, pr = [], []
    for odd in range(len(soft_s)):
        ds, prep = fit_soft_set(soft_s[:odd] + soft_s[odd+1:], cs)
        state = simulation(soft_s[odd], "SET", ds, prep, cs)["final"]
        ds, prep = fit_soft_reset(soft_r[:odd] + soft_r[odd+1:], cr, ds, prep, state)
        ps.append(simulation(soft_s[odd], "SET", ds, prep, cs))
        pr.append(simulation(soft_r[odd], "RESET", ds, prep, cr, state))
    old_s, old_r = _aligned_baseline(soft_s, soft_r, "leave_one_cycle_out")
    report["conditioned_1p1V"]["leave_one_cycle_out"] = {
        "static": {"SET": scores(soft_s, ps, "SET", cs), "RESET": scores(soft_r, pr, "RESET", cr)},
        "prior_model": {"SET": scores(soft_s, old_s, "SET", cs), "RESET": scores(soft_r, old_r, "RESET", cr)}}
    _record_curves(arrays, "loo_1p1_SET", soft_s, ps)
    _record_curves(arrays, "loo_1p1_RESET", soft_r, pr)
    report["conditioned_1p1V"]["common_trace_variability_floor"] = {
        "definition": "Retrospective lower bound for a single deterministic trace at fixed voltage samples, computed using observed currents in each evaluated group. It is not a prediction or a lower bound for a cycle-dependent model.",
        "cycles_4_to_53": {"SET": common_curve_floor(soft_s), "RESET": common_curve_floor(soft_r)},
        "cycles_36_to_53": {"SET": common_curve_floor(soft_s[32:]), "RESET": common_curve_floor(soft_r[32:])},
    }
    print("Finished 1.1 V chronological and cycle holdouts", flush=True)

    device, prep_soft = fit_soft_set(soft_s, cs)
    soft_initial = simulation(soft_s[0], "SET", device, prep_soft, cs)["final"]
    device, prep_soft = fit_soft_reset(soft_r, cr, device, prep_soft, soft_initial)
    report["gate_transfer_from_1p1V"] = {}
    for gate in (0.9, 1.3):
        cross = {}
        report["gate_transfer_from_1p1V"][str(gate)] = cross
        for phase in ("SET", "RESET"):
            key = ("1T1M", gate, phase)
            rows, circuit = groups[key], circuits[key]
            init = M.State() if phase == "SET" else soft_initial
            fixed = _batch(rows, phase, device, prep_soft, circuit, init)
            suffix = f"{gate:g}_{phase}"
            _record_curves(arrays, "gate_transfer_" + suffix, rows, fixed)
            boundary = len(rows) * 2 // 3
            # Refit only amplitudes on this gate's training sheets. Its
            # state/transport shapes, thresholds and hard path stay shared.
            if phase == "SET":
                ds, adapted = fit_soft_set(rows[:boundary], circuit, device, prep_soft, False)
            else:
                ds, adapted = fit_soft_reset(rows[:boundary], circuit, device, prep_soft, init, False)
            test = rows[boundary:]
            p = _batch(test, phase, ds, adapted, circuit, init)
            cross[phase] = {
                "no_gate_specific_fit_all_sheets": scores(rows, fixed, phase, circuit),
                "no_gate_specific_fit_held_out_sheets": scores(test, fixed[boundary:], phase, circuit),
                "amplitude_calibrated_held_out_sheets": scores(test, p, phase, circuit),
                "adapted_preparation": asdict(adapted),
            }
            _record_curves(arrays, "gate_amplitude_fit_" + suffix, test, p)

    bk, br = ("1M", None, "SET"), ("1M", None, "RESET")
    bare_s, bare_r = groups[bk], groups[br]
    bs, bc = circuits[bk], circuits[br]
    bare_dev, bare_prep = fit_bare_set(bare_s[:14], bs, device, prep_soft)
    # RESET sheets are a separate hard-LRS ensemble; no SET-to-RESET sheet
    # pairing is assumed. Initialize from hard programming, not target current.
    hard_initial = M.State(hard=1.0)
    device, bare_prep = fit_bare_reset(bare_r[:14], bc, bare_dev, bare_prep, hard_initial)
    set_predictions = _batch(bare_s[14:], "SET", device, bare_prep, bs)
    reset_predictions = _batch(bare_r[14:], "RESET", device, bare_prep, bc, hard_initial)
    report["standalone_held_out_sheets_15_to_22"] = {
        "SET": scores(bare_s[14:], set_predictions, "SET", bs),
        "RESET": scores(bare_r[14:], reset_predictions, "RESET", bc),
        "same_preparation_as_1p1V_SET": scores(
            bare_s[14:], _batch(bare_s[14:], "SET", device, prep_soft, bs), "SET", bs),
    }
    _record_curves(arrays, "bare_SET", bare_s[14:], set_predictions)
    _record_curves(arrays, "bare_RESET", bare_r[14:], reset_predictions)
    _record_curves(arrays, "bare_same_preparation_SET", bare_s[14:],
                   _batch(bare_s[14:], "SET", device, prep_soft, bs))

    report["standalone_training_population_band"] = {}
    for rows, test, phase, circuit in (
            (bare_s[:14], bare_s[14:], "SET", bs),
            (bare_r[:14], bare_r[14:], "RESET", bc)):
        samples = []
        training_parameters = []
        for row in rows:
            if phase == "SET":
                ds, prepared = fit_bare_set([row], circuit, device, bare_prep)
                initial = M.State()
            else:
                ds, prepared = fit_bare_reset([row], circuit, device, bare_prep, hard_initial)
                initial = hard_initial
            training_parameters.append({"sheet": row.label, "device": asdict(ds),
                                        "preparation": asdict(prepared)})
            samples.append(simulation(test[0], phase, ds, prepared, circuit, initial))
        scoring, band = score_ensemble(test, samples, phase, circuit)
        report["standalone_training_population_band"][phase] = {
            "held_out_scores": scoring, "training_parameter_samples": training_parameters}
        arrays["bare_population_" + phase + "_abs_current_quantiles_A"] = band
        arrays["bare_population_" + phase + "_training_currents_A"] = np.stack(
            [sample["current_A"] for sample in samples])

    report.update({
        "shared_model_parameters": asdict(device),
        "prepared_soft_amplitudes": asdict(prep_soft),
        "standalone_formation_amplitudes": asdict(bare_prep),
        "contract": {
            "soft_availability": "Hypothesized external preparation input, 1 for 1T1M conditioned sweeps and 0 for bare forming. Same-preparation bare SET is explicitly tested.",
            "hard_access_A": "0.5 mA chosen inside the bracket between the measured positive transistor maxima and 1 mA SMU setting. A rule based on available current, not a separately identified material constant.",
            "bare_set": "Fit off envelope and median censoring-interval center on sheets 1–14. Score transition only on uncensored points and separately report compliance onset. Hard formation sharpness is unresolved.",
            "bare_reset": "Fit a hard-path survival law on sheets 1–14, with an independently prepared hard LRS and the soft RESET leakage shape. SET and RESET sheet numbers are not treated as paired.",
            "holdouts": "All 1.1 V candidates are development evaluations on the existing archive. Other groups are sheet holdouts; no cycle ordering or pairing is inferred from sheet names alone.",
            "circuit": "Measured forward and reverse transistor tables, and a separate ideal SMU current cap for 1M SET. No physical time or measured transients in the state equations.",
            "provenance": {
                "paper": "https://doi.org/10.1038/s41586-023-05973-1",
                "source_data": "https://doi.org/10.5281/zenodo.7607096",
                "supplement": "https://repository.kaust.edu.sa/bitstreams/bdb101bf-745d-45d0-b250-3b2adbbda0a2/download",
                "scope": "User confirmed the local device sweeps are from Zhu et al. Main Fig. 2 contrasts nominally 0.053 µm² Au/Ti/h-BN/W structures and reports measurement-response / capacitance differences. Supplementary Fig. 6 separates forming from subsequent bipolar switching; Fig. 8 shows soft/filamentary coexistence for 1.5 V gate bias. Public archive listing does not map local workbook or sheet identities.",
                "hard_access_limitation": "A quasi-static bracket used only for the measured 0.9–1.3 V gates and the 1 mA bare protocol. It does not extrapolate to the mixed switching near 1.5 V. Transient current and actual preparation histories remain unresolved.",
            },
            "objective": "Equal total weight for eligible nonclamped points and transition points. Fit only training rows. Compare the same raw masks for all held-out candidates.",
        },
    })
    paths = [audit.ROOT / filename for _, _, _, filename in audit.WORKBOOKS]
    paths.extend(Path(__file__).with_name("data") / p for p in audit.TRANSISTOR_FILES)
    paths.extend([BASELINE, Path(__file__), Path(M.__file__), Path(audit.__file__)])
    report["sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (out / "shared_device_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    np.savez_compressed(out / "shared_device_curves.npz", **arrays)
    print("Saved shared_device_results.json and shared_device_curves.npz", flush=True)


if __name__ == "__main__":
    main()
