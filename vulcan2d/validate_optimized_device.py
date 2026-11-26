"""Retune SET timing and fit a common device law across gate conditions.

Run from VULCAN-2D: python3 -m vulcan2d.validate_optimized_device --output-dir PATH
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from . import cycle_quality
from . import model_shared_device as device_model
from . import validate_transition_regions as audit
from . import validate_shared_device as shared


SET = "SET"
RESET = "RESET"
GATES = (.9, 1.1, 1.3)
STRATEGIES = ("reference", "timing", "recent6", "recent12")
TIMING_SCALE_V = .25


def _event(voltage, current):
    return audit.prepare_curve(voltage, current, SET)[1].get("event_abs_voltage_V")


def fit_set(rows, circuit, method):
    if method not in STRATEGIES:
        raise ValueError(f"Unknown SET fit strategy: {method}")
    half_life = {"recent6": 6, "recent12": 12}.get(method)
    weights = np.ones(len(rows)) if half_life is None else np.exp2(
        np.arange(1-len(rows), 1) / half_life)
    prior, preparation = shared.fit_soft_set(rows, circuit, cycle_weights=weights)
    if method == "reference":
        return prior, preparation
    v, keep, target, weight = shared._training_data(rows, cycle_weights=weights)
    events = np.array([r.event.get("event_abs_voltage_V", np.nan) for r in rows])
    reported = np.isfinite(events)
    if not reported.any():
        return prior, preparation

    def evaluate(p):
        predicted = device_model.simulate_sweep(v, SET, p, preparation, circuit)["current_A"]
        when = _event(v, predicted)
        if when is None:
            return np.inf
        residual = weight * (np.log10(np.abs(predicted[keep])) - target)
        timing = np.average(np.abs(events[reported] - when), weights=weights[reported])
        return np.dot(residual, residual) + (TIMING_SCALE_V * timing)

    candidates = [prior]
    for shift in np.linspace(-.12, .12, 17):
        for scale in (.65, .85, 1., 1.25, 1.55):
            candidates.append(replace(prior, set_center_v=prior.set_center_v + shift,
                                      set_width_v=prior.set_width_v * scale))
    best = min(candidates, key=evaluate)
    candidates = [best] + [
        replace(best, set_center_v=best.set_center_v + shift,
                set_width_v=best.set_width_v * scale)
        for shift in (-.01, 0., .01) for scale in (.91, 1., 1.10)
    ]
    return min(candidates, key=evaluate), preparation


def _predict(rows, phase, device, preparation, circuit, initial=device_model.State()):
    return [shared.simulation(row, phase, device, preparation, circuit, initial) for row in rows]


def _metric(scored):
    return scored["use_transition"]["pooled_rmse_decade"], scored["event"]["mae_V"]


def _metric_detail(scored):
    activity, timing = _metric(scored)
    return {"transition_rmse_decade": activity,
            "transition_voltage_mae_V": timing,
            "n_events_scored": scored["event"]["n_compared"],
            "full_trace_rmse_decade": scored["eligible"]["pooled_rmse_decade"]}


def _pick_strategy(results):
    # Require simultaneous improvement over the reference on both training-only
    # validation metrics. This guards against buying timing with worse current.
    base_i, base_v = _metric(results["reference"])
    better = [k for k, v in results.items()
              if _metric(v)[1] is not None
              and _metric(v)[0] < base_i and _metric(v)[1] < base_v
              and v["event"]["n_compared"] >= results["reference"]["event"]["n_compared"]]
    return min(better, key=lambda k: sum(x/y for x, y in zip(_metric(results[k]), (base_i, base_v))),
               default="reference")


def _compare_blocks(rows, circuit, splits, online):
    comparisons = {}
    for method in STRATEGIES:
        tests, preds = [], []
        for stop, end in splits:
            if online:
                for index in range(stop, end):
                    params, prep = fit_set(rows[:index], circuit, method)
                    preds.extend(_predict(rows[index:index+1], SET, params, prep, circuit))
            else:
                params, prep = fit_set(rows[:stop], circuit, method)
                preds.extend(_predict(rows[stop:end], SET, params, prep, circuit))
            tests.extend(rows[stop:end])
        comparisons[method] = shared.scores(tests, preds, SET, circuit)
    return comparisons


def fit_all_gates(groups, circuits, limits, phase, starting_devices=None,
                  starting_preps=None, local_state="none", set_by_gate=None):
    """Fit one kinetics law with two bounded, gate-specific readout amplitudes."""
    template = shared.REFERENCE if starting_devices is None else starting_devices
    prep_by_gate = ({g: shared.PREPARATION for g in limits} if starting_preps is None
                    else starting_preps)
    data = {g: shared._training_data(groups["1T1M", g, phase][:counts[phase]])
            for g, counts in limits.items()}
    initial_states = {}
    for g in limits:
        if phase == SET:
            initial_states[g] = device_model.State()
        else:
            v = groups["1T1M", g, SET][0].voltage
            initial_states[g] = device_model.simulate_sweep(
                v, SET, template if set_by_gate is None else set_by_gate[g],
                prep_by_gate[g], circuits["1T1M", g, SET])["final"]
    if phase == SET:
        global_start = [template.beta_set, template.soft_exponent,
                        template.set_center_v, np.log(template.set_width_v)]
        global_low = [2, .6, .45, np.log(.008)]
        global_high = [10, 2.4, 2.1, np.log(.30)]
        off_name, on_name = "off_set_a", "soft_set_a"
        local_low, local_high = [-13.5, -6.7], [-7.5, -3.5]
    else:
        global_start = [template.beta_reset, np.log(template.reset_soft_alpha),
                        template.reset_center_v, np.log(template.reset_width_v)]
        global_low = [.3, np.log(.35), .15, np.log(.025)]
        global_high = [12, np.log(5.), 1.6, np.log(.55)]
        off_name, on_name = "off_reset_a", "soft_reset_a"
        local_low, local_high = [-15, -10], [-8.3, -4.3]

    offsets_per_gate = {"none": 0, "center": 1, "center_width": 2}[local_state]
    stride = 2 + offsets_per_gate

    def decode(p):
        beta, response, center, log_width = p[:4]
        if phase == SET:
            dev = replace(template, beta_set=beta, soft_exponent=response,
                          set_center_v=center, set_width_v=np.exp(log_width))
        else:
            dev = replace(template, beta_reset=beta, reset_soft_alpha=np.exp(response),
                          reset_center_v=center, reset_width_v=np.exp(log_width))
        preps, models = {}, {}
        for g, local in zip(limits, p[4:].reshape(-1, stride)):
            a, b = local[:2]
            preps[g] = replace(prep_by_gate[g], **{off_name: 10 ** a, on_name: 10 ** b})
            center_factor = np.exp(local[2]) if offsets_per_gate else 1.
            width_factor = np.exp(local[3]) if offsets_per_gate == 2 else 1.
            if phase == SET:
                models[g] = replace(dev, set_center_v=center*center_factor,
                                    set_width_v=np.exp(log_width)*width_factor)
            else:
                models[g] = replace(dev, reset_center_v=center*center_factor,
                                    reset_width_v=np.exp(log_width)*width_factor)
        return dev, preps, models

    def objective(p):
        _, preps, models = decode(p)
        errors = []
        for g, (v, use, expected, weights) in data.items():
            result = device_model.simulate_sweep(v, phase, models[g], preps[g],
                                                 circuits["1T1M", g, phase], initial_states[g])
            errors.append(weights * (np.log10(np.abs(result["current_A"][use])) - expected))
        if offsets_per_gate:
            offsets = p[4:].reshape(-1, stride)[:, 2:]
            penalty_scale = np.array([.20, .10][:offsets_per_gate])
            errors.append((offsets * penalty_scale).ravel())
        return np.concatenate(errors)

    local_starts = []
    for g in limits:
        p = prep_by_gate[g]
        local_starts.extend([np.log10(getattr(p, off_name)), np.log10(getattr(p, on_name))]
                            + [0.] * offsets_per_gate)
    bounds_low = local_low + [-.60, -1.10][:offsets_per_gate]
    bounds_high = local_high + [.60, 1.10][:offsets_per_gate]
    result = shared._least_squares(objective, global_start + local_starts,
                                  (global_low + bounds_low * len(limits),
                                   global_high + bounds_high * len(limits)))
    return decode(result)


def _compare_gates(groups, circuits, limits, local_state, ends=None, arrays=None):
    device, preps, set_models = fit_all_gates(
        groups, circuits, limits, SET, local_state=local_state)
    shared_device, preps, reset_models = fit_all_gates(
        groups, circuits, limits, RESET, device, preps, local_state, set_models)
    models = {g: replace(reset_models[g], set_center_v=set_models[g].set_center_v,
                         set_width_v=set_models[g].set_width_v) for g in limits}
    x = {}
    for phase in (SET, RESET):
        for gate, counts in limits.items():
            key = ("1T1M", gate, phase)
            test = groups[key][counts[phase]:None if ends is None else ends[gate][phase]]
            s_key = ("1T1M", gate, SET)
            model = models[gate]
            initial = (device_model.State() if phase == SET else device_model.simulate_sweep(
                groups[s_key][0].voltage, SET, model, preps[gate], circuits[s_key])["final"])
            predicted = _predict(test, phase, model, preps[gate], circuits[key], initial)
            x.setdefault(str(gate), {})[phase] = shared.scores(test, predicted, phase, circuits[key])
            if arrays is not None:
                shared._record_curves(arrays, f"joint_{local_state}_{gate:g}_{phase}", test, predicted)
    return x, shared_device, preps, models


def _score_gate_candidate(scores):
    """Macro-average transition RMSE, with equal weighting of gates and phases."""
    return float(np.mean([scores[str(g)][phase]["use_transition"]["pooled_rmse_decade"]
                          for phase in (SET, RESET) for g in GATES]))


def _gate_benchmarks(groups, circuits, limits, arrays):
    inner_limits = {g: {phase: n*3//4 for phase, n in counts.items()}
                    for g, counts in limits.items()}
    candidates = ("none", "center", "center_width")
    inner = {}
    for local_state in candidates:
        scores, _, _, _ = _compare_gates(groups, circuits, inner_limits, local_state, ends=limits)
        inner[local_state] = {"macro_rmse_decade": _score_gate_candidate(scores), "scores": scores}
        print("gates inner", local_state, inner[local_state]["macro_rmse_decade"], flush=True)
    pick = min(candidates, key=lambda k: inner[k]["macro_rmse_decade"])
    benchmarks = {}
    for local_state in dict.fromkeys(("none", pick)):
        scores, dev, preps, models = _compare_gates(groups, circuits, limits, local_state, arrays=arrays)
        benchmarks[local_state] = {"held_out": scores, "shared_device": asdict(dev),
                                   "circuits": {str(g): {"device": asdict(models[g]),
                                                         "prep": asdict(preps[g])} for g in GATES}}
    return {"training_counts": limits, "inner": inner, "selected": pick, "benchmarks": benchmarks}


def _standalone_bridge(groups, circuits, starting, preparation, arrays):
    """Refit measured bare-device anchors using the selected compact law."""
    sk, rk = ("1M", None, SET), ("1M", None, RESET)
    set_rows, reset_rows = groups[sk], groups[rk]
    hard_initial = device_model.State(hard=1.)
    model, bare = shared.fit_bare_set(set_rows[:14], circuits[sk], starting, preparation)
    model, bare = shared.fit_bare_reset(reset_rows[:14], circuits[rk], model, bare, hard_initial)
    result = {"device": asdict(model), "preparation": asdict(bare),
              "training_rows": 14, "independent_reset_initial": asdict(hard_initial)}
    for phase, rows, circuit, initial in (
            (SET, set_rows, circuits[sk], device_model.State()),
            (RESET, reset_rows, circuits[rk], hard_initial)):
        predicted = _predict(rows[14:], phase, model, bare, circuit, initial)
        result[phase] = shared.scores(rows[14:], predicted, phase, circuit)
        shared._record_curves(arrays, f"standalone_bridge_{phase}", rows[14:], predicted)
    return result


def _data_for_calibration():
    raw, circuits = shared.load_groups(include_abnormal_pairs=True)
    screen = cycle_quality.screen_groups(raw)
    groups, limits, summary = {}, {}, []
    for key, rows in raw.items():
        topology, gate, phase = key
        flags = screen[key]
        boundary = cycle_quality.TRAINING_BOUNDARY[gate]
        training = cycle_quality.split_with_screen(rows, flags, stop=boundary)
        validation = cycle_quality.split_with_screen(rows, flags, start=boundary)
        groups[key] = training + validation
        if topology == "1T1M":
            limits.setdefault(gate, {})[phase] = len(training)
        summary.append({
            "topology": topology, "gate_v": gate, "phase": phase,
            "training_before_screening": boundary, "training_retained": len(training),
            "validation_before_screening": len(rows) - boundary,
            "validation_retained": len(validation),
            "excluded": [asdict(flag) for flag in flags if flag.excluded],
        })
    return groups, circuits, limits, summary


def main(argv=None):
    fmt = argparse.ArgumentParser(description=__doc__)
    fmt.add_argument("--output-dir", type=Path, required=True)
    args = fmt.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups, circuits, limits, quality = _data_for_calibration()
    set_key, reset_key = ("1T1M", 1.1, SET), ("1T1M", 1.1, RESET)
    rows, circuit = groups[set_key], circuits[set_key]
    reset_rows, reset_circuit = groups[reset_key], circuits[reset_key]
    boundary = limits[1.1][SET]
    if [row.label[1:] for row in rows] != [row.label[1:] for row in reset_rows]:
        raise ValueError("1.1 V SET/RESET sweep labels no longer align")
    cuts = [int(boundary * fraction) for fraction in (.4375, .625, .8125, 1.)]
    splits = tuple(zip(cuts[:-1], cuts[1:]))
    report, arrays = {"quality_screen": quality}, {}
    for mode in ("frozen", "sequential"):
        online = mode == "sequential"
        inner = _compare_blocks(rows[:boundary], circuit, splits, online)
        strategy = _pick_strategy(inner)
        print(mode, "training-only selected", strategy,
              {k: _metric_detail(s) for k, s in inner.items()}, flush=True)
        entry = report[mode] = {"inner": inner, "selected": strategy, "held_out": {}}
        for strategy_name in dict.fromkeys(("reference", strategy)):
            fit, predicted, past_counts = [], [], []
            for j in range(boundary, len(rows)):
                n = j if online else boundary
                model, prep = (fit_set(rows[:n], circuit, strategy_name)
                               if not fit or online else fit[-1])
                fit.append((model, prep))
                predicted.append(shared.simulation(rows[j], SET, model, prep, circuit))
                past_counts.append(n)
            scored = shared.scores(rows[boundary:], predicted, SET, circuit)
            entry["held_out"][strategy_name] = {"SET": scored, "fits": [
                {"fit_sweeps": n, "device": asdict(item[0]), "prep": asdict(item[1])}
                for n, item in zip(past_counts, fit)]}
            shared._record_curves(arrays, mode + "_" + strategy_name + "_SET", rows[boundary:], predicted)
            if mode == "frozen":
                initializer = predicted[0]["final"]
                rd, rp = shared.fit_soft_reset(reset_rows[:boundary], reset_circuit, model, prep, initializer)
                rp_curves = _predict(reset_rows[boundary:], RESET, rd, rp, reset_circuit, initializer)
                entry["held_out"][strategy_name]["RESET"] = shared.scores(
                    reset_rows[boundary:], rp_curves, RESET, reset_circuit)
                entry["held_out"][strategy_name]["calibration"] = {
                    "device": asdict(rd), "prep": asdict(rp),
                    "set_to_reset_initial": asdict(initializer)}
                shared._record_curves(arrays, mode + "_" + strategy_name + "_RESET", reset_rows[boundary:], rp_curves)
            print(mode, strategy_name, _metric_detail(scored), flush=True)
    report["joint"] = _gate_benchmarks(groups, circuits, limits, arrays)
    frozen = report["frozen"]
    chosen = frozen["held_out"][frozen["selected"]]["calibration"]
    report["standalone_bridge"] = _standalone_bridge(
        groups, circuits, device_model.Device(**chosen["device"]),
        device_model.Preparation(**chosen["prep"]), arrays)
    report["common_trace_floor"] = shared.common_curve_floor(rows[boundary:])
    report["contract"] = {
        "frozen": "Fits stop at cycle 35; cycles 36–53 are all predicted with frozen parameters.",
        "sequential": "Each test cycle is predicted using only current traces from fully completed earlier cycles. Scores evaluate one-step forecasts, not fixed-parameter transfer.",
        "joint": "Fits a common transport law to all three gates. All candidates calibrate gate-specific preparation amplitudes. Optional regularized center and width shifts are chosen on a training-only split and require observations from the target gate.",
        "selection": "Selection uses cycles 4–35 only and requires better transition RMSE and switch-voltage MAE with no loss of event coverage.",
        "standalone": "The same device equations include independent bare hard-path anchors fit on the first 14 sheets. The last 8 sheets are scoring only. RESET starts at an independently supplied hard state. SET compliance remains censoring.",
        "raw_data": "Project workbooks used. The user confirmed all three gate conditions come from the same device.",
        "quality": "Use fixed branch-contrast criteria to exclude failed/nonstandard sweeps from normal-cycle fitting and scoring without moving raw training boundaries. Use only each group's training sheets to set relative conductance thresholds.",
    }
    np.savez_compressed(args.output_dir / "optimized_curves.npz", **arrays)
    (args.output_dir / "optimization_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Finished joint gate calibration", flush=True)


if __name__ == "__main__":
    main()
