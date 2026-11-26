"""Remove limiting segments, filter the fitting data, and audit RESET prediction."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from analysis import refit_iv41 as refit
from analysis import validate_programming_bridge as bridge
from vulcan2d import finalize_gate_model
from vulcan2d import model_shared_device as model
from vulcan2d import predict_gate as predictor
from vulcan2d import validate_gate_prediction as readpoint
from vulcan2d import validate_transition_regions as audit


def clean(rows, phase):
    return [replace(row, current=current, masks=masks, event=event)
            for row in rows
            for current, masks, event in [audit.filter_unclamped(
                row.voltage, row.current, phase, row.masks)]]


def matched_raw(original, cleaned):
    return [replace(filtered, current=row.current)
            for row, filtered in zip(original, cleaned)]


def mask_counts(rows):
    return {key: int(sum(np.count_nonzero(row.masks[key]) for row in rows))
            for key in ("eligible", "smu_compliance", "transistor_dominated", "lookup_mismatch",
                        "transistor_near_cap", "use_unclamped", "median_filter_changed")}


def reset_state_from_read(d, p, circuit, initial, observation):
    voltage, current = observation
    probe = np.array([0., -voltage, 0.])

    def residual(soft):
        state = replace(initial, soft=soft)
        read = model.simulate_sweep(probe, "RESET", d, p, circuit, state)["current_A"][1]
        return abs(read) - current

    left, right = residual(0.), residual(p.soft_available)
    if np.sign(left) == np.sign(right):
        raise ValueError("Early current is outside the fixed transport state's range")
    soft = brentq(residual, 0., p.soft_available, xtol=1e-10)
    return replace(initial, soft=soft)


def condition(cohort, raw, gate, d, p):
    written, erased = refit.simulate(gate, cohort, d, p)
    v = np.abs(raw[0].voltage)
    initial = written["final"]
    circuit = refit.circuits(gate)[1]
    errors = defaultdict(lambda: defaultdict(list))
    attempts = defaultdict(int)
    for row, original in zip(cohort["RESET"], raw):
        outward = row.masks["branch_outward"]
        prefix = outward & (v <= .20 + 1e-9)
        masks = {**row.masks, **{k: row.masks[k] & ~prefix
                                 for k in ("use_unclamped", "use_transition")}}
        scored = replace(original, masks=masks)
        candidates = {"voltage_only": erased}
        try:
            observation, _ = readpoint._read_and_mask(row)
            attempts["usable_0p20V_read"] += 1
        except ValueError:
            observation = None
        if observation is not None:
            # Require lookup headroom even when a lone point evades the
            # persistence criterion for excluding a clipped segment.
            a = np.flatnonzero(outward & np.isclose(v, .20))[0]
            if row.masks["current_to_transistor_cap"][a] < audit.NEAR_TRANSISTOR_CAP:
                for name in ("amplitude_read", "state_read"):
                    try:
                        if name == "amplitude_read":
                            prep = predictor._condition_reset(d, p, circuit, initial, observation)
                            state = initial
                        else:
                            prep = p
                            state = reset_state_from_read(d, p, circuit, initial, observation)
                        candidates[name] = model.simulate_sweep(
                            row.voltage, "RESET", d, prep, circuit, state)
                        attempts[name] += 1
                    except ValueError:
                        pass
        baseline = refit.error_scores([scored], "RESET", erased["current_A"])
        for name in ("voltage_only", "amplitude_read", "state_read"):
            result = (baseline if name not in candidates or name == "voltage_only" else
                      refit.error_scores([scored], "RESET", candidates[name]["current_A"]))
            for region in ("valid", "out", "return", "transition"):
                if region in result:
                    errors[name][region].append(result[region] ** 2)
                    if name != "voltage_only" and name in candidates:
                        errors[f"{name}_matched_baseline"][region].append(baseline[region]**2)
                        errors[f"{name}_matched"][region].append(result[region]**2)
    return {"available": dict(attempts),
            "rmse_decade": {name: {region: float(np.sqrt(np.mean(values)))
                                  for region, values in by_region.items()}
                            for name, by_region in errors.items()}}


def fit_reset_feedback(training, d, p):
    """Fit the existing local-voltage feedback law on the cleaned anchor scans."""
    gates = tuple(training)
    rows = {("1T1M", g, phase): group[phase] for g, group in training.items()
            for phase in refit.PHASES}
    circuits = {("1T1M", g, phase): c for g in gates
                for phase, c in zip(refit.PHASES, refit.circuits(g))}
    calibration = predictor.GateCalibration(gates, d, (p,) * len(gates))
    return finalize_gate_model._fit_reset_feedback(rows, circuits, calibration)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--bare-preparation", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    index = refit.origin_cache(args.workbook, args.cache)
    prior = predictor.GateCalibration.from_file(args.prior)
    gates = (*bridge.PROGRAM_GATES, *bridge.ASSESS_GATES)
    raw = {g: refit.load_cohort(index, args.cache, round((g - .9) * 10) + 5, g)[0] for g in gates}
    cleaned = {g: {phase: clean(rows[phase], phase) for phase in refit.PHASES}
               for g, rows in raw.items()}
    training = {g: {phase: refit.screen(cleaned[g][phase], g, phase)[0] for phase in refit.PHASES}
                for g in bridge.PROGRAM_GATES}
    shape = replace(prior.device, soft_growth_current_a=None)
    seed = prior.preparation(1.1)
    d, p = bridge.fit_preparation(training, shape, seed, True)

    feedback = fit_reset_feedback(training, d, p)
    bare_prior = predictor.GateCalibration.from_file(args.bare_preparation)
    _, bare, bare_rows = bridge.fit_bare_comparison(
        shape, bare_prior.preparation(1.1), preprocess=clean, score_on_raw=True)
    report = {
        "preprocessing": {g: {phase: mask_counts(rows[phase]) for phase in refit.PHASES}
                          for g, rows in cleaned.items()},
        "bare_preprocessing": {phase: mask_counts(bare_rows[phase]) for phase in refit.PHASES},
        "bare_holdout_scored_on_raw_selected_points": bare,
        "gate_holdout_scored_on_raw_selected_points": {},
        "source_sha256": refit.sha_file(args.workbook),
        "prior_calibration_sha256": refit.sha_file(args.prior),
        "bare_preparation_sha256": refit.sha_file(args.bare_preparation),
    }
    for gate in bridge.ASSESS_GATES:
        clean_rows = cleaned[gate]
        test = {phase: matched_raw(raw[gate][phase], clean_rows[phase]) for phase in refit.PHASES}
        a = refit.simulate(gate, test, prior.device, prior.preparation(1.1))
        b = refit.simulate(gate, test, d, p)
        score = {}
        for phase, old, new in zip(refit.PHASES, a, b):
            score[phase] = {
                "before_refit": refit.error_scores(test[phase], phase, old["current_A"]),
                "filtered_refit": refit.error_scores(test[phase], phase, new["current_A"]),
                "single_trace_floor": refit.deterministic_floor(test[phase], phase),
            }
        reset_c = refit.circuits(gate)[1]
        local = model.simulate_sweep(test["RESET"][0].voltage, "RESET", feedback.device,
                                     p, reset_c, b[0]["final"])
        score["RESET"]["local_voltage_feedback"] = refit.error_scores(
            test["RESET"], "RESET", local["current_A"])
        score["early_read"] = condition(clean_rows, test["RESET"], gate, d, p)
        report["gate_holdout_scored_on_raw_selected_points"][gate] = score
        example = {name: round(score["RESET"][name]["valid"], 4)
                   for name in ("before_refit", "filtered_refit", "local_voltage_feedback")}
        print(gate, example, score["early_read"]["available"], flush=True)
    (args.output / "filtered_reset_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    predictor.GateCalibration(bridge.PROGRAM_GATES, d, (p,) * len(bridge.PROGRAM_GATES)).to_file(
        args.output / "filtered_iv41_calibration.json")
    feedback.to_file(args.output / "filtered_iv41_feedback_experiment.json")


if __name__ == "__main__":
    main()
