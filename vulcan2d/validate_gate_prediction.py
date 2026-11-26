"""Withhold all 1.1 V device currents and test gate interpolation in its interior."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from . import cycle_quality as quality
from . import predict_gate as predictor
from . import validate_optimized_device as optimization
from . import validate_shared_device as shared

GATE_LOW, GATE_TEST, GATE_HIGH = .9, 1.1, 1.3
READ_V = .20


def fit_endpoints(groups, circuits):
    # Callers supply endpoint-only mappings. Fit neither endpoint thresholds
    # nor interpolation weights with any 1.1 V current observations.
    expected = {("1T1M", gate, phase) for gate in (GATE_LOW, GATE_HIGH)
                for phase in ("SET", "RESET")}
    if set(groups) != expected or set(circuits) != expected:
        raise ValueError("Endpoint calibration accepts only endpoint-gate data")
    flags = quality.screen_groups(groups)
    selected = {key: quality.split_with_screen(rows, flags[key])
                for key, rows in groups.items()}
    limits = {gate: {phase: len(selected["1T1M", gate, phase]) for phase in ("SET", "RESET")}
              for gate in (GATE_LOW, GATE_HIGH)}
    dev, prep, set_models = optimization.fit_all_gates(selected, circuits, limits, "SET")
    dev, prep, _ = optimization.fit_all_gates(
        selected, circuits, limits, "RESET", dev, prep, set_by_gate=set_models)
    c = predictor.GateCalibration((GATE_LOW, GATE_HIGH), dev, (prep[GATE_LOW], prep[GATE_HIGH]))
    return c, limits


def _read_and_mask(row):
    """Use only the first 0.20 V of the held-out outgoing branch as data input."""
    v = np.abs(row.voltage)
    candidate = np.flatnonzero(
        row.masks["branch_outward"] & row.masks["use_unclamped"] & np.isclose(v, READ_V))
    if len(candidate) != 1:
        raise ValueError(f"{row.label}: expected one valid early RESET read")
    idx = candidate[0]
    known_prefix = row.masks["branch_outward"] & (v <= v[idx] + 1e-12)
    scored_masks = {**row.masks, **{k: row.masks[k] & ~known_prefix
                                   for k in ("eligible", "use_unclamped", "use_transition")}}
    masked = replace(row, masks=scored_masks)
    return (float(v[idx]), float(abs(row.current[idx]))), masked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    out = parser.parse_args(argv).output_dir
    out.mkdir(parents=True, exist_ok=True)
    all_data, all_circuits = shared.load_groups(include_abnormal_pairs=True)
    endpoints = {k: rows for k, rows in all_data.items() if k[1] in (GATE_LOW, GATE_HIGH)}
    circuits = {k: c for k, c in all_circuits.items() if k in endpoints}
    calibration, training = fit_endpoints(endpoints, circuits)
    calibration.to_file(out / "gate_calibration.json")
    # Only after calibration is frozen, assemble the target-gate test rows.
    keys = [("1T1M", GATE_TEST, phase) for phase in ("SET", "RESET")]
    target_data = {k: all_data[k] for k in keys}
    flags = quality.screen_groups(target_data)
    written, erased = [quality.split_with_screen(target_data[k], flags[k]) for k in keys]
    if [s.label[1:] for s in written] != [r.label[1:] for r in erased]:
        raise ValueError("Target SET and RESET cycle identifiers do not align")
    target_circuits = tuple(all_circuits[k] for k in keys)
    reads_and_scored_rows = [_read_and_mask(r) for r in erased]
    subsets = {"all_50": slice(None), "cycles_36_to_53": slice(32, None)}
    report = {"calibration_gate_v": calibration.gates_v, "holdout_gate_v": GATE_TEST,
              "training_sweeps": training, "read_voltage_magnitude_v": READ_V,
              "holdout_labels": [r.label for r in erased],
              "conditional_protocol": "Measured current at |V|=0.2 V at outward RESET is used only by the optional conditioned forecast. Partial RESET scores remove outward |V|≤0.2 V from every comparison.",
              "predictions": {}}
    arrays = {}
    comparisons = {
        "fixed_low": (replace(calibration, preparations=(calibration.preparations[0],)*2), "linear", False),
        "fixed_high": (replace(calibration, preparations=(calibration.preparations[1],)*2), "linear", False),
        "log": (calibration, "log", False),
        "linear": (calibration, "linear", False),
        "log_read": (calibration, "log", True),
    }
    for method, (test_calibration, interpolation, conditioned) in comparisons.items():
        ws, rs, failures = [], [], []
        for s, r, (read, _) in zip(written, erased, reads_and_scored_rows):
            init = read if conditioned else None
            try:
                w, e = test_calibration.predict(GATE_TEST, s.voltage, r.voltage,
                                                method=interpolation,
                                                initial_read=init, circuits=target_circuits)
            except ValueError:
                # Keep the conditional comparison honest if a supplied read
                # lies outside the available load line.
                if not conditioned:
                    raise
                failures.append(r.label)
                w, e = test_calibration.predict(GATE_TEST, s.voltage, r.voltage,
                                                method=interpolation, circuits=target_circuits)
            ws.append(w)
            rs.append(e)
        entry = report["predictions"][method] = {"fallback_labels": failures, "scores": {}}
        shared._record_curves(arrays, method + "_SET", written, ws)
        shared._record_curves(arrays, method + "_RESET", erased, rs)
        for label, subset in subsets.items():
            partial_reset = [pair[1] for pair in reads_and_scored_rows][subset]
            entry["scores"][label] = {
                "SET": shared.scores(written[subset], ws[subset], "SET", target_circuits[0]),
                "RESET_prefix_excluded": shared.scores(
                    partial_reset, rs[subset], "RESET", target_circuits[1]),
            }
            if not conditioned:
                entry["scores"][label]["RESET"] = shared.scores(
                    erased[subset], rs[subset], "RESET", target_circuits[1])
    apex = int(np.argmax(np.abs(erased[0].voltage)))
    outward = slice(None, apex+1)
    report["initial_read_A"] = {
        "observed_median": float(np.median([r[0][1] for r in reads_and_scored_rows])),
        **{k: float(np.interp(READ_V, np.abs(arrays[k+"_RESET_voltage"][0, outward]),
                             np.abs(arrays[k+"_RESET_current_A"][0, outward])))
           for k in ("log", "linear")},
    }
    (out / "gate_prediction_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(out / "gate_predictions.npz", **arrays)


if __name__ == "__main__":
    main()
