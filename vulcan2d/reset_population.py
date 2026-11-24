"""Predict a RESET current distribution with the calibrated shared hBN device."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import model_shared_device as model
from . import predict_gate

READ_V = .15
LEVELS = (.10, .50, .90)
CURRENT_FLOOR_A = 1e-18


def simulate_ensemble(wave, device, prep, circuit, state, read_log_a):
    """Simulate one RESET waveform for an array of effective prepared currents."""
    v = np.abs(wave)
    a = int(np.argmax(v))
    log_amp = np.atleast_1d(read_log_a)
    if state.soft <= 0. or state.hard != 0.:
        raise ValueError("The RESET population is calibrated for a prepared soft state")
    prepared = model.Preparation(
        off_set_a=prep.off_set_a, soft_set_a=prep.soft_set_a, off_reset_a=prep.off_reset_a,
        soft_reset_a=10**log_amp[:, None] / (READ_V * state.soft), soft_available=prep.soft_available)
    outward = np.broadcast_to(v[:a + 1], (len(log_amp), a+1))
    returning = np.broadcast_to(v[a+1:], (len(log_amp), len(v)-a-1))
    out, _, soft, hard = model.reset_outward(outward, state, device, prepared, circuit)
    back, _ = circuit.operating_point(returning, lambda vh: model.device_current(
        vh, soft[:, -1:], hard[:, -1:], device, prepared, "RESET"))
    return np.concatenate((out, back), axis=1)


def prepared_distribution(distribution, program_current):
    delta = np.log10(program_current) - distribution["log_current_center"]
    center = distribution["offset"] + distribution["slope"] * delta
    return center + np.asarray(distribution["quantiles"])


def predict(calibration, gate):
    minimum, maximum = calibration["evaluated_gate_range_v"]
    if not minimum <= gate <= maximum:
        raise ValueError(f"Gate voltage is outside [{minimum}, {maximum}] V")
    d = model.Device(**calibration["device"])
    p = model.Preparation(**calibration["preparation"])
    protocol = calibration["protocol"]
    cs, cr = predict_gate.circuits_at_gate(gate)
    set_voltage = predict_gate.triangle(protocol["set_peak_v"], protocol["step_v"])
    reset_voltage = predict_gate.triangle(protocol["reset_peak_v"], protocol["step_v"])
    programmed = model.simulate_sweep(set_voltage, "SET", d, p, cs)
    program_current = float(np.max(programmed["current_A"]))
    samples = prepared_distribution(calibration["program_distribution"], program_current)
    draws = simulate_ensemble(reset_voltage, d, p, cr, programmed["final"], samples)
    log_i = np.log10(np.maximum(draws, CURRENT_FLOOR_A))
    return reset_voltage, 10**log_i.mean(axis=0), 10**np.quantile(log_i, LEVELS, axis=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--gate", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("schema_version") != 1:
        raise ValueError("Unsupported RESET population calibration")
    v, center, bounds = predict(calibration, args.gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("voltage_V", "center_abs_current_A", "p10_abs_current_A",
                         "p50_abs_current_A", "p90_abs_current_A"))
        writer.writerows(zip(v, center, *bounds))


if __name__ == "__main__":
    main()
