"""Gate-only IV41 SET/RESET prediction on the calibrated voltage protocol."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import model_shared_device
from . import predict_gate
from . import reset_population

PHASES = ("SET", "RESET")
LOG_FLOOR = -18.
MIN_LOCAL_V = .015


def log_abs(current):
    return np.log10(np.maximum(np.abs(current), 10**LOG_FLOOR))


def align(wave, source, values):
    """Resample the outgoing and return limbs independently."""
    v, u = np.abs(wave), np.abs(source)
    a, b = int(v.argmax()), int(u.argmax())
    return np.r_[np.interp(v[:a+1], u[:b+1], values[:b+1]),
                 np.interp(v[a+1:], u[b:][::-1], values[b:][::-1])]


def apparent_log_conductance(voltage, current, transistor):
    """Invert the monotone 1T load line and return a stable apparent I/V."""
    v, i = np.abs(voltage), np.abs(current)
    grid, drain = transistor
    allowed = np.interp(v, grid, drain)
    levels, index = np.unique(drain, return_index=True)
    local = v - np.interp(i, levels, grid[index])
    supported = (i<.95*allowed)&(local>=MIN_LOCAL_V)
    return log_abs(i)-np.log10(np.maximum(local, MIN_LOCAL_V)), supported


def original(calibration, gate, waveforms, circuits):
    model = model_shared_device
    old = calibration["physical_prior"]
    d = model.Device(**old["device"])
    p = model.Preparation(**old["preparation"])
    vset, _ = (waveforms[phase] for phase in PHASES)
    s = model.simulate_sweep(vset, "SET", d, p, circuits[0])
    reset_v, reset_i, _ = reset_population.predict(old, gate)
    reset_y = align(waveforms["RESET"],reset_v,log_abs(reset_i))
    return {"SET":log_abs(s["current_A"]), "RESET":reset_y}


def interpolate(gates, series, gate, edge_fraction=0.):
    left = int(np.clip(np.searchsorted(gates,gate)-1,0,len(gates)-2))
    t = (gate-gates[left])/(gates[left+1]-gates[left])
    if t < 0.:
        t *= edge_fraction
    elif t > 1.:
        t = 1.+(t-1.)*edge_fraction
    return series[left] + t*(series[left+1]-series[left])


def current_for_method(wave, circuit, base_log_i, anchors, gate, method, edge):
    if method == "prior":
        return np.sign(wave)*10**base_log_i
    key = "local_mean" if method.startswith("local") else "terminal_mean"
    if method.endswith("_delta"):
        key = "local_delta" if method.startswith("local") else "terminal_delta"
    gates = tuple(anchors)
    y = interpolate(gates, np.array([anchors[g][key] for g in gates]), gate, edge)
    v = np.abs(wave)
    if method == "terminal_delta":
        y += base_log_i
    elif method.startswith("local") and method.endswith("_delta"):
        y += apparent_log_conductance(v,10**base_log_i,circuit.transistor)[0]
        if method == "local_bounded_delta" and (gate < gates[0] or gate > gates[-1]):
            trained = np.stack([anchors[g]["local_mean"] for g in gates])
            y = np.clip(y, trained.min(axis=0), trained.max(axis=0))
    if method.startswith("local"):
        conductance = 10**np.clip(y, LOG_FLOOR, 0.)
        current, _ = circuit.operating_point(v,lambda vh: conductance*vh)
    else:
        current = np.minimum(10**np.clip(y, LOG_FLOOR, 0.),circuit.supply_bound(v))
    return np.sign(wave)*current


def erased_state_current(voltage, circuit, log_conductances, beta):
    field_g, retained_g = 10**np.asarray(log_conductances)
    current, _ = circuit.operating_point(
        voltage, lambda local: local * (field_g * np.exp(beta * np.sqrt(local)) + retained_g))
    return current


def current_with_reset_return(wave, circuit, baseline, gate, transport):
    """Use the erased-state I–V on the return, matching at the voltage apex."""
    anchors = sorted((float(g), terms) for g, terms in transport["log_conductances"].items())
    gates, coefficients = zip(*anchors)
    terms = interpolate(gates, np.asarray(coefficients), gate)
    v = np.abs(wave)
    apex = int(v.argmax())
    current = erased_state_current(v, circuit, terms, transport["beta_sqrt_v"])

    # Join the two predictions during the last tenth of the outward voltage
    # ramp so that both limbs use the same erased-state current at the apex.
    ramp = np.clip((v[:apex] / v[apex] - .9) / .1, 0., 1.)
    weight = ramp**2 * (3 - 2*ramp)
    changed = np.flatnonzero(weight)
    early = np.abs(baseline[:apex]).copy()
    early[changed] = 10**((1-weight[changed])*log_abs(early[changed])
                          + weight[changed]*log_abs(current[changed]))
    return np.sign(wave) * np.r_[early, current[apex:]]


def predict(calibration, gate):
    if calibration.get("schema_version") != 1:
        raise ValueError("Unsupported gate-only calibration")
    minimum, maximum = calibration["validated_gate_range_v"]
    if not minimum <= gate <= maximum:
        raise ValueError(f"Gate voltage must lie in [{minimum}, {maximum}] V")
    circuits = predict_gate.circuits_at_gate(gate)
    waveforms = {p:np.asarray(calibration["waveforms"][p]) for p in PHASES}
    base = original(calibration,gate,waveforms,circuits)
    result = {}
    for i, phase in enumerate(PHASES):
        method, edge = calibration["selected"][phase]
        anchors = {float(k):v[phase] for k,v in sorted(calibration["anchors"].items(),
                                                      key=lambda item:float(item[0]))}
        v = waveforms[phase]
        current = current_for_method(v,circuits[i],base[phase],anchors,gate,method,edge)
        if phase == "RESET" and "reset_return_transport" in calibration:
            current = current_with_reset_return(v, circuits[i], current, gate,
                                                calibration["reset_return_transport"])
        result[phase] = (v, current)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration",type=Path)
    parser.add_argument("--gate",type=float,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args(argv)
    result = predict(json.loads(args.calibration.read_text()),args.gate)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("phase","branch","voltage_V","predicted_current_A"))
        for phase,(wave,current) in result.items():
            apex = int(np.argmax(np.abs(wave)))
            writer.writerows((phase,"outward" if i <= apex else "return",v,c)
                             for i,(v,c) in enumerate(zip(wave,current)))


if __name__ == "__main__":
    main()
