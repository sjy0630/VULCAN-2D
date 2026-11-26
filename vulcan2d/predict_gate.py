"""Prospective gate interpolation within calibrated bounds for one 1T1M device.

Calibrate and validate with: python -m vulcan2d.validate_gate_prediction --output-dir RESULTS
Generate a predicted curve with: python -m vulcan2d.predict_gate CALIBRATION --gate V --output FILE
"""
import argparse
import csv
from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from . import model_shared_device as model
from . import validate_transition_regions as audit
from .transistor import load_default_lookup, load_reverse_lookup


@dataclass(frozen=True)
class GateCalibration:
    gates_v: tuple
    device: model.Device
    preparations: tuple
    anchor_devices: tuple = ()

    def __post_init__(self):
        if (not self.gates_v or len(self.gates_v)!=len(self.preparations)
                or not np.all(np.isfinite(self.gates_v))
                or np.any(np.diff(self.gates_v)<=0)
                or (self.anchor_devices and len(self.anchor_devices)!=len(self.gates_v))):
            raise ValueError("Gate anchors must be finite, ordered and match their calibrations")

    @classmethod
    def from_file(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(d["gates_v"]), model.Device(**d["device"]),
                   tuple(model.Preparation(**p) for p in d["preparations"]),
                   tuple(model.Device(**p) for p in d.get("anchor_devices", ())))

    def to_file(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + "\n",
                              encoding="utf-8")

    def _bracket(self, gate_v):
        first, last = self.gates_v[0], self.gates_v[-1]
        gate_v = float(gate_v)
        if not np.isfinite(gate_v) or not first <= gate_v <= last:
            raise ValueError(f"Gate voltage must be in the calibration bracket [{first}, {last}] V")
        right = int(np.searchsorted(self.gates_v,gate_v,side="left"))
        if self.gates_v[right] == gate_v:
            return right,right,0.
        left = right-1
        w = (gate_v-self.gates_v[left])/(self.gates_v[right]-self.gates_v[left])
        return left,right,w

    def device_at(self, gate_v):
        left,right,w = self._bracket(gate_v)
        if not self.anchor_devices:
            return self.device
        d1,d2 = self.anchor_devices[left],self.anchor_devices[right]
        if left==right:
            return d1
        p={}
        for fld in fields(model.Device):
            name=fld.name
            a,b=getattr(d1,name),getattr(d2,name)
            if a==b:
                p[name]=a
            elif isinstance(a,(float,int)) and isinstance(b,(float,int)):
                p[name]=(1-w)*a+w*b
            else:
                raise ValueError(f"Incompatible device law between gate anchors: {name}")
        return model.Device(**p)

    def preparation(self, gate_v, method="linear"):
        if method not in ("log", "linear"):
            raise ValueError("Amplitude interpolation must be log or linear")
        left,right,w = self._bracket(gate_v)
        a,b = self.preparations[left], self.preparations[right]
        if left==right:
            return a
        params = {}
        for field in ("off_set_a", "soft_set_a", "off_reset_a", "soft_reset_a"):
            lower, upper = getattr(a, field), getattr(b, field)
            params[field] = (10 ** ((1-w)*np.log10(lower) + w*np.log10(upper))
                             if method == "log" else (1-w)*lower + w*upper)
        params["soft_available"] = (1-w)*a.soft_available + w*b.soft_available
        return model.Preparation(**params)

    def predict(self, gate_v, set_voltage, reset_voltage, method="linear", initial_read=None,
                circuits=None, initial_state=model.State()):
        """Predict SET then RESET, optionally conditioning RESET on one early read.

        initial_read is (positive voltage magnitude, positive current magnitude)
        measured on the outgoing RESET branch before the area to be predicted.
        """
        prep = self.preparation(gate_v, method)
        device = self.device_at(gate_v)
        cs, cr = circuits_at_gate(gate_v) if circuits is None else circuits
        programmed = model.simulate_sweep(
            set_voltage, "SET", device, prep, cs, initial_state)
        initial = programmed["final"]
        if initial_read is not None:
            prep = _condition_reset(device, prep, cr, initial, initial_read)
        erased = model.simulate_sweep(reset_voltage, "RESET", device, prep, cr, initial)
        return programmed, erased


def circuits_at_gate(gate_v):
    return tuple(model.Circuit(transistor=audit.transistor_curve(lookup, gate_v))
                 for lookup in (load_default_lookup(), load_reverse_lookup()))


def _condition_reset(device, prep, circuit, initial, observation):
    voltage, observed = map(float, observation)
    if not (0 < voltage <= .25 and np.isfinite(observed) and observed > 0):
        raise ValueError("Use a finite positive current magnitude at |V| in (0, 0.25] V")
    waveform = np.array([0., -voltage, 0.])

    def error(log_scale):
        mod = replace(prep, soft_reset_a=prep.soft_reset_a*10 ** log_scale)
        result = model.simulate_sweep(waveform, "RESET", device, mod, circuit, initial)
        predicted = abs(result["current_A"][1])
        return np.log10(predicted) - np.log10(observed)

    if error(-4) > 0 or error(4) < 0:
        raise ValueError("Early RESET read is outside the device/circuit readout range")
    scale = brentq(error, -4, 4, xtol=1e-6)
    return replace(prep, soft_reset_a=prep.soft_reset_a*10 ** scale)


def triangle(peak, step):
    if not np.isfinite(peak) or not np.isfinite(step) or peak == 0 or step <= 0:
        raise ValueError("Need a finite nonzero peak and a positive step")
    up = np.linspace(0, peak, int(np.ceil(abs(peak)/step))+1)
    return np.r_[up, up[-2::-1]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--gate", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set-peak", type=float, default=5.)
    parser.add_argument("--reset-peak", type=float, default=-1.5)
    parser.add_argument("--step", type=float, default=.02)
    parser.add_argument("--amplitude", choices=("log", "linear"), default="linear")
    args = parser.parse_args(argv)
    cal = GateCalibration.from_file(args.calibration)
    written, erased = cal.predict(
        args.gate, triangle(args.set_peak, args.step), triangle(args.reset_peak, args.step),
        method=args.amplitude)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ("voltage_V", "current_A", "device_voltage_V", "soft_state", "hard_state")
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("phase", "gate_V", *columns))
        for phase, curve in (("SET", written), ("RESET", erased)):
            writer.writerows((phase, args.gate, *row) for row in zip(*(curve[c] for c in columns)))


if __name__ == "__main__":
    main()
