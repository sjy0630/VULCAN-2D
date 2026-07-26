"""Compare calibrated M0 with the measured-transistor M1 experiment.

This audit intentionally replaces only the SET-side transistor law.  The
available Fig.2b workbook has non-negative V_DS curves, while RESET terminal
orientation and gate bias are still unconfirmed.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from vulcan2d import features as F
from vulcan2d import model as M
from vulcan2d.transistor import load_default_lookup


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAL = os.path.join(ROOT, "vulcan2d", "vulcan2d_calibrated.npz")


def cv(values):
    values = np.asarray(values, float)
    return float(np.std(values) / abs(np.mean(values)) * 100)


def summarize(cycles):
    frame = F.extract_all(cycles)
    vh_at_vset = []
    for cycle, vset in zip(cycles, frame.Vset):
        apex = int(np.argmax(np.abs(cycle["Vs"])))
        index = int(np.argmin(np.abs(cycle["Vs"][:apex + 1] - vset)))
        vh_at_vset.append(abs(cycle["Vhs"][index]))
    return {
        "Vset": float(np.mean(frame.Vset)),
        "Vset_cv": cv(frame.Vset),
        "R_HRS": float(np.mean(frame.R_HRS)),
        "R_LRS": float(np.mean(frame.R_LRS)),
        "Icc": float(np.mean(frame.Icc)),
        "Icc_cv": cv(frame.Icc),
        "Vh_at_Vset": float(np.mean(vh_at_vset)),
        "window": float(np.median(frame.R_HRS / frame.R_LRS)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--ncycles", type=int, default=53)
    parser.add_argument(
        "--gates", type=float, nargs="+",
        default=[0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    args = parser.parse_args(argv)

    params = M.Params.from_npz(CAL)
    lookup = load_default_lookup()
    baseline, _ = M.simulate_cycles(
        params, n_cycles=args.ncycles, seed=args.seed)
    rows = [("M0 tanh", None, summarize(baseline))]
    for gate in args.gates:
        cycles, _ = M.simulate_cycles(
            params, n_cycles=args.ncycles, seed=args.seed,
            transistor_lookup=lookup, gate_voltage_set=gate)
        rows.append(("M1 lookup", gate, summarize(cycles)))

    print("SET-side transistor audit; RESET remains the calibrated empirical law")
    print(f"seed={args.seed}, cycles={args.ncycles}")
    print("-" * 109)
    print(f"{'model':<12}{'VG':>6}{'Vset':>10}{'CV':>8}{'R_HRS':>14}"
          f"{'R_LRS':>14}{'Icc':>13}{'Icc CV':>10}{'Vh@Vset':>11}{'window':>12}")
    for name, gate, row in rows:
        gate_text = "-" if gate is None else f"{gate:.1f}"
        print(f"{name:<12}{gate_text:>6}{row['Vset']:>10.3f}"
              f"{row['Vset_cv']:>7.1f}%{row['R_HRS']:>14.3e}"
              f"{row['R_LRS']:>14.3e}{row['Icc']:>13.3e}"
              f"{row['Icc_cv']:>9.1f}%{row['Vh_at_Vset']:>11.3f}"
              f"{row['window']:>12.0f}")


if __name__ == "__main__":
    main()
