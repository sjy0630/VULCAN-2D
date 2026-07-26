"""Measured standalone-transistor output characteristics for M1 experiments.

The raw advisor-provided workbook is intentionally kept out of version control.
``precompute_transistor`` converts it to a local, ignored JSON file.  This module
contains only the generic interpolation and validation logic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class _Curve:
    gate_voltage_v: float
    drain_voltage_v: np.ndarray
    drain_current_a: np.ndarray


class TransistorLookup:
    """Bilinear ``I_D(V_DS, V_G)`` lookup built from measured output curves.

    Each input curve may preserve a ``0 -> Vmax -> 0`` sweep.  The default
    ``mean`` branch averages the ascending and descending measurements on the
    ascending voltage grid, sets the physical origin to zero, and applies a
    cumulative maximum so the load-line solver sees a monotone transistor law.
    The raw measurements remain unchanged in the JSON payload.
    """

    def __init__(self, curves):
        if len(curves) < 2:
            raise ValueError("At least two gate-voltage curves are required")
        ordered = sorted(curves, key=lambda curve: curve.gate_voltage_v)
        gates = np.array([curve.gate_voltage_v for curve in ordered], dtype=float)
        if np.any(np.diff(gates) <= 0):
            raise ValueError("Gate voltages must be unique and increasing")
        self._curves = tuple(ordered)
        self.gate_voltages = gates
        self.vg_min = float(gates[0])
        self.vg_max = float(gates[-1])
        self.vds_min = max(float(curve.drain_voltage_v[0]) for curve in ordered)
        self.vds_max = min(float(curve.drain_voltage_v[-1]) for curve in ordered)

    @staticmethod
    def _collapse_sweep(vds, current, branch="mean"):
        vds = np.asarray(vds, dtype=float)
        current = np.asarray(current, dtype=float)
        if vds.ndim != 1 or current.ndim != 1 or len(vds) != len(current):
            raise ValueError("V_DS and I_D must be same-length one-dimensional arrays")
        if len(vds) < 3 or not np.all(np.isfinite(vds)) or not np.all(np.isfinite(current)):
            raise ValueError("Each curve needs at least three finite points")

        apex = int(np.argmax(vds))
        apex_last = len(vds) - 1 - int(np.argmax(vds[::-1]))
        up_v = vds[:apex + 1]
        up_i = current[:apex + 1]
        if np.any(np.diff(up_v) <= 0):
            raise ValueError("Ascending branch V_DS must be strictly increasing")

        if branch == "up" or apex == len(vds) - 1:
            selected = up_i
        elif branch == "mean":
            down_v = vds[apex_last:][::-1]
            down_i = current[apex_last:][::-1]
            if np.any(np.diff(down_v) <= 0):
                raise ValueError("Descending branch must become increasing when reversed")
            selected = 0.5 * (up_i + np.interp(up_v, down_v, down_i))
        else:
            raise ValueError(f"Unsupported branch policy: {branch}")

        selected = np.maximum(np.asarray(selected, dtype=float), 0.0)
        selected[0] = 0.0
        selected = np.maximum.accumulate(selected)
        return up_v, selected

    @classmethod
    def from_payload(cls, payload, branch="mean"):
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported transistor lookup schema")
        curves = []
        for raw in payload.get("curves", []):
            vds, current = cls._collapse_sweep(
                raw["drain_voltage_v"], raw["drain_current_a"], branch=branch)
            curves.append(_Curve(float(raw["gate_voltage_v"]), vds, current))
        return cls(curves)

    @classmethod
    def from_json(cls, path, branch="mean"):
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_payload(json.load(handle), branch=branch)

    def current(self, vds, gate_voltage):
        """Return measured/interpolated drain current in ampere.

        Extrapolation is rejected rather than silently clipping because the
        current dataset only validates ``0.5 <= V_G <= 3 V`` and
        ``0 <= V_DS <= 5 V``.
        """
        vds = float(vds)
        gate_voltage = float(gate_voltage)
        if not self.vds_min <= vds <= self.vds_max:
            raise ValueError(f"V_DS={vds:g} V is outside [{self.vds_min:g}, {self.vds_max:g}] V")
        if not self.vg_min <= gate_voltage <= self.vg_max:
            raise ValueError(f"V_G={gate_voltage:g} V is outside [{self.vg_min:g}, {self.vg_max:g}] V")

        hi = int(np.searchsorted(self.gate_voltages, gate_voltage, side="right"))
        if hi == 0:
            lo = hi = 0
        elif hi == len(self.gate_voltages):
            lo = hi = len(self.gate_voltages) - 1
        else:
            lo = hi - 1

        i_lo = float(np.interp(
            vds, self._curves[lo].drain_voltage_v, self._curves[lo].drain_current_a))
        if lo == hi:
            return i_lo
        i_hi = float(np.interp(
            vds, self._curves[hi].drain_voltage_v, self._curves[hi].drain_current_a))
        weight = ((gate_voltage - self.gate_voltages[lo]) /
                  (self.gate_voltages[hi] - self.gate_voltages[lo]))
        return float(i_lo + weight * (i_hi - i_lo))


def load_default_lookup(path=None, branch="mean"):
    default = Path(__file__).with_name("transistor_lookup.json")
    return TransistorLookup.from_json(default if path is None else path, branch=branch)
