"""Measured standalone-transistor output characteristics and circuit mapping.

The canonical source is the public Nature Figure 2b workbook archived at
Zenodo (DOI 10.5281/zenodo.7607096, CC BY 4.0). ``precompute_transistor``
converts it to the versioned JSON lookup used by v0.6.
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

    def voltage_for_current(self, current, gate_voltage, nbis=48):
        """Invert the monotone lookup and return ``|V_DS|`` in volt.

        The processed curves are monotone by construction, so a bounded
        bisection is more robust than interpolating current as an independent
        coordinate when adjacent measured samples have the same current.
        """
        current = float(current)
        gate_voltage = float(gate_voltage)
        if current < 0.0:
            raise ValueError("Drain-current magnitude must be non-negative")
        maximum = self.current(self.vds_max, gate_voltage)
        tolerance = max(1e-15, 1e-10 * max(maximum, 1e-15))
        if current > maximum + tolerance:
            raise ValueError(
                f"I_D={current:g} A exceeds the measured maximum "
                f"{maximum:g} A at V_G={gate_voltage:g} V")
        if current <= 0.0:
            return self.vds_min

        lo, hi = self.vds_min, self.vds_max
        for _ in range(nbis):
            mid = 0.5 * (lo + hi)
            if self.current(mid, gate_voltage) < current:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def curves(self):
        """Return defensive copies of the processed monotone output curves."""
        return tuple({
            "gate_voltage_v": curve.gate_voltage_v,
            "drain_voltage_v": curve.drain_voltage_v.copy(),
            "drain_current_a": curve.drain_current_a.copy(),
        } for curve in self._curves)


def load_default_lookup(path=None, branch="mean"):
    default = Path(__file__).with_name("data") / "fig2b_transistor_lookup.json"
    return TransistorLookup.from_json(default if path is None else path, branch=branch)


def load_reverse_lookup(path=None, branch="mean"):
    """Load the locally measured negative-``V_DS`` transistor magnitude.

    The JSON stores ``|V_DS|`` and ``|I_D|`` while retaining the negative-bias
    provenance.  Sign handling remains the responsibility of the surrounding
    1T1M circuit equations.
    """
    default = (Path(__file__).with_name("data") /
               "bookoutput2_reverse_transistor_lookup.json")
    return TransistorLookup.from_json(default if path is None else path, branch=branch)


@dataclass(frozen=True)
class TransistorMapping:
    """Map the standalone Fig. 2b transistor onto the 1T1R circuit.

    ``gate_gain`` and ``gate_offset_v`` are electrical mapping parameters, not
    h-BN material parameters. ``series_resistance_ohm`` represents unresolved
    access/contact resistance between the measured 1T and the 1T1R drain node.

    The historical magnitude API permits an explicit positive-curve RESET
    proxy. The signed ``drain_current`` API instead requires measured reverse
    data for negative V_DS. Use ``measured_transistor_mapping`` for the shared
    1T / 1T1R research path.
    """

    lookup: TransistorLookup
    gate_gain: float = 1.0
    gate_offset_v: float = 0.0
    current_scale: float = 1.0
    series_resistance_ohm: float = 0.0
    reset_policy: str = "magnitude_proxy"

    # Optional measured reverse-bias branch.  Its gate voltages are the actual
    # applied values, so it has its own mapping rather than inheriting the
    # positive Fig. 2b interface correction.
    reset_lookup: TransistorLookup | None = None
    reset_gate_gain: float = 1.0
    reset_gate_offset_v: float = 0.0

    def effective_gate_voltage(self, gate_voltage, side="set"):
        if side == "reset" and self.reset_lookup is not None:
            return (self.reset_gate_gain * float(gate_voltage) +
                    self.reset_gate_offset_v)
        return self.gate_gain * float(gate_voltage) + self.gate_offset_v

    def _active_lookup(self, side):
        if side not in ("set", "reset"):
            raise ValueError("side must be 'set' or 'reset'")
        if side == "set":
            return self.lookup
        if self.reset_policy == "measured_reverse":
            if self.reset_lookup is None:
                raise ValueError("Measured RESET policy requires reset_lookup")
            return self.reset_lookup
        if self.reset_policy == "magnitude_proxy":
            return self.lookup
        raise ValueError(f"Unsupported RESET policy: {self.reset_policy}")

    def drain_current(self, voltage_ds_v, gate_voltage):
        """Signed total drain current at the original physical source reference.

        V_DS=V_D-V_S and gate_voltage=V_G-V_S, with the measured physical
        source held at zero. Negative V_DS does not swap source/drain labels
        or change V_GS. Body/well conditions are inherited from the data;
        arbitrary source/body bias and channel/body current separation are
        not identified by this lookup. Positive and reverse datasets have
        not yet been confirmed to describe the same physical transistor.
        """
        voltage = float(voltage_ds_v)
        gate = float(gate_voltage)
        if not np.all(np.isfinite([voltage, gate])):
            raise ValueError("Terminal voltages must be finite")
        side = "reset" if np.signbit(voltage) else "set"
        if side == "reset" and self.reset_policy != "measured_reverse":
            raise ValueError("Signed negative bias requires measured reverse data")
        magnitude = self.current(abs(voltage), gate, side=side)
        return -magnitude if np.signbit(voltage) else magnitude

    def current(self, terminal_vds, gate_voltage, side="set", nbis=32):
        terminal_vds = abs(float(terminal_vds))
        lookup = self._active_lookup(side)
        effective_gate = self.effective_gate_voltage(gate_voltage, side=side)
        if self.series_resistance_ohm <= 0.0:
            return self.current_scale * lookup.current(
                terminal_vds, effective_gate)

        # Solve I = scale * I_lookup(Vterminal - I*Rs, Vg_eff).  The clipped
        # intrinsic voltage keeps every query inside the measured V_DS range.
        lo = 0.0
        hi = self.current_scale * lookup.current(terminal_vds, effective_gate)
        for _ in range(nbis):
            current = 0.5 * (lo + hi)
            intrinsic_vds = max(terminal_vds - current * self.series_resistance_ohm, 0.0)
            target = self.current_scale * lookup.current(intrinsic_vds, effective_gate)
            if current < target:
                lo = current
            else:
                hi = current
        return 0.5 * (lo + hi)

    def voltage_for_current(self, current, gate_voltage, side="set"):
        """Return terminal ``|V_DS|`` required to carry ``|I_D|``."""
        current = abs(float(current))
        if self.current_scale <= 0.0:
            raise ValueError("current_scale must be positive")
        lookup = self._active_lookup(side)
        effective_gate = self.effective_gate_voltage(gate_voltage, side=side)
        intrinsic = lookup.voltage_for_current(
            current / self.current_scale, effective_gate)
        return intrinsic + current * max(self.series_resistance_ohm, 0.0)


def measured_transistor_mapping(positive_lookup=None, reverse_lookup=None):
    """One measured fixed-source constitutive law for standalone and series use.

    Both gates are literal applied values. No 1T1R peak current or h-BN
    parameter is used to recalibrate this transistor. This joins two measured
    bias domains; it is not a microscopic MOS model or proof of device match.
    """
    return TransistorMapping(
        lookup=load_default_lookup() if positive_lookup is None else positive_lookup,
        reset_lookup=load_reverse_lookup() if reverse_lookup is None else reverse_lookup,
        reset_policy="measured_reverse",
    )
