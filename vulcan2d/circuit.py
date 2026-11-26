"""Quasi-static series KCL with the fixed physical source as voltage reference."""
import numpy as np


def solve_series_load_line(applied_voltage_v, gate_voltage, hbn_current,
                          mapping, nbis=44):
    """Return signed (I, V_hBN) solving I_hBN(V_h)=I_D(V_app-V_h,V_GS).

    The supplied frozen-state h-BN law must be continuous, passive and
    monotone on the applied-polarity interval, with I_hBN(0)=0. The transistor
    law must be monotone there too. Gate voltage keeps its original physical
    source reference; the internal drain node is V_app-V_hBN, not V_app.
    Only the measured voltage/gate domain is supported. No target current
    enters this calculation. Historical magnitude-only mapping objects are
    accepted with the polarity passed explicitly for backwards compatibility.
    """
    applied = float(applied_voltage_v)
    gate = float(gate_voltage)
    if not np.all(np.isfinite([applied, gate])):
        raise ValueError("Applied and gate voltages must be finite")
    if int(nbis) != nbis or nbis < 1:
        raise ValueError("nbis must be a positive integer")
    sign = -1.0 if np.signbit(applied) else 1.0
    magnitude = abs(applied)
    side = "reset" if sign < 0.0 else "set"
    if float(hbn_current(0.0)) != 0.0 or mapping.current(0.0, gate, side=side) != 0.0:
        raise ValueError("Passive static component laws must pass through the origin")

    def transistor(voltage):
        historical_proxy = (side == "reset"
                            and getattr(mapping, "reset_policy", None) == "magnitude_proxy")
        if hasattr(mapping, "drain_current") and not historical_proxy:
            return sign * mapping.drain_current(sign * voltage, gate)
        return mapping.current(voltage, gate, side=side)

    if magnitude <= 1e-15:
        return 0.0, 0.0
    lo, hi = 0.0, magnitude
    for _ in range(int(nbis)):
        vh = 0.5 * (lo + hi)
        ih = sign * float(hbn_current(sign * vh))
        it = transistor(magnitude - vh)
        if not np.all(np.isfinite([ih, it])) or ih < 0.0 or it < 0.0:
            raise ValueError("Load-line currents must be finite and passive")
        if ih < it:
            lo = vh
        else:
            hi = vh
    vh = sign * (0.5 * (lo + hi))
    return float(hbn_current(vh)), vh
