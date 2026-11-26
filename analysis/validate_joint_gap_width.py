"""Test a common effective gap width in SET transport and field-driven growth."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from analysis import validate_joint_gap_set as joint
from analysis.validate_single_gap_series_set import GapRamp


MIN_WIDTH_NM = .20
ACTION_PER_NM_SQRT_EV = 10.25  # 2 sqrt(2 m_e eV) / hbar, multiplied by 1 nm


def action_per_nm(voltage, barrier):
    """WKB action across a linearly tilted barrier, using injection energy."""
    top = np.sqrt(barrier)
    end = np.sqrt(np.maximum(barrier - voltage, 0.))
    trapezoid = (2. / 3.) * (barrier + top * end + end**2) / (top + end)
    triangle = (2. / 3.) * barrier * top / np.maximum(voltage, barrier)
    return ACTION_PER_NM_SQRT_EV * np.where(voltage <= barrier, trapezoid, triangle)


class WidthRamp(GapRamp):
    """One gap determines both the tunneling action and the local writing field."""

    def predict(self, initial, parameters, *, iterations=28):
        barrier, log_resistance, log_closed_g, log_rate, field_scale = parameters
        resistance = np.exp(log_resistance)
        contact_g = np.exp(log_closed_g + action_per_nm(0., barrier) * MIN_WIDTH_NM)
        anchor_current = initial * .5
        anchor_voltage = .5 - resistance * anchor_current
        width = np.log(contact_g * anchor_voltage / anchor_current) / action_per_nm(anchor_voltage, barrier)
        if not np.all(np.isfinite(width) & (width >= MIN_WIDTH_NM)):
            raise ValueError("The common conductance range does not include all read anchors")

        def solve(applied, width):
            bound = self.circuit.supply_bound(applied)
            lo = np.zeros_like(width + applied)
            hi = np.broadcast_to(applied, lo.shape)
            for _ in range(iterations):
                mid = (lo + hi) * .5
                current = contact_g * mid * np.exp(-width * action_per_nm(mid, barrier))
                below = (current < bound) & (
                    mid + resistance * current + self.transistor_drop(current) < applied)
                lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
            local = .5 * (lo + hi)
            current = contact_g * local * np.exp(-width * action_per_nm(local, barrier))
            return np.minimum(current, bound), local

        def advance(width, local, delta):
            exponent = log_rate + field_scale * (local / width - 1.)
            rate = np.exp(np.minimum(exponent, 50.))
            return MIN_WIDTH_NM + (width - MIN_WIDTH_NM) * np.exp(-rate * delta)

        trace = [width]
        for applied, step in zip(self.grid[:-1], np.diff(self.grid)):
            _, local = solve(applied, width)
            middle = advance(width, local, step / 2.)
            _, local = solve(applied + step / 2., middle)
            width = advance(width, local, step)
            trace.append(width)
        trajectory = np.asarray(trace).T
        mapped = np.array([np.interp(self.terminal[:self.apex + 1], self.grid, row) for row in trajectory])
        mapped = np.c_[mapped, np.broadcast_to(width[:, None], (len(initial), len(self.terminal) - self.apex - 1))]
        current, local = solve(self.terminal, mapped)
        return current, local, mapped


def fit_joint_width(train, max_resistance):
    blocks = joint.setup_fit(train, .075, WidthRamp)

    def residual(params):
        terms = []
        for ramp, initial, targets in blocks:
            current, _, _ = ramp.predict(initial, params, iterations=22)
            terms.extend(weight * (np.log10(current[keep]) - actual) for keep, actual, weight in targets)
        return np.concatenate(terms)

    bounds = ([.25, np.log(1.), np.log(.006), -27., .2],
              [4., np.log(max_resistance), np.log(2.), 4., 39.])
    starts = ([1.1, np.log(400.), np.log(.03), -8., 12.],
              [2.3, np.log(120.), np.log(.015), -13., 22.])
    fits = [least_squares(residual, start, bounds=bounds, diff_step=2e-3,
                          ftol=5e-4, max_nfev=70) for start in starts]
    successful = [run for run in fits if run.success and np.isfinite(run.cost)]
    if not successful:
        raise RuntimeError(f"Gap-width fit failed to converge: {[run.message for run in fits]}")
    return min(successful, key=lambda run: run.cost), [float(run.cost) for run in fits]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "baseline", "joint-reference", "output", "figure"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prior, reference = (json.loads(path.read_text())["fit"] for path in (args.baseline, args.joint_reference))
    old, previous = ([item["beta_per_V"], np.log(item["resistance_ohm"]),
                      *item["log_growth_rate_field_slope_capacity"]] for item in (prior, reference))
    train, test, max_r, fallback = joint.load_split(args.cache, old[0])
    fit, costs = fit_joint_width(train, max_r)
    result, panels = joint.score(test, np.asarray(old), fit.x, previous, WidthRamp)
    barrier, log_r, log_closed, log_rate, field_scale = fit.x
    report = {
        "protocol": "Same chronological/workbook splits, masks, SET inputs and training weights as joint gap study. "
                    "Current Gc*V*exp(-WKB action) uses the integral through a tilted effective barrier. "
                    "The effective width relaxes toward 0.20 nm at a rate activated by Vgap/width. "
                    "Evolution is per applied ramp increment, since absolute sweep times are unavailable. "
                    "All material parameters shared; only the external circuit and observed low-bias initial "
                    "state vary. WKB electron mass and minimum width are fixed modeling assumptions.",
        "bare_anchor_fallback_cycles": fallback,
        "fit": {"effective_barrier_eV": barrier, "resistance_ohm": np.exp(log_r),
                "closed_low_bias_conductance_S": np.exp(log_closed), "log_rate_V_inv": log_rate,
                "field_sensitivity_nm_per_V": field_scale, "training_cost": fit.cost, "start_costs": costs},
        "results": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    joint.draw(args.figure, panels, "Joint fit with evolving gap width")
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
