"""Fit a single SET gap law to 1M and 1T1M using measured low-bias initial states."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from analysis import audit_log_soft as reader
from analysis import validate_compact_set_physics as compact
from analysis.validate_single_gap_series_set import GapRamp
from vulcan2d import validate_shared_device as shared


def bare_items(rows, reference_beta):
    items, fallback = [], []
    for index, row in enumerate(rows):
        voltage, current = row.voltage, np.abs(row.current)
        indices = np.flatnonzero(row.masks["use_unclamped"] & row.masks["branch_outward"])
        prefix = indices[voltage[indices] <= .55]
        anchor = reader.reference_current(prefix, voltage, current)
        if anchor is None:
            use = prefix[voltage[prefix] >= .25]
            if not len(use):
                raise ValueError(f"No low-bias observations for bare cycle {row.label}")
            ratio = np.sinh(reference_beta * voltage[use]) / np.sinh(reference_beta * .5)
            anchor = float(np.exp(np.median(np.log(current[use] / ratio))))
            fallback.append(row.label)
        items.append((index, row, anchor / compact.REFERENCE_V))
    return items, fallback


def observations(items, circuit):
    actual, mask, initial = compact.data_matrix(items, circuit if circuit.transistor is not None else None)
    voltage = items[0][1].voltage
    if any(not np.array_equal(voltage, row.voltage) for _, row, _ in items):
        raise ValueError("SET voltage grids differ within a group")
    return voltage, actual, mask & (voltage > .55), initial


def regions(voltage, bare):
    if bare:
        outward = np.arange(len(voltage)) <= np.argmax(voltage)
        return {"pre_switch": outward & (voltage < 2.2),
                "transition": outward & (voltage >= 2.2),
                "return": ~outward}
    return {key: mask for key, mask in compact.branch_windows(voltage).items()
            if key in ("low_outward", "transition_outward", "high_outward", "return")}


def clamp_indices(voltage, current, limit):
    capped = current >= .99 * limit
    persistent = capped & np.r_[capped[1:], False]
    indices = np.flatnonzero(persistent & (np.arange(len(voltage)) <= voltage.argmax()))
    return indices[0] if len(indices) else None


def setup_fit(groups, step, ramp_type=GapRamp):
    blocks = []
    for name, (items, circuit) in groups.items():
        bare = name == "1M"
        if not bare:
            sample = np.linspace(0, len(items) - 1, min(15, len(items)), dtype=int)
            items = [items[index] for index in sample]
        voltage, actual, mask, initial = observations(items, circuit)
        subsets = list(regions(voltage, bare).values())
        targets = [(mask & subset, 1.) for subset in subsets if np.any(mask & subset)]
        if bare:
            onset = np.zeros_like(mask)
            for index, row in enumerate(actual):
                point = clamp_indices(voltage, row, circuit.compliance_a)
                if point is None:
                    raise ValueError("No compliance onset in a bare training cycle")
                onset[index, point] = True
            targets.append((onset, 2.))
        topology_weight = .5 if bare else .5 / (len(groups) - 1)
        total_weight = sum(weight for _, weight in targets)
        blocks.append((ramp_type(voltage, circuit, step), initial,
                       [(keep, np.log10(actual[keep]), np.sqrt(weight * topology_weight / (total_weight * keep.sum())))
                        for keep, weight in targets]))
    return blocks


def fit_joint(groups, old, max_resistance, reference=None):
    blocks = setup_fit(groups, .05)

    def residual(params):
        terms = []
        for ramp, initial, targets in blocks:
            current, _, _ = ramp.predict(initial, params, iterations=24)
            terms.extend(weight * (np.log10(current[keep]) - actual) for keep, actual, weight in targets)
        return np.concatenate(terms)

    low = [1., np.log(1.), -32., .2, 1.]
    high = [18., np.log(max_resistance), 8., 32., 30.]
    starts = (old, [8., np.log(300.), -14., 12., 18.], [5., np.log(30.), -23., 18., 25.])
    if reference is not None:
        low[2] = -45.
        low.append(0.)
        high.append(2.)
        starts = ([*reference, .03], [*reference, .55], [*reference, 1.1])
    fits = [least_squares(residual, np.clip(start, np.array(low) + .001, np.array(high) - .001),
                          bounds=(low, high), diff_step=1e-3, ftol=4e-4, max_nfev=60)
            for start in starts]
    successful = [fit for fit in fits if fit.success and np.isfinite(fit.cost)]
    if not successful:
        raise RuntimeError("Joint SET fit failed to converge")
    return min(successful, key=lambda fit: fit.cost), [float(fit.cost) for fit in fits]


def score(groups, old, fitted, reference=None, ramp_type=GapRamp):
    result, panels = {}, []
    smaller_resistance = old.copy()
    smaller_resistance[1] = fitted[1]
    for name, (items, circuit) in groups.items():
        bare = name == "1M"
        voltage, actual, mask, initial = observations(items, circuit)
        fast = ramp_type(voltage, circuit, .025)
        tight = ramp_type(voltage, circuit, .0125)
        baseline_ramp = GapRamp(voltage, circuit, .0125)
        old_values = baseline_ramp.predict(initial, old)[0]
        resistance_only = baseline_ramp.predict(initial, smaller_resistance)[0]
        new_values, gap, _ = tight.predict(initial, fitted)
        reference_values = baseline_ramp.predict(initial, reference)[0] if reference is not None else None
        refinement = fast.predict(initial, fitted)[0]
        voltage_sum = gap + np.exp(fitted[1]) * new_values + tight.transistor_drop(new_values)
        balance_mask = mask & (new_values < .999 * circuit.supply_bound(voltage))
        assert np.max(np.abs((voltage_sum - voltage)[balance_mask])) < 1e-5
        assert np.all(voltage_sum <= voltage + 1e-5)

        subsets = {"all_unclamped": np.ones_like(voltage, dtype=bool),
                   "outward": np.arange(len(voltage)) <= voltage.argmax(),
                   **regions(voltage, bare)}
        summary = {}
        comparisons = {"prior": old_values, "resistance_only": resistance_only, "joint": new_values}
        if reference is not None:
            comparisons["previous_joint"] = reference_values
        for key, subset in subsets.items():
            use = mask & subset
            if np.any(use):
                summary[key] = {"points": int(use.sum()),
                                **{label: float(np.sqrt(np.mean(np.log10(vals[use] / actual[use]) ** 2)))
                                   for label, vals in comparisons.items()}}
        data = {"scans": len(items), "rmse_decade": summary,
                "max_refinement_decade": float(np.max(np.abs(np.log10(refinement[mask] / new_values[mask])))),
                "max_voltage_balance_error_V": float(np.max(np.abs((voltage_sum - voltage)[balance_mask])))}
        if bare:
            def onsets(current):
                indices = [clamp_indices(voltage, row, circuit.compliance_a) for row in current]
                return [float(voltage[index]) if index is not None else None for index in indices]
            onset = {label: onsets(values) for label, values in {"measured": actual, **comparisons}.items()}
            delta = [abs(a - b) for a, b in zip(onset["measured"], onset["joint"]) if b is not None]
            data.update(cycles=[row.label for _, row, _ in items], clamp_onset_V=onset,
                        joint_clamp_mae_V=float(np.mean(delta)) if delta else None)
        result[name] = data
        if name in ("1M", "0.9", "1.4", "1.5"):
            panels.append((name, voltage, mask, actual, old_values, new_values, reference_values))
    return result, panels


def draw(path, panels, candidate_label="Joint fit with state-dependent rate"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), layout="constrained")
    for ax, (name, voltage, mask, observed, prior, fitted, reference) in zip(axes.flat, panels):
        apex = voltage.argmax()
        outward = np.arange(len(voltage)) <= apex
        measured = (np.any(mask, axis=0) | ((name == "1M") & (voltage > 2.4))) & outward
        test = (voltage > .55) & outward
        lo, median, hi = np.quantile(observed[:, measured], [.1, .5, .9], axis=0) * 1e6
        ax.fill_between(voltage[measured], lo, hi, color=".7", alpha=.25, label="Measured p10–p90")
        ax.plot(voltage[measured], median, color=".12", label="Measured median")
        curves = [(prior, "Prior 1T1M fit", "#486d93", "--"),
                  (fitted, "Joint fit", "#a1263d", "-")]
        if reference is not None:
            curves[1:] = [(reference, "Previous joint fit", ".45", ":"),
                          (fitted, candidate_label, "#a1263d", "-")]
        for value, label, color, style in curves:
            ax.plot(voltage[test], np.median(value[:, test], axis=0) * 1e6,
                    color=color, ls=style, label=label)
        ax.set(title="Bare hBN" if name == "1M" else f"1T1M · {name} V gate",
               yscale="log", xlabel="Applied SET voltage (V)", ylabel="Current magnitude (µA)")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("SET transfer | held-out scans with measured low-bias initial state")
    fig.savefig(path, dpi=175)
    plt.close(fig)


def load_split(cache, reference_beta):
    groups, circuits = shared.load_groups()
    key = ("1M", None, "SET")
    bare, fallback = bare_items(groups[key], reference_beta)
    circuit = circuits[key]
    train, test = {"1M": (bare[:14], circuit)}, {"1M": (bare[14:], circuit)}
    onset = [shared._clip_voltage(row.voltage, row.current) for _, row, _ in bare[:14]]
    if any(value is None for value in onset):
        raise ValueError("Bare training requires a measured compliance onset per cycle")
    max_resistance = .97 * min(onset) / circuit.compliance_a
    index = json.loads((cache / "index.json").read_text())
    for gate, (all_rows, rows, circuit) in compact.get_cohorts(index, cache).items():
        cut = int(.7 * len(all_rows)) if gate <= 1.3 else 0
        if cut:
            train[str(gate)] = ([item for item in rows if item[0] < cut], circuit)
        test[str(gate)] = ([item for item in rows if item[0] >= cut], circuit)
    return train, test, max_resistance, fallback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "baseline", "output", "figure"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--state-dependent", type=Path,
                        help="Fit a state-dependent rate using this earlier joint result as comparator")
    args = parser.parse_args()
    prior = json.loads(args.baseline.read_text())["fit"]
    old = np.array([prior["beta_per_V"], np.log(prior["resistance_ohm"]),
                    *prior["log_growth_rate_field_slope_capacity"]])
    train, test, max_resistance, fallback = load_split(args.cache, old[0])
    reference = None
    if args.state_dependent is not None:
        previous = json.loads(args.state_dependent.read_text())["fit"]
        reference = [previous["beta_per_V"], np.log(previous["resistance_ohm"]),
                     *previous["log_growth_rate_field_slope_capacity"]]
    fit, costs = fit_joint(train, old, max_resistance, reference)
    result, panels = score(test, old, fit.x, reference)
    report = {"protocol": "Shared gap law, resistance and write evolution per applied sweep increment. "
                          "Per-scan low-bias reference sets the initial amplitude; bare scans without "
                          "a valid 0.5 V bracket use the median normalized current on 0.25–0.55 V "
                          "with the prior transport slope for initialization only. "
                          "Bare first 14 measured cycles and first 70% in workbook order at 0.9–1.3 V "
                          "used for fitting; all 1.4/1.5 V and remaining cycles assessed separately. "
                          "Point errors exclude observations used for anchors and masked current-limited points. "
                          "Bare compliance onset is a separate metric and used only on training 1M to fit. "
                          "Resistance-only comparison replaces the prior resistance with the joint fit's "
                          "while preserving the prior transport and evolution parameters. SET only.",
              "training_scans": {name: len(items) for name, (items, _) in train.items()},
              "bare_anchor_fallback_cycles": fallback, "max_resistance_ohm": max_resistance,
              "fit": {"beta_per_V": fit.x[0], "resistance_ohm": np.exp(fit.x[1]),
                      "log_growth_rate_field_slope_capacity": fit.x[2:5].tolist(),
                      "training_cost": fit.cost, "start_costs": costs},
              "results": result}
    if reference is not None:
        report["fit"]["state_rate_coupling"] = fit.x[5]
        report["protocol"] += (" The growth rate also scales with the instantaneous gap conductance "
                               "to a fitted common power, relative to 0.1 microsiemens at 0.5 V.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    draw(args.figure, panels)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
