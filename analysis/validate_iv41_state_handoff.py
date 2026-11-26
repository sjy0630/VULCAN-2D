"""Replay the IV41 Origin-order sequence with controlled state handoffs."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from analysis import refit_iv41 as iv
from vulcan2d import model_shared_device as model
from vulcan2d import validate_transition_regions as audit


GATES = (.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5)
MODES = {
    "independent": ("Restart every phase", "#9c631f", ":"),
    "within_pair": ("Carry SET to RESET", "#317493", "--"),
    "continuous": ("Carry within and between pairs", "#a72c46", "-"),
}
READ_V = .15
READ_CURRENT_FLOOR = 1e-14


def origin_cycle(name):
    """The unsuffixed Origin column is the first member of its series."""
    return int(name.rsplit("@", 1)[1]) if "@" in name else 1


def metadata_by_curve(index, sheet):
    return {str(row[4]): row for row in index["curves"][1:]
            if row[13] == index["names"][sheet - 1]}


def map_phase(rows, metadata):
    mapped = {}
    for row in rows:
        meta = metadata[row.label]
        suffixes = [origin_cycle(name) for name in meta[7:10]]
        if len(set(suffixes)) != 1 or suffixes[0] in mapped:
            raise ValueError(f"Origin ID is ambiguous for curve {row.label}: {meta[7:10]}")
        mapped[suffixes[0]] = row
    return mapped


def terminal_conductance(voltage, current, outward):
    v, i = np.abs(voltage), np.abs(current)
    peak = np.argmax(v)
    region = slice(0, peak + 1) if outward else slice(None, peak - 1, -1)
    v, i = v[region], i[region]
    if not v[0] <= READ_V <= v[-1]:
        raise ValueError("Waveform does not span the selected terminal read voltage")
    return max(float(np.interp(READ_V, v, i)), READ_CURRENT_FLOOR) / READ_V


def correlation(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def transition_data(sets, resets):
    common = sorted(sets.keys() & resets.keys())
    send = np.log10([terminal_conductance(*sets[k], False) for k in common])
    rstart = np.log10([terminal_conductance(*resets[k], True) for k in common])
    successive = np.asarray(np.diff(common) == 1)
    bridges = [k for k in common if k + 1 in sets]
    rend = np.log10([terminal_conductance(*resets[k], False) for k in bridges])
    sstart = np.log10([terminal_conductance(*sets[k + 1], True) for k in bridges])
    return {
        "SET_count": len(sets), "RESET_count": len(resets), "matching_origin_ids": len(common),
        "adjacent_bridges": len(bridges),
        "SET_end_to_RESET_start_r": correlation(send, rstart),
        "RESET_start_lag_minus_one_r": correlation(send[1:][successive], rstart[:-1][successive]),
        "RESET_start_lag_plus_one_r": correlation(send[:-1][successive], rstart[1:][successive]),
        "within_pair_first_difference_r": correlation(np.diff(send)[successive], np.diff(rstart)[successive]),
        "within_pair_median_absolute_decade": float(np.median(np.abs(send - rstart))),
        "RESET_end_to_next_SET_start_r": correlation(rend, sstart),
        "across_pair_median_absolute_decade": float(np.median(np.abs(rend - sstart))),
    }


def source_checks(workbook, index, cache):
    book = load_workbook(workbook, read_only=True, data_only=True)
    try:
        table = book["比较2对应关系"]
        selected = {row[1]: origin_cycle(row[10])
                    for row in table.iter_rows(min_row=2, values_only=True)
                    if row[0] and str(row[0]).startswith("Fig.2c")}
    finally:
        book.close()
    matching = [k for k in range(1, 54) if selected[f"S{k}"] == selected[f"R{k}"]]

    def raw_sequences(sheet):
        mapping = metadata_by_curve(index, sheet)
        rows = {p: {} for p in ("正", "负")}
        with np.load(cache / f"s{sheet}.npz") as arrays:
            for key in arrays.files:
                meta = mapping[key]
                origin = origin_cycle(meta[7])
                if origin in rows[meta[6]]:
                    raise ValueError(f"Repeated Origin ID in {sheet}/{meta[6]}/{origin}")
                rows[meta[6]][origin] = tuple(arrays[key][:, 1:].T)
        return rows["正"], rows["负"]

    old_s, old_r = raw_sequences(19)
    export_s = {k: old_s[selected[f"S{k}"]] for k in range(1, 54)}
    export_r = {k: old_r[selected[f"R{k}"]] for k in range(1, 54)}
    high_s, high_r = raw_sequences(12)
    high = transition_data(high_s, high_r)
    for sign, phase in (("SET", high_s), ("RESET", high_r)):
        high[f"{sign}_median_peak_uA"] = float(np.median([np.max(np.abs(i)) for _, i in phase.values()]) * 1e6)
    return {
        "historical_fig2c_matching_origin_labels": matching,
        "historical_display_to_origin": selected,
        "historical_display_labels": transition_data(export_s, export_r),
        "full_322_origin_labels": transition_data(old_s, old_r),
        "IV41_5V_origin_labels": high,
    }


def replay(pairs, circuits, device, preparation, empty, mode):
    set_circuit, reset_circuit = circuits
    records, last_id, carry = [], None, empty
    for number, set_row, reset_row in pairs:
        set_initial = carry if mode == "continuous" and last_id == number - 1 else empty
        st = model.simulate_sweep(set_row.voltage, "SET", device, preparation, set_circuit, set_initial)
        reset_initial = empty if mode == "independent" else st["final"]
        rt = model.simulate_sweep(reset_row.voltage, "RESET", device, preparation, reset_circuit, reset_initial)
        last_id, carry = number, rt["final"]
        records.append((st, rt, set_initial, reset_initial))
    return records


def scores(rows, predicted):
    """Identical project masks for all arms, preserving branch labels."""
    samples = {}
    for name in ("all", "outward", "return"):
        observations = []
        for row, current in zip(rows, predicted):
            mask = row.masks["use_unclamped"].copy()
            if name != "all":
                mask &= row.masks["branch_outward"] == (name == "outward")
            observations.append((np.abs(row.current[mask]), np.abs(current[mask])))
        summary = audit._summarize_residuals(observations)
        summary["n_invalid_predictions"] = int(summary["n_invalid_predictions"])
        samples[name] = summary
    return samples


def operating_point_check(rows, predictions, circuit):
    total = 0.
    for row, prediction in zip(rows, predictions):
        v = np.abs(row.voltage) - np.abs(prediction["device_voltage_V"])
        limit = circuit.transistor_current(np.maximum(v, 0.))
        total = max(total, float(np.max(np.abs(limit - np.abs(prediction["current_A"])))))
    return total


def state_readbacks(pairs, predictions):
    for j, (number, s, r) in enumerate(pairs):
        fields = {"origin_id": number, "SET_curve_id": s.label, "RESET_curve_id": r.label}
        reads = (("SET_start", s, 0, True), ("SET_end", s, 0, False),
                 ("RESET_start", r, 1, True), ("RESET_end", r, 1, False))
        for label, row, phase, outward in reads:
            fields[f"measured_{label}_uS"] = 1e6 * terminal_conductance(row.voltage, row.current, outward)
            for mode, predicted in predictions.items():
                fields[f"{mode}_{label}_uS"] = 1e6 * terminal_conductance(
                    row.voltage, predicted[j][phase]["current_A"], outward)
        for mode, predicted in predictions.items():
            st, rt, set_initial, reset_initial = predicted[j]
            for prefix, state in (("SET_initial", set_initial), ("SET_final", st["final"]),
                                  ("RESET_initial", reset_initial), ("RESET_final", rt["final"])):
                for key, value in asdict(state).items():
                    fields[f"{mode}_{prefix}_{key}"] = value
        yield fields


def draw(output, panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(panels), 2, figsize=(10, 3.0 * len(panels)), layout="constrained")
    history, haxes = plt.subplots(len(panels), 2, figsize=(10, 2.6 * len(panels)), layout="constrained")
    for j, (gate, pairs, predicted, rows) in enumerate(panels):
        for p, phase in enumerate(("SET", "RESET")):
            ax = axes[j, p]
            samples = [pair[p + 1] for pair in pairs]
            voltage = np.abs(samples[0].voltage)
            observed = np.stack([np.abs(r.current) for r in samples])
            kept = np.stack([r.masks["use_unclamped"] for r in samples])
            outward = samples[0].masks["branch_outward"]
            for branch, style in ((outward, "-"), (~outward, "--")):
                ids = np.flatnonzero(branch & kept.any(axis=0))
                lo, median, hi = 1e6 * np.nanquantile(
                    np.where(kept[:, ids], observed[:, ids], np.nan), [.1, .5, .9], axis=0)
                ax.fill_between(voltage[ids], lo, hi, color=".87")
                ax.plot(voltage[ids], median, style, color=".15", label="Measured median" if style == "-" else None)
                for key, (label, color, line) in MODES.items():
                    current = np.stack([np.abs(record[p]["current_A"][ids]) for record in predicted[key]])
                    m = 1e6 * np.nanmedian(np.where(kept[:, ids], current, np.nan), axis=0)
                    ax.plot(voltage[ids], m, color=color, ls=line, lw=1.25,
                            alpha=1. if style == "-" else .62, label=label if style == "-" else None)
            ax.set(title=f"{gate:g} V gate | {phase}", yscale="log",
                   xlabel="Applied voltage magnitude (V)", ylabel="Current magnitude (µA)")
            ax.grid(alpha=.2)

            ax = haxes[j, p]
            label = f"{phase}_start"
            cycle = [r["origin_id"] for r in rows]
            ax.plot(cycle, [r[f"measured_{label}_uS"] for r in rows], color=".14", lw=1.,
                    label="Measured")
            for key in ("within_pair", "continuous"):
                legend, color, line = MODES[key]
                ax.plot(cycle, [r[f"{key}_{label}_uS"] for r in rows], color=color, ls=line, label=legend)
            ax.set(title=f"{gate:g} V gate | {phase} entry", yscale="log",
                   xlabel="Origin sequence index", ylabel="Terminal I/V at 0.15 V (µS)")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    haxes[0, 0].legend(fontsize=7)
    fig.suptitle("Frozen hBN model | three state handoff rules")
    history.suptitle("Per-scan terminal readout | measured and replayed")
    fig.savefig(output / "iv_curves.png", dpi=165)
    history.savefig(output / "entry_history.png", dpi=165)
    plt.close(fig)
    plt.close(history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("workbook", "cache", "baseline", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    index = iv.origin_cache(args.workbook, args.cache)
    baseline = json.loads(args.baseline.read_text())
    device = model.Device(**baseline["device"])
    preparation = model.Preparation(**baseline["preparation"])
    empty = model.State(**baseline["initial"])
    summary = {"workbook_sha256": (args.cache / "origin_sha256.txt").read_text().strip(),
               "baseline_sha256": iv.sha_file(args.baseline),
               "frozen_parameters": {k: baseline[k] for k in ("device", "preparation", "initial", "training_cycles")},
               "readout": {"terminal_voltage_V": READ_V, "floor_A_for_read_correlations": READ_CURRENT_FLOOR},
               "score_current_floors_A": audit.CURRENT_FLOOR_A,
               "protocol": "Fixed archived 322 1.1 V parameters. Pair same Origin suffix on modal-grid, accepted IV41 "
                           "sweeps. Only applied sweep waveforms and existing transistor load curves enter replay. "
                           "Sequence starts at the original zero state; full carry restarts across missing pairs. "
                           "Score the same measured-point use_unclamped masks in all three replay modes. "
                           "5 V and 322 source comparisons are descriptive only.",
               "source_checks": source_checks(args.workbook, index, args.cache), "gates": {}}
    panels, histories = [], []
    pooled = {mode: {phase: [0, 0.] for phase in iv.PHASES} for mode in MODES}
    for gate in GATES:
        sheet = round((gate - .9) * 10) + 5
        cohort, excluded = iv.load_cohort(index, args.cache, sheet, gate)
        meta = metadata_by_curve(index, sheet)
        set_rows, reset_rows = (map_phase(cohort[phase], meta) for phase in iv.PHASES)
        common = sorted(set_rows.keys() & reset_rows.keys())
        pairs = [(k, set_rows[k], reset_rows[k]) for k in common]
        desc = transition_data(
            {k: (v.voltage, v.current) for k, v in set_rows.items()},
            {k: (v.voltage, v.current) for k, v in reset_rows.items()},
        )
        circuits = iv.circuits(gate)
        predicted = {mode: replay(pairs, circuits, device, preparation, empty, mode) for mode in MODES}
        results = {}
        for mode, traces in predicted.items():
            results[mode] = {}
            for p, phase in enumerate(iv.PHASES):
                rows = [pair[p + 1] for pair in pairs]
                measures = scores(rows, [value[p]["current_A"] for value in traces])
                measures["max_KCL_residual_A"] = operating_point_check(rows, [value[p] for value in traces], circuits[p])
                results[mode][phase] = measures
                all_points = measures["all"]
                n = all_points["n_points"]
                pooled[mode][phase][0] += n
                pooled[mode][phase][1] += n * all_points["pooled_rmse_decade"] ** 2
        changes = {}
        for p, phase in enumerate(iv.PHASES):
            dx = []
            for pair, within, full in zip(pairs, predicted["within_pair"], predicted["continuous"]):
                mask = pair[p + 1].masks["use_unclamped"]
                dx.extend(np.log10(np.abs(full[p]["current_A"][mask] / within[p]["current_A"][mask])))
            changes[phase] = {"rms_prediction_change_decade": float(np.sqrt(np.mean(np.square(dx)))),
                              "max_prediction_change_decade": float(np.max(np.abs(dx)))}
        rows = list(state_readbacks(pairs, predicted))
        histories.extend({"gate_V": gate, **r} for r in rows)
        summary["gates"][str(gate)] = {"pairs": len(pairs), "excluded": excluded,
                                       "states_reset_at": [a for a, prev in zip(common[1:], common[:-1]) if a != prev + 1],
                                       "measured_pair_evidence": desc, "scores": results,
                                       "continuous_vs_within_pair": changes,
                                       "continuous_state_ranges": {
                                           col: [min(r["continuous_" + col] for r in rows),
                                                 max(r["continuous_" + col] for r in rows)]
                                           for col in ("SET_initial_soft", "SET_final_soft", "RESET_final_soft",
                                                       "SET_initial_hard", "SET_final_hard", "RESET_final_hard")}}
        if gate in (.9, 1.4, 1.5):
            panels.append((gate, pairs, predicted, rows))
    summary["pooled"] = {mode: {phase: {"points": n, "rmse_decade": (sse / n) ** .5}
                                for phase, (n, sse) in phase_scores.items()}
                          for mode, phase_scores in pooled.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "state_history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(histories[0]))
        writer.writeheader()
        writer.writerows(histories)
    draw(args.output, panels)
    (args.output / "scores.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary["pooled"], indent=2))


if __name__ == "__main__":
    main()
