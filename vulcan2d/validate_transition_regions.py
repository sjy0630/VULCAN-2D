"""Prepare point masks from the measured sweeps and score held-out predictions.

Run from the repository root:
    python -m vulcan2d.validate_transition_regions --output PATH.zip

The ZIP contains point-level data, event metadata and a scoring summary.
The default inputs are the eight local measurement workbooks and any saved
postconditioning / RESET-refined held-out prediction archives present locally.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from openpyxl import load_workbook

from .transistor import load_default_lookup, load_reverse_lookup


ROOT = Path(__file__).resolve().parents[1]
WORKBOOKS = (
    ("1M", None, "SET", "1R数据.xlsx"),
    ("1M", None, "RESET", "裸RRESET.xlsx"),
    ("1T1M", 0.9, "SET", "0.9V栅压.xlsx"),
    ("1T1M", 0.9, "RESET", "0.9V栅压RESET.xlsx"),
    ("1T1M", 1.1, "SET", "1T1M写入.xlsx"),
    ("1T1M", 1.1, "RESET", "1T1M擦除.xlsx"),
    ("1T1M", 1.3, "SET", "1.3V栅压.xlsx"),
    ("1T1M", 1.3, "RESET", "1.3V栅压RESET.xlsx"),
)
DEFAULT_ARCHIVES = (
    "postconditioning_refit_cycle4plus_curves.npz",
    "reset_platform_terminal_refinement_curves.npz",
)
TRANSISTOR_FILES = (
    "fig2b_transistor_lookup.json", "bookoutput2_reverse_transistor_lookup.json"
)
CURRENT_FLOOR_A = {"SET": 1e-11, "RESET": 1e-12}
SMU_LIMIT_A = 1e-3
SMU_TOLERANCE = 0.01
LOAD_THRESHOLD = 0.80
LOAD_HALF_WINDOW_V = 0.08
BRANCH_MATCH_DECADE = 0.025
EVENT_MIN_V = 0.15
EVENT_STEP_V = 0.10
EVENT_MIN_CHANGE_DECADE = 0.15
EVENT_HALF_WINDOW_V = 0.20
SENSITIVITY_WINDOWS_V = (0.15, 0.30)
NEAR_TRANSISTOR_CAP = 0.95


def persistent(mask, minimum=3):
    """Keep whole runs of at least `minimum` consecutive observations."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    result = np.zeros(len(mask), dtype=bool)
    for start, end in edges.reshape(-1, 2):
        if end - start >= minimum:
            result[start:end] = True
    return result


def median_preview(values):
    """Three-point median within uninterrupted finite samples."""
    result = values.copy()
    if len(values) < 3:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(values, 3)
    good = np.all(np.isfinite(windows), axis=1)
    result[1:-1][good] = np.median(windows[good], axis=1)
    return result


def _unique_ascending(v, indices):
    return indices[np.r_[True, np.diff(v[indices]) > 1e-10]]


def transistor_curve(lookup, gate):
    grid = np.unique(np.concatenate([c["drain_voltage_v"] for c in lookup.curves()]))
    grid = grid[(grid >= lookup.vds_min) & (grid <= lookup.vds_max)]
    return grid, np.array([lookup.current(v, gate) for v in grid])


def load_dominance(v, current, apex, curve, ripple):
    """Estimate the fraction of an incremental terminal voltage across 1T."""
    grid, idrain = curve
    cap = np.interp(v, grid, idrain, left=np.nan, right=np.nan)
    ratio = np.divide(current, cap, out=np.full_like(v, np.nan), where=cap > 0)
    levels, indices = np.unique(idrain, return_index=True)
    vds = np.interp(current, levels, grid[indices], left=np.nan, right=np.nan)
    physically_supported = (ratio <= 1.05) & (vds <= v)
    vds[~physically_supported] = np.nan
    # Do not differentiate the outward branch while its state can be evolving.
    # Transfer locally measured return slope only where the branches agree.
    return_indices = _unique_ascending(v, np.arange(len(v) - 1, apex - 1, -1))
    vr = v[return_indices]
    vdr = vds[return_indices]
    if len(vr) < 3:
        return cap, np.full_like(v, np.nan), ratio
    before = np.interp(v - LOAD_HALF_WINDOW_V, vr, vdr, left=np.nan, right=np.nan)
    after = np.interp(v + LOAD_HALF_WINDOW_V, vr, vdr, left=np.nan, right=np.nan)
    fraction = (after - before) / (2 * LOAD_HALF_WINDOW_V)
    reference = np.interp(v[:apex + 1], vr, ripple[return_indices],
                          left=np.nan, right=np.nan)
    agree = np.abs(ripple[:apex + 1] - reference) <= BRANCH_MATCH_DECADE
    # Use the entire slope window to establish overlap on the outward branch.
    mismatch = np.r_[0, np.cumsum(~agree)]
    outward = v[:apex + 1]
    lo = np.searchsorted(outward, outward - LOAD_HALF_WINDOW_V, side="left")
    hi = np.searchsorted(outward, outward + LOAD_HALF_WINDOW_V, side="right")
    fraction[:apex + 1][mismatch[hi] != mismatch[lo]] = np.nan
    fraction[~physically_supported | (fraction < 0) | (fraction > 1.10)] = np.nan
    return cap, fraction, ratio


def locate_event(v, log_g, apex, phase):
    """Find an outward interval with a sustained directional conductance change."""
    x = v[:apex + 1]
    y = log_g[:apex + 1]
    lo = np.searchsorted(x, x - EVENT_STEP_V + 1e-9, side="right") - 1
    hi = np.searchsorted(x, x + EVENT_STEP_V - 1e-9, side="left")
    centers = np.flatnonzero((lo >= 0) & (hi < len(x)) & (x >= EVENT_MIN_V))
    if not len(centers):
        return None
    lo, hi = lo[centers], hi[centers]
    missing = np.r_[0, np.cumsum(~np.isfinite(y))]
    good = (missing[hi + 1] == missing[lo]) & (x[lo] >= EVENT_MIN_V)
    sign = 1 if phase == "SET" else -1
    change = sign * (y[hi] - y[lo])
    slopes = np.where(good & (change >= EVENT_MIN_CHANGE_DECADE),
                      change / (x[hi] - x[lo]), -np.inf)
    if not np.any(np.isfinite(slopes)):
        return None
    # Equivalent intervals around an abrupt step can tie. Use their midpoint.
    tied = np.flatnonzero(np.isclose(slopes, np.max(slopes), atol=1e-9, rtol=0))
    best = int(tied[len(tied) // 2])
    return int(centers[best]), int(lo[best]), int(hi[best]), float(change[best])


def prepare_curve(voltage, current, phase, smu_limit=None, transistor=None):
    v, i = np.abs(voltage), np.abs(current)
    apex = int(np.nanargmax(v))
    finite_v = np.isfinite(v)
    if not np.all(finite_v):
        raise ValueError("A sweep with missing voltages cannot be aligned")
    if np.any(np.diff(v[:apex + 1]) < -1e-9) or np.any(np.diff(v[apex:]) > 1e-9):
        raise ValueError("Expected a single outward and return voltage ramp")
    sample = np.arange(len(v))
    eligible = np.isfinite(i) & (i >= CURRENT_FLOOR_A[phase]) & (v > 1e-12)
    log_i = np.full_like(v, np.nan)
    log_i[eligible] = np.log10(i[eligible])
    preview = np.r_[median_preview(log_i[:apex + 1]),
                    median_preview(log_i[apex:])[1:]]
    log_g = np.full_like(v, np.nan)
    log_g[eligible] = preview[eligible] - np.log10(v[eligible])
    smu = np.zeros(len(v), dtype=bool)
    if smu_limit is not None:
        smu = persistent(eligible & (np.abs(i / smu_limit - 1.0) <= SMU_TOLERANCE))
    cap = load_share = ratio = np.full_like(v, np.nan)
    if transistor is not None:
        cap, load_share, ratio = load_dominance(v, i, apex, transistor, preview)
    load = persistent(eligible & (load_share >= LOAD_THRESHOLD))
    mismatch = eligible & (ratio > 1.05)
    keep = eligible & ~smu & ~load & ~mismatch
    detection = locate_event(v, log_g, apex, phase)
    event_distance = np.full_like(v, np.inf)
    detail = {}
    if detection is not None:
        index, start, end, change = detection
        event_distance[:apex + 1] = np.abs(v[:apex + 1] - v[index])
        detail = {
            "event_abs_voltage_V": float(v[index]),
            "event_change_decade": change,
            "event_core_intersects_compliance": bool(np.any((smu | load)[start:end + 1])),
        }
    event = event_distance <= EVENT_HALF_WINDOW_V + 1e-9
    return {
        "branch_outward": sample <= apex,
        "current_preview_abs_A": 10 ** preview,
        "eligible": eligible,
        "smu_compliance": smu,
        "transistor_cap_A": cap,
        "current_to_transistor_cap": ratio,
        "transistor_incremental_voltage_share": load_share,
        "transistor_dominated": load,
        "lookup_mismatch": mismatch,
        "use_unclamped": keep,
        "event_window": event,
        "event_distance_V": event_distance,
        "use_transition": event & keep,
    }, detail


def filter_unclamped(voltage, current, phase, masks):
    """Filter the usable log current without crossing branches or masked points."""
    v = np.abs(voltage)
    outward = masks["branch_outward"]
    near = masks["eligible"] & (masks["current_to_transistor_cap"] >= NEAR_TRANSISTOR_CAP)
    near = persistent(near & outward) | persistent(near & ~outward)
    usable = masks["use_unclamped"] & ~near
    log_i = np.full_like(v, np.nan, dtype=float)
    log_i[usable] = np.log10(np.abs(current[usable]))
    filtered = log_i.copy()
    for branch in (outward, ~outward):
        filtered[branch] = median_preview(log_i[branch])
    changed = usable & (filtered != log_i)
    cleaned = np.asarray(current, dtype=float).copy()
    cleaned[changed] = np.sign(current[changed]) * 10 ** filtered[changed]

    apex = int(np.argmax(v))
    log_g = filtered.copy()
    log_g[usable] -= np.log10(v[usable])
    event = locate_event(v, log_g, apex, phase)
    distance = np.full_like(v, np.inf, dtype=float)
    detail = {}
    if event is not None:
        index, _, _, change = event
        distance[:apex + 1] = np.abs(v[:apex + 1] - v[index])
        detail = {"event_abs_voltage_V": float(v[index]), "event_change_decade": change,
                  "event_core_intersects_compliance": False}
    window = distance <= EVENT_HALF_WINDOW_V + 1e-9
    return cleaned, {**masks, "use_unclamped": usable, "transistor_near_cap": near,
                     "median_filter_changed": changed, "event_window": window,
                     "event_distance_V": distance, "use_transition": window & usable}, detail


def read_sweeps(path):
    with path.open("rb") as handle:
        workbook = load_workbook(handle, read_only=True, data_only=True)
        try:
            for sheet in workbook.worksheets:
                values = []
                rows = []
                for number, row in enumerate(sheet.iter_rows(min_row=2, max_col=2,
                                                             values_only=True), 2):
                    if not isinstance(row[0], (int, float)):
                        if row[0] is None and row[1] is None:
                            continue
                        raise ValueError(f"{path.name}/{sheet.title}, row {number}: invalid voltage")
                    rows.append(number)
                    values.append((row[0], row[1] if isinstance(row[1], (int, float))
                                   else np.nan))
                yield sheet.title, np.asarray(rows), np.asarray(values, dtype=float).T
        finally:
            workbook.close()


def collect_curves():
    forward, reverse = load_default_lookup(), load_reverse_lookup()
    for topology, gate, phase, file in WORKBOOKS:
        transistor = None
        if gate is not None:
            transistor = transistor_curve(forward if phase == "SET" else reverse, gate)
        instrument_limit = SMU_LIMIT_A if topology == "1M" and phase == "SET" else None
        for sheet, rows, (voltage, current) in read_sweeps(ROOT / file):
            masks, event = prepare_curve(voltage, current, phase, instrument_limit,
                                         transistor)
            yield (topology, gate, phase, file, sheet), rows, voltage, current, masks, event


def _summarize_residuals(observations):
    residuals = []
    cycle_mse = []
    invalid = 0
    total = 0
    for measured, predicted in observations:
        good = np.isfinite(predicted) & (predicted > 0)
        total += len(measured)
        invalid += np.count_nonzero(~good)
        if np.any(good):
            error = np.log10(predicted[good]) - np.log10(measured[good])
            residuals.extend(error)
            cycle_mse.append(float(np.mean(error ** 2)))
    return {
        "n_points": total,
        "n_cycles_with_valid_residuals": len(cycle_mse),
        "n_invalid_predictions": invalid,
        "pooled_rmse_decade": (float(np.sqrt(np.mean(np.square(residuals))))
                               if residuals and not invalid else None),
        "equal_cycle_rmse_decade": (float(np.sqrt(np.mean(cycle_mse)))
                                    if residuals and not invalid else None),
    }


def score_archive(path, measured_curves, cycle_errors):
    output = {}
    with np.load(path, allow_pickle=False) as data:
        for split, first in (("leave_one_cycle_out", 4),
                             ("chronological_first35_to_later", 36)):
            output[split] = {}
            for phase in ("SET", "RESET"):
                scores = {}
                for cycle in range(first, 54):
                    v, i, masks = measured_curves[phase, cycle]
                    side = phase.lower()
                    raw = f"observed_cycle_{cycle}_{side}"
                    target = f"{split}_same_protocol_cycle_{cycle}_{side}"
                    if (not np.array_equal(v, data[raw + "_voltage"], equal_nan=True)
                            or not np.array_equal(i, data[raw + "_current"], equal_nan=True)
                            or not np.array_equal(v, data[target + "_voltage"])):
                        raise ValueError(f"Archive and workbook are misaligned: {path.name}/{target}")
                    pred = np.abs(data[target + "_current"])
                    if pred.shape != i.shape:
                        raise ValueError(f"Invalid prediction shape: {path.name}/{target}")
                    subsets = {name: masks[name]
                               for name in ("eligible", "use_unclamped", "use_transition")}
                    for half_window in SENSITIVITY_WINDOWS_V:
                        subsets[f"transition_half_window_{half_window:.2f}_V"] = (
                            masks["use_unclamped"]
                            & (masks["event_distance_V"] <= half_window + 1e-9)
                        )
                    for subset, use in subsets.items():
                        values = np.abs(i[use]), pred[use]
                        scores.setdefault(subset, []).append(values)
                        if subset in ("eligible", "use_unclamped", "use_transition"):
                            metric = _summarize_residuals([values])
                            cycle_errors.append((path.name, split, phase, cycle, subset,
                                                 metric["n_points"],
                                                 metric["n_invalid_predictions"],
                                                 metric["pooled_rmse_decade"]))
                output[split][phase] = {
                    subset: _summarize_residuals(series)
                    for subset, series in scores.items()
                }
    return output


def _csv_string(rows):
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)
    return buffer.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", action="append", type=Path,
                        help="Saved target-excluding prediction NPZ (repeatable)")
    args = parser.parse_args(argv)
    archives = (args.archive if args.archive is not None else [
        ROOT / "analysis" / name for name in DEFAULT_ARCHIVES
        if (ROOT / "analysis" / name).is_file()
    ])
    curves = {}
    meta = ("topology", "gate_V", "phase", "workbook", "sheet")
    points = []
    events = []
    cycle_errors = []
    counts = {}
    for key, rows, v, i, masks, event in collect_curves():
        topology, gate, phase, file, sheet = key
        failed_set = gate == 1.1 and int(sheet[1:]) < 4
        masks["use_transition_for_fit"] = masks["use_transition"] & (not failed_set)
        if not points:
            points.append((*meta, "known_failed_set", "excel_row", "voltage_V", "raw_current_A", *masks))
        points.extend((*key, failed_set, *[None if not np.isfinite(x) else x for x in record])
                      for record in zip(rows, v, i, *masks.values()))
        count_fields = ("eligible", "smu_compliance", "transistor_dominated",
                        "lookup_mismatch", "use_unclamped", "use_transition",
                        "use_transition_for_fit")
        per_curve = {name: int(np.count_nonzero(masks[name])) for name in count_fields}
        counts.setdefault(file, {"points": 0, "sweeps": 0, "events": 0,
                                 **dict.fromkeys(count_fields, 0)})
        counts[file]["points"] += len(v)
        counts[file]["sweeps"] += 1
        counts[file]["events"] += bool(event)
        for name, n in per_curve.items():
            counts[file][name] += n
        events.append({"topology": topology, "gate_V": gate, "phase": phase,
                       "workbook": file, "sheet": sheet, "known_failed_set": failed_set,
                       **event, **per_curve})
        if gate == 1.1:
            curves[phase, int(sheet[1:])] = v, i, masks
    event_columns = tuple(dict.fromkeys(k for event in events for k in event))
    input_paths = [ROOT / file for _, _, _, file in WORKBOOKS]
    input_paths += [Path(__file__).with_name("data") / file for file in TRANSISTOR_FILES]
    input_paths += archives
    summary = {
        "method": {
            "current_floor_A": CURRENT_FLOOR_A,
            "filter": "Three-point median of log current magnitude, per sweep branch, only across finite samples above the floor. Used only to locate events and match branches. Errors use original measured currents.",
            "smu": {"limit_A": SMU_LIMIT_A, "relative_tolerance": SMU_TOLERANCE,
                    "min_consecutive_points": 3, "applies_to": "1M SET only"},
            "transistor": {"threshold": LOAD_THRESHOLD,
                           "half_window_V": LOAD_HALF_WINDOW_V,
                           "branch_match_decade": BRANCH_MATCH_DECADE,
                           "min_consecutive_points": 3,
                           "rule": "Approximate load dominance from local dVds/dVapp on return; transferred to outward only where the measured branches agree across the window. Requires a valid independent 1T lookup and assumes the return state is approximately frozen."},
            "mismatch": "Measured magnitude > 105% of I_T at full |Vapp|. Excluded separately from device scores. Missing differential classification does not imply compliance.",
            "event": {"min_voltage_V": EVENT_MIN_V, "slope_half_window_V": EVENT_STEP_V,
                      "min_change_decade": EVENT_MIN_CHANGE_DECADE,
                      "score_half_window_V": EVENT_HALF_WINDOW_V,
                      "sensitivity_half_windows_V": SENSITIVITY_WINDOWS_V,
                      "rule": "Maximum outward slope of log10(|I|/|V|) in the SET (up) or RESET (down) direction. A terminal proxy for state change; a compliance-censored event cannot determine the intrinsic switching interval."},
            "fit_mask": "use_transition_for_fit excludes the known failed SET cycles 1–3 and their RESET sweeps at 1.1 V. Other datasets retain all otherwise usable cycles; no new adequacy labels are inferred.",
            "prediction_scope": "1.1 V valid cycles 4-53 for leave-one-out and 36-53 for chronological test. Other datasets are labeled, without held-out model comparison."
        },
        "datasets": counts,
        "held_out_scores": {path.name: score_archive(path, curves, cycle_errors)
                            for path in archives},
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in input_paths},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("points.csv", _csv_string(points))
        archive.writestr("events.csv", _csv_string([
            event_columns, *([event.get(column) for column in event_columns]
                             for event in events)
        ]))
        archive.writestr("cycle_scores.csv", _csv_string([
            ("archive", "split", "phase", "cycle", "subset", "n_points",
             "n_invalid_predictions", "rmse_decade"),
            *cycle_errors,
        ]))
        archive.writestr("summary.json", json.dumps(
            summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
