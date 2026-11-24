"""IV41 gate refit, 1M transfer, and grouped device holdout audits.

Run from the repository: python -m analysis.refit_iv41 WORKBOOK FROZEN_ZIP OUTPUT_DIR
Use --cache to reuse the verified Origin-export arrays from a previous run.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from functools import lru_cache
from hashlib import sha256
import csv
import io
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
from openpyxl import load_workbook
from scipy.optimize import least_squares

from vulcan2d import cycle_quality as quality
from vulcan2d import finalize_gate_model as finalizer
from vulcan2d import model_shared_device as model
from vulcan2d import predict_gate as predictor
from vulcan2d import validate_gate_prediction as gate_validation
from vulcan2d import validate_optimized_device as optimization
from vulcan2d import validate_shared_device as shared
from vulcan2d import validate_transition_regions as audit

PHASES = ("SET", "RESET")
ANCHORS = (.9, 1.1, 1.3)
TUNE = (1., 1.2)
EXTERNAL = (1.4, 1.5)
PHASE_FIELD = {
    "SET": ("off_set_a", "soft_set_a"),
    "RESET": ("off_reset_a", "soft_reset_a"),
}
COLS = (("A", r"\d+"), ("B", r"\d+"), ("C", "[^<]+"),
        ("D", "[^<]+"), ("E", "[^<]+"), ("F", "[^<]+"))
ORIGIN_ROW = re.compile(
    (r'<x:row r="\d+"[^>]*>' + "".join(
        rf'<x:c r="{col}\d+"[^>]*><x:v>({value})</x:v></x:c>'
        for col, value in COLS
    ) + "</x:row>").encode()
)


def sha_file(path):
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def origin_cache(workbook, cache):
    """Extract the regular six-column data sheets and reconcile their counts."""
    cache.mkdir(parents=True, exist_ok=True)
    stamp = cache / "origin_sha256.txt"
    digest = sha_file(workbook)
    if stamp.exists() and stamp.read_text().strip() != digest:
        raise ValueError("Parsed data cache belongs to a different workbook")
    if not stamp.exists() and (cache / "index.json").exists():
        raise ValueError("Use an empty cache directory or a cache with source verification")
    index_file = cache / "index.json"
    sheets = range(5, 29)
    if index_file.exists() and all((cache / f"s{n}.npz").exists() for n in sheets):
        return json.loads(index_file.read_text())

    book = load_workbook(workbook, read_only=True, data_only=True)
    try:
        index = dict(names=book.sheetnames, objects=list(book.worksheets[0].values),
                     curves=list(book.worksheets[1].values))
    finally:
        book.close()
    records = {row[9]: row for row in index["objects"][1:]}
    with ZipFile(workbook) as bundle:
        for n in sheets:
            rows = defaultdict(list)
            with bundle.open(f"xl/worksheets/sheet{n}.xml") as handle:
                pending = b""
                for chunk in iter(lambda: handle.read(2**20), b""):
                    pending += chunk
                    end = pending.rfind(b"</x:row>") + len(b"</x:row>")
                    if end < len(b"</x:row>"):
                        continue
                    for ident, point, _, _, voltage, current in ORIGIN_ROW.findall(pending[:end]):
                        rows[int(ident)].append((int(point), float(voltage), float(current)))
                    pending = pending[end:]
            expected = records[index["names"][n - 1]]
            if (len(rows), sum(map(len, rows.values()))) != tuple(expected[6:8]):
                raise ValueError(f"Data count mismatch in {expected[9]}")
            np.savez_compressed(cache / f"s{n}.npz",
                                **{str(key): np.asarray(series) for key, series in rows.items()})
    index_file.write_text(json.dumps(index, ensure_ascii=False))
    stamp.write_text(digest + "\n")
    return index


@lru_cache(maxsize=None)
def circuits(gate):
    return predictor.circuits_at_gate(gate)


def load_cohort(index, cache, sheet, gate):
    name = index["names"][sheet - 1]
    rows = {str(row[4]): row for row in index["curves"][1:] if row[13] == name}
    result = {phase: defaultdict(list) for phase in PHASES}
    rejects = Counter()
    seen = set()
    with np.load(cache / f"s{sheet}.npz") as data:
        for ident in sorted(data.files, key=int):
            order, v, i = data[ident].T
            metadata = rows[ident]
            if len(v) != metadata[11] or not np.all(np.diff(order) > 0) or metadata[5] != gate:
                raise ValueError(f"Invalid curve index in {name}/{ident}")
            phase = {"正": "SET", "负": "RESET"}[metadata[6]]
            if not np.isfinite(v).all() or not np.isfinite(i).all():
                rejects[f"{phase}_nonfinite"] += 1
                continue
            key = (phase, v.tobytes(), i.tobytes())
            if key in seen:
                rejects[f"{phase}_duplicate"] += 1
                continue
            seen.add(key)
            circuit = circuits(gate)[PHASES.index(phase)]
            try:
                masks, event = audit.prepare_curve(v, i, phase, transistor=circuit.transistor)
            except ValueError:
                rejects[f"{phase}_nontriangular"] += 1
                continue
            if v.min() < -circuit.transistor[0][-1] - 1e-8 or v.max() > circuit.transistor[0][-1] + 1e-8:
                rejects[f"{phase}_lookup_range"] += 1
                continue
            result[phase][v.tobytes()].append(shared.Sweep(ident, v, i, masks, event))
    cohort = {}
    for phase, grids in result.items():
        if not grids:
            raise ValueError(f"No {phase} waveforms in {name}")
        cohort[phase] = max(grids.values(), key=len)
        rejects[f"{phase}_nonmodal_grid"] = sum(map(len, grids.values())) - len(cohort[phase])
    return cohort, dict(rejects)


def screen(rows, gate, phase):
    """Apply the persistent read contrast screen without 322-only labels."""
    reads = [quality._read_conductances(row, phase) for row in rows]
    on = np.asarray(reads)[:, :, 1 if phase == "SET" else 0]
    levels = np.array([np.median(x[np.isfinite(x)]) if np.isfinite(x).any() else np.nan
                       for x in on.T])
    flags = [quality._record(("1T1M", gate, phase), row, r, levels)
             for row, r in zip(rows, reads)]
    return [row for row, flag in zip(rows, flags) if not flag.excluded], [
        flag.sheet for flag in flags if flag.excluded]


def fit_gates(cohorts, reset_anchors):
    gates = tuple(cohorts)
    rows = {("1T1M", g, p): cohorts[g][p] for g in gates for p in PHASES}
    pairs = {("1T1M", g, p): circuit
             for g in gates for p, circuit in zip(PHASES, circuits(g))}
    limits = {g: {p: len(cohorts[g][p]) for p in PHASES} for g in gates}
    device, preparation, set_models = optimization.fit_all_gates(rows, pairs, limits, "SET")
    device, preparation, _ = optimization.fit_all_gates(
        rows, pairs, limits, "RESET", device, preparation, set_by_gate=set_models)
    reference = predictor.GateCalibration(gates, device, tuple(preparation[g] for g in gates))
    if reset_anchors:
        return finalizer.review._fit_reset_anchors(rows, pairs, reference)
    return reference


def compare_reset_fits(anchors):
    """Check the transport anchors on the final 25 indexed scans at each gate."""
    development, validation = (
        {gate: {phase: rows[phase][portion] for phase in PHASES}
         for gate, rows in anchors.items()}
        for portion in (slice(None, -25), slice(-25, None))
    )
    scores = {}
    for method in ("joint", "transport_anchors"):
        calibration = fit_gates(development, reset_anchors=(method == "transport_anchors"))
        scores[method] = {}
        for gate, rows in validation.items():
            _, erased = simulate(gate, rows, calibration.device, calibration.preparation(gate))
            scores[method][gate] = error_scores(rows["RESET"], "RESET", erased["current_A"])
    return scores


def log_i(current):
    return np.log10(np.maximum(np.abs(current), 1e-18))


def align(target, source, values):
    """Interpolate independently on increasing and decreasing sweep limbs."""
    v, u = np.abs(target), np.abs(source)
    a, b = np.argmax(v), np.argmax(u)
    return np.r_[np.interp(v[:a + 1], u[:b + 1], values[:b + 1]),
                 np.interp(v[a + 1:], u[b:][::-1], values[b:][::-1])]


def error_scores(rows, phase, simulated):
    v = rows[0].voltage
    if any(not np.array_equal(v, row.voltage) for row in rows):
        raise ValueError("Scores require a consistent voltage grid")
    results = defaultdict(list)
    events = []
    y_pred = log_i(simulated)
    for row in rows:
        delta = (log_i(row.current) - y_pred) ** 2
        kept = row.masks["use_unclamped"]
        out = row.masks["branch_outward"]
        ranges = {
            "valid": kept, "transition": row.masks["use_transition"],
            "out": kept & out, "return": kept & ~out,
            "early_out": kept & out & (np.abs(v) >= .15) & (np.abs(v) <= .45),
        }
        for region, mask in ranges.items():
            if np.any(mask):
                results[region].append(float(np.mean(delta[mask])))
        if "event_abs_voltage_V" in row.event:
            events.append(row.event["event_abs_voltage_V"])
    _, pred = audit.prepare_curve(v, simulated, phase)
    event_v = pred.get("event_abs_voltage_V")
    return {region: float(np.sqrt(np.mean(values))) for region, values in results.items()} | {
        "n": len(rows), "transition_n": len(results["transition"]), "event_n": len(events),
        "event_median_V": float(np.median(events)) if events else None,
        "event_pred_V": event_v,
        "event_mae_V": float(np.mean(np.abs(np.asarray(events) - event_v)))
        if events and event_v is not None else None,
    }


def deterministic_floor(rows, phase):
    """In-sample lower bound for one common trace, with the evaluation weights."""
    obs = np.stack([log_i(row.current) for row in rows])
    use = np.stack([row.masks["use_unclamped"] for row in rows])
    valid = use.any(axis=1)
    n = use.sum(axis=1)
    weight = use / np.maximum(n, 1)[:, None]
    total = weight.sum(axis=0)
    mean = np.divide((weight * obs).sum(axis=0), total,
                     out=np.full_like(total, -18.), where=total > 0)
    score = error_scores(rows, phase, 10 ** mean)["valid"]
    events = [row.event["event_abs_voltage_V"] for row in rows
              if "event_abs_voltage_V" in row.event]
    return {"valid_rmse_decade": score, "n": int(valid.sum()),
            "event_q10_q50_q90_V": np.quantile(events, [.1, .5, .9]).tolist() if events else None}


def interpolation(anchors, gate, phase, wave):
    gates = tuple(anchors)
    hi = int(np.searchsorted(gates, gate))
    left, right = gates[hi - 1:hi + 1]
    a, b = (np.median([align(wave, r.voltage, log_i(r.current)) for r in anchors[g][phase]], axis=0)
            for g in (left, right))
    weight = (gate - left) / (right - left)
    return 10 ** ((1 - weight) * a + weight * b)


def simulate(gate, cohort, d, p):
    vset, vreset = (cohort[phase][0].voltage for phase in PHASES)
    return model.simulate_cycle(vset, vreset, d, p, *circuits(gate))


def condition_reset(gate, cohort, device, prep, baseline):
    """Score the future part of RESET with and without one observed early read."""
    branch = defaultdict(list)
    unavailable = {}
    set_result, unconditioned = baseline
    reset_circuit = circuits(gate)[1]
    for row in cohort["RESET"]:
        try:
            read, scored = gate_validation._read_and_mask(row)
            adjusted = predictor._condition_reset(device, prep, reset_circuit,
                                                  set_result["final"], read)
        except ValueError as error:
            unavailable[row.label] = str(error)
            continue
        conditioned = model.simulate_sweep(row.voltage, "RESET", device, adjusted,
                                           reset_circuit, set_result["final"])
        for name, curve in (("without_read", unconditioned), ("with_read", conditioned)):
            for region, error in error_scores([scored], "RESET", curve["current_A"]).items():
                if region in ("valid", "out", "return", "transition") and error is not None:
                    branch[name, region].append(error**2)
    report = {name: {region: {"rmse_decade": float(np.sqrt(np.mean(values))), "n": len(values)}
                     for (candidate, region), values in branch.items() if candidate == name}
              for name in ("without_read", "with_read")}
    report["unavailable"] = unavailable
    return report


def calibrate_and_audit(index, args):
    anchors, anchor_counts, raw_anchors = {}, {}, {}
    for gate in ANCHORS:
        rows, exclusions = load_cohort(index, args.cache, round((gate - .9) * 10) + 5, gate)
        raw_anchors[gate] = rows
        anchors[gate], ids = {}, {}
        for phase in PHASES:
            anchors[gate][phase], ids[phase] = screen(rows[phase], gate, phase)
        anchor_counts[gate] = {"raw": {p: len(rows[p]) for p in PHASES},
                               "quality_excluded_ids": ids, "other_rejects": exclusions}
    print("Loaded IV41 anchors", anchor_counts, flush=True)
    cal = fit_gates(anchors, reset_anchors=True)
    cal.to_file(args.output / "iv41_calibration.json")
    print("Same-IV41 recalibration complete", flush=True)

    scores, curves, conditioned = {}, {}, {}
    with ZipFile(args.frozen) as bundle:
        for gate in TUNE:
            cohort, rejects = load_cohort(index, args.cache, round((gate - .9) * 10) + 5, gate)
            parsed = list(csv.DictReader(io.StringIO(
                bundle.read(f"prediction_{gate:.1f}V.csv").decode())))
            device, preparation = cal.device_at(gate), cal.preparation(gate)
            predictions = simulate(gate, cohort, device, preparation)
            conditioned[gate] = condition_reset(gate, cohort, device, preparation, predictions)
            for phase, prediction in zip(PHASES, predictions):
                rows, wave = cohort[phase], cohort[phase][0].voltage
                old_v, old_i = np.asarray([
                    (float(r["voltage_V"]), float(r["current_A"]))
                    for r in parsed if r["phase"] == phase]).T
                options = {
                    "frozen": 10 ** align(wave, old_v, log_i(old_i)),
                    "iv41": prediction["current_A"],
                    "adjacent": interpolation(raw_anchors, gate, phase, wave),
                }
                key = f"{gate:.1f}_{phase}"
                scores[key] = {name: error_scores(rows, phase, current)
                               for name, current in options.items()}
                scores[key]["other_rejects"] = rejects
                scores[key]["deterministic_floor"] = deterministic_floor(rows, phase)
                curves[key] = (wave, np.quantile([log_i(r.current) for r in rows], [.1, .5, .9], axis=0),
                               {name: log_i(current) for name, current in options.items()})
                print(key, {k: round(v["valid"], 4) for k, v in scores[key].items()
                            if "valid" in v}, flush=True)
    return cal, anchors, anchor_counts, scores, curves, conditioned


def shared_population(training):
    """Fit one coefficient set with equal loss weight for each training device."""
    cs, cr = circuits(1.1)
    reference, prep = shared.REFERENCE, shared.PREPARATION

    def buckets(phase):
        by_grid = defaultdict(list)
        for cohort in training:
            v, use, target, w = shared._training_data(cohort[phase])
            by_grid[v.tobytes()].append((v, use, target, w))
        result = []
        for data in by_grid.values():
            v = data[0][0]
            total = np.zeros_like(v)
            numerator = total.copy()
            for _, use, target, w in data:
                total[use] += w**2
                numerator[use] += w**2 * target
            valid = total > 0
            result.append((v, valid, numerator[valid] / total[valid], np.sqrt(total[valid])))
        return result

    def fit(phase, data, initial, dev, preparation):
        if phase == "SET":
            names = ("beta_set", "soft_exponent", "set_center_v", "set_width_v")
            lower, upper = [0., .6, .22, .008], [12., 2.8, 2.2, .50]
            circuit = cs
        else:
            names = ("beta_reset", "reset_off_power", "reset_center_v", "reset_width_v",
                     "reset_on_power")
            dev = replace(dev, reset_on_power=1.3)
            lower, upper = [0., .55, .16, .025, .60], [12., 3.2, 1.8, .55, 3.]
            circuit = cr
        amp = PHASE_FIELD[phase]
        start = [getattr(dev, k) for k in names] + [np.log10(getattr(preparation, k)) for k in amp]

        def decode(p):
            return (replace(dev, **dict(zip(names, p))),
                    replace(preparation, **dict(zip(amp, 10 ** p[len(names):]))))

        def residual(p):
            d, a = decode(p)
            errors = []
            for v, use, target, weight in data:
                current = model.simulate_sweep(v, phase, d, a, circuit, initial)["current_A"]
                errors.append(weight * (log_i(current[use]) - target))
            return np.concatenate(errors)

        sol = least_squares(residual, start, bounds=(lower + [-18., -18.], upper + [-3., -3.]),
                            x_scale="jac", diff_step=3e-5, max_nfev=220)
        if not sol.success:
            raise RuntimeError(f"Population {phase} fitting failed: {sol.message}")
        return decode(sol.x)

    sets = buckets("SET")
    d, p = fit("SET", sets, model.State(), reference, prep)
    initial = model.simulate_sweep(sets[0][0], "SET", d, p, cs)["final"]
    return fit("RESET", buckets("RESET"), initial, d, p)


def few_read_fit(cohort, device, prep, mode, n_read=5):
    """Use only terminal read windows, with the fitted dynamics held fixed."""
    windows = {}
    for phase in PHASES:
        reads = defaultdict(list)
        for row in cohort[phase][:n_read]:
            v = np.abs(row.voltage)
            for outward in (True, False):
                mask = (row.masks["use_unclamped"] & (row.masks["branch_outward"] == outward)
                        & (v >= .15 - 1e-9) & (v <= .25 + 1e-9))
                for index in np.flatnonzero(mask):
                    reads[index].append(log_i(row.current[index]))
        if not reads:
            return None, {}
        windows[phase] = {k: float(np.median(values)) for k, values in sorted(reads.items())}

    amp = tuple(field for phase in PHASES for field in PHASE_FIELD[phase])
    x0 = np.array([np.log10(getattr(prep, field)) for field in amp])
    distinct = mode == "independent"
    count = len(amp) if distinct else len(PHASES)

    def offsets(x):
        return x if distinct else np.repeat(x, 2)

    def decode(x):
        return replace(prep, **dict(zip(amp, 10 ** (x0 + offsets(x)))))

    def residual(x):
        results = simulate(1.1, cohort, device, decode(x))
        errors = []
        for phase, result in zip(PHASES, results):
            values = windows[phase]
            indices = list(values)
            actual = np.array(list(values.values()))
            out = cohort[phase][0].masks["branch_outward"][indices]
            for branch in (out, ~out):
                if np.any(branch):
                    errors.append((log_i(result["current_A"][indices])[branch] - actual[branch])
                                  / np.sqrt(np.count_nonzero(branch)))
        # Weak ridge stabilizes amplitude decompositions in a narrow read band.
        return np.r_[np.concatenate(errors), (.05 if distinct else .35) * x]

    bound = 2. if distinct else .7
    result = least_squares(residual, np.zeros(count), bounds=(-bound, bound),
                           diff_step=3e-5, max_nfev=100, loss="soft_l1", f_scale=.25)
    if not result.success:
        return None, {}
    return decode(result.x), {
        "read_points": {p: len(w) for p, w in windows.items()},
        "scale": dict(zip(amp, (10 ** offsets(result.x)).tolist())),
    }


def leave_group_out(index, cache):
    data, rejects = {}, {}
    for sheet in range(14, 29):
        name = index["names"][sheet - 1].split("_", 1)[1]
        data[name], rejects[name] = load_cohort(index, cache, sheet, 1.1)
    results = {}
    for held, sample in data.items():
        train = [{phase: screen(group[phase], 1.1, phase)[0] for phase in PHASES}
                 for name, group in data.items() if name != held]
        d, p = shared_population(train)
        test = {phase: sample[phase][5:] for phase in PHASES}
        need = [name for name, portion in test.items() if not portion]
        if need:
            raise ValueError(f"No held-out {need} scans in {held}")
        preds = {"zero_shot": simulate(1.1, test, d, p)}
        metadata = {}
        for method in ("independent", "common_scale"):
            adapted, metadata[method] = few_read_fit(sample, d, p, mode=method)
            if adapted is not None:
                preds[f"five_read_{method}"] = simulate(1.1, test, d, adapted)
        results[held] = {
            "calibration_reads": {phase: [r.label for r in sample[phase][:5]] for phase in PHASES},
            "other_rejects": rejects[held], "adaptation": metadata,
            "results": {
                name: {phase: error_scores(test[phase], phase, pred["current_A"])
                       for phase, pred in zip(PHASES, pair)}
                for name, pair in preds.items()
            },
        }
        print("Held group", held, {name: [round(r["valid"], 4) for r in record.values()]
                                   for name, record in results[held]["results"].items()}, flush=True)
    return results


def bare_bridge(cal):
    """Compare common and independently fitted SET transport on the same 1M holdout."""
    groups, pairs = shared.load_groups(include_abnormal_pairs=True)
    first, last, circuits_1m = {}, {}, {}
    for phase in PHASES:
        key = ("1M", None, phase)
        first[phase], last[phase] = groups[key][:14], groups[key][14:]
        circuits_1m[phase] = pairs[key]
    device, prep = cal.device, cal.preparation(1.1)
    device, prep = shared.fit_bare_set(first["SET"], circuits_1m["SET"], device, prep)
    verified = model.simulate_sweep(first["SET"][0].voltage, "SET", device, prep,
                                   circuits_1m["SET"])["final"]
    device, prep = shared.fit_bare_reset(first["RESET"], circuits_1m["RESET"],
                                        device, prep, verified)
    local_d, local_p = finalizer._fit_bare_pretransport(first["SET"], device, prep)
    reports = {}
    plot = {}
    for name, d, p in (("shared_SET_shape", device, prep), ("local_SET_shape", local_d, local_p)):
        voltage = [last[phase][0].voltage for phase in PHASES]
        sim = model.simulate_cycle(*voltage, d, p, *(circuits_1m[phase] for phase in PHASES))
        report = {}
        for phase, predicted in zip(PHASES, sim):
            report[phase] = error_scores(last[phase], phase, predicted["current_A"])
            if phase == "SET":
                report[phase]["pre_compliance_rmse_decade"] = finalizer._bare_set_score(
                    last[phase], d, p, circuits_1m[phase])[0]["pre_compliance"]["pooled_rmse_decade"]
                points = np.flatnonzero(audit.persistent(
                    (predicted["voltage_V"] > 0) & (predicted["current_A"] >= .99e-3)))
                clips = [shared._clip_voltage(row.voltage, row.current) for row in last[phase]]
                report[phase]["clip_pred_V"] = float(predicted["voltage_V"][points[0]]) if len(points) else None
                report[phase]["clip_mae_V"] = float(np.mean(np.abs(
                    np.asarray(clips) - report[phase]["clip_pred_V"]))) if len(points) else None
        reports[name] = {"device": asdict(d), "preparation": asdict(p), "results": report,
                         "RESET_initial_state": asdict(sim[0]["final"])}
        plot[name] = [log_i(predicted["current_A"]) for predicted in sim]
    for idx, phase in enumerate(PHASES):
        rows = last[phase]
        plot[f"1M_{phase}"] = (
            rows[0].voltage,
            np.quantile([log_i(r.current) for r in rows], [.1, .5, .9], axis=0),
            {name: plot[name][idx] for name in reports},
        )
    return reports, {key: plot[key] for key in ("1M_SET", "1M_RESET")}


def extrapolate(index, args, cal):
    """Exercise target-gate transistor data with hBN readout held at the edge."""
    scores, plots = {}, {}
    for gate in EXTERNAL:
        cohort, rejections = load_cohort(index, args.cache, round((gate - .9) * 10) + 5, gate)
        inherited = cal.preparation(ANCHORS[-1])
        results = simulate(gate, cohort, cal.device, inherited)
        for phase, predicted in zip(PHASES, results):
            rows = cohort[phase]
            key = f"{gate:.1f}_{phase}"
            scores[key] = error_scores(rows, phase, predicted["current_A"])
            scores[key]["other_rejects"] = rejections
            plots[key] = (rows[0].voltage,
                          np.quantile([log_i(r.current) for r in rows], [.1, .5, .9], axis=0),
                          {"edge": log_i(predicted["current_A"])})
    return scores, plots


def make_plot(curves, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"frozen": ("Archive", "#818181"), "iv41": ("Same-IV41 refit", "#008365"),
             "adjacent": ("Adjacent I–V interpolation", "#c2712e"),
             "edge": ("1.3 V hBN + target 1T", "#008365"),
             "shared_SET_shape": ("Shared SET transport", "#008365"),
             "local_SET_shape": ("1M-calibrated SET transport", "#c2712e")}
    ncols = 2
    nrows = (len(curves) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(10.8, 3.5*nrows))
    handles = {}
    for ax, (label, (voltage, band, model_values)) in zip(axes.flat, curves.items()):
        v = np.abs(voltage)
        apex = int(np.argmax(v))
        for points, ls in ((slice(None, apex+1), "-"), (slice(apex, None), "--")):
            x, logs = v[points], band[:, points] + 6
            fill = ax.fill_between(x, logs[0], logs[2], color="#b2c5d8", alpha=.35)
            line, = ax.plot(x, logs[1], ls, color="#23486c", lw=1.65)
            handles.update({"Measured median": line, "Measured 10–90%": fill})
            for name, values in model_values.items():
                legend, color = style[name]
                line, = ax.plot(x, values[points] + 6, ls, color=color, lw=1.45)
                handles[legend] = line
        ax.set_title(label.replace("_", " V  |  ") if label[0].isdigit() and not label.startswith("1M")
                     else label.replace("_", "  |  "), loc="left", weight="bold")
        ax.set_ylabel(r"$\log_{10}(|I| / \mathrm{\mu A})$")
        ax.set_xlabel("Terminal voltage magnitude (V)")
        ax.grid(alpha=.15)
    fig.suptitle(title, weight="bold", fontsize=15)
    fig.legend(handles.values(), handles.keys(), ncol=3, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, .10 if nrows == 1 else .06, 1, .94))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cache is None:
        args.cache = args.output / "cache"
    index = origin_cache(args.workbook, args.cache)
    cal, anchor_data, anchors, gate_scores, curves, conditioned = calibrate_and_audit(index, args)
    reset_comparison = compare_reset_fits(anchor_data)
    make_plot(curves, args.output / "Vulcan_IV41_refit.png",
              "IV41 | In-range prediction after protocol-matched calibration")
    bare, bare_plot = bare_bridge(cal)
    make_plot(bare_plot, args.output / "Vulcan_1M_bridge.png",
              "Bare 1M | Held-out SET and RESET")
    far, far_plot = extrapolate(index, args, cal)
    make_plot(far_plot, args.output / "Vulcan_IV41_extrapolation.png",
              "IV41 | Out-of-range gate test with hBN readout held at 1.3 V")
    devices = leave_group_out(index, args.cache)
    report = {
        "provenance": {"workbook_sha256": sha_file(args.workbook), "archive_sha256": sha_file(args.frozen),
                       "transistor_lookup_sha256": {name: sha_file(audit.ROOT / "vulcan2d/data" / name)
                                                    for name in audit.TRANSISTOR_FILES}},
        "anchor_counts": anchors, "anchor_reset_comparison": reset_comparison,
        "gate_validation": gate_scores, "conditioned_reset": conditioned, "bare_1M": bare,
        "gate_extrapolation": far, "leave_device_group_out": devices,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
