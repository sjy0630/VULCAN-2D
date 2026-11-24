"""Fit and validate the RESET transport and prepared-state distribution.

The reset prefix estimates one effective connected-channel conductance, and
the remaining trace estimates shared RESET kinetics. At prediction time the
prefix is optional: the population is learned only from calibration gates.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from analysis import diagnose_reset_error as cleaning
from analysis import refit_iv41 as refit
from analysis import validate_programming_bridge as bridge
from vulcan2d import model_shared_device as model
from vulcan2d import predict_gate as predictor
from vulcan2d.reset_population import LEVELS, READ_V, prepared_distribution, simulate_ensemble

PHASE = "RESET"
PREFIX_END = .23


def load_data(index, cache):
    data = {}
    for gate in (*bridge.PROGRAM_GATES, *bridge.ASSESS_GATES):
        raw, _ = refit.load_cohort(index, cache, round((gate - .9) * 10) + 5, gate)
        filtered = {phase: cleaning.clean(rows, phase) for phase, rows in raw.items()}
        data[gate] = {phase: (rows, cleaning.matched_raw(raw[phase], rows))
                      for phase, rows in filtered.items()}
    return data


def estimate_initial(row, circuit):
    """Inverse load-line estimate of I_hBN at a standard local read voltage."""
    v, i = np.abs(row.voltage), np.abs(row.current)
    grid, currents = circuit.transistor
    levels, idx = np.unique(currents, return_index=True)
    local = v - np.interp(i, levels, grid[idx])
    use = (row.masks["use_unclamped"] & row.masks["branch_outward"]
           & (v >= .10 - 1e-9) & (v <= PREFIX_END + 1e-9)
           & (i < .95 * np.interp(v, grid, currents)) & (local >= .03))
    if use.sum() < 4:
        return np.nan
    return float(np.median(np.log10(i[use] * READ_V / local[use])))


def truncate_prefix(row):
    learned = row.masks["branch_outward"] & (np.abs(row.voltage) <= PREFIX_END + 1e-9)
    masks = dict(row.masks)
    for key in ("use_unclamped", "use_transition"):
        masks[key] = masks[key] & ~learned
    return replace(row, masks=masks)


def descriptors(data, prior):
    result = {}
    for g, phases in data.items():
        v = phases["SET"][0][0].voltage
        written = model.simulate_sweep(v, "SET", prior.device, prior.preparation(1.1),
                                       refit.circuits(g)[0])
        if not (written["final"].soft > 0 and written["final"].hard == 0.):
            raise ValueError(f"Expected active soft state without strong connection at {g} V")
        reset = phases[PHASE][0]
        result[g] = {
            "program_current_a": float(np.max(written["current_A"])),
            "initial": written["final"],
            "read_log_a": np.array([estimate_initial(row, refit.circuits(g)[1]) for row in reset]),
        }
    return result


def normalized_design(cohort):
    kept, tran, back = [], [], []
    for row in cohort:
        use = row.masks["use_unclamped"]
        kept.append(use)
        tran.append(row.masks["use_transition"])
        back.append(use & ~row.masks["branch_outward"])
    arrays = [np.asarray(x, dtype=float) for x in (kept, tran, back)]
    terms = [(arr / arr.sum(), w) for arr, w in zip(arrays, (.45, .25, .30)) if arr.any()]
    weight = sum(arr*w for arr,w in terms)
    total = weight.sum(axis=0)
    target = (weight * refit.log_i(np.stack([row.current for row in cohort]))).sum(axis=0)
    mean = np.divide(target, total, out=np.zeros_like(target), where=total > 0)
    return mean, np.sqrt(total)


def fit_packets(data, info, gates):
    packets = {}
    for gate in gates:
        rows = data[gate][PHASE][0]
        selected = {row.label for row in refit.screen(rows, gate, PHASE)[0]}
        initial = info[gate]["read_log_a"]
        good = [i for i in np.argsort(initial) if np.isfinite(initial[i]) and rows[i].label in selected]
        if len(good) < 15:
            raise ValueError(f"Insufficient usable RESET prefix reads at {gate} V")
        chunks = np.array_split(good, 7)
        packets[gate] = (np.array([np.mean(initial[chunk]) for chunk in chunks]),
                         np.stack([normalized_design([truncate_prefix(rows[i]) for i in chunk])
                                   for chunk in chunks], axis=1))
    return packets


def fit_kinetics(data, info, prior, gates, method):
    prep0 = prior.preparation(1.1)
    device0 = replace(prior.device, reset_on_power=1., reset_hazard_per_v=None)
    packets = fit_packets(data, info, gates)
    x0 = [np.log10(prep0.off_reset_a), device0.beta_reset, device0.reset_off_power,
          1.12, .08, .12, .35]
    lo = [-17., 0., .4, .18, .015, 0., .04]
    hi = [-5., 12., 4., 1.75, .60, .75, 1.]
    if method != "field":
        x0 += [.16]
        lo += [0.]
        hi += [4.]

    def decode(x):
        off, beta, exponent, center, width, late, gap, *assist = x
        d = replace(device0, beta_reset=beta, reset_off_power=exponent,
                    reset_center_v=center, reset_width_v=width,
                    reset_late_fraction=late, reset_center_gap_v=gap,
                    reset_power_assist_v=assist[0] if assist else 0.,
                    reset_tail_power_fraction=0. if method == "fast_power" else 1.)
        return d, replace(prep0, off_reset_a=10**off)

    def residual(x):
        d, p = decode(x)
        result = []
        for g, (scales, (target, w)) in packets.items():
            current = simulate_ensemble(data[g][PHASE][0][0].voltage, d, p,
                                     refit.circuits(g)[1], info[g]["initial"], scales)
            result.append((w * (refit.log_i(current) - target)).ravel())
        return np.concatenate(result)

    fit = least_squares(residual, x0, bounds=(lo, hi), x_scale="jac",
                        max_nfev=125, diff_step=8e-5, ftol=2e-6, xtol=2e-6, gtol=2e-6)
    if not np.all(np.isfinite(fit.x)):
        raise RuntimeError(fit.message)
    return (*decode(fit.x), {"parameters": fit.x.tolist(), "fit_converged": bool(fit.success),
                            "iterations": fit.nfev, "fit_loss": float(fit.cost)})


def fit_program_distribution(info, gates):
    """Fit a shared response to simulated peak programming current."""
    current = np.log10([info[g]["program_current_a"] for g in gates])
    observed = [info[g]["read_log_a"][np.isfinite(info[g]["read_log_a"])] for g in gates]
    centers = [np.median(x) for x in observed]
    origin = -4.2
    slope, offset = np.polyfit(current-origin, centers, 1)
    offsets = np.concatenate([np.quantile(x-m, (np.arange(41)+.5)/41)
                              for x,m in zip(observed,centers)])
    quantiles = np.quantile(offsets, (np.arange(61)+.5)/61)
    return {"log_current_center": origin, "slope": float(slope), "offset": float(offset),
            "quantiles": quantiles.tolist(),
            "anchor_median_log_a": dict(zip(gates, map(float, centers)))}


def observed_bands(rows):
    values = np.stack([np.where(row.masks["use_unclamped"], refit.log_i(row.current), np.nan)
                       for row in rows])
    ok = np.any(np.isfinite(values), axis=0)
    voltage = rows[0].voltage
    apex = np.argmin(voltage)
    x = np.arange(len(voltage))
    ranges = (x[:apex+1], x[apex+1:])
    bands = np.full((len(LEVELS), len(voltage)), np.nan)
    bands[:, ok] = np.nanquantile(values[:, ok], LEVELS, axis=0)
    for branch in ranges:
        has = branch[ok[branch]]
        if not len(has):
            raise ValueError("Missing all eligible data on one limb")
        bands[:, branch] = [np.interp(branch, has, y[has]) for y in bands]
    return bands


def interpolation(train_bands, gate):
    ordered = sorted(train_bands)
    i = int(np.searchsorted(ordered, gate))
    if i == 0 or i == len(ordered):
        return train_bands[ordered[0 if i == 0 else -1]]
    lo, hi = ordered[i-1:i+1]
    t = (gate-lo)/(hi-lo)
    return (1-t)*train_bands[lo] + t*train_bands[hi]


def interval_scores(rows, bounds):
    cover, width, score = [], [], []
    for row in rows:
        keep = row.masks["use_unclamped"]
        if not np.any(keep):
            continue
        obs = refit.log_i(row.current)[keep]
        lo, _, hi = bounds[:, keep]
        span = hi-lo
        cover.append(np.mean((obs>=lo)&(obs<=hi)))
        width.append(np.mean(span))
        score.append(np.mean(span + 10*(np.maximum(lo-obs, 0.) + np.maximum(obs-hi, 0.))))
    return dict(coverage_80=float(np.mean(cover)), width_decade=float(np.mean(width)),
                interval_score_80=float(np.mean(score)))


def paired_block_interval(rows, prediction, reference, block_size=10):
    """Measure uncertainty of the RMSE change under contiguous sweep resampling."""
    losses = []
    for row in rows:
        keep = row.masks["use_unclamped"]
        if not np.any(keep):
            continue
        y = refit.log_i(row.current)[keep]
        losses.append([np.mean((y-curve[keep])**2) for curve in (prediction, reference)])
    losses = np.asarray(losses)
    n = len(losses)
    block = min(block_size, n)
    rng = np.random.default_rng(31141)
    start = rng.integers(0, n-block+1, size=(3000, int(np.ceil(n/block))))
    sample = (start[:, :, None] + np.arange(block)).reshape(len(start), -1)[:, :n]
    errors = np.sqrt(np.mean(losses[sample], axis=1))
    delta = errors[:, 0] - errors[:, 1]
    return {"delta_rmse_decade": float(np.sqrt(losses.mean(0))[0]-np.sqrt(losses.mean(0))[1]),
            "block_scans": block, "percentile_95": np.quantile(delta, [.025,.975]).tolist()}


def evaluate(data, info, prior, d, p, train, assess, distribution):
    empirical = {g: observed_bands(data[g][PHASE][0]) for g in train}
    report, plotting = {}, {}
    for g in assess:
        raw = data[g][PHASE][1]
        v = raw[0].voltage
        circuit = refit.circuits(g)[1]
        state = info[g]["initial"]
        log_amp = prepared_distribution(distribution, info[g]["program_current_a"])
        draws = simulate_ensemble(v, d, p, circuit, state, log_amp)
        logs = refit.log_i(draws)
        bands = np.quantile(logs, LEVELS, axis=0)
        baseline = model.simulate_sweep(v, PHASE, prior.device, prior.preparation(1.1),
                                        circuit, state)["current_A"]
        predicted = {
            "prior": refit.log_i(baseline),
            "empirical": interpolation(empirical, g)[1],
            "population": logs.mean(axis=0),
        }
        scores = {k: refit.error_scores(raw, PHASE, 10**y) for k,y in predicted.items()}
        scores["population_interval"] = interval_scores(raw, bands)
        scores["empirical_interval"] = interval_scores(raw, interpolation(empirical, g))
        scores["paired_block_comparison"] = {
            name: paired_block_interval(raw, predicted["population"], predicted[name])
            for name in ("prior", "empirical")
        }
        scores["single_curve_floor"] = refit.deterministic_floor(raw, PHASE)
        # Conditional forecasts use the prefix as an input and omit it from scoring.
        valid = np.isfinite(info[g]["read_log_a"])
        cond = simulate_ensemble(v, d, p, circuit, state, info[g]["read_log_a"][valid])
        targets = [truncate_prefix(r) for r, include in zip(raw, valid) if include]
        scores["read_conditioned_n"] = len(targets)
        if targets:
            def paired_score(curves):
                result = [refit.error_scores([row], PHASE, prediction)
                          for row,prediction in zip(targets, curves)]
                return {k: float(np.sqrt(np.mean([r[k]**2 for r in result if k in r])))
                        for k in ("valid", "out", "return", "transition")}
            scores["read_conditioned"] = paired_score(cond)
            scores["read_matched_unconditional"] = paired_score(
                np.broadcast_to(10**predicted["population"], cond.shape))
        report[g] = scores
        plotting[g] = (v, observed_bands(raw), bands, predicted)
    return report, plotting


def observation_correlations(data, info):
    result = {}
    for g, cohort in data.items():
        reads = info[g]["read_log_a"]
        rows = cohort[PHASE][0]
        post = []
        for row in rows:
            v = np.abs(row.voltage)
            use = (row.masks["use_unclamped"] & ~row.masks["branch_outward"]
                   & (v >= .55-1e-9) & (v <= .65+1e-9))
            post.append(float(np.median(refit.log_i(row.current[use]))) if use.sum() >= 4 else np.nan)
        valid = np.isfinite(reads) & np.isfinite(post)
        result[g] = {"n": int(valid.sum()), "rho": float(spearmanr(reads[valid], np.array(post)[valid]).statistic)}
    return result


def figure(path, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), sharex=True)
    for ax, (g, (voltage, obs, band, line)) in zip(axes.flat, groups.items()):
        a = int(np.argmin(voltage))
        nonzero = np.flatnonzero(np.abs(voltage) > 1e-12)
        for branch, ls in ((nonzero[nonzero <= a], "-"), (nonzero[nonzero >= a], "--")):
            x = -voltage[branch]
            observed = ax.fill_between(x, obs[0,branch]+6, obs[2,branch]+6, color="#c9d6e6",
                                       alpha=.42, label="Measured 10–90%")
            predict = ax.fill_between(x, band[0,branch]+6, band[2,branch]+6, color="#78baa8",
                                      alpha=.25, label="Predicted 80% interval")
            centered, = ax.plot(x, obs[1,branch]+6, ls, color="#244d71", label="Measured median")
            old, = ax.plot(x, line["prior"][branch]+6, ls, color="#777", label="Previous")
            central, = ax.plot(x, line["population"][branch]+6, ls, color="#096e57", label="Population center")
        ax.set_title(f"{g:g} V gate | RESET", weight="bold", loc="left")
        ax.set_xlabel("Terminal voltage magnitude (V)")
        ax.set_ylabel(r"$\log_{10}(|I| / \mu \mathrm{A})$")
        ax.grid(alpha=.18)
    handles = [Line2D([], [], color=h.get_color(), label=h.get_label()) for h in (centered, old, central)]
    handles += [observed, predict, Line2D([], [], color="#555", linestyle="--", label="Return sweep")]
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0,.07,1,1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index = refit.origin_cache(args.workbook, args.cache)
    prior = predictor.GateCalibration.from_file(args.prior)
    data = load_data(index, args.cache)
    info = descriptors(data, prior)
    anchors = bridge.PROGRAM_GATES
    development = {name: {} for name in ("field", "power", "fast_power")}
    for held in anchors:
        train = tuple(g for g in anchors if g != held)
        distribution = fit_program_distribution(info, train)
        for name in development:
            d, p, diagnostic = fit_kinetics(data, info, prior, train, name)
            r, _ = evaluate(data, info, prior, d, p, train, (held,), distribution)
            development[name][held] = {"metrics": r[held], "fit": diagnostic}
            print("development", name, held, round(r[held]["population"]["valid"], 4),
                  round(d.reset_power_assist_v, 4), diagnostic["fit_converged"], flush=True)
    mean = {name: float(np.mean([r["metrics"]["population"]["valid"] for r in records.values()]))
            for name, records in development.items()}
    chosen = next(name for name in development if mean[name] <= 1.01 * min(mean.values()))
    print("model selected on anchor folds", chosen, mean, flush=True)
    evaluation, coefficients = {}, {}
    distribution = fit_program_distribution(info, anchors)
    for name in development:
        d, p, diagnostic = fit_kinetics(data, info, prior, anchors, name)
        evaluation[name], curves = evaluate(data, info, prior, d, p, anchors,
                                            bridge.ASSESS_GATES, distribution)
        coefficients[name] = {"device": asdict(d), "preparation": asdict(p), "fit": diagnostic}
        print("assessment", name, [(g, round(r["population"]["valid"], 4),
                                    round(r["population_interval"]["coverage_80"], 3))
                                   for g,r in evaluation[name].items()], flush=True)
        if name == chosen:
            figure(args.output / "Vulcan_RESET_population.png", curves)
    report = {
        "source_sha256": refit.sha_file(args.workbook), "prior_sha256": refit.sha_file(args.prior),
        "development": development, "mean_dev_rmse_decade": mean, "selected": chosen,
        "assessment": evaluation, "read_return_correlation": observation_correlations(data, info),
        "program_distribution": distribution, "coefficients": coefficients,
    }
    calibration = {"schema_version": 1, "selected": chosen,
                   "device": coefficients[chosen]["device"],
                   "preparation": coefficients[chosen]["preparation"],
                   "program_distribution": distribution,
                   "evaluated_gate_range_v": [min(anchors), max(bridge.ASSESS_GATES)],
                   "training_gates_v": list(anchors),
                   "protocol": {"set_peak_v": 5., "reset_peak_v": -1.5, "step_v": .01}}
    (args.output / "reset_population_calibration.json").write_text(
        json.dumps(calibration, indent=2, allow_nan=False) + "\n")
    (args.output / "reset_population_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
