"""Training-only terminal-curve constraints declared before the next fit.

No previous model predictions, fitted parameters, test records, timestamps or
invented cell identities enter these calculations. Slopes describe measured
trajectories, not necessarily frozen-state constitutive derivatives.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "analysis/shared_hbn_2026-11-03/data/native_observations.npz"
TRAIN = {"1r": [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
         "1t1r": list(range(4, 36))}
FLOORS = {"set": 1e-11, "reset": 1e-12}
COMPLIANCE = 1e-3
LIMIT_MASK_A = .995e-3
CENTRAL_QUANTILES = [.10, .90]
# Windows are fixed here for the next fit. They span low-voltage readout,
# increasing-voltage SET curvature, and low/mid/late RESET; they are not
# adapted to a candidate's residual or optimized one window at a time.
SET_WINDOWS = [(.1, .2), (.2, .5), (.5, 1.), (1., 2.), (2., 4.)]
RESET_WINDOWS = [(.1, .2), (.2, .4), (.4, .8), (.8, 1.2), (1., 1.4), (1.4, 1.7)]
RESET_FLATNESS_WINDOWS = [(.6, 1.), (1., 1.4), (1.4, 1.7)]


def tag(value):
    return f"{value:g}".replace(".", "p")


def branch_indices(voltage, branch):
    apex = int(np.argmax(abs(voltage)))
    return np.arange(apex+1) if branch == "outward" else np.arange(apex+1, len(voltage))


def read_at(voltage, current, magnitude, branch):
    indices = branch_indices(voltage, branch)
    index = indices[np.argmin(abs(abs(voltage[indices])-magnitude))]
    if abs(abs(voltage[index])-magnitude) > 3e-7:
        raise ValueError("Requested native voltage sample absent; interpolation forbidden")
    return float(abs(current[index])), int(index)


def distribution(values, transform):
    value = np.array(values, float)
    coordinate = np.log10(value) if transform == "log10" else value
    return {
        "n": int(len(value)), "min": float(value.min()), "q10": float(np.quantile(value, .1)),
        "q25": float(np.quantile(value, .25)), "median": float(np.median(value)),
        "q75": float(np.quantile(value, .75)), "q90": float(np.quantile(value, .9)),
        "max": float(value.max()), "scoring_coordinate": transform,
        "coordinate_mean_target": float(coordinate.mean()),
        "coordinate_population_sd": float(coordinate.std()),
        "coordinate_median_absolute_deviation": float(np.median(abs(coordinate-np.median(coordinate)))),
        "back_transformed_mean_target": float(10.**coordinate.mean()) if transform == "log10" else float(coordinate.mean()),
    }


def curve_scatter(v, observed, phase, topology):
    valid = (abs(v)[None, :] > 1e-12) & np.isfinite(observed) & (observed >= FLOORS[phase])
    up = np.arange(len(v)) <= np.argmax(abs(v))
    masks = {"all_eligible": valid, "outward": valid & up, "return": valid & ~up}
    if (topology, phase) == ("1r", "set"):
        free = observed < LIMIT_MASK_A
        masks.update(nonlimited=valid & free, nonlimited_outward=valid & free & up,
                     nonlimited_return=valid & free & ~up, limited=valid & ~free)
    result = {}
    for name, mask in masks.items():
        count = mask.sum(axis=0)
        logs = np.zeros_like(observed)
        logs[mask] = np.log10(observed[mask])
        mean = logs.sum(axis=0)/np.maximum(count, 1)
        delta = np.where(mask, logs-mean, 0.)
        spread = float(np.sqrt(np.sum(delta**2)/mask.sum()))
        result[name] = {"n": int(mask.sum()), "empirical_within_index_rms_decade": spread,
                        "proposed_systematic_log_mean_curve_rmse_target_decade": spread,
                        "equivalent_fixed_waveform_total_rmse_target_decade": float(np.sqrt(2.)*spread)}
        if name == "all_eligible":
            result[name]["native_points"] = [{"index": int(j), "voltage_V": float(v[j]),
                "branch": "outward" if up[j] else "return", "n_eligible": int(count[j]),
                "mean_log10_current_A": float(mean[j]),
                "population_sd_log10_current": float(np.sqrt(np.sum(delta[:, j]**2)/count[j])),
                "q10_current_A": float(np.quantile(observed[mask[:, j], j], .1)),
                "q90_current_A": float(np.quantile(observed[mask[:, j], j], .9))}
                for j in np.flatnonzero(count)]
    return result


def main():
    raw, loaded = {}, hashlib.sha256()
    with np.load(DATA) as archive:
        for top, ids in TRAIN.items():
            for cid in ids:
                for phase in ("set", "reset"):
                    prefix = f"{top}_cycle_{cid}_{phase}"
                    v, i = archive[prefix+"_voltage"].copy(), archive[prefix+"_current"].copy()
                    raw[top, cid, phase] = (v, abs(i))
                    for suffix, array in (("voltage", v), ("stored_current", i)):
                        loaded.update((prefix+suffix).encode()); loaded.update(array.tobytes())
    features = {}

    def add(identifier, rows, *, description, unit, transform="log10", primary=False,
            acceptance_mode="central_quantile_interval", **metadata):
        eligible = [row["value"] for row in rows if row["eligible"]]
        if not eligible:
            raise ValueError(f"Empty feature {identifier}")
        stats = distribution(eligible, transform)
        interval = [0., stats["q90"]] if acceptance_mode == "upper_flatness_bound" else [stats["q10"], stats["q90"]]
        features[identifier] = {"description": description, "unit": unit, "primary_gate": primary,
            "distribution": stats, "proposed_empirical_acceptance_interval": interval,
            "acceptance_mode": acceptance_mode,
            "n_censored_or_ineligible_records": len(rows)-len(eligible), "records": rows, **metadata}

    # Native read points and same-polarity return/outward ratios.
    read_points = {"1r": {"set": [.05, .1, .2, 1., 2., 4.], "reset": [.025, .05, .1, .2, .5, 1.]},
                   "1t1r": {"set": [.1, .2, .5, 1., 2., 4.], "reset": [.1, .2, .4, .8, 1.2, 1.4, 1.6]}}
    primary_reads = {("1r", "set", "outward", 2.), ("1r", "reset", "return", .2),
        ("1t1r", "set", "return", .2), ("1t1r", "set", "return", 2.),
        ("1t1r", "set", "return", 4.), ("1t1r", "reset", "outward", .2),
        ("1t1r", "reset", "return", .2)}
    for top, phases in read_points.items():
        for phase, points in phases.items():
            for x in points:
                for branch in ("outward", "return"):
                    rows = []
                    for cid in TRAIN[top]:
                        v, i = raw[top, cid, phase]; value, index = read_at(v, i, x, branch)
                        rows.append({"record": cid, "value": value, "eligible": value >= FLOORS[phase],
                            "native_index": index, "voltage_V": float(v[index]),
                            "at_bare_SET_compliance": bool(top == "1r" and phase == "set" and value >= LIMIT_MASK_A)})
                    add(f"{top}.{phase}.{branch}.I_{tag(x)}V", rows,
                        description=f"{top} {phase} {branch} terminal current magnitude at |Vcmd|={x:g}V",
                        unit="A", primary=(top, phase, branch, x) in primary_reads,
                        topology=top, phase=phase, branch=branch, magnitude_V=x)
                rows = []
                for cid in TRAIN[top]:
                    v, i = raw[top, cid, phase]
                    outward, oi = read_at(v, i, x, "outward"); returning, ri = read_at(v, i, x, "return")
                    rows.append({"record": cid, "value": returning/outward,
                        "eligible": min(returning, outward) >= FLOORS[phase], "outward_index": oi, "return_index": ri})
                add(f"{top}.{phase}.return_over_outward_{tag(x)}V", rows,
                    description=f"Same-polarity hysteresis ratio |Ireturn|/|Ioutward| at |Vcmd|={x:g}V",
                    unit="ratio", topology=top, phase=phase, magnitude_V=x)

    # Local trajectory exponents, including terminal sublinear SET return.
    for top in TRAIN:
        for phase in ("set", "reset"):
            windows = (SET_WINDOWS if phase == "set" else RESET_WINDOWS) if top == "1t1r" else (
                [(.1, .5), (.5, 1.), (1., 2.), (2., 2.4)] if phase == "set" else [(.05, .1), (.1, .2), (.2, .4), (.8, 1.2)])
            for branch in ("outward", "return"):
                for a, b in windows:
                    rows = []
                    for cid in TRAIN[top]:
                        v, i = raw[top, cid, phase]; ia, ja = read_at(v, i, a, branch); ib, jb = read_at(v, i, b, branch)
                        ratio = ib/ia; alpha = np.log(ratio)/np.log(abs(v[jb]/v[ja]))
                        rows.append({"record": cid, "value": float(alpha),
                            "eligible": min(ia, ib) >= FLOORS[phase], "current_ratio_high_over_low": ratio,
                            "apparent_terminal_R_ratio_high_over_low": abs(v[jb]/v[ja])/ratio,
                            "indices": [ja, jb], "any_bare_SET_compliance": bool(top == "1r" and phase == "set" and max(ia, ib) >= LIMIT_MASK_A)})
                    primary = top == "1t1r" and (
                        phase == "set" and branch == "return" and (a, b) in [(.1, .2), (1., 2.), (2., 4.)]
                        or phase == "reset" and branch == "outward" and (a, b) == (1., 1.4))
                    add(f"{top}.{phase}.{branch}.alpha_{tag(a)}_{tag(b)}V", rows,
                        description="Native-endpoint trajectory exponent dlog|I|/dlog|Vcmd| (secant), not a frozen-state derivative",
                        unit="dimensionless", transform="identity", primary=primary,
                        topology=top, phase=phase, branch=branch, window_abs_V=[a, b])

    # RESET plateau-like windows are tested for flatness, not presumed flat.
    for a, b in RESET_FLATNESS_WINDOWS:
        data_rows = {name: [] for name in ("geomean_current", "log_current_q90_q10_span", "ols_log_slope")}
        paired = []
        for cid in TRAIN["1t1r"]:
            v, i = raw["1t1r", cid, "reset"]
            inds = branch_indices(v, "outward"); inds = inds[(abs(v[inds]) >= a-1e-9) & (abs(v[inds]) <= b+1e-9)]
            good = np.isfinite(i[inds]) & (i[inds] >= FLOORS["reset"])
            selected = inds[good]; ly = np.log10(i[selected]); lx = np.log10(abs(v[selected]))
            slope = np.dot(lx-lx.mean(), ly-ly.mean())/np.dot(lx-lx.mean(), lx-lx.mean())
            values = {"geomean_current": float(10.**ly.mean()),
                      "log_current_q90_q10_span": float(np.quantile(ly, .9)-np.quantile(ly, .1)),
                      "ols_log_slope": float(slope)}
            for key, value in values.items():
                data_rows[key].append({"record": cid, "value": value, "eligible": bool(np.all(good)),
                                       "n_points": len(inds), "n_above_floor": int(good.sum())})
            sv, si = raw["1t1r", cid, "set"]
            paired.append({"record": cid, "value": values["geomean_current"]/float(si[np.argmax(abs(sv))]),
                           "eligible": bool(np.all(good))})
        for name, rows in data_rows.items():
            add(f"1t1r.reset.outward.{name}_{tag(a)}_{tag(b)}V", rows,
                description=f"RESET outward window |Vcmd|={a:g}–{b:g}V: {name}",
                unit="A" if name == "geomean_current" else "decade" if name.startswith("log_current") else "dimensionless",
                transform="log10" if name == "geomean_current" else "identity",
                primary=(a, b) == (1., 1.4) and name in ("geomean_current", "log_current_q90_q10_span"),
                acceptance_mode="upper_flatness_bound" if name == "log_current_q90_q10_span" else "central_quantile_interval",
                topology="1t1r", phase="reset", branch="outward", window_abs_V=[a, b])
        add(f"1t1r.reset.plateau_to_SET_apex_{tag(a)}_{tag(b)}V", paired,
            description="Same-record RESET window geometric mean divided by measured SET-apex current; not an inferred transistor cap",
            unit="ratio", topology="1t1r", phase="reset", window_abs_V=[a, b])

    # RESET peak distribution, including endpoint maxima rather than deleting them.
    for top in TRAIN:
        peaks = {name: [] for name in ("peak_current", "peak_abs_voltage")}
        for cid in TRAIN[top]:
            v, i = raw[top, cid, "reset"]; inds = branch_indices(v, "outward"); j = int(inds[np.argmax(i[inds])])
            for name, value in (("peak_current", float(i[j])), ("peak_abs_voltage", float(abs(v[j])))):
                peaks[name].append({"record": cid, "value": value, "eligible": True,
                    "native_index": j, "endpoint_maximum": j == int(np.argmax(abs(v)))})
        for name, rows in peaks.items():
            add(f"{top}.reset.outward.{name}", rows, description="Outward RESET peak; endpoint maxima retained",
                unit="A" if name == "peak_current" else "V", transform="log10" if name == "peak_current" else "identity",
                primary=True, topology=top, phase="reset", branch="outward")

    # Bare SET transition intervals and the only universally nonlimited low read.
    for threshold in (.1e-3, .5e-3, LIMIT_MASK_A):
        rows = []
        for cid in TRAIN["1r"]:
            v, i = raw["1r", cid, "set"]; outward = branch_indices(v, "outward")
            j = int(outward[np.flatnonzero(i[outward] >= threshold)[0]])
            rows.append({"record": cid, "value": float((v[j-1]+v[j])/2.), "eligible": True,
                "last_below_voltage_V": float(v[j-1]), "first_crossing_voltage_V": float(v[j]),
                "native_index": j, "bracket_width_V": float(v[j]-v[j-1])})
        add(f"1r.set.transition_midpoint_at_{tag(threshold*1e3)}mA", rows,
            description="First outward current-threshold crossing bracket midpoint; voltage uncertainty remains the full native bracket",
            unit="V", transform="identity", primary=threshold == LIMIT_MASK_A,
            topology="1r", phase="set", current_threshold_A=threshold)
    read_resistance, reset_fit, reset_fit_25, first_anomaly = [], [], [], []
    for cid in TRAIN["1r"]:
        sv, si = raw["1r", cid, "set"]; i05, j = read_at(sv, si, .05, "return")
        read_resistance.append({"record": cid, "value": float(abs(sv[j])/i05),
            "eligible": FLOORS["set"] <= i05 < LIMIT_MASK_A, "current_A": i05, "native_index": j})
        rv, ri = raw["1r", cid, "reset"]; up = branch_indices(rv, "outward")
        for minimum, rows in ((.05, reset_fit), (.025, reset_fit_25)):
            keep = up[(abs(rv[up]) >= minimum-3e-7) & (abs(rv[up]) <= .2+3e-7)]
            resistance = np.dot(rv[keep], rv[keep])/np.dot(abs(rv[keep]), ri[keep])
            rows.append({"record": cid, "value": float(resistance), "eligible": True,
                         "window_abs_V": [minimum, .2], "n_points": len(keep)})
        i25, j25 = read_at(rv, ri, .025, "outward"); i50, j50 = read_at(rv, ri, .05, "outward")
        first_anomaly.append({"record": cid, "value": float((abs(rv[j25])/i25)/(abs(rv[j50])/i50)),
                              "eligible": True, "current_25mV_A": i25, "current_50mV_A": i50})
    add("1r.set.return.R_0p05V", read_resistance, description="Below-compliance retained read resistance at+0.05V",
        unit="ohm", primary=True, topology="1r", phase="set", branch="return")
    add("1r.reset.outward.R_fit_0p05_0p2V", reset_fit, description="Origin-constrained low-voltage I=G|V| fit, R=1/G",
        unit="ohm", primary=True, topology="1r", phase="reset", branch="outward")
    add("1r.reset.outward.R_fit_0p025_0p2V_sensitivity", reset_fit_25,
        description="Same resistance fit including initial25mV point; mandatory sensitivity, not silent data deletion",
        unit="ohm", topology="1r", phase="reset", branch="outward")
    add("1r.reset.outward.R_25mV_over_R_50mV", first_anomaly,
        description="Initial25mV anomaly diagnostic; no assigned instrument cause",
        unit="ratio", topology="1r", phase="reset", branch="outward")

    scatter = {}
    for top, ids in TRAIN.items():
        scatter[top] = {}
        for phase in ("set", "reset"):
            v = raw[top, ids[0], phase][0]
            obs = np.array([raw[top, cid, phase][1] for cid in ids])
            if not all(np.array_equal(raw[top, cid, phase][0], v) for cid in ids):
                raise ValueError("Native grids differ")
            scatter[top][phase] = curve_scatter(v, obs, phase, top)

    assumptions = [
        {"id": "A1", "statement": "Use recovered Git workbooks as current experimental records; native cache values are unchanged.",
         "source": "Recovered workbook/native-cache equality audit; current training-array digest", "need": "Data provenance", "evidence": "No interpolation, clipping or resampling in this script", "if_violated": "Rebuild constraints from corrected original source."},
        {"id": "A2", "statement": "Treat the regular voltage column as nominal terminal command for circuit modeling; separately expose actual hBN voltage under compliance.",
         "source": "Native regular ramps and1mA plateau; voltage header AV", "need": "Avoid driving material with5V while reporting1mA through~100ohm", "evidence": "Strong circuit consistency, not independent instrument readback documentation", "if_violated": "Use verified readback voltage and measurement model; preserve both columns."},
        {"id": "A3", "statement": "Bare SET has1mA compliance; bare RESET is not clipped at1mA.",
         "source": "Prior confirmed protocol and raw RESET peaks above1mA", "need": "Correct external circuit closure", "evidence": "Raw peak/current distributions in this report", "if_violated": "Replace with documented instrument settings; do not change intrinsic law by topology."},
        {"id": "A4", "statement": "Measured branch slopes are trajectory features; they do not establish a frozen hidden state or a constitutive derivative.",
         "source": "No state-resolved measurements accompany these sweeps", "need": "Prevent overinterpreting sublinear slopes/plateau", "evidence": "Same-polarity hysteresis is explicitly measured", "if_violated": "Use independent retention/pulse constraints before asserting frozen-state laws."},
        {"id": "A5", "statement": "1T1R numbered SET/RESET pairs4-35 are treated as an ordered sequence; first3 remain outside primary normal-operation targets.",
         "source": "Existing confirmed S/R pairing and advisor exclusion", "need": "State continuity and fixed training scope", "evidence": "Native labels retained", "if_violated": "Revise sequence metadata; never infer missing elapsed time."},
        {"id": "A6", "statement": "Bare same-label SET-to-RESET pairing is conditional; chronology between bare sheets and physical cell identity are not asserted.",
         "source": "Workbook labels; absent independent chronology/cell map", "need": "Avoid invented cross-record state history", "evidence": "Low read consistency is evidence of compatibility only", "if_violated": "Use verified cell/sequence metadata, retaining raw labels."},
        {"id": "A7", "statement": "Same nominal stack/geometry motivates a shared material equation; individual microscopic states need not be identical.",
         "source": "Zhu Nature Methods and SI chip layout, previously verified", "need": "Define the unification contract", "evidence": "Common nominal Au-Ti/hBN/W geometry, distinct physical cells", "if_violated": "Document measured structural differences before allowing corresponding parameters."},
        {"id": "A8", "statement": "No physical dwell time, sweep speed, thermal transient or cell identity is inferred from voltage-point spacing.",
         "source": "Available records contain voltage/current only", "need": "Bound interpretation of any dynamics", "evidence": "No time columns used", "if_violated": "Add measured time/protocol data before calibrating rates in seconds."},
        {"id": "A9", "statement": "Historical phase current floors are reporting/censor thresholds, not certified instrument detection limits.",
         "source": "Existing1e-11A SET/1e-12A RESET scoring convention", "need": "Comparable finite log scores", "evidence": "Censored raw values remain present and counts are reported", "if_violated": "Repeat with documented noise/offset model and preserve threshold sensitivity."},
        {"id": "A10", "statement": "Empirical Q10-Q90 feature bands and within-index scatter are descriptive training variability, not confidence intervals or irreducible noise.",
         "source": "Fixed repeated-record feature definitions before next fit", "need": "Data-scaled prospective acceptance proposal", "evidence": "No iid/population or new-device coverage claim", "if_violated": "Use validated stochastic/state model and independent experiments for population claims."},
        {"id": "A11", "statement": "The positive SET transistor mapping at nominalVG=1.1V is interpolated between1.0V/1.5V gate curves. The reverse lookup contains a1.1V node; this does not independently establish same-transistor identity. Internal hBN resistance trends are conditional on the appropriate mapping.",
         "source": "Independent direct lookup inventory: SET gates0.5,1.0,1.5,...; reverse table0.1V gate spacing includes1.1V", "need": "Separate raw terminal-shape evidence from deembedded-material inference", "evidence": "Reviewer reports SET mapping scale~0.8 or effectivegate~1.03V can change the inferred internal resistance trend; these are assumption tests, not established calibration uncertainty bounds", "if_violated": "Keep raw terminal feature gates; report polarity-specific mapping-conditioned inference and profile justified gate interpolation before introducing new material physics."},
    ]
    for row in assumptions:
        row.update(human_confirmed_necessary_or_simplifying=None, decision_id=None,
                   registry_status="documented_evidence_or_proposal_not_human_finalized")
    primary = [identifier for identifier, feature in features.items() if feature["primary_gate"]]
    result = {
        "status": "training_only_prefit_feature_constraints_proposed",
        "training_ids": TRAIN, "test_arrays_loaded": False,
        "training_arrays_sha256": loaded.hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "definitions": {"units": "voltage V; current A; resistance ohm; alpha natural-log ratio (base independent)",
            "branch_rule": "First maximum|V| index belongs to outward; all subsequent indices belong to return, including repeated apex.",
            "current_rule": "Use stored magnitude; negative RESET voltage does not imply stored current has physical sign.",
            "interpolation": "None: all readpoints are exact native samples within3e-7V float storage tolerance.",
            "feature_bands": "Uniform empirical Q10-Q90 across eligible training records, chosen before the next fit; not confidence intervals.",
            "flatness_exception": "Within-window log-current span is bounded above by its empiricalQ90 only; a smooth model is allowed to be flatter than individual noisy/fluctuating records and is not forced to reproduce their minimum variation.",
            "censoring": "Below-floor endpoint features retain raw values but do not define primary log-feature distributions; counts explicit.",
            "first25mV": "Preserved everywhere, included in full-curve score; low-resistance fit with/without it reported separately.",
            "bare_transition": "Interval midpoint is descriptive; original last-below/first-crossing bracket retained.",
            "plateau": "Window flatness is evaluated; constant current and state freezing are not assumed."},
        "features": features, "curve_scatter_targets": scatter,
        "proposed_success_criteria_before_next_fit": {
            "status": "proposed_acceptance_contract_freeze_before_parameter_search",
            "structural_gates": ["One common material current law and state evolution for both topologies; no topology-conditioned intrinsic equation.",
                "Measured transistor and instrument compliance enter only external circuit closure; use solved material voltage/current.",
                "Continuous1T state propagation including SET return, RESET and cycle boundaries; no per-cycle fitted resets.",
                "If using a common initial bare state per conditional record, declare it; do not invent bare inter-record chronology.",
                "KCL/KVL, passivity and numerical refinement checks must pass before comparing accuracy."],
            "primary_feature_gate": "For every listed primary feature, the median of model-generated training-record features must lie inside its fixed acceptance interval. These are empiricalQ10-Q90 bands, except plateau fluctuation span uses only the empiricalQ90 upper bound (zero model fluctuation is allowed). Report every failure individually; global RMSE cannot compensate. Also report full observed range and model median, not only pass/fail.",
            "primary_feature_ids": primary,
            "joint_shape_gate": "Simultaneously satisfy1T SET returnI(0.2V),I(2V),I(4V), the1-2/2-4V slope change, RESET peak current and1.0-1.4V plateau level/flatness. These must arise in one continuously propagated trajectory with one material parameter set.",
            "score_reporting": "Report true ordered record-level log RMSE for training1T4-35, all bare records, each phase/branch; separate bare SET limited/nonlimited. Keep raw scatter in totals. Do not substitute a two-cycle surrogate or median curve for these scores.",
            "secondary_data_scaled_curve_target": "Proposed smooth-curve accuracy target: systematic deviation of predicted mean log curve from observed mean log curve no larger than the corresponding empirical within-index RMS spread. This is a training reference, not a guaranteed noise floor. For a fixed common curve the equivalent total-RMSE reference is sqrt(2) times spread; cycle-dependent predictions require exact bias/variance/covariance decomposition.",
            "noncompensation": "A failed low-read/plateau/transition feature remains failed even if other voltage regions improve. Quantization-level compliance-platform variance is not a stand-alone intrinsic-material acceptance target.",
            "model_discovery_limit": "Passing these curve gates supports mathematical consistency and descriptive adequacy, not unique microscopic mechanism identification.",
            "transistor_mapping_condition": "All numerical feature gates here concern measured terminal curves and do not require any particular deembeddedRh trend. Treat positive-SET internalRh increase/decrease as conditional on forward gate interpolation/current-scale assumptions; the reverse table has a1.1V node. No±20% calibration confidence is claimed. A legitimate external mapping profile precedes rejection of the shared material law on that internal trend.",
        },
        "assumption_registry": assumptions,
    }
    (OUT/"feature_constraints.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    def feature_median(identifier):
        return features[identifier]["distribution"]["median"]

    set_alpha_high = features["1t1r.set.return.alpha_2_4V"]["distribution"]
    lines = ["# 统一1R/1T1R模型：下一轮拟合前的实测约束", "",
        "仅使用固定训练集：裸1R前14个原编号记录；1T1R第4–35圈。未读测试数组，未读旧模型预测或拟合参数。所有量都是端口曲线特征；局部斜率不等同于冻结状态的本构导数。", "",
        "## 建议先冻结的验收口径", "",
        "1. 两接法共用材料电流方程、状态方程及参数；晶体管和仪器限流只进入外电路。实际hBN压降必须参与演化。",
        "2. 1T1R连续传播状态，不能每圈重置。裸R只按已声明的条件SET/RESET配对处理，不假定表间时间或同一cell。",
        "3. 以下主特征的模型中位数应同时落入预先固定的验收区间：通常为训练经验Q10–Q90；平台波动幅度只设经验Q90上限，允许平滑模型更平。逐项列失败，不能以总RMSE抵消。该范围不是置信区间。",
        "4. 同时报告真实逐记录RMSE、各分支及裸SET非限流/限流分数；保留记录散布。两圈替代目标不能称为完整顺序损失。",
        "5. 电学守恒和数值收敛通过后再谈精度。通过电学约束仍不等于证明微观机制。", "",
        "## 关键联合形状", "",
        f"1T1R SET返扫的端口指数从低压超线性转为高压次线性：0.1–0.2V指数中位{feature_median('1t1r.set.return.alpha_0p1_0p2V'):.3f}，1–2V为{feature_median('1t1r.set.return.alpha_1_2V'):.3f}，2–4V为{set_alpha_high['median']:.3f}；后一区间Q10–Q90仅{set_alpha_high['q10']:.3f}–{set_alpha_high['q90']:.3f}。2V与4V电流中位分别{feature_median('1t1r.set.return.I_2V')*1e6:.3f}µA、{feature_median('1t1r.set.return.I_4V')*1e6:.3f}µA，必须与0.2V的{feature_median('1t1r.set.return.I_0p2V')*1e6:.3f}µA同时复现。",
        f"RESET外扫1.0–1.4V窗口的端口电流近似平台：窗口几何平均电流中位{feature_median('1t1r.reset.outward.geomean_current_1_1p4V')*1e6:.3f}µA，端点指数中位{feature_median('1t1r.reset.outward.alpha_1_1p4V'):.5f}；但每条曲线有波动，不能把它解释成已经证明的静态恒流元件。整个RESET峰中位{feature_median('1t1r.reset.outward.peak_current')*1e6:.3f}µA。",
        "以上都是不依赖旧材料参数的端口观测。正向SET的VG=1.1V晶体管曲线来自1.0/1.5V之间插值；反向表含1.1V节点，但该事实不证明是同一只晶体管。内部Rh随电流上升与否受相应映射影响，不能作为无条件材料否决条件。正向尺度0.8或有效栅压1.03V只能作为有记录的假设敏感性情景，不能称为已知±20%校准误差。", "",
        "## 主特征训练范围", "", "| 特征 | 中位数 | 验收区间 | 单位 |", "|---|---:|---:|---|"]
    for identifier in primary:
        feature = features[identifier]; d = feature["distribution"]
        lo, hi = feature["proposed_empirical_acceptance_interval"]
        lines.append(f"| `{identifier}` | {d['median']:.6g} | {lo:.6g} – {hi:.6g} | {feature['unit']} |")
    lines += ["", "电流单位A。完整逐记录值、跨越区间、原始索引、被floor筛出的数量及全部辅助特征均在同名JSON中。", "",
        "## 曲线散布参考", "", "| 数据/相位 | 经验逐点log散布RMS | 共同确定曲线总RMSE参考√2×散布 |", "|---|---:|---:|"]
    for top in TRAIN:
        for phase in ("set", "reset"):
            d = scatter[top][phase]["all_eligible"]
            lines.append(f"| {top} {phase} | {d['empirical_within_index_rms_decade']:.6f} | {d['equivalent_fixed_waveform_total_rmse_target_decade']:.6f} |")
    lines += ["", "上述参考来自训练记录变化，不是不可约测量噪声。循环状态模型可能解释其中一部分；应使用偏差、观测散布、预测散布和协方差的精确分解。平台点极小散布不应被当作材料精度硬指标。", "",
        "## 数据处理边界", "",
        "裸RESET最初−25mV点没有删掉；主低阻读数用−0.05至−0.2V，同时保存包含−25mV的拟合敏感性。零偏及低于既有floor的电流仍在原始记录；特征统计显示筛选数。没有把floor当成实测电流，也没有插值生成点。", "",
        "裸SET的阈值给出最后低于阈值与首次达到阈值的电压区间；不能把区间中点解释成精确的原子势垒或材料演化停止阈值。SET限流平台不能用于直接推断本征硬态电阻。", "",
        "## 假设登记", "",
        "下列条目记录依据、使用需求与失效后的处理；没有替研究者确认其“必要/简化”属性。", ""]
    for a in assumptions:
        lines += [f"### {a['id']}", "", a["statement"], "", f"依据：{a['source']}；用途：{a['need']}。证据：{a['evidence']}。若不成立：{a['if_violated']}", ""]
    (OUT/"feature_constraints.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"n_features": len(features), "n_primary": len(primary),
        "primary": {identifier: features[identifier]["distribution"] for identifier in primary},
        "training_arrays_sha256": loaded.hexdigest()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
