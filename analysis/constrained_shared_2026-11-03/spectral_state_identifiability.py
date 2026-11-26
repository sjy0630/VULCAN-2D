"""Training-only fixed-shape gamma oracle and necessary direction checks.

No ODE, new material parameter, test record, inferred elapsed time or cell ID.
Native mean paths are diagnostics. Pair error bounds use terminal circuit
closure (not fixed observed-current deembedding) and apply only to the frozen
shape, nominal transistor mapping and stated gamma direction constraint.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from vulcan2d.transistor import measured_transistor_mapping
from spectral_readout import spectral_current

DATA = ROOT / "analysis/shared_hbn_2026-11-03/data/native_observations.npz"
TRAIN = {"1r": [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
         "1t1r": list(range(4, 36))}
REPRESENTATIVE = {"1r": [1, 8, 15], "1t1r": [4, 19, 35]}
FLOORS = {"set": 1e-11, "reset": 1e-12}
LIMIT = .995e-3  # Existing conservative classification of nominal 1 mA compliance.
ORDER = 512


class Audit:
    def __init__(self, parameters):
        self.p = dict(parameters)
        self.reference_gamma = self.p.pop("gamma_ev")
        self.mapping = measured_transistor_mapping()

    def intrinsic(self, v, loggamma, order=ORDER):
        return np.abs(spectral_current(v, 10.**loggamma, order=order, **self.p))

    def invert(self, voltage, current):
        if abs(voltage) < 1e-14:
            return None, "zero_internal_voltage"
        if current >= abs(voltage)/self.p["ron_ohm"]:
            return None, "at_or_above_ohmic_ceiling"
        def residual(lg):
            return math.log10(float(self.intrinsic(voltage, lg))/current)
        if residual(-16) > 0:
            return None, "below_numerical_gamma_bracket"
        if residual(5) < 0:
            return None, "above_numerical_gamma_bracket"
        lg = brentq(residual, -16., 5., xtol=2e-11)
        return lg, "pointwise_oracle"

    def terminal(self, top, phase, command, loggamma, order=ORDER):
        """Forward current; observed current is not used here."""
        if top == "1r":
            return float(self.intrinsic(command, loggamma, order))
        sign = 1. if phase == "set" else -1.
        mag = abs(command)
        def kcl(local):
            return float(self.intrinsic(sign*local, loggamma, order)) - self.mapping.current(mag-local, 1.1, phase)
        local = brentq(kcl, 0., mag, xtol=1e-13, rtol=1e-13)
        return float(self.intrinsic(sign*local, loggamma, order))

    def profile(self, top, phase, voltage, current, compliance_fraction=None):
        apex = int(np.argmax(abs(voltage)))
        points = []
        for j, (command, observed) in enumerate(zip(voltage, current)):
            point = {"native_index": j, "branch": "outward" if j <= apex else "return",
                     "command_V": float(command), "raw_current_A": float(observed),
                     "below_scoring_floor": bool(observed < FLOORS[phase]),
                     "gamma_ev": None, "log10_gamma_ev": None}
            limited = 0. if compliance_fraction is None else float(compliance_fraction[j])
            point["compliance_fraction"] = limited
            if abs(command) < 1e-14:
                point["status"] = "zero_voltage_nonidentifiable"
                point["internal_voltage_V"] = 0.
            elif 0 < limited < 1:
                point["status"] = "mean_mixes_compliant_and_noncompliant_records"
                point["internal_voltage_V"] = None
            else:
                target = LIMIT if limited == 1 else float(observed)
                if top == "1t1r":
                    try:
                        vt = self.mapping.voltage_for_current(target, 1.1, phase)
                        local = float(np.sign(command)*(abs(command)-vt))
                    except ValueError:
                        point["status"] = "exceeds_transistor_lookup"
                        point["internal_voltage_V"] = None
                        points.append(point)
                        continue
                    if abs(command)-vt <= 0:
                        point["status"] = "no_passive_internal_voltage"
                        point["internal_voltage_V"] = local
                        points.append(point)
                        continue
                else:
                    local = float(command)
                point["internal_voltage_V"] = local
                lg, status = self.invert(local, target)
                point["status"] = status
                if lg is not None:
                    point["gamma_ev"] = 10.**lg
                    point["log10_gamma_ev"] = lg
                    if limited == 1:
                        point["status"] = "compliance_gamma_lower_bound"
                        point["bound_uses_current_A"] = LIMIT
            points.append(point)
        return points

    def witness(self, top, phase, points, *, branch=None, min_voltage=.05):
        # Raw-path ordering, including repeated apex, is preserved. A pair with
        # a reversed state order is sufficient for a conditional no-go witness.
        eligible = [p for p in points if p["status"] == "pointwise_oracle"
                    and not p["below_scoring_floor"]
                    and abs(p["command_V"]) >= min_voltage-1e-10
                    and (branch is None or p["branch"] == branch)]
        direction = 1 if phase == "set" else -1
        violation, selected = 0., None
        for i, a in enumerate(eligible):
            for b in eligible[i+1:]:
                value = direction*(a["log10_gamma_ev"]-b["log10_gamma_ev"])
                if value > violation:
                    violation, selected = value, (a, b)
        adjacent = sum(direction*(a["log10_gamma_ev"]-b["log10_gamma_ev"]) > 1e-8
                       for a, b in zip(eligible[:-1], eligible[1:]))
        result = {"direction": "nondecreasing" if direction == 1 else "nonincreasing",
                  "branch": branch or "complete_phase", "min_abs_command_V": min_voltage,
                  "n_eligible_points": len(eligible), "n_adjacent_violations": int(adjacent),
                  "max_oracle_direction_violation_decade": violation}
        if selected is None:
            return result
        a, b = selected
        # For monotone I(gamma), the constrained two-point optimum touches
        # gamma_a=gamma_b. Equal and opposite log errors give the exact minimax
        # pair error. Unlike fixed-Vh fits, series calculations close KCL anew.
        def errors(lg, order=ORDER):
            return [math.log10(self.terminal(top, phase, p["command_V"], lg, order)/p["raw_current_A"])
                    for p in (a, b)]
        lo, hi = sorted([a["log10_gamma_ev"], b["log10_gamma_ev"]])
        optimum = brentq(lambda lg: sum(errors(lg)), lo, hi, xtol=2e-10)
        error = errors(optimum)
        refined = errors(optimum, 1024)
        result.update({"pair": [{k: p[k] for k in ("native_index", "branch", "command_V", "raw_current_A", "internal_voltage_V", "gamma_ev")} for p in (a, b)],
                       "common_gamma_at_pair_minimax_ev": 10.**optimum,
                       "terminal_pair_minimax_abs_log_error_lower_bound_decade": max(abs(x) for x in error),
                       "terminal_pair_minimax_log_errors_decade": error,
                       "quadrature_512_to_1024_pair_error_max_change_decade": max(abs(x-y) for x, y in zip(error, refined)),
                       "scope": "Exact two-point minimax bound conditional on fixed shape and nominal transistor mapping; not a global RMSE bound or statistical rejection."})
        return result

    def frozen_fit(self, top, phase, voltage, observed, branch, low, high):
        apex = int(np.argmax(abs(voltage)))
        branch_mask = np.arange(len(voltage)) <= apex
        if branch == "return":
            branch_mask = ~branch_mask
        mask = branch_mask & (abs(voltage) >= low-1e-10) & (abs(voltage) <= high+1e-10)
        if top == "1r" and phase == "set":
            mask &= np.all(observed < LIMIT, axis=0)
        mask &= np.all(observed >= FLOORS[phase], axis=0)
        command = voltage[mask]
        samples = observed[:, mask]
        logmean = np.log10(samples).mean(axis=0)
        mean = 10.**logmean
        if top == "1r":
            local = command
        else:
            drop = np.array([self.mapping.voltage_for_current(x, 1.1, phase) for x in mean])
            local = np.sign(command)*(abs(command)-drop)
        def residual(lg):
            return np.log10(self.intrinsic(local, lg))-logmean
        fit = minimize_scalar(lambda lg: float(np.mean(residual(lg)**2)), bounds=(-16., 5.), method="bounded", options={"xatol": 2e-9})
        err = residual(fit.x)
        refined = np.log10(self.intrinsic(local, fit.x, 1024))-logmean
        return {"topology": top, "phase": phase, "branch": branch,
                "abs_command_window_V": [low, high], "n_native_points": int(mask.sum()),
                "gamma_ev": 10.**float(fit.x), "mean_curve_rmse_decade": float(np.sqrt(np.mean(err**2))),
                "max_abs_mean_error_decade": float(np.max(abs(err))),
                "empirical_within_index_rms_decade": float(np.sqrt(np.mean((np.log10(samples)-logmean)**2))),
                "quadrature_512_to_1024_max_log_change_decade": float(np.max(abs(err-refined))),
                "native_indices": np.flatnonzero(mask).tolist(), "command_V": command.tolist(),
                "mean_raw_current_A": mean.tolist(), "mean_log_residual_decade": err.tolist(),
                "error_definition": "Intrinsic current evaluated at observed-current deembedded voltage; diagnostic only, not series terminal prediction." if top == "1t1r" else "Intrinsic/terminal raw log-current error, noncompliant points only."}


def main():
    source = OUT / "spectral_readout_screen.json"
    family = json.loads(source.read_text())["families"]["gaussian_ensemble"]
    parameters = family["fits"][family["selected_start"]]["parameters"]
    audit = Audit(parameters)
    loaded_hash = hashlib.sha256()
    raw, profiles, representative, fits, witnesses = {}, {}, {}, {}, {}
    with np.load(DATA) as archive:
        for top, ids in TRAIN.items():
            for phase in ("set", "reset"):
                voltages, currents = [], []
                for cid in ids:
                    prefix = f"{top}_cycle_{cid}_{phase}"
                    v = archive[prefix+"_voltage"].copy()
                    original = archive[prefix+"_current"].copy()
                    for suffix, array in (("voltage", v), ("stored_current", original)):
                        loaded_hash.update((prefix+suffix).encode()); loaded_hash.update(array.tobytes())
                    voltages.append(v); currents.append(abs(original))
                if not all(np.array_equal(voltages[0], v) for v in voltages):
                    raise ValueError("Training native waveforms differ")
                v, y = voltages[0], np.asarray(currents)
                raw[top, phase] = (v, y)
                mean = 10.**np.log10(y).mean(axis=0)
                limited = (y >= LIMIT).mean(axis=0) if (top, phase) == ("1r", "set") else None
                key = f"{top}.{phase}"
                points = audit.profile(top, phase, v, mean, limited)
                profiles[key] = {"mean_definition": "Geometric mean of raw current magnitudes, no floor replacement; not a realizable individual trajectory.",
                                 "status_counts": dict(Counter(p["status"] for p in points)),
                                 "n_raw_below_scoring_floor": int((y < FLOORS[phase]).sum()),
                                 "n_raw_zero_command": int(len(ids)*np.sum(abs(v) < 1e-14)),
                                 "n_raw_at_or_above_ohmic_ceiling_nonzero_command": int(np.sum((y >= abs(v)[None, :]/parameters["ron_ohm"]) & (abs(v)[None, :] > 1e-14))) if top == "1r" else None,
                                 "points": points}
                witnesses[key] = {b or "complete_phase": audit.witness(top, phase, points, branch=b)
                                  for b in (None, "outward", "return")}
                if (top, phase) == ("1r", "reset"):
                    witnesses[key]["complete_including_25mV"] = audit.witness(top, phase, points, min_voltage=.025)
                representative[key] = {}
                for cid in REPRESENTATIVE[top]:
                    row = y[ids.index(cid)]
                    fractions = (row >= LIMIT).astype(float) if (top, phase) == ("1r", "set") else None
                    curve = audit.profile(top, phase, v, row, fractions)
                    representative[key][str(cid)] = {"status_counts": dict(Counter(p["status"] for p in curve)),
                        "complete_phase_witness": audit.witness(top, phase, curve), "points": curve}
                print(key, "profile complete", flush=True)
    for name, specification in {
        "bare_HRS_SET": ("1r", "set", "outward", .1, 2.4),
        "bare_RESET_tail": ("1r", "reset", "outward", 1.8, 2.5),
        "series_RESET_plateau": ("1t1r", "reset", "outward", 1., 1.4),
    }.items():
        top, phase, branch, low, high = specification
        v, y = raw[top, phase]
        fits[name] = audit.frozen_fit(top, phase, v, y, branch, low, high)
    # Same pair witnesses in every training record; no test records are read.
    replicate_witnesses = {}
    for top, phase, branch in [("1r", "set", "outward"), ("1r", "reset", "outward"),
                               ("1t1r", "set", "return"), ("1t1r", "reset", "outward")]:
        key = f"{top}.{phase}"
        witness = witnesses[key][branch]
        if "pair" not in witness:
            continue
        indices = [p["native_index"] for p in witness["pair"]]
        v, y = raw[top, phase]
        rows = []
        for cid, row in zip(TRAIN[top], y):
            pair = audit.profile(top, phase, v[indices], row[indices])
            # Fix branch/native index after two-point profiling; both come from
            # one actual native branch, not an artificial two-point waveform.
            for p, j in zip(pair, indices):
                p["native_index"] = j; p["branch"] = branch
            result = audit.witness(top, phase, pair, min_voltage=.025)
            rows.append({"record": cid, **result})
        replicate_witnesses[key+"."+branch] = rows
    series_v, series_i = raw["1t1r", "set"]
    return_mask = ((np.arange(len(series_v)) > np.argmax(abs(series_v)))
                   & (abs(series_v) >= .06-1e-10))
    return_logs = np.log10(series_i[:, return_mask])
    return_scatter = float(np.sqrt(np.mean((return_logs-return_logs.mean(axis=0))**2)))
    # Independently allow two gamma variables instead of enforcing equality.
    # This checks that the necessary SET pair optimum really occurs at contact.
    key_witness = witnesses["1t1r.set"]["return"]
    key_pair = key_witness["pair"]
    def pair_constraints(q):
        errors = np.array([math.log10(audit.terminal("1t1r", "set", p["command_V"], q[k])/p["raw_current_A"])
                           for k, p in enumerate(key_pair)])
        return np.r_[q[2]-errors, q[2]+errors, q[1]-q[0]]
    g0 = math.log10(key_witness["common_gamma_at_pair_minimax_ev"])
    independent = minimize(lambda q: q[2], [g0-.02, g0+.02, .2], method="SLSQP",
                           bounds=[(-16, 5), (-16, 5), (0, 5)],
                           constraints=[{"type": "ineq", "fun": pair_constraints}],
                           options={"ftol": 1e-11, "maxiter": 100})
    if not independent.success or np.min(pair_constraints(independent.x)) < -1e-8:
        raise RuntimeError("Independent monotone-pair verification failed")
    report = {
        "status": "training_only_fixed_spectral_shape_gamma_identifiability_no_ODE_no_new_material_parameters",
        "training_ids": TRAIN, "representative_record_ids_preselected_by_record_position": REPRESENTATIVE,
        "selected_static_start": family["selected_start"], "fixed_parameters": parameters,
        "profiles": profiles, "representative_profiles": representative,
        "frozen_gamma_fits": fits, "mean_path_direction_witnesses": witnesses,
        "same_native_pair_in_each_training_record": replicate_witnesses,
        "series_SET_return_above_60mV_empirical_within_index_rms_decade": return_scatter,
        "independent_two_gamma_minimax_check": {
            "optimizer": "SLSQP with two independently varying gamma coordinates, two-sided error constraints, and gamma_a<=gamma_b",
            "success": bool(independent.success), "gamma_pair_ev": (10.**independent.x[:2]).tolist(),
            "terminal_minimax_error_decade": float(independent.fun),
            "difference_from_common_gamma_root_decade": float(independent.fun-key_witness["terminal_pair_minimax_abs_log_error_lower_bound_decade"]),
            "minimum_constraint": float(np.min(pair_constraints(independent.x)))},
        "loaded_training_arrays_sha256": loaded_hash.hexdigest(),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__), source, OUT/"spectral_readout.py", ROOT/"vulcan2d/transistor.py", DATA]},
        "scope_limits": [
            "Only gamma varies. epsilon, eta, width, Ron and temperature are frozen at the selected series soft-LRS screen.",
            "SET gamma nondecreasing and RESET gamma nonincreasing are explicit subhypotheses; electrical passivity does not impose them.",
            "The geometric mean of distinct records need not be an individual state trajectory. Individual pair checks are also reported.",
            "An oracle inverts observed current and is not a predictive switching model. Zero voltage gives no gamma information.",
            "Bare SET compliance yields only a conservative state lower bound. Mixed compliant/noncompliant mean points are not inverted.",
            "The scoring floors are historical numerical scoring choices, not independently established measurement detection limits. Raw currents are preserved.",
            "A frozen-shape failure does not reject all common-shape spectral models or prove a need for a state-dependent energy shift.",
            "Gamma_c/u, microscopic state identity, kinetic times and cell matching are not identified by these static current curves.",
        ]}
    (OUT/"spectral_state_identifiability.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    write_markdown(report)
    print(json.dumps({"frozen_gamma_fits": {k: {a: b for a, b in x.items() if not isinstance(b, list)} for k, x in fits.items()},
                      "mean_witnesses": witnesses}, indent=2))


def write_markdown(report):
    fits = report["frozen_gamma_fits"]
    w = report["mean_path_direction_witnesses"]
    rows = []
    for name, x in fits.items():
        rows.append(f"| {name} | {x['gamma_ev']:.5g} | {x['mean_curve_rmse_decade']:.4f} | {x['max_abs_mean_error_decade']:.4f} | {x['empirical_within_index_rms_decade']:.4f} |")
    wr = []
    for name, branch in [("1r.set", "outward"), ("1r.reset", "outward"),
                         ("1r.reset", "complete_including_25mV"), ("1t1r.set", "return"),
                         ("1t1r.reset", "outward"), ("1t1r.reset", "return")]:
        x = w[name][branch]
        if "pair" not in x:
            continue
        a, b = x["pair"]
        wr.append(f"| {name} {branch} | {a['command_V']:g} → {b['command_V']:g} | {a['gamma_ev']:.4g} → {b['gamma_ev']:.4g} | {x['terminal_pair_minimax_abs_log_error_lower_bound_decade']:.4f} |")
    individual = []
    for name, rows_ in report["same_native_pair_in_each_training_record"].items():
        valid = [x for x in rows_ if x["n_eligible_points"] == 2]
        violating = [x for x in valid if "pair" in x]
        values = [x["terminal_pair_minimax_abs_log_error_lower_bound_decade"] for x in violating]
        individual.append(f"- {name}：相同训练均值见证点对在 {len(valid)}/{len(rows_)} 条记录可反演，其中 {len(violating)} 条违反假设方向；有冲突记录的点对误差下界中位数为 {np.median(values):.4f} decade。" if values else f"- {name}：{len(valid)} 条可反演记录中该点对无方向冲突。")
    text = """# 固定谱形、仅 gamma 变化：训练集状态可辨识性审计

日期：2026-11-03。此报告只使用预先固定的 1R 与 1T1R 训练记录；不读取测试记录、不拟合 ODE、不增加材料参数。

## 结论

当前 soft-LRS 拟合得到的谱形，建议先做联合静态筛选再进入完整动力学拟合。它可以逐点反演出多个数量级的有效耦合，但部分原始路径要求 gamma 先减后增，与“正偏 SET 全程只增加 gamma、负偏 RESET 全程只减小 gamma”的候选单向律精确复现曲线的要求冲突。较明确的例子是 1T1R 的 SET 回扫。这个结论只针对本次冻结的 epsilon、eta、width、Ron、T 和默认晶体管映射；不排除重新联合约束后的共同谱形，也不支持立即添加状态相关能级位移。是否满足一个允许误差的实用模型验收标准，还应按预先约定的特征和误差门槛判断，不能把任何非零残差都当作物理反证。

下一步优先做共同谱形的廉价联合静态筛选：soft-LRS 与裸 1R 的非限流 HRS 同时约束 epsilon、eta、width，初筛每个声明的状态窗口各用一个固定 gamma；并同时检查沿路径的 gamma 方向约束。裸 HRS 若不能在合理误差内保持固定 gamma，应先检验既有 gamma 的渐进 SET 演化能否解释曲率。静态筛选通过后，再谈动力学。不能把两个接法各拟合一套谱形称为统一。

## 冻结 gamma 的局部曲线拟合

下表为同一个固定谱形下，每个指定窗口只优化一个 gamma。1T1R 的这一项使用观测电流去嵌入后的内部电压，属于形状诊断；不是端口预测误差。经验散布是现有训练记录的逐点离散度，不是不可约噪声或正式置信区间。不同窗口的 gamma 也不代表已经证明存在多个物理状态。

| 窗口 | gamma (eV) | 均值曲线 RMSE (decade) | 最大误差 (decade) | 逐点经验 RMS 散布 (decade) |
|---|---:|---:|---:|---:|
""" + "\n".join(rows) + """

裸 HRS 的均值曲线误差小于该窗口较大的经验离散度，因此这项冻结 gamma 拟合本身不能构成一个有统计依据的家族否定结论。它指出的是需要联合约束的曲率问题。

## 路径方向的必要条件

固定电压时，谱电流幅度随 gamma 严格增加。将实测电流逐点反演得到 gamma 后，SET 应沿采集顺序不减，RESET 应不增；此处明确检验的是这个额外单向动力学假设，电学被动性本身并不要求它。保留回扫与重复顶点，不按电压排序覆盖采集顺序。

下表选取每条训练几何均值路径中 gamma 逆序差最大的点对。给出的误差下界是满足该点对单调状态约束时，至少有一个点必须承受的绝对 log-current 误差。它通过共同 gamma 的平衡误差求得；1T1R 每次都重新求解实际串联 KCL，未冻结去嵌入电压。因此它是固定谱形及标称晶体管映射条件下的端口两点 minimax 必要下界，不是完整曲线 RMSE 下界，也不是统计显著性。

| 路径 | 原始顺序命令电压 (V) | oracle gamma (eV) | 两点端口 minimax 下界 (decade) |
|---|---:|---:|---:|
""" + "\n".join(wr) + """

默认方向检查保留 |V|≥0.05 V。裸 RESET 的 -25 mV 初始点另外单独纳入，明确展示其影响；没有静默删除。JSON 保留完整原生网格上的均值轮廓、每类预选 3 条代表记录的逐点结果，以及低于历史评分阈值和无解的标记；全部训练原始记录仍在未改动的数据缓存中。

几何均值未必对应任何一条实际轨迹，因此又在每条训练记录中检验同一原始点对：

""" + "\n".join(individual) + """

作为量级参照，1T1R SET 回扫 |V|≥0.06 V 的逐点训练经验 RMS 散布为 """ + f"{report['series_SET_return_above_60mV_empirical_within_index_rms_decade']:.4f}" + """ decade。均值回扫见证的约 0.102 decade 最小最大误差比这个散布大，且同一方向冲突在全部 32 条记录中复现；这支持优先修改共同谱形的筛选条件，而不能直接证明某个微观过程。两点最大误差与全路径 RMS 是不同指标，此处仅作量级参照。

数学依据：若 SET 中先后两点的 oracle 值 gamma_a > gamma_b，而约束要求 gamma_a'≤gamma_b'，两点最优的最大绝对 log 电流误差在公共 gamma_* 处达到。求解 log[I_a(gamma_*)/I_a,obs] + log[I_b(gamma_*)/I_b,obs] = 0 即使两点误差大小相同、方向相反；RESET 的相反序关系同理。I_a、I_b 均为各点命令电压下重新闭合电路的端口电流。数值积分加密 512→1024 阶在所有均值见证误差上的最大变化低于 3×10^-8 decade。

## 可辨识与不可辨识

- 固定谱形与内部电压后，严格位于 0 与 |Vh|/Ron 之间的电流唯一确定 gamma；这是条件可辨识，不是微观机制识别。gamma 很小时，gamma 主要改变幅度，曲率基本由共同谱形决定。
- V=0 的测量偏移不能识别 gamma。裸 SET 达到约 1 mA compliance 时，真实材料电压低于命令电压，仅可给出 gamma 的保守下界；混合限流与非限流记录的几何均值不做状态反演。Ron 固定为训练估计时，个别低阻原始记录可超过 Ohmic 上界，必须报告为不可达，不能强行反演。
- 裸 HRS 0.1–2.4 V 的 gamma 要随电压显著增加，冻结状态拟合差并不自动排除渐进 SET 激活。裸 RESET 高压尾和系列平台也不能凭静态 gamma 的不同就宣称不同材料理论。
- gamma_c 与 u 的坐标选择没有独立定标；这些电学曲线不能直接给出通道长度、缺陷数、势垒迁移速率或等待时间。不能把 oracle 轨迹作为预测性动力学结果。

## 真正需要补充的约束

1. 正向独立晶体管 VG=1.1 V 的实测曲线。目前正向是 1.0/1.5 V 插值；反向已有 1.1 V 节点。插值敏感性结果不是未知曲线的置信区间。
2. 独立晶体管正反向扫描与 1T1R 中晶体管是否同一物理器件，以及裸 1R 与串联记录是否来自同一 h-BN 单元的实验来源说明。理论相同不等于已经证明所有样品数值参数完全相同。
3. 各采样间隔、扫描速率、两极性之间等待时间和实际测试顺序。没有这些信息只能辨识按采样步的有效演化，不能报告物理秒单位速率。
4. 裸 1R 记录之间的真实连续性与初始化过程。记录编号不能代替连续状态传播的依据。

这些缺项限制物理定量解释；现有数据仍足够先排除本次冻结谱形的部分单向假设，不需要等待补数据才开始这一必要筛选。所有可复算的点、见证对、单独记录敏感性、数值加密结果及输入散列见 `spectral_state_identifiability.json`。
"""
    (OUT/"SPECTRAL_STATE_IDENTIFIABILITY.md").write_text(text)


if __name__ == "__main__":
    main()
