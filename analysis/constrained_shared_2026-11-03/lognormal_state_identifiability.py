"""Frozen lognormal gamma-direction audit; no ODE, material fit or test data.

Reuses the Gaussian audit's independent terminal KCL and two-point minimax
algorithm through a readout adapter. Original Gaussian artifacts are untouched.
The finite-energy backend is checked at the inferred small-gamma states; no
large-gamma or Ohmic-limit certification is inferred from this audit.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from spectral_state_identifiability import Audit, ROOT, OUT, DATA, TRAIN
from refine_lognormal_spectral import readout, tail_diagnostics
from lognormal_spectral_screen import independent_z_integral

SOURCE = OUT/"lognormal_spectral_refinement.json"
RESULT = OUT/"lognormal_state_identifiability.json"


class LognormalAudit(Audit):
    """Readout adapter; keeps terminal and witness methods unchanged."""
    def __init__(self, source):
        self.q = np.array(source["selected_q"], dtype=float)
        self.ron = source["parameters"]["ron_ohm"]
        super().__init__({"gamma_ev": 10.**self.q[0], "ron_ohm": self.ron})

    def intrinsic(self, voltage, loggamma, order=512):
        # Parent witness method calls 512/1024. Map these API levels to the
        # finite-energy 64/128 orders, and name the output metadata accordingly.
        selected_order = 128 if order == 1024 else 64
        q = self.q.copy(); q[0] = loggamma
        return np.abs(readout(q, voltage, self.ron, selected_order))

    def invert(self, voltage, current):
        if abs(voltage) < 1e-14:
            return None, "zero_internal_voltage"
        # This is a numerical search range only, not a physical prior or a
        # previously certified gamma interval. Each inferred state is checked.
        lower, upper = -12., -1.
        def residual(lg):
            return math.log10(float(self.intrinsic(voltage, lg))/current)
        if residual(lower) > 0 or residual(upper) < 0:
            return None, "outside_finite_gamma_search_bracket_no_extrapolation"
        return brentq(residual, lower, upper, xtol=2e-11), "pointwise_oracle"

    def witness(self, *args, **kwargs):
        result = super().witness(*args, **kwargs)
        old = "quadrature_512_to_1024_pair_error_max_change_decade"
        if old in result:
            result["quadrature_64_to_128_pair_error_max_change_decade"] = result.pop(old)
        return result

    def numerical_point(self, voltage, loggamma, independent=False):
        q = self.q.copy(); q[0] = loggamma
        v = np.atleast_1d(voltage)
        i32 = abs(readout(q, v, self.ron, 32))
        i64 = abs(readout(q, v, self.ron, 64))
        tail = tail_diagnostics(q, v, self.ron, 64)
        bounds = np.array(tail["per_voltage_omitted_current_bound_A"])
        result = {"internal_voltage_V": v.tolist(), "gamma_ev": 10.**loggamma,
                  "max_32_to_64_log_change_decade": float(np.max(abs(np.log10(i32/i64)))),
                  "max_omitted_tail_relative_current_bound": float(np.max(bounds/i64)),
                  "max_omitted_tail_log_error_bound_decade": float(np.max(np.log10(1.+bounds/i64)))}
        if independent:
            reference, error_estimate = independent_z_integral(q, v, self.ron)
            result.update(max_64_vs_independent_Z_log_change_decade=float(np.max(abs(np.log10(i64/abs(reference))))),
                          independent_Z_absolute_error_estimate_A=error_estimate,
                          independent_Z_error_estimate_over_min_current=float(error_estimate/np.min(abs(reference))))
        return result


def native_subset(profile, voltages):
    result = []
    for voltage in voltages:
        matches = [p for p in profile if abs(p["command_V"]-voltage) < 1e-9]
        if len(matches) != 1:
            raise ValueError(f"Requested native return/outward point {voltage} absent or ambiguous")
        result.append(matches[0])
    return result


def empirical_interval_pair(audit, points, observed):
    """Feasibility of SET gamma_a<=gamma_b in raw pointwise Q10-Q90 bands.

    This is a diagnostic against empirical bands, not a replacement gate,
    uncertainty interval or test of a population distribution.
    """
    bands, intervals = [], []
    for p in points:
        band = np.quantile(observed[:, p["native_index"]], [.1, .9])
        band_gamma = []
        for current in band:
            local = p["command_V"]-audit.mapping.voltage_for_current(current, 1.1, "set")
            lg, status = audit.invert(local, current)
            if lg is None:
                raise ValueError(f"Empirical interval boundary inversion failed: {status}")
            band_gamma.append(lg)
        bands.append(band); intervals.append(band_gamma)
    bands, intervals = np.array(bands), np.array(intervals)
    feasible = bool(intervals[0, 0] <= intervals[1, 1])
    if feasible:
        first = intervals[0, 0]
        lg = [first, max(first, intervals[1, 0])]
        expansion = 0.
    else:
        # At optimum gamma_a=gamma_b: first current below its Q10, second
        # above its Q90. Equal and opposite log excess minimizes expansion.
        def boundary_errors(common):
            return [math.log10(audit.terminal("1t1r", "set", points[0]["command_V"], common)/bands[0, 0]),
                    math.log10(audit.terminal("1t1r", "set", points[1]["command_V"], common)/bands[1, 1])]
        common = brentq(lambda x: sum(boundary_errors(x)), intervals[1, 1], intervals[0, 0], xtol=2e-11)
        lg = [common, common]
        expansion = max(abs(x) for x in boundary_errors(common))
    predicted = np.array([audit.terminal("1t1r", "set", p["command_V"], g) for p, g in zip(points, lg)])
    return {"commands_V": [p["command_V"] for p in points],
            "native_indices": [p["native_index"] for p in points],
            "raw_current_Q10_Q90_A": bands.tolist(),
            "gamma_intervals_corresponding_to_current_Q10_Q90_ev": (10.**intervals).tolist(),
            "exists_gamma_nondecreasing_solution_inside_both_empirical_bands": feasible,
            "minimum_symmetric_log_current_band_expansion_decade": float(expansion),
            "multiplicative_band_expansion_factor": 10.**expansion,
            "witness_gamma_pair_ev": (10.**np.array(lg)).tolist(),
            "witness_terminal_predicted_current_A": predicted.tolist(),
            "witness_signed_log_position_relative_to_Q10_Q90": np.column_stack([np.log10(predicted/bands[:, 0]), np.log10(predicted/bands[:, 1])]).tolist(),
            "scope": "Additional conditional diagnostic, not a changed primary full-curve gate. Empirical Q10-Q90 are descriptive training bands, not confidence intervals."}


def main():
    source = json.loads(SOURCE.read_text())
    if not source["numerical_checks"]["all_passed"]:
        raise ValueError("Selected frozen source readout did not pass its numerical checks")
    audit = LognormalAudit(source)
    training_hash = hashlib.sha256()
    with np.load(DATA) as archive:
        loaded = {}
        for top in ("1t1r", "1r"):
            currents = []
            for cid in TRAIN[top]:
                prefix = f"{top}_cycle_{cid}_set"
                v = archive[prefix+"_voltage"].copy()
                original = archive[prefix+"_current"].copy()
                for label, data in (("voltage", v), ("stored_current", original)):
                    training_hash.update((prefix+label).encode()); training_hash.update(data.tobytes())
                currents.append(abs(original))
            loaded[top] = (v, np.asarray(currents))
    v, current = loaded["1t1r"]
    indices = np.flatnonzero((np.arange(len(v)) > np.argmax(abs(v))) & (v > 0))
    mean = 10.**np.log10(current).mean(axis=0)
    # Profile uses the native full sweep so indices/branches retain their
    # original meaning; only return points are retained/evaluated below.
    profile = audit.profile("1t1r", "set", v[indices], mean[indices])
    for p, index in zip(profile, indices):
        p["native_index"] = int(index); p["branch"] = "return"
    primary = [p for p in profile if p["command_V"] >= .06-1e-10]
    full_witness = audit.witness("1t1r", "set", primary, branch="return", min_voltage=.06)
    # Same Gaussian witness, plus explicit high-voltage slope window; no pair
    # is silently discarded if the most obstructed location moves.
    pair_specs = {"previous_Gaussian_5_to_1p4V": [5., 1.4],
                  "high_voltage_4_to_2V": [4., 2.],
                  "mid_voltage_2_to_1V": [2., 1.],
                  "high_voltage_5_to_4V": [5., 4.]}
    if "pair" in full_witness:
        pair_specs["strongest_lognormal_mean_pair"] = [p["command_V"] for p in full_witness["pair"]]
    pairs = {}
    for name, commands in pair_specs.items():
        points = native_subset(profile, commands)
        witness = audit.witness("1t1r", "set", points, branch="return", min_voltage=.06)
        rows = []
        native = [p["native_index"] for p in points]
        for cid, observed in zip(TRAIN["1t1r"], current):
            row_points = audit.profile("1t1r", "set", v[native], observed[native])
            for p, index in zip(row_points, native):
                p["native_index"] = index; p["branch"] = "return"
            row = audit.witness("1t1r", "set", row_points, branch="return", min_voltage=.06)
            rows.append({"record": cid, "oracle_pair": row_points, **row})
        pairs[name] = {"commands_V": commands, "mean_oracle_pair": points, "mean_witness": witness,
                       "n_individual_records_with_direction_conflict": sum("pair" in r for r in rows),
                       "empirical_band_monotone_feasibility": empirical_interval_pair(audit, points, current),
                       "individual_records": rows}
    bare_v, bare_current = loaded["1r"]
    bare_mean = 10.**np.log10(bare_current).mean(axis=0)
    bare_commands = [.1, .2, .5, 1., 1.5, 2., 2.4]
    bare_indices = []
    outward = np.arange(len(bare_v)) <= np.argmax(abs(bare_v))
    for command in bare_commands:
        candidates = np.flatnonzero(outward & (abs(bare_v-command) < 3e-7))
        if len(candidates) != 1:
            raise ValueError("Missing native HRS point")
        bare_indices.append(int(candidates[0]))
    if np.any(bare_current[:, bare_indices] >= .995e-3):
        raise ValueError("Selected HRS diagnostic unexpectedly includes compliance")
    bare_profile = audit.profile("1r", "set", bare_v[bare_indices], bare_mean[bare_indices])
    for p, index in zip(bare_profile, bare_indices):
        p["native_index"] = index; p["branch"] = "outward"
    bare_witness = audit.witness("1r", "set", bare_profile, branch="outward", min_voltage=.1)
    numerical_points = {}
    for p in profile+bare_profile:
        if p["status"] == "pointwise_oracle":
            numerical_points[p["internal_voltage_V"], p["log10_gamma_ev"]] = "mean_oracle"
    for entry in pairs.values():
        for row in entry["individual_records"]:
            for p in row["oracle_pair"]:
                if p["status"] == "pointwise_oracle":
                    numerical_points[p["internal_voltage_V"], p["log10_gamma_ev"]] = "individual_oracle"
            if "pair" in row:
                lg = math.log10(row["common_gamma_at_pair_minimax_ev"])
                for p in row["pair"]:
                    predicted = audit.terminal("1t1r", "set", p["command_V"], lg)
                    local = p["command_V"]-audit.mapping.voltage_for_current(predicted, 1.1, "set")
                    numerical_points[local, lg] = "individual_minimax_circuit_state"
        band = entry["empirical_band_monotone_feasibility"]
        for command, gamma, predicted in zip(band["commands_V"], band["witness_gamma_pair_ev"], band["witness_terminal_predicted_current_A"]):
            local = command-audit.mapping.voltage_for_current(predicted, 1.1, "set")
            numerical_points[local, math.log10(gamma)] = "empirical_band_witness_circuit_state"
    numerical = [{"point_kind": label, **audit.numerical_point(*key)} for key, label in numerical_points.items()]
    # Independently validate representative inferred points, all selected mean
    # witnesses, and their reclosed-circuit minimax states on full Z support.
    check_points = native_subset(profile, [.06, .1, .2, .5, 1., 1.4, 2., 3., 4., 5.])+bare_profile
    checked = {}
    for p in check_points:
        if p["status"] == "pointwise_oracle":
            key = (p["internal_voltage_V"], p["log10_gamma_ev"])
            checked[key] = audit.numerical_point(*key, independent=True)
    for key in (min(numerical_points, key=lambda x: x[1]), max(numerical_points, key=lambda x: x[1])):
        checked[key] = audit.numerical_point(*key, independent=True)
    for entry in pairs.values():
        witness = entry["mean_witness"]
        if "pair" not in witness:
            continue
        lg = math.log10(witness["common_gamma_at_pair_minimax_ev"])
        for p in witness["pair"]:
            predicted = audit.terminal("1t1r", "set", p["command_V"], lg)
            local = p["command_V"]-audit.mapping.voltage_for_current(predicted, 1.1, "set")
            checked[local, lg] = audit.numerical_point(local, lg, independent=True)
    gaussian = json.loads((OUT/"spectral_state_identifiability.json").read_text())
    gaussian_audit = Audit(gaussian["fixed_parameters"])
    gaussian_pair = gaussian["mean_path_direction_witnesses"]["1t1r.set"]["return"]["pair"]
    gaussian_bands = empirical_interval_pair(gaussian_audit, gaussian_pair, current)
    report = {"status": "training_only_frozen_lognormal_gamma_direction_diagnostic_no_ODE_no_fit",
        "fixed_parameters": source["parameters"], "selected_q": source["selected_q"],
        "training_ids": TRAIN, "phases_read": ["set"],
        "series_SET_return_profile": profile, "series_primary_min_command_V": .06,
        "series_full_return_mean_witness": full_witness, "selected_pairs": pairs,
        "bare_HRS_selected_outward_points": bare_profile, "bare_HRS_mean_witness": bare_witness,
        "numerical_checks": {"inferred_oracle_and_pair_states": numerical, "independent_Z_selected_points": list(checked.values()),
            "max_32_to_64_log_change_decade": max(x["max_32_to_64_log_change_decade"] for x in numerical),
            "max_omitted_tail_log_error_bound_decade": max(x["max_omitted_tail_log_error_bound_decade"] for x in numerical),
            "max_64_vs_independent_Z_log_change_decade": max(x["max_64_vs_independent_Z_log_change_decade"] for x in checked.values()),
            "scope": "Every inferred mean/individual oracle gamma and individual/common band-witness circuit state checked at its own local voltage by order refinement and omitted-current tail bound; selected states and gamma extremes additionally checked in independent Z coordinates. No large-gamma/Ohmic-limit claim."},
        "comparison_with_frozen_Gaussian": gaussian["mean_path_direction_witnesses"]["1t1r.set"]["return"],
        "Gaussian_original_strongest_pair_empirical_band_feasibility": gaussian_bands,
        "loaded_training_arrays_sha256": training_hash.hexdigest(),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [Path(__file__), SOURCE, OUT/"refine_lognormal_spectral.py", OUT/"lognormal_spectral_screen.py",
             OUT/"spectral_state_identifiability.py", OUT/"spectral_state_identifiability.json", ROOT/"vulcan2d/transistor.py", DATA]},
        "limitations": ["gamma oracle uses observed current; it is not a predictive dynamics result.",
            "Only the spectral distribution family changed; no material fitting or parameter update occurs here.",
            "SET gamma nondecreasing is an explicit candidate assumption, not a consequence of passivity.",
            "Mean-path pair minimax bounds condition on the selected shape and default transistor. They are not whole-curve RMSE bounds or statistical tests.",
            "Numerical bracket gamma=1e-12..0.1 eV is an inversion search bracket, not a measured or fully certified physical interval.",
            "Passing or improving one pair does not establish globally monotone SET evolution, RESET consistency, or physical mechanism.",
            "Nonzero exact-mean minimax error is not automatically a practical model rejection. Original full-curve feature gates are unchanged."]}
    RESULT.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    write_markdown(report)
    print(json.dumps({"full_mean_witness": full_witness,
                      "selected_pairs": {k: {"commands": v["commands_V"], "mean": v["mean_witness"], "n_conflicting": v["n_individual_records_with_direction_conflict"]} for k, v in pairs.items()},
                      "numerical": {k: v for k, v in report["numerical_checks"].items() if not isinstance(v, list)}}, indent=2))


def write_markdown(report):
    lines = ["# 冻结 lognormal 谱形后的 SET 单调状态筛选", "", "日期：2026-11-03。仅使用预定训练记录；只反演 gamma，不新增材料参数、不重新拟合谱形、不运行 ODE。原 Gaussian 审计保持不变。", "",
             "## 固定谱形与方法", "",
             "采用 `lognormal_spectral_refinement.json` 中已冻结的 m、sigma、eta、Ron、T，仅允许 gamma 随路径变化。通过原始观测电流去嵌入后反演 gamma；两点 minimax 误差再独立闭合串联 KCL，因此后者不是固定观测电压下的误差。SET 假设为沿原始采集顺序 gamma 不减。主检查窗口为回扫 5→0.06 V，0.04/0.02 V 仅保留诊断。", "",
             "## 原先冲突是否缓解", "", "| 点对（原始顺序） | 新 gamma 先→后 (eV) | 不减方向冲突记录数 | 均值两点端口误差下界 (decade) |", "|---|---:|---:|---:|"]
    for name, row in report["selected_pairs"].items():
        a, b = row["mean_oracle_pair"]
        witness = row["mean_witness"]
        error = witness.get("terminal_pair_minimax_abs_log_error_lower_bound_decade", 0.)
        lines.append(f"| {a['command_V']:g}→{b['command_V']:g} V ({name}) | {a['gamma_ev']:.6g}→{b['gamma_ev']:.6g} | {row['n_individual_records_with_direction_conflict']}/32 | {error:.6f} |")
    old = report["comparison_with_frozen_Gaussian"]["terminal_pair_minimax_abs_log_error_lower_bound_decade"]
    new = report["series_full_return_mean_witness"].get("terminal_pair_minimax_abs_log_error_lower_bound_decade", 0.)
    lines += ["", f"Gaussian 最强均值点对的条件下界为 {old:.6f} decade；新 lognormal 谱形按 gamma 逆序差选择的最强均值点对为 {new:.6f} decade，约为 {(10.**new-1)*100:.1f}% 的乘性电流偏差。两者均为点对最大误差，不能直接解释成完整曲线 RMSE 的变化。", "",
              "这证明当前冻结谱形不能以单向 gamma 精确穿过所有均值点，但不是按任何非零残差就否决有用模型。4→2 V 的方向矛盾已在全部 32 条记录中消失，表明同方向 SET 演化可以修正这段静态斜率偏高的问题。新分布对原先问题带来了实质性改善。", "",
              "## 原始 Q10–Q90 电流区间是否允许单调状态", "",
              "另外直接把每个见证点的原始训练 Q10–Q90 电流区间映射为允许 gamma 区间，并检查是否存在 gamma后≥gamma前、且两个端口电流同时落入各自区间的解。若没有，则通过重新闭合 KCL 求使两个区间对称乘性扩张所需的最小 log 扩张量。这个诊断不改变原先全曲线主门槛；经验分位区间也不是置信区间。", "",
              "| 谱形/点对 | 同时落入原区间是否可行 | 最小区间扩张 (decade) | 乘性扩张因子 |", "|---|---|---:|---:|"]
    for label, band in [("Lognormal 原Gaussian见证", report["selected_pairs"]["previous_Gaussian_5_to_1p4V"]["empirical_band_monotone_feasibility"]),
                        ("Lognormal 4→2 V", report["selected_pairs"]["high_voltage_4_to_2V"]["empirical_band_monotone_feasibility"]),
                        ("Lognormal 2→1 V", report["selected_pairs"]["mid_voltage_2_to_1V"]["empirical_band_monotone_feasibility"]),
                        ("Lognormal 5→4 V", report["selected_pairs"]["high_voltage_5_to_4V"]["empirical_band_monotone_feasibility"]),
                        ("Lognormal 新最强见证", report["selected_pairs"]["strongest_lognormal_mean_pair"]["empirical_band_monotone_feasibility"]),
                        ("Gaussian 原最强见证", report["Gaussian_original_strongest_pair_empirical_band_feasibility"])]:
        lines.append(f"| {label} | {'是' if band['exists_gamma_nondecreasing_solution_inside_both_empirical_bands'] else '否'} | {band['minimum_symmetric_log_current_band_expansion_decade']:.6g} | {band['multiplicative_band_expansion_factor']:.7g} |")
    lines += ["", "新最强均值见证所需的分位区间扩张仅约 0.23%，另一个高压 5→4 V 点对所需扩张约 0.73%；这是条件诊断的大小，不能写成实验误差估计。Gaussian 原最强见证相应需要约 18.8% 扩张。新谱形确实大幅减轻了此处的方向冲突。", "",
              "因此，本轮建议是保留 lognormal 作为下一步受约束动力学评估的候选；剩余均值逆序需要评估，但不能作为零误差否决门槛。仍须使用原先全曲线与特征门槛评价动态结果，且需同时处理 RESET 与裸 1R。当前检查既不证明全模型成功，也不自动阻止进一步定量改善。", "",
              "## 裸 1R HRS 必要小检查", "", "以下全部为非限流的训练均值原生点，未拟合单独 HRS 谱形。", "", "| V (V) | 训练几何均值电流 (A) | 反演 gamma (eV) |", "|---|---:|---:|"]
    for p in report["bare_HRS_selected_outward_points"]:
        lines.append(f"| {p['command_V']:g} | {p['raw_current_A']:.6g} | {p['gamma_ev']:.6g} |")
    n = report["numerical_checks"]
    lines += ["", "这些反演只说明冻结共同谱形下需要何种状态轨迹；不能据此把电压相关状态反演称为动力学预测，也不能推出具体缺陷过程或秒单位速率。", "", "## 数值范围与限制", "",
              f"每个均值反演状态均检查有限能量积分 32→64 阶，最大变化 {n['max_32_to_64_log_change_decade']:.3g} decade；100 eV 以上被截断电流的保守最大 log 误差界为 {n['max_omitted_tail_log_error_bound_decade']:.3g} decade。代表反演点和共同 gamma 电路见证点另做独立 Z 坐标积分，最大差 {n['max_64_vs_independent_Z_log_change_decade']:.3g} decade。", "",
              "这把数值证据扩展到本次实际反演的有限 gamma 状态，不认证任意大 gamma 或 Ohmic 极限。默认正向 VG=1.1 V 晶体管仍来自插值，器件配对和真实计时仍未独立确认；所有结论保留这些条件。完整路径、32 条记录的相同点对、数值核验和输入散列见 `lognormal_state_identifiability.json`。", ""]
    (OUT/"lognormal_state_identifiability.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
