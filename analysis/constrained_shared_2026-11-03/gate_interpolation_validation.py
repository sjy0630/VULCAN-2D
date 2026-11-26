"""Standalone positive-transistor-only gate interpolation cross-validation.

No hBN measurements or fitted material parameters are opened. Production
lookup/mapping remains unchanged. Interpolation coordinates are compared by
leaving each interior measured gate out; an optional power is selected only
on these standalone folds, with a separate nested-LOO diagnostic.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vulcan2d.transistor import load_default_lookup

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "vulcan2d/data/fig2b_transistor_lookup.json"
FLOORS = (1e-11, 1e-10, 1e-9)
PRIMARY_FLOOR = 1e-10
POWER_BOUNDS = (0., 2.)
QUERY_VDS = np.array([.1, .5, 1., 2., 4., 5.])
REFERENCE_SCALE = .789  # comparison only; never enters p selection/objective
METHODS = {"linear_I": 1., "sqrt_I": .5, "log_I": 0., "pchip_I": None}


def summarize(x):
    value = np.asarray(x, float)
    return {"n": len(value), "min": float(value.min()), "q10": float(np.quantile(value, .1)),
            "median": float(np.median(value)), "q90": float(np.quantile(value, .9)),
            "max": float(value.max())}


def predict(curves, target_gate, query_vds, method, power=None):
    gates = np.array([c["gate_voltage_v"] for c in curves])
    if not gates[0] < target_gate < gates[-1]:
        raise ValueError("Only interior gate interpolation is allowed")
    vds = np.asarray(query_vds, float)
    for c in curves:
        if vds.min() < c["drain_voltage_v"][0] or vds.max() > c["drain_voltage_v"][-1]:
            raise ValueError("Drain-voltage extrapolation is forbidden")
    currents = np.array([np.interp(vds, c["drain_voltage_v"], c["drain_current_a"]) for c in curves])
    if method == "pchip_I":
        return np.asarray(PchipInterpolator(gates, currents, axis=0, extrapolate=False)(target_gate))
    right = int(np.searchsorted(gates, target_gate)); left = right-1
    weight = (target_gate-gates[left])/(gates[right]-gates[left])
    lower, upper = currents[left], currents[right]
    p = METHODS[method] if method in METHODS else float(power)
    result = np.zeros_like(vds)
    zero_origin = (lower == 0.) & (upper == 0.)
    nonzero = ~zero_origin
    if np.any((lower[nonzero] <= 0.) | (upper[nonzero] <= 0.)):
        raise ValueError("Power/log interpolation requires positive non-origin anchors; no artificial floor substituted")
    la, lb = np.log(lower[nonzero]), np.log(upper[nonzero])
    if abs(p) < 1e-6:
        # Stable p->0 generalized-mean limit; correction is negligible at p=0.
        log_prediction = (1.-weight)*la+weight*lb+.5*p*weight*(1.-weight)*(lb-la)**2
    else:
        log_prediction = np.logaddexp(np.log1p(-weight)+p*la, np.log(weight)+p*lb)/p
    result[nonzero] = np.exp(log_prediction)
    return result


def score(vds, observed, predicted, floor):
    eligible = (vds > 0.) & np.isfinite(observed) & (observed >= floor)
    invalid = eligible & (~np.isfinite(predicted) | (predicted <= 0.))
    if np.any(invalid) or not np.any(eligible):
        raise ValueError("Invalid prediction or empty standalone-transistor fold")
    error = np.log10(predicted[eligible]/observed[eligible])
    correction = observed[eligible]/predicted[eligible]
    return {"n_scored": int(eligible.sum()), "n_nonzero_below_floor": int(np.sum((vds > 0.) & ~eligible)),
            "rmse_decade": float(np.sqrt(np.mean(error**2))), "mse_decade2": float(np.mean(error**2)),
            "signed_mean_log_error_decade": float(error.mean()),
            "true_over_predicted_correction": summarize(correction),
            "n_correction_at_or_below_reference_scale": int(np.count_nonzero(correction <= REFERENCE_SCALE))}


def cross_validate(curves, method, floor, power=None, include_curves=False):
    folds = []
    for j in range(1, len(curves)-1):
        target = curves[j]; retained = curves[:j]+curves[j+1:]
        v, obs = target["drain_voltage_v"], target["drain_current_a"]
        pred = predict(retained, target["gate_voltage_v"], v, method, power)
        row = {"held_gate_V": target["gate_voltage_v"],
               "neighbor_gates_V": [curves[j-1]["gate_voltage_v"], curves[j+1]["gate_voltage_v"]],
               "scores": score(v, obs, pred, floor)}
        if include_curves:
            row.update(voltage_ds_V=v.tolist(), observed_current_A=obs.tolist(), predicted_current_A=pred.tolist())
        folds.append(row)
    return {"gate_equal_mse_decade2": float(np.mean([r["scores"]["mse_decade2"] for r in folds])),
            "gate_equal_rmse_decade": float(np.sqrt(np.mean([r["scores"]["mse_decade2"] for r in folds]))),
            "folds": folds}


def optimize_power(curves, floor):
    answer = minimize_scalar(lambda p: cross_validate(curves, "power_cv", floor, p)["gate_equal_mse_decade2"],
                             bounds=POWER_BOUNDS, method="bounded", options={"xatol": 1e-9, "maxiter": 100})
    if not answer.success:
        raise RuntimeError("Standalone power-coordinate optimization did not converge")
    return {"power": float(answer.x), "gate_equal_rmse_decade": float(np.sqrt(answer.fun)),
            "n_objective_evaluations": int(answer.nfev), "bounded_interval": list(POWER_BOUNDS)}


def main():
    payload = json.loads(SOURCE.read_text())
    lookup = load_default_lookup()
    curves = list(lookup.curves())
    if [c["gate_voltage_v"] for c in curves] != [.5, 1., 1.5, 2., 2.5, 3.]:
        raise ValueError("Unexpected source gates")
    evaluations = {}
    for floor in FLOORS:
        methods = {method: cross_validate(curves, method, floor, include_curves=floor == PRIMARY_FLOOR)
                   for method in METHODS}
        optimized = optimize_power(curves, floor)
        methods["power_cv"] = cross_validate(curves, "power_cv", floor, optimized["power"],
                                             include_curves=floor == PRIMARY_FLOOR)
        evaluations[str(floor)] = {"methods": methods, "optimized_power": optimized}
    primary = evaluations[str(PRIMARY_FLOOR)]
    p = primary["optimized_power"]["power"]

    # The selected p's CV score reuses folds for hyperparameter selection.
    # This nested check excludes each outer gate entirely from p selection.
    nested = []
    for j in range(1, len(curves)-1):
        outer = curves[j]; retained = curves[:j]+curves[j+1:]
        inner = optimize_power(retained, PRIMARY_FLOOR)
        prediction = predict(retained, outer["gate_voltage_v"], outer["drain_voltage_v"], "power_cv", inner["power"])
        nested.append({"outer_held_gate_V": outer["gate_voltage_v"],
            "inner_available_gates_V": [c["gate_voltage_v"] for c in retained],
            "inner_selected_power": inner["power"],
            "inner_cv_rmse_decade": inner["gate_equal_rmse_decade"],
            "outer_scores": score(outer["drain_voltage_v"], outer["drain_current_a"], prediction, PRIMARY_FLOOR)})

    # Curves and requested voltages at1.1V: this is interpolation output, not truth.
    grid = np.unique(np.concatenate([c["drain_voltage_v"] for c in curves]))
    grid = grid[(grid >= lookup.vds_min) & (grid <= lookup.vds_max)]
    default_grid = predict(curves, 1.1, grid, "linear_I")
    default_points = predict(curves, 1.1, QUERY_VDS, "linear_I")
    np.testing.assert_allclose(default_points, [lookup.current(v, 1.1) for v in QUERY_VDS], rtol=3e-15, atol=0.)
    interpolated = {}
    for method in [*METHODS, "power_cv"]:
        values = predict(curves, 1.1, grid, method, p)
        points = predict(curves, 1.1, QUERY_VDS, method, p)
        positive = grid > 0.
        ratios = values[positive]/default_grid[positive]
        interpolated[method] = {"gate_voltage_V": 1.1, "power": p if method == "power_cv" else METHODS.get(method),
            "query_Vds_V": QUERY_VDS.tolist(), "query_current_A": points.tolist(),
            "query_ratio_to_default_linear_I": (points/default_points).tolist(),
            "full_voltage_ds_V": grid.tolist(), "full_current_A": values.tolist(),
            "full_ratio_to_default_nonzero_voltage": ratios.tolist(),
            "ratio_summary_nonzero_voltage": summarize(ratios),
            "n_ratio_at_or_below_reference_scale": int(np.count_nonzero(ratios <= REFERENCE_SCALE))}

    lower_gate = {}
    for method, detail in primary["methods"].items():
        first = next(f for f in detail["folds"] if f["held_gate_V"] == 1.)
        others = [f for f in detail["folds"] if f["held_gate_V"] > 1.]
        lower_gate[method] = {"held_1p0V": first["scores"],
            "other_three_gates_equal_rmse_decade": float(np.sqrt(np.mean([f["scores"]["mse_decade2"] for f in others])))}
    result = {
        "status": "standalone_positive_transistor_only_interpolation_validation_no_mapping_change",
        "opened_measurement_file": str(SOURCE.relative_to(ROOT)),
        "hbn_measurements_or_fits_opened": False,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "transistor_implementation_sha256": hashlib.sha256((ROOT/"vulcan2d/transistor.py").read_bytes()).hexdigest(),
        "source_metadata": {key: value for key, value in payload.items() if key != "curves"},
        "preprocessing": "Existing source convention, independently per measured gate: average up/down sweeps, physical zero origin, nonnegative monotone cumulative maximum. CV validates interpolation of this processed measurement law, not raw-point noise.",
        "measured_gate_voltages_V": [c["gate_voltage_v"] for c in curves],
        "held_interior_gates_V": [1., 1.5, 2., 2.5],
        "methods": {"linear_I": "Adjacent retained gates linear in current (p=1)",
            "sqrt_I": "Adjacent retained gates linear in sqrt(current), then square (p=0.5)",
            "log_I": "Adjacent retained gates linear in log current, then exponentiate (p=0 limit)",
            "pchip_I": "PCHIP in gate/current coordinates using all retained measured gates for local shape-preserving derivatives",
            "power_cv": "One global generalized-mean exponent p optimized only on standalone-transistor held-gate CV; no hBN objective"},
        "scoring": "Mean log-current squared error within each held gate at nonzeroVds, then equal mean over gates; current floors only select observed target points, never replace anchor currents.",
        "primary_floor_A": PRIMARY_FLOOR,
        "floor_sensitivity": evaluations,
        "nested_power_validation": {"scope": "Each outer held gate is excluded from every inner anchor and inner target used to choose p",
            "folds": nested, "outer_gate_equal_rmse_decade": float(np.sqrt(np.mean([r["outer_scores"]["mse_decade2"] for r in nested])))},
        "lower_gate_fold_distinction": lower_gate,
        "interpolated_VG1p1": interpolated,
        "reference_scale_discussion": {"reference_scale": REFERENCE_SCALE, "fractional_decrease": 1.-REFERENCE_SCALE,
            "role": "Externally supplied21.1% current-scale sensitivity reference; comparison only, not a fitted target or uncertainty assumption",
            "evidence": ["Default linear interpolation overpredicts the held1.0V curve strongly; its correction ratios can reach or exceed a21.1% reduction at someVds.",
                "That fold bridges0.5V and1.5V (1.0V gate gap), whereas the actual1.1V query bridges1.0V and1.5V (0.5V gap). Those errors cannot be transferred as a confidence band.",
                "The tested data-supported coordinate changes at1.1V produce the ratios recorded here; none reaches0.789 on the nonzero nativeVds grid.",
                "No direct1.1V truth is available. This cannot prove that a cell change is required, or distinguish interpolation error from unmeasured standalone-to-series transistor differences. It does not establish±21.1% calibration uncertainty." ]},
        "limitations": ["Four interior gate curves from the same source are not independent device replications.",
            "The globalp CV score is a hyperparameter-selection score; nestedLOO is supplied separately, still on a small sparse-gate dataset.",
            "No true measured1.1V positive output curve exists in this source, so1.1V ratios are method sensitivity, not predictive error bars.",
            "Endpoint gate extrapolation, reverse-bias calibration and shared-cell identity are outside this audit.",
            "Production transistor law and hBN material parameters remain unchanged."],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT/"gate_interpolation_validation.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    lines = ["# 仅用独立1T数据核验栅压插值", "",
        "只读取Nature Fig.2b正向独立晶体管查表，未读取hBN数据、目标器件拟合或修改生产映射。实测栅压为0.5、1.0、1.5、2.0、2.5、3.0V。逐次留出四个内部栅压，按各栅压等权的log电流MSE评分。", "",
        "## 留栅压结果", "", "| 方法 | 四折RMSE(decade) | 留出1.0V | 其余三折 |", "|---|---:|---:|---:|"]
    for method, detail in primary["methods"].items():
        lower = lower_gate[method]
        lines.append(f"| {method} | {detail['gate_equal_rmse_decade']:.6f} | {lower['held_1p0V']['rmse_decade']:.6f} | {lower['other_three_gates_equal_rmse_decade']:.6f} |")
    lines += ["", f"全局p只由独立1T交叉验证选出：p={p:.6f}。该优化后的CV分数不能当作未参与调参的测试分数；进一步把外层留出栅压彻底排除后，嵌套LOO的四折RMSE为{result['nested_power_validation']['outer_gate_equal_rmse_decade']:.6f} decade。每个外层折的p与结果见JSON。", "",
        "电流floor为1e−11/1e−10/1e−9A的三组结果一致：所有非零电压目标点都高于这些floor，故这里没有真正考验弱信号检测阈值。零电压不计入log分数，原点保持0。", "",
        "## VG=1.1V各方法相对默认linear-I的电流比", "",
        "| 方法 | Vds0.1V | 0.5V | 1V | 2V | 4V | 5V |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method, values in interpolated.items():
        lines.append("| "+method+" | "+" | ".join(f"{v:.6f}" for v in values["query_ratio_to_default_linear_I"])+" |")
    lines += ["", "这张表是插值方法差异，**不是1.1V实测误差或置信区间**。完整101个原生电压位置的电流曲线、各折观测/预测数组保存在JSON。", "",
        "## 21.1%尺度变化能否归因于插值", "",
        "低栅压1.0V折确实显示默认插值可能产生约20%甚至更大的过预测。但该折是跨0.5→1.5V的1V宽缺口；实际1.1V查询位于已测1.0→1.5V的0.5V宽间隔，不能直接照搬误差。",
        "本次测试的linear/sqrt/log/PCHIP及1T-CV选择的p，在1.1V的整个非零Vds网格均没有把电流降至默认值的0.789。因而不能仅凭这些插值替代方案，把所需21.1%缩放宣称为已获实测支持的校准修正。",
        "同时，没有1.1V真值就无法证明该缩放必然来自器件变化，也不能排除更复杂的插值误差。要区分插值、独立晶体管与阵列晶体管差异，需要对应的1.1V实测/器件映射证据；当前只能做有边界的假设敏感性分析。", "",
        "预处理沿用原查表：同一栅压正反扫描取均值、原点归零并单调化，逐栅压独立执行。这里检验的是该处理后测量定律的栅压插值，不是原始噪声或反向晶体管模型。"]
    (OUT/"gate_interpolation_validation.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"cv_rmse": {k: v["gate_equal_rmse_decade"] for k, v in primary["methods"].items()},
        "global_power": p, "nested_rmse": result["nested_power_validation"]["outer_gate_equal_rmse_decade"],
        "VG1p1_ratios": {k: v["query_ratio_to_default_linear_I"] for k, v in interpolated.items()}}, indent=2))


if __name__ == "__main__":
    main()
