"""Prepared-only fixed-m=2 cooperative integral initialization; no KCL or ODE.

Requires explicit parent authorization before invoking --run-authorized.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import signal
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INPUT = ROOT / "analysis/soft_drive_structure_2026-11-03/DRIVE_STRUCTURE_INVERSION.json"
INPUT_SHA = "f56c1405b503179233db1131675732a5abed76191806990020121778dafa9fe5"
PLAN_MD = HERE / "COOPERATIVE_INITIALIZATION_PLAN.md"
LAMBDAS = np.arange(41, dtype=float) / 20.0
RANK_RCOND = 1e-12
QUADRATURE_REL_LIMIT = 1e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def nnls_two_columns(x, target):
    """Exact finite active-set enumeration for two nonnegative coefficients."""
    norms = np.linalg.norm(x, axis=0)
    if not np.all(np.isfinite(x)) or not np.all(norms > 0):
        raise ValueError("Nonfinite or zero design column; preserve failure")
    z = x / norms
    unconstrained, _, rank, sv = np.linalg.lstsq(z, target, rcond=RANK_RCOND)
    candidates = [("origin", np.zeros(2))]
    for j, name in enumerate(("alpha_axis", "beta_axis")):
        c = np.zeros(2)
        c[j] = max(float(np.dot(z[:, j], target) / np.dot(z[:, j], z[:, j])), 0.0)
        candidates.append((name, c))
    if np.all(unconstrained >= 0):
        candidates.append(("two_column_feasible", unconstrained))
    scored = [(float(np.mean((z @ c - target) ** 2)), order, name, c)
              for order, (name, c) in enumerate(candidates)]
    loss, _, active, scaled = min(scored, key=lambda item: (item[0], item[1]))
    coef = scaled / norms
    residual = x @ coef - target
    gradient = z.T @ residual
    kkt = [abs(float(gradient[j])) if scaled[j] > 0 else max(-float(gradient[j]), 0.0)
           for j in range(2)]
    finite_positive = bool(rank == 2 and np.all(coef > 0)
                           and np.all(np.isfinite(coef))
                           and math.isfinite(float(coef[0] / coef[1])))
    condition = float(sv[0] / sv[1]) if sv[1] > 0 else None
    return coef, {
        "mean_squared_average_rate_mismatch_per_v2": loss,
        "average_rate_mismatch_rms_per_v": math.sqrt(loss),
        "alpha_kappa_per_v2": float(coef[0]),
        "beta_kappa_over_J_per_a_v2": float(coef[1]),
        "J_positive_a": float(coef[0] / coef[1]) if finite_positive else None,
        "finite_positive_seed_mapping": finite_positive,
        "active_set": active,
        "numerical_rank": int(rank),
        "column_norms": norms.tolist(),
        "normalized_column_singular_values": sv.tolist(),
        "normalized_column_condition_number": condition,
        "normalized_column_inner_product": float(np.dot(z[:, 0], z[:, 1])),
        "normalized_KKT_max_violation": max(kkt),
        "unconstrained_scaled_coefficients": unconstrained.tolist(),
        "boundary_or_degenerate": not finite_positive,
    }


def integral_columns(points, pairs, ds, nodes):
    roots, weights = np.polynomial.legendre.leggauss(nodes)
    t = (roots + 1.0) / 2.0
    w = weights / 2.0
    a, b = pairs[:, 0], pairs[:, 1]
    v0, v1 = points["material_voltage_v"][a], points["material_voltage_v"][b]
    u0, u1 = points["u"][a], points["u"][b]
    log_i = np.log(points["current_a"])
    v = v0[:, None] + (v1 - v0)[:, None] * t
    u = u0[:, None] + (u1 - u0)[:, None] * t
    currents = np.exp(log_i[a, None] + (log_i[b] - log_i[a])[:, None] * t)
    c = ds * np.sum(w * v * currents * np.expm1(u), axis=1)
    e = ds[None, :] * np.sum(
        w[None, None, :] * v[None, :, :]
        * np.exp(LAMBDAS[:, None, None] * v[None, :, :] - u[None, :, :]), axis=2)
    if not (np.all(np.isfinite(c)) and np.all(c > 0)
            and np.all(np.isfinite(e)) and np.all(e > 0)):
        raise ValueError("Invalid integral columns; no alternative interpolation")
    return e, c


def profile(e, c, ds, delta_u, interval_branch):
    target = delta_u / ds
    records, coefficients, residuals = [], [], []
    for k, lam in enumerate(LAMBDAS):
        x = np.column_stack((e[k] / ds, -c / ds))
        coef, row = nnls_two_columns(x, target)
        integrated = coef[0] * e[k] - coef[1] * c
        residual = integrated - delta_u
        row.update({
            "lambda_per_v": float(lam),
            "profile_index": k,
            "integrated_change_surrogate": integrated.tolist(),
            "integrated_residual": residual.tolist(),
            "integrated_residual_rms": float(np.sqrt(np.mean(residual ** 2))),
            "integrated_residual_max_absolute": float(np.max(np.abs(residual))),
            "integrated_residual_sum": float(np.sum(residual)),
            "branches": {},
        })
        for branch in ("outward", "return"):
            mask = interval_branch == branch
            row["branches"][branch] = {
                "intervals": int(np.sum(mask)),
                "average_rate_mismatch_rms_per_v": float(np.sqrt(np.mean((residual[mask] / ds[mask]) ** 2))),
                "integrated_residual_rms": float(np.sqrt(np.mean(residual[mask] ** 2))),
                "integrated_residual_max_absolute": float(np.max(np.abs(residual[mask]))),
                "integrated_residual_sum": float(np.sum(residual[mask])),
            }
        records.append(row)
        coefficients.append(coef)
        residuals.append(residual)
    return records, np.asarray(coefficients), np.asarray(residuals)


def band_winners(records):
    bands = (("low", 0.0, 2.0 / 3.0), ("middle", 2.0 / 3.0, 4.0 / 3.0), ("high", 4.0 / 3.0, 2.0))
    choices = []
    for name, lo, hi in bands:
        eligible = [r for r in records if r["lambda_per_v"] >= lo
                    and (r["lambda_per_v"] < hi or (name == "high" and r["lambda_per_v"] <= hi))]
        winner = min(eligible, key=lambda r: (r["mean_squared_average_rate_mismatch_per_v2"], r["lambda_per_v"]))
        choices.append((name, winner))
    return choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-authorized", action="store_true")
    args = parser.parse_args()
    if not args.run_authorized:
        raise SystemExit("Prepared only. Root must authorize before --run-authorized execution.")
    outputs = {ext: HERE / f"cooperative_initialization.{ext}"
               for ext in ("plan.json", "json", "npz", "md")}
    if any(p.exists() for p in outputs.values()):
        raise FileExistsError("Preserve prior/partial artifacts; no automatic rerun or overwrite")
    raw = INPUT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA:
        raise ValueError("Input SHA differs from frozen 26-point source")
    source = json.loads(raw)
    observations = source["observations"]
    if len(observations) != 26 or any(o["status"] != "inverted" for o in observations):
        raise ValueError("Expected exactly 26 previously inverted observations")
    fields = ("native_index", "command_v", "material_voltage_v", "current_a", "u", "gamma_ev")
    points = {key: np.asarray([o[key] for o in observations], dtype=float) for key in fields}
    branches = np.asarray([o["branch"] for o in observations])
    if set(branches) != {"outward", "return"}:
        raise ValueError("Unexpected branch identifiers")
    for key in fields:
        if not np.all(np.isfinite(points[key])):
            raise ValueError(f"Nonfinite input field {key}")
    if not (np.all(points["current_a"] > 0) and np.all(points["u"] > 0)
            and np.all(points["material_voltage_v"] > 0)
            and np.all(points["material_voltage_v"] <= points["command_v"])):
        raise ValueError("Observed conditional state is outside declared passive positive domain")
    frozen = source["plan"]["frozen"]
    gamma_c = float(frozen["gamma_c_ev"])
    q = np.exp(-points["u"])
    if not np.allclose(q, points["gamma_ev"] / (points["gamma_ev"] + gamma_c), rtol=1e-11, atol=0.0):
        raise ValueError("Saved u and gamma state coordinates disagree")
    pairs, interval_branches = [], []
    branch_indices = {}
    for branch in ("outward", "return"):
        ids = np.flatnonzero(branches == branch)
        ids = ids[np.argsort(points["native_index"][ids])]
        if len(ids) != 13 or np.any(np.diff(points["native_index"][ids]) <= 0):
            raise ValueError("Expected 13 uniquely chronological observations per branch")
        delta_a = np.diff(points["command_v"][ids])
        if not np.all(delta_a > 0 if branch == "outward" else delta_a < 0):
            raise ValueError("Branch voltage direction does not match chronology")
        branch_indices[branch] = ids
        pairs.extend(zip(ids[:-1], ids[1:]))
        interval_branches.extend([branch] * 12)
    pairs = np.asarray(pairs, dtype=int)
    interval_branch = np.asarray(interval_branches)
    ds = np.abs(points["command_v"][pairs[:, 1]] - points["command_v"][pairs[:, 0]])
    delta_u = points["u"][pairs[:, 1]] - points["u"][pairs[:, 0]]
    if len(ds) != 24 or not np.all(ds > 0):
        raise ValueError("Expected exactly 24 nonzero branch-local intervals")
    peak_out = int(branch_indices["outward"][-1])
    peak_return = int(branch_indices["return"][0])
    if points["command_v"][peak_out] != 5.0 or points["command_v"][peak_return] != 5.0:
        raise ValueError("Expected separate 5 V branch endpoints")
    plan = {
        "status_at_declaration": "authorized_bounded_cooperative_initialization",
        "equation": "du/ds=alpha*v*exp(lambda*v-u)-beta*v*I*expm1(u), alpha=kappa, beta=kappa/J; fixed recovery exponent m=2",
        "path_clock": "s=cumulative absolute command-voltage distance inside each branch; no measured seconds",
        "observations": 26, "branch_local_intervals": 24,
        "fixed_recovery_exponent_m": 2, "fitted_parameter_count_in_followup_model": 3,
        "exponent_is_fitted": False,
        "q_equation": "dq/ds=kappa*v*((I/J)*(1-q)-exp(lambda*v)*q**2)",
        "changed_integral_only": "E=int(v*exp(lambda*v-u) ds); C unchanged",
        "method_reference": "analysis/reversible_family_2026-11-03/integral_initialization.py",
        "lambda_grid_per_v": LAMBDAS.tolist(),
        "lambda_range_scope": "Finite effective exploration domain, not microscopic or experimental bound; no expansion",
        "interpolation": "piecewise linear u, v, log I; surrogate path does not enforce intermediate KCL/readout",
        "quadrature_rules": [8, 16], "primary_rule": 16,
        "loss": "mean(((alpha*E-beta*C-delta_u)/delta_s)^2), 24 equal interval weights; not current RMSE",
        "regression": "82 two-column nonnegative linear regressions, finite active-set enumeration, no iterative optimizer",
        "rank_relative_cutoff": RANK_RCOND,
        "seed_rule": "each fixed low/middle/high lambda band overall minimum; if zero/degenerate no replacement; at most three",
        "seed_bands": "[0,2/3),[2/3,4/3),[4/3,2]",
        "quadrature_relative_limit_for_forward_readiness": QUADRATURE_REL_LIMIT,
        "compute_budget_s": 30,
        "budget_start": "after frozen input validation, source hashes and plan persistence",
        "new_inverse_problems": 0, "KCL_node_problems": 0, "new_ODE": 0,
        "nonlinear_parameter_fits": 0, "holdout_accesses": 0,
        "maximum_profile_regressions": 82,
        "maximum_lambda_dependent_quadrature_nodes": 23616,
        "peak_state_scope": "two observed peak endpoints kept separate for interpolation; actual forward must still connect continuously",
        "no_restart_or_extension": True,
    }
    outputs["plan.json"].write_text(encode(plan))
    hashes = {str(p): sha(p) for p in (INPUT, PLAN_MD, Path(__file__), outputs["plan.json"],
                                              ROOT / "analysis/reversible_family_2026-11-03/integral_initialization.py")}
    report = {"status": "running_cooperative_initialization", "plan": plan,
              "source_sha256": hashes, "fixed_material_parameters": {
                  "Ron_ohm": float(frozen["Ron_ohm"]), "gamma_c_ev": gamma_c,
                  "parallel_path_a_s": 0.0},
              "new_inverse_problems": 0, "KCL_node_problems": 0, "new_ODE": 0,
              "nonlinear_parameter_fits": 0, "holdout_accesses": 0,
              "seeds": [], "seed_band_decisions": []}
    started = time.monotonic()
    def timeout(*_):
        raise TimeoutError("Fixed 30-second integral initialization budget exhausted; no restart")
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 30.0)
    try:
        e8, c8 = integral_columns(points, pairs, ds, 8)
        e16, c16 = integral_columns(points, pairs, ds, 16)
        p8, coef8, residual8 = profile(e8, c8, ds, delta_u, interval_branch)
        p16, coef16, residual16 = profile(e16, c16, ds, delta_u, interval_branch)
        rel_e = np.abs(e16 - e8) / np.abs(e16)
        rel_c = np.abs(c16 - c8) / np.abs(c16)
        quad_ok = bool(max(float(np.max(rel_e)), float(np.max(rel_c))) <= QUADRATURE_REL_LIMIT)
        winners8 = dict(band_winners(p8))
        seen = set()
        for band, winner in band_winners(p16):
            k = winner["profile_index"]
            decision = {"band": band, "selected_profile_index": k,
                        "selected_lambda_per_v": winner["lambda_per_v"],
                        "coarse_best_lambda_per_v": winners8[band]["lambda_per_v"],
                        "coarse_best_same_profile": winners8[band]["profile_index"] == k,
                        "replacement_allowed": False,
                        "coarse_at_primary_profile": p8[k]}
            if winner["finite_positive_seed_mapping"]:
                key = (winner["lambda_per_v"], winner["alpha_kappa_per_v2"], winner["J_positive_a"])
                if key in seen:
                    decision["decision"] = "duplicate_not_replaced"
                else:
                    seen.add(key)
                    decision["decision"] = "finite_positive_conditional_initialization"
                    report["seeds"].append({
                        "seed_id": f"cooperative_{band}", "band": band, "bin": band, "profile_index": k,
                        "selection": "Lowest primary 16-point integral rate mismatch in fixed lambda bin; zero/degenerate bin winner not replaced",
                        "lambda_per_v": key[0], "kappa_per_v2": key[1], "J_positive_a": key[2],
                        "initial_u_at_command_0p2_v": float(points["u"][branch_indices["outward"][0]]),
                        "initial_q_at_command_0p2_v": float(q[branch_indices["outward"][0]]),
                        "average_rate_mismatch_rms_per_v": winner["average_rate_mismatch_rms_per_v"],
                        "ready_for_forward_initialization": quad_ok,
                        "fixed_recovery_exponent_m": 2,
                        "scope": "Fixed-m=2 conditional integral initializer, not an ODE trajectory or current prediction",
                    })
            else:
                decision["decision"] = "zero_boundary_or_degenerate_not_replaced"
            report["seed_band_decisions"].append(decision)
        report.update({
            "status": "completed_bounded_cooperative_initialization_no_ODE_no_KCL",
            "compute_elapsed_s": time.monotonic() - started,
            "profile_regressions": 82,
            "lambda_dependent_quadrature_nodes": 23616,
            "quadrature_check": {"max_relative_E_change": float(np.max(rel_e)),
                                 "max_relative_C_change": float(np.max(rel_c)),
                                 "within_declared_numerical_limit": quad_ok,
                                 "interpolation_error_not_checked": True},
            "profiles_gauss8": p8, "profiles_gauss16": p16,
            "unjoined_peak_observation": {
                "outward_u": float(points["u"][peak_out]),
                "return_u": float(points["u"][peak_return]),
                "difference_return_minus_outward": float(points["u"][peak_return] - points["u"][peak_out]),
                "interpretation": "Reported only; no peak-state reset or jump is authorized"},
            "interpretation": "All scores are fixed-spectrum conditional surrogate-path integral mismatches, never 500-point log-current RMSE",
        })
        arrays = {f"observed_{k}": v for k, v in points.items()}
        arrays.update(observed_branch=branches, observed_q=q, interval_pairs=pairs,
                      interval_branch=interval_branch, interval_delta_s_v=ds,
                      interval_delta_u=delta_u, lambda_per_v=LAMBDAS,
                      E_gauss8=e8, E_gauss16=e16, C_gauss8=c8, C_gauss16=c16,
                      coefficients_gauss8=coef8, coefficients_gauss16=coef16,
                      integrated_residuals_gauss8=residual8, integrated_residuals_gauss16=residual16,
                      E_relative_quadrature_difference=rel_e, C_relative_quadrature_difference=rel_c)
        np.savez_compressed(outputs["npz"], **arrays)
        lines = ["# 固定 m=2 协作恢复候选的积分初始化结果", "",
                 "这是固定 m=2、固定谱、固定晶体管下 26 个既有条件反演状态的初始化诊断；指数未拟合，没有 ODE、KCL、新反演或保留集访问。", "",
                 "24 个分支内区间的平均速率错配等权。下列分数单位 V⁻¹，不是电流 RMSE；两个峰值观测没有在此强行连接。", "",
                 "| λ 分段 | 最优 λ / V⁻¹ | κ / V⁻² | J+ / A | 平均速率错配 RMS / V⁻¹ | 状态 |",
                 "|---|---:|---:|---:|---:|---|"]
        for decision in report["seed_band_decisions"]:
            row = p16[decision["selected_profile_index"]]
            jtext = "无有限正映射" if row["J_positive_a"] is None else f'{row["J_positive_a"]:.9g}'
            lines.append(f'| {decision["band"]} | {row["lambda_per_v"]:.2f} | {row["alpha_kappa_per_v2"]:.9g} | {jtext} | {row["average_rate_mismatch_rms_per_v"]:.9g} | {decision["decision"]} |')
        lines += ["", f'Gauss 8→16：E 最大相对差 {np.max(rel_e):.6g}，C 最大相对差 {np.max(rel_c):.6g}；声明的数值检查通过：{quad_ok}。这不检验观测插值的真实性。', "",
                  "所有 41 点、零边界、设计矩阵条件数、KKT 残差、24 项积分残差和分支分量见 JSON/NPZ。最多三项初始化，零边界或退化最优不补位；范围/最优边界均不触发扩大扫描。", "",
                  "这些种子尚未通过连续电路动力学或原 500 点电流评分。中间插值状态一般不满足 KCL/材料读出，低回归残差不证明真实轨迹可达；大残差亦不能普遍排除固定 m=2 候选。任何后续 forward 或拟合须由 root 另行决定。", "",
                  f'状态：{report["status"]}；计算耗时 {report["compute_elapsed_s"]:.6f} s，82 次有限两系数回归，未作非线性拟合。']
        outputs["md"].write_text("\n".join(lines) + "\n")
    except Exception as exc:
        report.update(status="failed_bounded_cooperative_initialization_no_restart",
                      error_type=type(exc).__name__, error=str(exc),
                      compute_elapsed_s=time.monotonic() - started)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        outputs["json"].write_text(encode(report))
    print(encode({k: report[k] for k in ("status", "compute_elapsed_s", "quadrature_check", "seeds")}))


if __name__ == "__main__":
    main()
