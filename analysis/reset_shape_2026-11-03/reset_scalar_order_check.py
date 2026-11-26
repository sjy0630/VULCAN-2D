"""One-pass, no-fit scalar RESET order audit on saved training-only arrays.

The observed negative-polarity 50 mV outward read is the sole order anchor.
No SET boundary, spectral subtraction, model fitting, or heldout arrays enter.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "mechanism_selection_2026-11-03/field_rupture_trial.npz"
IDS = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
DELTAS = [0.0, 0.05, 0.10, 0.20]


def main():
    with np.load(SOURCE, allow_pickle=False) as z:
        ids = z["record_ids"].copy()
        voltage = z["command_voltage_v"].copy()
        observed = z["observed_reset_abs_current_a"].copy()
    assert ids.tolist() == IDS
    assert observed.shape == (14, len(voltage))
    assert np.all(np.isfinite(voltage))
    apex = int(np.argmax(abs(voltage)))
    anchor = int(np.argmin(abs(abs(voltage[:apex + 1]) - 0.05)))
    assert abs(abs(voltage[anchor]) - 0.05) < 3e-7
    nodes = np.arange(anchor + 1, apex + 1)
    nodes = nodes[abs(voltage[nodes]) > 0.0]
    assert np.all(np.diff(abs(voltage[:apex + 1])) >= 0.0)
    assert np.all(np.isfinite(observed[:, np.r_[anchor, nodes]]))
    assert np.all(observed[:, np.r_[anchor, nodes]] > 0.0)

    plan = dict(
        input_saved_training_npz=str(SOURCE), training_labels=IDS,
        anchor="Same-polarity RESET outward native 50 mV sample",
        endpoints="Every later outward nonzero native voltage through first apex",
        relative_current_envelopes=DELTAS,
        envelope="Each original current independently ranges in [(1-delta) I, (1+delta) I]",
        witness="Two records have strictly disjoint current intervals in one order at 50 mV and strictly reversed order at a later common voltage",
        pair_selection="Exhaust all 91 unordered record pairs; no observed-data-dependent subset",
        strongest="Largest delta_contact=min((R_anchor-1)/(R_anchor+1), (R_later-1)/(R_later+1)) among nominal reversals",
        interpretation="Diagnostic sensitivity envelopes, not measured noise or confidence intervals; pair-node witnesses are dependent",
        constraints="No fit, ODE, SET inputs, background subtraction, residual floor, original heldout records or plots",
    )
    (HERE / "RESET_SCALAR_ORDER_PLAN.json").write_text(json.dumps(plan, indent=2) + "\n")
    witnesses = []
    evaluated = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ai, aj = float(observed[i, anchor]), float(observed[j, anchor])
            for node in nodes:
                bi, bj = float(observed[i, node]), float(observed[j, node])
                evaluated += 1
                if (ai - aj) * (bi - bj) >= 0.0:
                    continue
                lo, hi = (i, j) if ai < aj else (j, i)
                early_low, early_high = float(observed[lo, anchor]), float(observed[hi, anchor])
                late_low_record, late_high_record = float(observed[lo, node]), float(observed[hi, node])
                r0 = early_high / early_low
                r1 = late_low_record / late_high_record
                d0 = (r0 - 1.0) / (r0 + 1.0)
                d1 = (r1 - 1.0) / (r1 + 1.0)
                assert r0 > 1.0 and r1 > 1.0
                witnesses.append(dict(
                    initially_lower_record=int(ids[lo]), initially_higher_record=int(ids[hi]),
                    anchor_native_index=anchor, anchor_command_voltage_v=float(voltage[anchor]),
                    later_native_index=int(node), later_command_voltage_v=float(voltage[node]),
                    initial_lower_current_a=early_low, initial_higher_current_a=early_high,
                    later_current_of_initially_lower_record_a=late_low_record,
                    later_current_of_initially_higher_record_a=late_high_record,
                    anchor_current_ratio=r0, later_reversed_current_ratio=r1,
                    anchor_interval_contact_delta=d0, later_interval_contact_delta=d1,
                    joint_interval_contact_delta=min(d0, d1),
                ))
    rows = []
    for delta in DELTAS:
        qualifying = []
        for w in witnesses:
            early_gap = (1.0 - delta) * w["initial_higher_current_a"] - (1.0 + delta) * w["initial_lower_current_a"]
            later_gap = (1.0 - delta) * w["later_current_of_initially_lower_record_a"] - (1.0 + delta) * w["later_current_of_initially_higher_record_a"]
            strict = early_gap > 0.0 and later_gap > 0.0
            assert strict == (delta < w["joint_interval_contact_delta"])
            if strict:
                qualifying.append(dict(w, anchor_disjoint_interval_gap_a=early_gap, later_reversed_disjoint_interval_gap_a=later_gap))
        pairs = {tuple(sorted((w["initially_lower_record"], w["initially_higher_record"]))) for w in qualifying}
        affected_nodes = {w["later_native_index"] for w in qualifying}
        strongest = max(qualifying, key=lambda w: w["joint_interval_contact_delta"]) if qualifying else None
        if strongest:
            strongest = dict(strongest,
                anchor_lower_record_interval_a=[(1.0 - delta) * strongest["initial_lower_current_a"], (1.0 + delta) * strongest["initial_lower_current_a"]],
                anchor_higher_record_interval_a=[(1.0 - delta) * strongest["initial_higher_current_a"], (1.0 + delta) * strongest["initial_higher_current_a"]],
                later_initially_lower_record_interval_a=[(1.0 - delta) * strongest["later_current_of_initially_lower_record_a"], (1.0 + delta) * strongest["later_current_of_initially_lower_record_a"]],
                later_initially_higher_record_interval_a=[(1.0 - delta) * strongest["later_current_of_initially_higher_record_a"], (1.0 + delta) * strongest["later_current_of_initially_higher_record_a"]],
            )
        rows.append(dict(delta=delta, n_strict_reversal_pair_nodes=len(qualifying),
            n_distinct_record_pairs=len(pairs), n_distinct_later_nodes=len(affected_nodes),
            strongest_witness=strongest))
    digest = hashlib.sha256()
    for key, array in (("record_ids", ids), ("command_voltage_v", voltage), ("observed_reset_abs_current_a", observed)):
        digest.update(key.encode()); digest.update(array.tobytes())
    report = dict(status="completed_one_pass_no_fit_same_polarity_scalar_order_check", plan=plan,
        n_records=len(ids), n_record_pairs=91, n_later_nodes=len(nodes),
        n_pair_nodes_evaluated=evaluated, n_nominal_strict_reversals=len(witnesses),
        diagnostics=rows,
        selected_training_arrays_sha256=digest.hexdigest(),
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_holdout_arrays_accessed=False,
        interpretation=[
            "A common deterministic scalar continuous flow with unique solutions preserves state ordering under the same subsequent protocol.",
            "If initial and later current readouts are monotone in that scalar state with the same orientation, strict current order reversal is impossible; a shared additive background at a node cannot change this.",
            "This uses only same-polarity RESET currents and does not require SET-to-RESET boundary continuity or conductance equality across polarity.",
            "Pure state-dependent rupture rates do not by themselves escape scalar order preservation. State-dependent loss may alter peak positions while retaining fixed-node order.",
            "Counterexamples may arise from unequal protocols, stochastic dynamics, additional state, nonmonotone readout, or non-order-preserving jumps. These are alternatives, not identified mechanisms or a proof of a two-state minimum.",
            "Unknown timing/dwell are not assumed identical by the data: common protocol is a conditional model assumption. Finite sampling means the scalar order test concerns measured anchor and endpoint currents, not full continuous observations.",
            "Sensitivity deltas are independent observation envelopes, not confidence/noise estimates. Counts share records and nodes and are not independent experimental replication.",
        ])
    (HERE / "RESET_SCALAR_ORDER_CHECK.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# 同极性 RESET 的标量状态排序检查", "",
        "仅用上一阶段保存的 14 条训练 RESET。以各圈外扫 −50 mV 电流排序，对全部随后外扫原生非零电压节点逐对检查，不使用 SET 读态、不减背景、不拟合。", "",
        f"共 91 对记录 × {len(nodes)} 个后续节点 = {evaluated} 个配对节点；它们相互依赖，不能视为独立实验重复。", "",
        "| 原始电流相对敏感性窗 | 严格反序配对节点 | 不同记录对 | 不同后续节点 |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| ±{100 * row['delta']:.0f}% | {row['n_strict_reversal_pair_nodes']} | {row['n_distinct_record_pairs']} | {row['n_distinct_later_nodes']} |")
    strongest = rows[0]["strongest_witness"]
    if strongest:
        w = strongest
        lines += ["", "最强单个见证（按两端区间首次接触所需的相对宽度排序）：", "",
            f"- 在 |V|={abs(w['anchor_command_voltage_v']):.6g} V，记录 {w['initially_lower_record']} 为 {w['initial_lower_current_a'] * 1e3:.9g} mA，低于记录 {w['initially_higher_record']} 的 {w['initial_higher_current_a'] * 1e3:.9g} mA。",
            f"- 到 |V|={abs(w['later_command_voltage_v']):.6g} V，记录 {w['initially_lower_record']} 为 {w['later_current_of_initially_lower_record_a'] * 1e3:.9g} mA，高于记录 {w['initially_higher_record']} 的 {w['later_current_of_initially_higher_record_a'] * 1e3:.9g} mA。",
            f"- 该反序到 δ={w['joint_interval_contact_delta']:.9g} 才首次有一端区间接触；这是必要敏感性宽度，不是测得的噪声水平。各指定窗口的实际区间与间隙保存在 JSON。"]
    lines += ["", "共同确定性标量连续动力学若解唯一，且电流读出对该状态保持相同单调方向，就保留初始排序；把损失率改为依赖 a 本身也不能产生这种反序。共同加性背景在同节点只平移全部电流，不会改变排序。", "",
        "该结论以相同后续测量协议、合适的电流观测模型为条件。协议差异、随机性、额外状态、非单调读出或不保序的跳变均可能规避；当前检查不识别这些解释，也不证明必须引入第二个连续状态。±0/5/10/20% 只是预声明敏感性窗口，不是仪器误差或置信区间。"]
    (HERE / "RESET_SCALAR_ORDER_CHECK.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(status=report["status"], diagnostics=rows), indent=2))


if __name__ == "__main__":
    main()
