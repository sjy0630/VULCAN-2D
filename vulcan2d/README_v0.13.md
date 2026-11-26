# VULCAN-2D v0.13：状态交接适用域与完整样本审查

**2026-10-29 增量修订（v0.13.1）：** 独立 1T 与 1T1R 两段现在共用
`measured_transistor_mapping()`；通过 `transistor_mapping=` 一次传入完整循环，
禁止正反向静默采用不同电流比例、访问电阻或栅压参考。默认电流曲线不变。
新主要评价范围按老师意见改为第 4～53 圈，前三圈保留原始记录；本页下文的
全 53 对审查、训练数和旧评分仍是历史结果，并未改名为新范围的重新拟合。
见[修复报告](../analysis/UNIFIED_TRANSISTOR_REPAIR_2026-10-29.md)。

本轮针对朱凯晨等 Nature h-BN 1T1M 器件的准静态仿真与物理解释。v0.13 保留 v0.12 的数值方程和标定参数，增加可执行的交接边界及全样本验证。它不引入新的微观机理或部分 SET 的预测律。

2026-09-20 完整运行已完成：84 项测试通过，53 对留一和后 18 对顺序验证在两模式下共 142 条记录均成功生成并评分。原 51 对/顺序验证的 80 个可比数值保留三位小数后与 v0.12 一致。全部 53 对同协议 RESET 汇总 RMSE 为 0.370 decade（旧 51 对为 0.293）；最大误差集中在 R1/R2/R3。详细数值、图和下一步见 [本轮进展报告](../analysis/V13_PROGRESS_2026-09-20.md)。

## 1. 为什么没有直接把两个状态设成相等

SET 和 RESET 分别采用

\[
I_S(v,x)=(1-x)P(v)+xC(v),\qquad
I_R(v,p)=(1-p)H(v)+pL(v).
\]

其中 P/C 是正向 pre/connected 包络，H/L 是反向 HRS/LRS 包络。两状态的归一化依据、极性和读出协议不同，不能由它们都落在 [0,1] 推出 p=x。

现有交接为

\[
A_{L,R}=\eta\frac{C(v_r)}{\sinh(\alpha_Lv_r)},\qquad v_r=0.1\,\mathrm V.
\]

`A_L,R` 是 A 单位的电流幅值，旧代码中的 `g_lrs` 名字不能使它变成 S 单位电导；小信号电导约为 `alpha_L*A_L,R`。η 是配对数据中的跨协议换算系数，不是迁移效率。

将 C 改成最终 I_S(v_r,x_end) 可以制造末态依赖，但它额外假定未连接分量也服从相同 η。在 x→0 时，默认参数下换算电流甚至会低于既有 RESET HRS 包络；借助 HRS cap 保持数值排序并不能验证这个假设。因此本轮不默认采用这一外推。

## 2. 可执行的充分 SET 条件

完整循环接口检查：

\[
1-x_{end}\leq10^{-3},\qquad
\delta_I=\frac{|I_S(v_r,x_{end})-C(v_r)|}{C(v_r)}\leq10^{-3}.
\]

另要求末态有限且在 [0,1]、v_r>0、参考处 C>P>0 且有限。容差是充分 SET 包络近似的实现要求，**不是从实验识别出的材料阈值，也不是总预测误差上限**。对有序线性包络，电流条件不会严于状态条件；保留它是为了报告采用完整包络的电流近似量。状态条件还避免两个包络接近时误把不完整状态当成饱和。

不满足饱和条件时抛出 `HandoffDomainError`，附带末态与参考电流诊断；非有限参数、非法状态或包络则抛出普通 `ValueError`。单独的 `run_autonomous_set()` 仍可用于 SET 诊断。人工将 SET 转变中心增加 100 V² 的测试仅用于检验域外行为，不是实验反例。

这个检查针对**模型生成的末态**，用于保证充分 SET 包络近似的数值自洽；它不能证明每条实测循环都达到相同物理状态，也不能自动预测异常 RESET 的发生。

充分 SET 下保持旧版电流和分段状态逐点不变。相同标定包络、相同 RESET 输入下，充分 SET 的不同路径仍给出相同 RESET；更细的历史依赖尚未建模。

## 3. 当前 API 与兼容性

```python
from vulcan2d import simulate_calibrated_cycle, HandoffDomainError

result = simulate_calibrated_cycle(set_voltage, reset_voltage, gate_voltage=1.1)
set_current = result["set"]["Is"]
reset_current = result["reset"]["Is"]
set_state = result["set"]["xs"]
reset_state = result["reset"]["pbs"]
handoff_check = result["handoff"]
```

包顶层 `simulate_calibrated_cycle` 和 `run_autonomous_full_cycle` 现在调用 M13。**新结果不再包含拼接的 `connected_fraction`**，避免把两段局部包络坐标当作统一物理态；电压和电流的完整数组 `Vs/Is` 仍保留。需要严格复现旧接口结构时，显式导入 `vulcan2d.model_v12`。

`same_protocol` 与 `cross_gate_prior` 两种 RESET 模式保留。显式覆盖 RESET 阈值/宽度时，来源标记为 `explicit_parameter_override`。NaN/无穷参数和非有限输出会被拒绝，避免旧的单调状态更新掩盖无效参数。

## 4. 全 53 对验证的口径

训练资格仍采用旧的包络可分性规则，51 对有效 RESET；**评分集合包含全部 53 对**。每个目标都先排除，再拟合：

| 留出目标 | SET 训练数 | RESET/配对训练数 |
| --- | ---: | ---: |
| 旧 51 对中的一个 | 52 | 50 |
| S2/R2 或 S3/R3 | 52 | 51 |

R2/R3 无须反演出有效隐状态即可直接比较预测电流。顺序划分仍是 35 SET、33 RESET 训练，测试后 18 对；R2/R3 不被包装成新增晚期样本。

同一套预测分别汇总旧 51 对、全 53 对和原拒绝 2 对。报告保留逐循环/逐支路误差、训练编号、失败状态和电流门槛。组内存在失败或无效预测时，不静默改报成功子集的汇总精度。

逐点误差使用实际采样点及 SET `1e-11 A`、RESET `1e-12 A` 的实测电流门槛。中位回线误差采用独立口径：把各支路绝对电流截到 `1e-13 A` 后，在对数域插值到从 |V|=0.1 V 开始、间隔 0.02 V 的网格，再比较预测/实测中位回线；它不等于逐循环 RMSE 的中位数。

零偏置点另行报告：固定按 `abs(V_app)<=1e-12 V` 分离，不依赖预测电流大小。静态模型在这些点给出零电流，而实测存在非零读数，无法直接计算有限的对数误差。因此报告各循环的零偏置点数及实测绝对电流中位数/最大值；它们不进入对数评分。非零电压且达到实测电流门槛的零/无效预测仍被标为失败，任何点的非有限预测也不得忽略。现有数据的这一显式口径对应旧评分中隐式跳过的零电流预测，但不把非零偏置预测失败当作正常筛选。

回顾性漂移诊断固定使用前 35 次训练得到的输运包络，再从各条实测 RESET 电流识别 E50/width。它只用于观察分布变化，绝不传给目标预测。两个模式、原电路映射与公式选择均经过现有数据开发，整个审查仍是内部评估。

PF 状态识别程序即使对旧筛选未通过的 R2/R3 给出数值，也不证明这些循环存在可解释的完整 RESET 转变。漂移对比继续单独报告旧有效集合，图中另行标识 R2/R3 的诊断点；电流评分则保留所有循环。

## 5. 复跑与产物

本机在 Python 3.13.12 下复现；依赖见 `requirements-research.txt`。使用包含这些依赖的 Python：

```bash
python -m unittest discover -s tests -v
python -m vulcan2d.validate_v13
python -m vulcan2d.plot_validation_v13
```

`validate_v13` 可用 `--output-dir` 指定输出目录；绘图入口通过 `--input-dir` 读取已保存的结果，不重新拟合。审查产物与旧版分开保存：

- `analysis/full_cycle_validation_audit_v13.json`：完整汇总、逐循环记录、来源及源码/输入 SHA256。
- `analysis/full_cycle_validation_audit_v13_cycles.csv`：可筛选的逐循环表。
- `analysis/full_cycle_validation_audit_v13_curves.npz`：原始与预测数组，避免绘图时重跑模型。
- `analysis/figures/21_v13_cycle_error_and_reset_drift.{png,pdf}`：循环误差和回顾性阈值诊断。
- `analysis/figures/22_v13_previously_excluded_pairs.{png,pdf}`：S2/R2、S3/R3 实测和预测。

逐目标 LOO 和顺序验证的拟合/预测失败均会逐条记录；若前 35 次共同校准失败，则晚期目标保留失败记录，回顾诊断说明缺少固定参照。成功报告中的组分数不包含被静默移除的失败预测。

## 6. 物理解释的交付范围

[方程与证据对应表](../analysis/V12_PHYSICS_EVIDENCE_2026-09-20.md) 列出了每个现行方程的数据来源、单位、解释和边界。v0.13 沿用这些数值律，并修正交接输出的语义。旧版 patch、层间通道和损伤叙述不能当作当前执行路径的微观实现。

后续最有价值的研究对照是：简化输运包络、用更简单的 SET 电压驱动替代曝光坐标、用独立 RESET 初态分布替代配对 η 交接。它们应分别在相同划分下重新拟合并评分，以判断现有复杂度的必要性。拟合胜出支持有效模型的选择，不能唯一证明离子物种。

本轮未将桌面 UI 切换到研究模型，也未增加脉冲/时间、热或任意器件外推要求。
