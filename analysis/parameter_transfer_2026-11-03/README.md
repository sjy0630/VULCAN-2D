# 同方程训练分组诊断接口

> 来源更新：用户随后确认裸记录是同一器件的多次循环。本文的 `1r` 单对轨迹
> 接口只保留作历史条件诊断，不可作为完成多圈连续验证的接口。实际顺序已确认，
> 4/23是测过但未提供的循环；无任意初态重置的重复协议检查见 `../bare_history_2026-11-03/`。

`group_problem.py` 用于分别检查固定材料族在裸1R、1T1R训练组的可接受参数域。
这不是最终共享参数模型，分组可拟合也不等于参数可以迁移或机制已验证。
默认工厂固定原谱形/表、β=1、A_RESET=0及原电路，不增加cell或逐圈参数。

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("analysis/parameter_transfer_2026-11-03").resolve()))
from group_problem import make_problem

problem = make_problem("1t1r", mode="periodic")  # 或 "1r"
# q6 = [log10_gamma_c, log10_JS, log10_JR, log10_kS, log10_kR, u0]
# result = problem.evaluate(q6)
# residual = problem.residual(q6)
metadata = problem.checkpoint_metadata()
```

可用 `problem.from_joint_q(q7)` 提取原7维向量对应的6维坐标；`to_joint_q(q6)`
仅为旧接口补齐坐标，未选组初态固定0且不会模拟该组。
路径速率没有秒单位，两个独立校准值的差别不能直接当作材料差异。

评分完全调用原 `SpectralJointProblem.score_predictions`：只在临时兼容输入中
用未选组的既有训练观测填其空槽，使该块残差严格为0，随后立即丢弃。
这些占位不是模型预测，不会暴露在输出误差、特征、details、预测或导出文件中。
原训练加载器仍读取两个训练组，以保留冻结的Ron和源数据记录；不读取留出数组。

| 选组 | 原27,023维向量切片 | 轨迹策略 |
|---|---|---|
| 1r | `[0:5583]` | 一个prepared SET/RESET pair对原14条记录评分 |
| 1t1r | `[5583:27023]` | cycle4使用一次u0，连续传播至35；原三对周期检查失败则full32 |

保留原掩码及权重，包括bare SET的三个等权子组；没有把两支组内cost重新归一化。
在同一材料参数和对应初态下，两组cost之和等于原联合cost（浮点求和误差范围内）。

`compare_modes(q6)` 只比较选组；最终series候选仍须full32与periodic对照。
`export_training_evaluation(q6, path, mode="full")` 默认导出选组完整预测，
可用 `compare_modes=True` 附带对照。不会导出未选组占位或留出记录。

`validate_group_problem.py` 的10项检查已通过，结果见 `group_validation.json`：
原始joint与分组残差逐字节相同；单组模拟、原初态交接、periodic/full32、失败回退、
仅选组指标/导出及训练数组访问范围均已检查。该验证使用合成积分器；
本次真实ODE次数为0、优化次数为0，不构成任何组已可拟合的证据。
