# 条件协议时钟适配器

`protocol_problem.py` 提供 `make_problem(alpha, integrator_options=None, mode='periodic')`。
它只增加一个外部固定的协议情景：`alpha = r_series / r_bare`。
以累计绝对指令电压路径 `s` 为坐标，裸 1R 使用 `du/ds = H`，1T1R 使用
`du/ds = H/alpha`，SET 与 RESET 使用同一个系数。

**alpha 不是已测扫速，不是材料拟合参数；没有引入物理秒、等待时间或逐圈时钟。**
两种接法共用同一个 `SpectralLaw` / `TabulatedSpectralLaw` 基础材料对象，
冻结谱形与传输表不变，`A_RESET = 0`。电流、逆电流、阈值、状态边界、限流与
KCL 仍调用原实现。`alpha=1` 直接注入原积分器，不经过导数包装。

## 使用

在仓库根目录运行：

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("analysis/conditional_clock_2026-11-03").resolve()))
from protocol_problem import make_problem

problem = make_problem(alpha=25.0, mode="periodic")
metadata = problem.checkpoint_metadata()
# 本例只构建问题，不执行拟合或全 32 圈积分。
```

`q` 保持 7 维及原顺序：

1. `log10_gamma_c_ev`
2. `log10_J_SET_A`
3. `log10_J_RESET_A`
4. `log10_k_SET_per_V`
5. `log10_k_RESET_per_V`
6. `initial_u_1r`
7. `initial_u_1t1r_cycle4`

随后可由已批准的调用方使用继承的 `evaluate(q, mode='periodic')`、
`evaluate(q, mode='full')` 或 `export_training_evaluation(...)`。
适配器本身不执行这些操作。原始训练记录掩码、权重、32 圈连续传播、
仅在第 4 圈使用一次的初态，以及周期近似失败后的 full32 回退均保持原实现。
周期近似仍需按原约定对最终候选进行 full32 验证。

保存候选时应同时保存 `problem.checkpoint_metadata()`：其中包含 alpha、
两个路径导数系数、冻结谱形、表路径及源文件 SHA256。记录 alpha 情景
不能替代真实扫速测量，也不能修复单条 SET 返扫中冻结谱必要界的违例。

## 本次验证

运行 `validate_protocol_problem.py` 得到 `protocol_validation.json`，12 项检查通过。

- alpha=1：两种接法、两种极性的短谱轨迹数值数组逐字节一致；仅运行耗时不比较。
- 独立解析线性 ODE `du/ds = ±k u`：alpha=.5、1、4，验证 series 的
  `1/alpha` 系数、bare 的单位系数及两种极性；不是只测试包装器返回值。
- 验证相位状态交接、原训练掩码与初态索引、限流、电压分配、零状态边界及
  上界越域报错保持不变。
- 本次没有运行 full32、参数优化或留出集评估。

这验证的是时钟适配器，不构成 h-BN 机制或任何 alpha 数值的物理证据。
