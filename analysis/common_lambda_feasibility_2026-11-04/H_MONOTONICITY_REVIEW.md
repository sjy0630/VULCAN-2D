# 共同乘积根单调性：定向独立复核

**180/180 项保存算术检查通过；0 项失败。**

仅核对 H_MONOTONICITY_CHECK.json 与本阶段计划，以及前阶段保存的四配置、16 个 SET4/RESET4 外扫窗口。实际解码 NPZ 的 local_voltage_v、weights_path_v、F 共 48 个唯一成员；不解码其他窗口数组。没有新 λ 求值、标量根、材料/电路调用、反演、ODE 或拟合。

独立重算正权重、A、加权电压均值、支持端点、R0/S/T、支持间隔和导数界；全部四配置满足 P 支 y<0、R_P0>S_P>T_P>0，N 支 y_H>y_L>0、R_N0<S_N<T_N，以及两组支持严格分离和 μ_NH0−v_NL,max>0。

| 配置 | Cplus / V | Cminus / V | H′ 上界 / V |
|---|---:|---:|---:|
| linear/Gauss8 | 3.6044529019 | 31.9762451120 | -28.3717922101 |
| linear/Gauss16 | 3.6056684880 | 31.9494170830 | -28.3437485950 |
| pchip/Gauss8 | 3.6037858561 | 32.0096465468 | -28.4058606907 |
| pchip/Gauss16 | 3.6052627158 | 31.9880736240 | -28.3828109082 |

导数符号复核：P 支 log(κr)=常数−log A_PL+log(R_P−T_P)−2log(R_P−S_P)；N 支 log(αβ)=常数−log A_NL+log(T_N−R_N)−2log(R_N−S_N)。相减得到计划中的 H′。
舍去 H′ 的两个非正项 −μ_NL 与 −R_P′/(R_P−T_P)，P 的剩余项不超过 Cplus。N 项使用 μ_NH 随 λ 非减、μ_NL≤v_NL,max，以及 min[1/(D−x)+2/x]=(1+√2)²/D，得到下界 Cminus。因此 H′≤Cplus−Cminus<0。
在两个有限 N 边界，κr 连续、有限且正；R_N→S_N+ 时 αβ→∞，R_N→T_N− 时 αβ→0。故每个指定有限正和问题在完整 RESET 正系数域内，确有且仅有一个 H=0 根。
该存在唯一性结论只属于固定读出、固定重建轨迹、固定 λ0 节点及共同路径时钟。它不保证 b∈[0,ln10]，不保证其余窗口闭合或连续前向曲线准确，也不是未知实验轨迹、噪声或读出误差的置信证书。普通双精度保存算术复核不是形式化区间算术证明。既定31点诊断算法保持不变。

实际读取的4个输入文件 SHA256：

- `/Users/sujiangyu/Desktop/SIM2RRAM/VULCAN-2D/analysis/common_lambda_feasibility_2026-11-04/H_MONOTONICITY_CHECK.json`：`5c9b9ecacf06afa343ffc754a78b345b24ce01d33c2c9ac886e3750873fd6e9e`
- `/Users/sujiangyu/Desktop/SIM2RRAM/VULCAN-2D/analysis/common_lambda_feasibility_2026-11-04/H_MONOTONICITY_PLAN.md`：`d098cb742d7bae6fee3772174b0353f3db7e919d34253804667baf2feb7d6f86`
- `/Users/sujiangyu/Desktop/SIM2RRAM/VULCAN-2D/analysis/empirical_integral_2026-11-04/empirical_integral_diagnostic.json`：`b491cca8e7b0524c57f9093cb814bc2cbb96ab5d0251ee2d4189947d8516e9ba`
- `/Users/sujiangyu/Desktop/SIM2RRAM/VULCAN-2D/analysis/empirical_integral_2026-11-04/empirical_integral_arrays.npz`：`a4e00c5292c1c802105481cacf3db80f417ad85a83351e2487f2421403dc7da9`

全部4个输入在复核末尾重验未变。仅新建本报告，不修改输入或冻结源码。
