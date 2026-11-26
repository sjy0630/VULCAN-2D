# 在手跨栅压数据：来源、可用约束与边界

2026-11-03。范围仅VG=0.9/1.3V；没有读取1.1V留出电流参与选择，没有拟合或新增物理机制。

## 可直接复用的数据

| gate | SET源文件 | RESET源文件 | 记录数 |
|---|---|---|---:|
|0.9V|`0.9V栅压.xlsx`|`0.9V栅压RESET.xlsx`|100+100|
|1.3V|`1.3V栅压.xlsx`|`1.3V栅压RESET.xlsx`|101+101|

四个xlsx当前都是本机云端占位。以下旧缓存已经包含足以进行低偏单曲线检验的观测字段，不必先恢复整个工作簿：

- `analysis/cross_polarity_physics_native_points.csv`：gate、record、phase、raw_index、原始端口电压/电流**幅值**；另外保存旧去嵌入/剔除字段。本轮仅取前述观测字段，重新用当前统一mapping求内部电压。
- `analysis/cross_polarity_physics_cycles.csv`：每条完整原扫描的零偏参考最大值、最大电压、来源配对标签；仅复用零偏参考，不把旧拟合指标当新数据。
- `analysis/cross_polarity_physics_eta_grid.csv`：已做过的相邻原生点插值和条件跨极性比值，可作历史参照，不能用它绕开配对未知。
- `analysis/cross_polarity_physics_summary.json`：源工作簿哈希、窗口与处理说明。
- 提取程序 `analysis/diagnose_cross_polarity_physics.py:70` 从两列读端口I/V；`:101` 将SET最终峰值之后的返回支及RESET首次峰值之前的初始支截取为低偏观测；没有用材料模型生成这些观测。

native缓存SET覆盖外加幅值0–0.60V，RESET覆盖0–0.45V；均保留原raw_index。每条SET有61个、RESET有46个缓存点，包括被旧guard排除的点。mean与up记录的原始观测重复，不能当成新增实验；本轮只读mean记录的观测列。

四份工作簿的归档哈希（本轮未重新读取占位文件验证）：

- 0.9 SET：`98d602294dd0448ae996a07c742890530b5a0e8b4ffc42662ba8262aa6b28dcd`
- 0.9 RESET：`3e794ee648326e88dd4daf4413610357d73462b442a2f077d6eb60c9323786e3`
- 1.3 SET：`be26ededc91f9e037f3d06cb40c743da162d6932ba3652d9beb799d26104f43e`
- 1.3 RESET：`38096c46c16c65f98449ffffaec9b212ecd2cbdeec653e56af4f46a088e94269`

## 哪些比较有依据

`diagnose_cross_polarity_physics.py:26–28` 区分：1.1V为已确认Si/Ri配对；0.9/1.3V仅为同号Sheet的探索性配对。`RESET_HISTORY_RESULTS_2026-09-21.md:153` 还明确，跨栅压同cell、采集顺序、扫描时间与SET—RESET间隔没有确认；0.9/1.3的RESET终点1.5V也不同于1.1的1.7V。

因此：

1. 单条SET或RESET内的记录顺序可以用；不要求其他记录或其他gate扫速相同。
2. 相同Sheet编号不能直接升级为SET尾→RESET头的同一状态。即使配对后来确认，低偏不自动保证冻结，也还需保留两次扫描之间状态松弛/等待的可能。
3. 各gate的群体曲线可以检验同一读出方程是否支持这些观测；不能把群体中位差异称为纯栅压因果作用，也不能直接要求不同器件具有相同初态。
4. 0.9/1.3数据已有历史模型探索及先验使用，属于额外数据约束，不是从未使用过的外部盲测集。

## 已执行的小型检验

程序：`cross_gate_lowbias_order.py`；逐记录结果：`cross_gate_lowbias_order.json`；汇总：`cross_gate_lowbias_order.md`。

固定本轮lognormal谱形和现行正负实测T映射。对每条曲线重新求
`Vh = sign(Vapp) * (|Vapp| − VT(|I|,VG))`，在名义内部|Vh|=0.10–0.30V窗口逐点反演 `I=F(Vh,gamma)`。窗口及T压降比例≤0.25、容量比例≤0.5、逆条件数≤1沿用旧低偏诊断；不按是否支持模型选择窗口。

按原raw_index递增，SET返回要求gamma不减，RESET初始要求gamma不增。分别考察原值和`[I/(1+p),I(1+p)]`的p=1%、5%敏感性；每个端点重新去嵌入，不固定旧Vh。对SET区间`[l_j,h_j]`，存在非减gamma序列的充要条件是每点`max_{k≤j}l_k≤h_j`；RESET相应为`l_j≤min_{k≤j}h_k`。这个区间排序计算不拟合速率或状态轨迹，也不使用秒级时间。

数值上仅以已有transport表提供反演初值，再调用精确谱积分精化、检查反演两侧夹逼，并抽查64阶积分。无法稳定反演的点记未知。数据floor分为“全部正电流”与“大于该完整扫描最大零偏参考的3倍”，分列输出，不删除低于floor的记录以改善总体数字。少于5个合格原生点记覆盖不足。

该检验能区分“单向状态方向已与低偏观测不相容”与“缺乏可判数据/仅有细小噪声违例”，提供超出1.1V单组的约束。它不能把通过叫成动力学验证，也不能把失败唯一归因于材料：固定谱形、T插值和测量控制仍属于条件。若1%/5%和floor处理消除原值微小违例，应如实报告敏感性，不宣称机制无解。
