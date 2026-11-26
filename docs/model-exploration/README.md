# VULCAN-2D 模型探索记录

更新于 2026-11-25。这里记录为 Zhu 等人的 Au–Ti/h-BN–W 忆阻器建立统一模型的历次尝试。目标是在裸 1M 和不同栅压的 1T1M 中共用 hBN 的电流方程、状态演化和物理参数。裸器件使用仪器限流，1T1M 使用独立的晶体管负载线。状态可以随曲线的写入历史而不同。 [Nature 原文](https://doi.org/10.1038/s41586-023-05973-1) · [器件和数据来源核对](../../analysis/NATURE_1R_1T1R_PROVENANCE_2026-11-03.md)

## 迭代记录

后续实验按[模型迭代协议](PROTOCOL.md)建档。复制[迭代模板](ITERATION_TEMPLATE.md)，完成后在表尾加一条记录。

| 阶段 | 主要内容 | 记录 |
|---|---|---|
| 早期器件模型，v0.2–v0.13 | 从缺陷、导电斑块和 soft/hard 状态，到实测晶体管分压、SET→RESET 状态交接和预测适用域 | [版本索引与关键图](#v02v013-器件模型) |
| 2026-09 至 11-05 | 输运机制、保留电荷与接触、随机多域；统一 1R/1T1R 的 GapLaw、谱输运、可逆捕获和 q/a 连续历史 | [逐版对照](../../analysis/MODEL_ATTEMPTS_COMPARISON_2026-11-05.md) · [48 篇原报告汇编](../../analysis/MODEL_ATTEMPTS_SOURCE_REPORTS_2026-11-05.md) · [原报告清单](../../analysis/MODEL_ATTEMPTS_SOURCES_2026-11-05.json) · [全部历史报告与图](history-index.md) |
| 2026-11-24 交接 | 独立晶体管负载、跨栅压留出基准、能量约束的物理候选 | [交接记录和复现入口](../../../VULCAN-2D-handoff-2026-11-24/docs/RESEARCH_HANDOFF_2026-11-24.md) |
| 2026-11-24 至 11-25 | 冻结迁移、soft 电导范围、RESET 去回程、IV41 与高导通态、SET 输运拆分、裸 1M/1T1M 联合间隙模型 | [逐项记录、完整误差和原图](2026-11-24-to-25.md) · [图集](figures.md) |
| 2026-11-25-01 | 方向诊断：确定性评分地板（RESET 仅波形约 0.77–1.18 decade），IV41 高导通态按行序整段切换，裸/1T1M 联合缺过冲与热 | [诊断与建议](iterations/2026-11-25-01-方向诊断/README.md) |
| 2026-11-25-02 | 数据配对勘误（`Fig.2c` 仅 2/53 同圈 vs `IV41` 严格连续配对）、平滑 1T 去嵌与前序 `|V_reset|` 耗尽标度的电热双通道统一模型：0.9 V SET 去程 `1.411→0.301`、1.4 V `0.979→0.361 decade`，裸 1M 开通 MAE `0.100 V`，零样本跑通同器件 `IV41 5.0 V` 与 15 颗阵列器件 | [完整记录、评分与四组对照图](iterations/2026-11-25-02-数据配对与电热双通道统一模型/README.md) |
| 2026-11-25-02 | 按 Origin 序号重建 IV41 的 704 对曲线。圈内状态传递改善总体 RESET；跨圈传递抬高了下一圈 SET 的低压电流 | [配对、评分和图](iterations/2026-11-25-02-IV41状态交接/README.md) |
| 2026-11-25-03 | 去鼓包陡跳 SET、跨圈交接与缩并几何热阻 $R_{\text{th}}(G)$ 闭环 RESET：消除裸 1M、1.1–1.5 V 与 5.0 V 开通前上拱鼓包并统一红线；裸 1M 去程降至 `1.012 decade`（2.65 V 垂直跳变），1.1/1.2/1.4/1.5 V 降至 `0.165/0.130/0.285/0.285 decade`，5.0 V 降至 `0.321 decade`，7 档栅压配对 RESET 全面改善 | [完整记录、评分与对比图](iterations/2026-11-25-03-去鼓包陡跳SET与几何热阻闭环RESET/README.md) |
| 2026-11-26-01 | **12 参数双向合一精简模型**（参数由 24 砍半至 12，正负偏压与零偏弛豫合并为单一奇对称方程）：裸 1M SET 去程降至 `0.996 decade`，`0.9 V` SET 去程由 `0.581` 骤降至 `0.173 decade`，`0.9–1.5 V` 全七档去程统一落在 `0.13–0.31 decade`，同时 `0.9/1.0/1.4/1.5 V` 及裸 1M 配对 RESET 进一步改善 | [完整记录、12参数清单与对比图](iterations/2026-11-26-01-双向合一精简模型/README.md) |
| 2026-11-26-02 | **12 参数统一模型 v4.1（回程对齐与平滑步长收敛）**：零新增参数下解耦普适量子电导底数 $e^2/h$ 与零偏弛豫门槛，`0.9–1.2 V` SET 全程降至 `0.168–0.189 decade`（去程 `0.115–0.137`、回程 `0.194–0.240`），裸 1M 降至 `0.988 decade`；平滑 `5.0 V` 折角（`0.328 decade`），全工况步长减半误差降至 `0.0002–0.0013 decade` | [完整记录与收敛审计图](iterations/2026-11-26-02-回程对齐与平滑步长收敛/README.md) |

## 现在能确认什么

- **两种接法的模型接口已经统一。** 同一个 hBN 电流与状态模型接入不同外电路。较好的单栅压或局部条件分数，仍对应各自的评分任务。 [联合对照](2026-11-24-to-25.md#裸-1m-与-1t1m-共同拟合)
- **裸 hBN 给了固定串联电阻一条硬约束。** 裸器件约在 2.5–2.8 V 达到 1 mA，而旧版的 23 kΩ 固定串联电阻，即使承担全部 2.75 V，也最多通过约 0.12 mA。联合拟合把阻值降下来后可以达到毫安级，但部分 1T1M 的 SET 仍开通太晚。 [裸器件曲线](evidence/bare_handoff_iv_comparison.png) · [只改电阻对照](evidence/joint_gap_set.json)
- **目前的主要分歧集中在 SET 开通前和开通段。** 最新的有效间隙宽度版本改善了部分 1T1M 曲线和裸器件达到限流的电压，裸器件开通前误差增加，0.9/1.4 V 的 1T1M 开通仍明显偏晚。 [最新曲线](evidence/joint_gap_width.png) · [全部分段评分](evidence/joint_gap_width.json)
- **RESET 的入口、去程和回程各有自己的证据。** 补足入口电导的表示范围后，整条 RESET 的预测仍受后续状态演化影响。实测终点能显著压低某些回程误差，这个结果用于定位瓶颈。 [RESET 记录](2026-11-24-to-25.md#reset-读出与擦除动力学)

早期关于前三圈与第 3→4 圈落差的机制讨论保留在原报告中，后续实验核对表明 1.1 V 启动的前三圈 SET 未成功。正式工作循环从第 4 圈开始。裸 1M 的编号按同一器件的真实循环顺序排列，4 和 23 已测但缺失。每份结果按实际协议注明是否跨圈传递状态。 [启动更正与历史口径](../../analysis/MODEL_ATTEMPTS_COMPARISON_2026-11-05.md)

本阶段结束时，q/a 分支作为历史对照保留。当前接手的是交接快照的模型，最新联合间隙版本保留为 SET 研究候选。此前讨论的“按相近低压电导匹配裸 1M 与 1T1M、比较开通前的本征曲线”记为待做分析。

## v0.2–v0.13 器件模型

下面是各版主要改动。每版 README 中保留其原始测量条件、分段误差和图名。右栏列出原图、历史复跑图和原始评分。体电荷、缺陷通道及导电路径等均按当时模型的物理解释记录。

| 版本 | 主要改动与证据 | 当前可打开的图或数据 |
|---|---|---|
| [v0.2](../../vulcan2d/README_v0.2.md) | 多缺陷组占据控制陷阱辅助输运，串入晶体管限流，尝试把阈值分布与电热反馈联系起来。记录了实测和模拟的转折等特征 | [复跑图](evidence/history/09_vulcan_v2_validation.png) · [建模方案](../../analysis/v2_synthesis.json) |
| [v0.3](../../vulcan2d/README_v0.3.md) | 更新导电斑块分布及晶体管查表。该版代表性序列的 SET 去/回程为 0.146/0.106，RESET 为 0.278/0.181 decade | [复跑图](evidence/history/10_vulcan_v3_validation.png) · [斑块分布](../../analysis/figures/11_afm_conductive_spot_distribution.png) |
| [v0.4](../../vulcan2d/README_v0.4.md) | 用 XTEM 线索约束 soft/hard 两种导电状态，覆盖三档栅压和裸 1R。硬态特征与受实验条件约束的 RESET 结果分开评估 | [多工况对比](../../analysis/figures/12_vulcan_v4_multiregime_validation.png) · [评分](../../analysis/v4_validation_summary.json) |
| [v0.5](../../vulcan2d/README_v0.5.md) | 检查强条件对结果的影响。撤去实测约束的 RESET 终态后，原低误差升至约 0.98/1.06 decade | [约束对比](../../analysis/figures/13_vulcan_v5_constraints_audit.png) · [评分](../../analysis/v5_validation_summary.json) |
| [v0.6](../../vulcan2d/README_v0.6.md) | 使用公开晶体管输出查表求实际分压。正偏测量的覆盖和反偏 RESET 的查表条件分别审查 | [负载线](../../analysis/figures/14_vulcan_v6_transistor_loadline.png) · [评分](../../analysis/v6_validation_summary.json) |
| [v0.7](../../vulcan2d/README_v0.7.md) | 先从已测 RESET 反推状态，再检验共享局部场转变。留出的归一化状态 RMSE 约 0.041，单位是状态值 | [复跑图](evidence/history/15_vulcan_v7_reset_identification.png) · [反演结果](../../analysis/v7_reset_identification_summary.json) |
| [v0.8](../../vulcan2d/README_v0.8.md) | 用 RESET 开头的低压读出初始化。1.1 V 去/回程 0.234/0.397 decade，属于已知入口的预测 | [复跑图](evidence/history/16_vulcan_read_initialized_reset.png) · [评分](../../analysis/v8_read_initialized_reset_summary.json) |
| [v0.9](../../vulcan2d/README_v0.9.md) | 利用实测 SET 刻画接续的 RESET 入口。后 16 圈入口电导 RMSE 由 0.362 降到 0.304 decade | [重建 I–V](evidence/history/17_vulcan_full_cycle_link_reconstructed.png) · [全循环记录](../../analysis/full_cycle_validation_summary.json) |
| [v0.10](../../vulcan2d/README_v0.10.md) | RESET 终态改用同协议高阻终点约束，和固定 HRS/LRS 比例作对照 | [复跑图](evidence/history/18_vulcan_full_cycle_hrs_attractor.png) · [终点与回程评分](../../analysis/full_cycle_hrs_attractor_summary.json) |
| [v0.11](../../vulcan2d/README_v0.11.md) | 高阻态使用有效场辅助输运。留出后 16 圈的 RESET 回程 0.308→0.130，去程 0.229→0.238 decade | [复跑图](evidence/history/19_vulcan_pf_hrs_refinement.png) · [场辅助输运评分](../../analysis/full_cycle_pf_refinement_summary.json) |
| [v0.12](../../vulcan2d/README_v0.12.md) | 用电压波形推演 SET→RESET；留一的中位回线 SET 去/回程 0.078/0.085，RESET 0.192/0.124 decade | [I–V 对比原图](evidence/20_vulcan_autonomous_loop.png) · [评分](../../analysis/full_cycle_autonomous_summary.json) |
| [v0.13](../../vulcan2d/README_v0.13.md) | 审计状态交接适用域与完整覆盖。增加启动圈后 RESET 评分由 51 对 0.293 变为 53 对 0.370 decade | [逐圈误差](../../analysis/figures/21_v13_cycle_error_and_reset_drift.png) · [新增覆盖曲线](../../analysis/figures/22_v13_previously_excluded_pairs.png) |
| [v0.13.1 电路修正](../../analysis/UNIFIED_TRANSISTOR_REPAIR_2026-10-29.md) | 单独 1T 和串联器件共享有符号晶体管输出，统一正反支路的电路口径。材料模型统一另见 11 月的联合工作 | [修正报告与图](../../analysis/UNIFIED_TRANSISTOR_REPAIR_2026-10-29.md) |

更早的机制分类、失败对照、数学界限和每版的实验图入口都保留在 [11 月 5 日历史对照](../../analysis/MODEL_ATTEMPTS_COMPARISON_2026-11-05.md) 及其[原报告汇编](../../analysis/MODEL_ATTEMPTS_SOURCE_REPORTS_2026-11-05.md) 中。v0.2、v0.3、v0.7–v0.11 缺的版本图已补为复跑图或重建图，来源见 [图集](figures.md#历史版本补图)。
