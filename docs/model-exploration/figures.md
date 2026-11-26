# 模型对比图

[模型探索总目录](README.md) · [11 月 24–25 日逐项记录](2026-11-24-to-25.md) · [历史原报告与图目录](history-index.md)

各图保留原始坐标、拟合线、实测中位数和分布。读图时先看横轴：固定态检查用扣除晶体管压降后的 hBN 电压，其余 SET 去程图多用外加端电压。高导通组的图使用实测回读结果分组。点位筛选、误差和模型用到的输入可从对应记录查到。

## 整圈与 RESET

| 原图 | 主要看哪里 |
|---|---|
| [v0.12 自主回线](evidence/20_vulcan_autonomous_loop.png) | 实测和原自主模型的 SET/RESET 四个分支 |
| [早期三档 SET+RESET 留出](evidence/overall_heldout_iv.png) | 比较原基准和逐步写入、连续擦除候选；RESET 尾段误差大 |
| [裸器件 SET/RESET](evidence/bare_handoff_iv_comparison.png) | 冻结 1T1M 参数和裸校准的差异；绿线给出 23 kΩ 的电流上限 |
| [IV41 新增四档冻结检验](evidence/iv41_frozen_iv.png) | 随栅压改变的 SET 和 RESET 失配位置 |
| [RESET 场激活擦除](evidence/activated_reset.png) | 1.1 和 1.3 V 的去程改善，0.9 V 仍偏离 |
| [RESET 平台读出](evidence/reset_platform_readout.png) | 场辅助电流如何改变平台以及回程 |
| [RESET 延迟擦除](evidence/delayed_reset.png) | 1.3 V 的平台、回程和反推状态 |
| [RESET 终点加权](evidence/endpoint_constrained_reset.png) | 降低终点偏差后，去程和回程怎样变化 |
| [RESET 全程连续状态](evidence/continuous_reset.png) | 1.3 V 去、回程及顶点后的状态衰减 |

## 高导通态与 SET

| 原图 | 主要看哪里 |
|---|---|
| [IV41 高导通分布](evidence/iv41_high_conductance.png) | 分别在端口和 hBN 局部电压上看到高导通组 |
| [首次形成概率及高态 I–V](evidence/hard_formation_validation.png) | 高态 I–V 近似欧姆，形成概率的高栅压排序失配 |
| [初态与负载线](evidence/hard_initial_state_loadline.png) | 初始读数能区分部分样本，总体形成比例仍有偏差 |
| [SET 去程背景对照](evidence/set_outward_refinement.png) | 根号场和跳跃背景在开通前、后的差异 |
| [跳跃背景＋连通增长](evidence/compact_set_physics.png) | 0.9 和 1.4 V 局部好转，1.5 V 的开通前电流偏高 |
| [高栅压分组与关闭增长](evidence/compact_set_high_gate_groups.png) | 1.5 V 低电导组的过预测几乎全来自背景项 |
| [初始保留连接](evidence/set_retained_links.png) | 加入近欧姆残留通路只带来局部改善 |
| [独立低压斜率](evidence/set_independent_slope.png) | 起点与涨幅解开后，1.5 V 的低组高压外推更接近实测 |
| [有限传输速率](evidence/set_current_bottleneck.png) | 开通窗改善，部分开通后平台偏低 |
| [单间隙和固定电阻](evidence/single_gap_series_set.png) | 高偏压较贴近，0.9 V 的快速开通推迟 |
| [SET 固定态近似诊断](evidence/set_fixed_state_readout_audit.png) | 同一限制项在 0.9 V 和 1.5 V 去程的效果相反 |
| [裸与串联的联合固定电阻](evidence/joint_gap_set.png) | 裸器件达到毫安，0.9/1.4 V 的开通晚 |
| [共享状态影响写入速率](evidence/joint_gap_state_rate.png) | 新旧联合曲线几乎重合 |
| [有效间隙宽度](evidence/joint_gap_width.png) | 开通略提前，裸器件开通前漏电偏高 |

## 历史版本补图

下列图在隔离目录中用项目保留的历史代码和测量表生成。v0.2 用提交 `58e45e1` 的模型和已存标定，v0.3、v0.7、v0.8、v0.10、v0.11 用提交 `87b6a15` 中保留的对应验证模块。v0.3 的四段误差复现到原报告的三位小数，v0.7、v0.8、v0.10、v0.11 的复跑 JSON 与原数值一致（浮点容差内）。v0.9 从 v0.10 的代码保留的“实测 SET + 固定 HRS/LRS 比例”对照重建 I–V，去回程 RMSE 与 v0.9 原报告相同，图的版式为这次重绘。

| 图 | 复跑入口或来源 |
|---|---|
| [v0.2](evidence/history/09_vulcan_v2_validation.png) | `analysis/extract_features.py` 后运行 `vulcan2d.validate` |
| [v0.3](evidence/history/10_vulcan_v3_validation.png) | `vulcan2d.validate` |
| [v0.7](evidence/history/15_vulcan_v7_reset_identification.png) | `vulcan2d.validate_v7` |
| [v0.8](evidence/history/16_vulcan_read_initialized_reset.png) | `vulcan2d.validate_v8` |
| [v0.9 重建](evidence/history/17_vulcan_full_cycle_link_reconstructed.png) | `vulcan2d.validate_full_cycle` 中的 `set_conditioned_cycles` 对照 |
| [v0.10](evidence/history/18_vulcan_full_cycle_hrs_attractor.png) | `vulcan2d.validate_full_cycle` |
| [v0.11](evidence/history/19_vulcan_pf_hrs_refinement.png) | `vulcan2d.validate_v11` |
