# 0.9/1.3 V低偏单支路状态方向检验

0.9/1.3 only; no fit, no1.1 residuals or holdout selection; original cached port I/V re-deembedded with current mapping.

重新使用当前统一晶体管映射去嵌入缓存中的原始端口I/V，未读取旧内部电压作为观测，未拟合。窗口为名义内部|Vh|=0.10–0.30V；保护阈值沿用原低偏诊断。SET返回按原raw_index递增要求gamma不减，RESET起始要求gamma不增。

| gate | 支路 | floor | 电流预算 | 可判点不足 | 数值/域未知 | 区间方向可行 | 条件违例 |
|---:|---|---|---:|---:|---:|---:|---:|
| 0.9 | set | all_positive | 0% | 4 | 0 | 0 | 96 |
| 0.9 | set | all_positive | 1% | 4 | 0 | 0 | 96 |
| 0.9 | set | all_positive | 5% | 4 | 0 | 6 | 90 |
| 0.9 | set | greater_than_3_full_scan_zero_reference | 0% | 4 | 0 | 0 | 96 |
| 0.9 | set | greater_than_3_full_scan_zero_reference | 1% | 4 | 0 | 0 | 96 |
| 0.9 | set | greater_than_3_full_scan_zero_reference | 5% | 4 | 0 | 6 | 90 |
| 0.9 | reset | all_positive | 0% | 7 | 0 | 0 | 93 |
| 0.9 | reset | all_positive | 1% | 7 | 0 | 0 | 93 |
| 0.9 | reset | all_positive | 5% | 7 | 0 | 1 | 92 |
| 0.9 | reset | greater_than_3_full_scan_zero_reference | 0% | 7 | 0 | 0 | 93 |
| 0.9 | reset | greater_than_3_full_scan_zero_reference | 1% | 7 | 0 | 0 | 93 |
| 0.9 | reset | greater_than_3_full_scan_zero_reference | 5% | 7 | 0 | 1 | 92 |
| 1.3 | set | all_positive | 0% | 32 | 0 | 0 | 69 |
| 1.3 | set | all_positive | 1% | 32 | 0 | 14 | 55 |
| 1.3 | set | all_positive | 5% | 32 | 0 | 34 | 35 |
| 1.3 | set | greater_than_3_full_scan_zero_reference | 0% | 32 | 0 | 0 | 69 |
| 1.3 | set | greater_than_3_full_scan_zero_reference | 1% | 32 | 0 | 14 | 55 |
| 1.3 | set | greater_than_3_full_scan_zero_reference | 5% | 32 | 0 | 34 | 35 |
| 1.3 | reset | all_positive | 0% | 37 | 0 | 0 | 64 |
| 1.3 | reset | all_positive | 1% | 37 | 0 | 1 | 63 |
| 1.3 | reset | all_positive | 5% | 37 | 0 | 12 | 52 |
| 1.3 | reset | greater_than_3_full_scan_zero_reference | 0% | 37 | 0 | 0 | 64 |
| 1.3 | reset | greater_than_3_full_scan_zero_reference | 1% | 37 | 0 | 1 | 63 |
| 1.3 | reset | greater_than_3_full_scan_zero_reference | 5% | 37 | 0 | 12 | 52 |

逐记录raw_index、保留点、去嵌入电压、每种预算的gamma区间及失效理由见JSON。正向/负向读出采用各自符号；不同gate不配对，不假设扫速相同。

可行只表示存在满足方向的区间选择，不能证明完整动力学。违例只限制固定谱形+当前T映射+单向状态；尤其原值的小违例可由噪声或查表误差产生，须看1%/5%与datafloor敏感性。

缓存来源及配对限制见 CROSS_GATE_DATA_CONSTRAINT_PLAN.md；原始xlsx为云端占位，本轮不声称重新核对原表。

精确反演认证 32835/32835，64阶抽查最大电流残差 3.88e-10 decade；耗时 47.04s。
