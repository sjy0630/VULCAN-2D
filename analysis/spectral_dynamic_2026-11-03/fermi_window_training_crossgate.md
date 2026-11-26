# 训练及跨栅压 SET 返扫：有限温度全谱类成对必要界

纯代数及当前晶体管映射检验，无谱态反演、拟合或 ODE。只将 1.1 V 编号4–35训练数据，以及0.9/1.3 V编号≥4的原生端口观测纳入数值数组。

名义内部窗口 .10–.30 V；沿用三项 T 保护阈值。高信号为 I>max(100×完整扫描最大零偏参考,1e-10 A)。5% 为 [I/1.05,I×1.05]，每次电流变化联合去嵌入。任意 η∈[0,1]、300 K。

主跨度筛选：原生 raw_index 间隔≥3 且端口电压跨度≥.05 V；同时报告全部点对与≥.10 V跨度敏感性。要求整个电流区间的高低内部电压顺序一致。

| Gate | 电流门槛 | 跨度 | 条件违例 | 未见点对违例 | 点不足 | 跨度不足 | 检查点对数 |
|---:|---|---|---:|---:|---:|---:|---:|
| 0.9 | original_positive | all_native_pairs | 55 | 38 | 4 | 0 | 17805 |
| 0.9 | original_positive | span_ge_3indices_and_0p05V | 15 | 78 | 4 | 0 | 11328 |
| 0.9 | original_positive | span_ge_3indices_and_0p10V | 3 | 89 | 4 | 1 | 5278 |
| 0.9 | high_signal | all_native_pairs | 55 | 38 | 4 | 0 | 17805 |
| 0.9 | high_signal | span_ge_3indices_and_0p05V | 15 | 78 | 4 | 0 | 11328 |
| 0.9 | high_signal | span_ge_3indices_and_0p10V | 3 | 89 | 4 | 1 | 5278 |
| 1.1 | original_positive | all_native_pairs | 0 | 32 | 0 | 0 | 1440 |
| 1.1 | original_positive | span_ge_3indices_and_0p05V | 0 | 32 | 0 | 0 | 896 |
| 1.1 | original_positive | span_ge_3indices_and_0p10V | 0 | 32 | 0 | 0 | 480 |
| 1.1 | high_signal | all_native_pairs | 0 | 32 | 0 | 0 | 1440 |
| 1.1 | high_signal | span_ge_3indices_and_0p05V | 0 | 32 | 0 | 0 | 896 |
| 1.1 | high_signal | span_ge_3indices_and_0p10V | 0 | 32 | 0 | 0 | 480 |
| 1.3 | original_positive | all_native_pairs | 6 | 61 | 31 | 0 | 13717 |
| 1.3 | original_positive | span_ge_3indices_and_0p05V | 1 | 66 | 31 | 0 | 8907 |
| 1.3 | original_positive | span_ge_3indices_and_0p10V | 0 | 67 | 31 | 0 | 4321 |
| 1.3 | high_signal | all_native_pairs | 6 | 61 | 31 | 0 | 13717 |
| 1.3 | high_signal | span_ge_3indices_and_0p05V | 1 | 66 | 31 | 0 | 8907 |
| 1.3 | high_signal | span_ge_3indices_and_0p10V | 0 | 67 | 31 | 0 | 4321 |

## 高信号且至少 .05 V / 3 原生索引跨度的最强见证

| Gate/编号 | raw_index | 端口V早→晚 | 电流A早→晚 | 5%最小电流比 | 5%最大上界 | 超界倍数 |
|---|---|---|---|---:|---:|---:|
| 0.9/12 | 973→978 | 0.28→0.23 | 1.08218e-06→1.09366e-08 | 89.750851 | 4.5118344 | 19.89232 |
| 1.3/52 | 978→983 | 0.23→0.18 | 7.4616e-06→2.4692e-06 | 2.7409247 | 2.586616 | 1.0596566 |

## 解释边界

多点后验筛查，不是统计显著性；点对互相关，5%不是测得噪声置信区间。未见违例只表示保留低压点对没有否决此必要条件，不能证明整个模型或动力学有效。违反的是冻结非负谱/固定线性刚性移位/返扫传输不减/等温300K/当前有效准静态去嵌入的前提交集，不能唯一归因于正向状态回退。

逐记录保留点、实际电流/电压、5%联合去嵌入端点、最强点对和缺覆盖原因见JSON。没有使用旧缓存内部电压；未重新打开原工作簿；跨栅压不配对，不假定同器件或扫速。

计算耗时 5.799s；独立正向回代最大相对残差 4.44e-16。
