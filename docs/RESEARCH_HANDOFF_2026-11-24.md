# Vulcan-2D 研究交接

本次交接后的实验、逐版差异、原始分数与曲线见 [模型探索记录](../../VULCAN-2D/docs/model-exploration/README.md)。后续试验按[模型迭代协议](../../VULCAN-2D/docs/model-exploration/PROTOCOL.md)归档。

## 研究问题

针对 Zhu 等人在 Nature 发表的 Au–Ti/hBN/W 忆阻器，建立一套能解释 SET 写入、RESET 擦除和低压读出，并能在不同外接电路中复用的器件模型。裸 1M 和 CMOS 1T1M 是标定和验证场景，光敏晶体管串联是后续的迁移检验。

## 当前版本和证据

| 版本 | 用途与证据 |
| --- | --- |
| 跨栅压预测基准 | `vulcan2d/model_shared_device.py` 和 `vulcan2d/gate_only.py`。裸 1M 使用仪器限流，1T1M 使用独立实测晶体管负载线。0.9、1.1、1.3 V 标定，历史留出 1.0、1.2、1.4、1.5 V。完整回线的对数电流 RMSE 从 0.814 降到 0.793，端口直接插值基线为 0.815。1.5 V 典型 RESET 回扫由 1.556 降到 1.202，1.4 V 由 0.682 升到 0.835。导纳修正、回扫幅值插值和转折点平滑属于经验校准。 |
| 能量约束的物理候选 | `vulcan2d/activated_configuration.py` 和 `vulcan2d/retained_charge_configuration.py`。保留构型和电荷占据的历史，按同一能量函数约束正反跃迁速率，并用独立负载线求端口电流。原验证采用固定直流扫压的电压路径，实际跃迁时标还待独立标定。它尚未在与 0.793 基准完全相同的四档跨栅压任务上验证。 |
| 稳态 1.1 V 对照 | 该批数据第 1–3 圈经实验确认 SET 未成功，第 4–53 圈为有效循环。使用旧版统一连通模型，50 圈留一评分为 SET 0.166、RESET 0.261。这个结果对应同一栅压、不同循环的测试。 |

共用方程比按拓扑切换输运支路更贴合研究问题。当前裸 1M 仍需要单独标定部分导电幅度，跨拓扑的参数迁移尚待检验。Poole–Frenkel 型曲线形状也不足以确认微观机制。以均匀场、300 K 和 6 nm 为条件，SET 背景拟合系数反推出的表观介电常数约为 138，明显高于文献中的 hBN 测量值。温度和非扰动固定态读出是区分输运机制的直接约束。

## 接手顺序

1. 先复现历史留出基准。训练按原规则排除限流点并做三点中值滤波，评分用相同掩码下的原始电流。分别报告 SET、RESET、去程、回扫、RESET 翻转位置和低压读出电导。
2. 用相同数据划分和晶体管查表评估物理候选。将 SET 终态传入 RESET，令回扫电流只由模型的擦除终态产生。把 0.793 作为预测基准。
3. 测量同一颗器件改变 SET 条件的序列。优先核对是否已有固定写入幅值、固定栅压、不同脉宽的记录。每圈保存初态、实际施加波形及写入后和擦除后的低压电导。用部分脉宽标定，其余脉宽留作预测。若按栅压做对照，可用 0.95、1.15、1.35 V SET 栅压，RESET 栅压固定 1.10 V。测量前冻结参数和预测。
4. 固定电阻态后测不同温度下的低扰动 I–V，比较候选输运规律。1M 与 1T1M 的对照优先使用同一颗器件和可比的初始状态、读出与编程历史。
5. 独立测光敏晶体管的暗态和照明输出特性。冻结已验证的 hBN 参数后，只替换外接晶体管的负载特性，用留出的光照条件检验串联回线。

## 复现入口

- 多栅压基准：`python3 -m analysis.validate_gate_only`
- 擦除回扫候选：`python3 -m analysis.validate_reset_return`
- 新实验冻结及评分：`python3 -m analysis.validate_paper_claim` 和 `python3 -m analysis.score_paper_measurements`
- 1.1 V 有效循环：`python3 -m vulcan2d.validate_postconditioning`
- 物理候选的能量约束和限制：`analysis/ACTIVATED_CONFIGURATION_PHYSICS_2026-09-21.md` 和 `analysis/RETAINED_TRANSPORT_RESULTS_2026-09-21.md`

上述命令用 `--help` 查看输入文件和输出目录。源数据、经过滤波的训练输入、原始评分数据和输出图表应注明来源及筛选口径。纸面上注明哪些结果来自历史留出，哪些来自新采集的独立测量。

## 文献

- [Zhu et al., Hybrid 2D–CMOS microchips for memristive applications, Nature 618, 57–62 (2023)](https://doi.org/10.1038/s41586-023-05973-1)
- [Ducry et al., An ab initio study on resistance switching in hexagonal boron nitride (2022)](https://doi.org/10.1038/s41699-022-00340-6)
