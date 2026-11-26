# 晶体管映射：原文、原始来源与既有确认的有限审计

日期：2026-11-03。只审查当前查表来源、端子定义和可校准范围；没有修改 core、查表、已有拟合或多人正在编辑的报告。

**结论：没有发现支持把共同电流尺度改成 c≈0.84/0.88 的独立标定证据，也没有发现当前正负极性选择或固定源极参考的新代码错误。确实发现了后续报告遗漏早期用户确认的问题，必须先修正证据状态，避免重复索要已确认信息。**

## 1. 先恢复被后续报告遗漏的确认

工作区根目录 [`PROJECT_HANDOFF.md`](../../../PROJECT_HANDOFF.md) 第178–183行明确记载：

> 已经由原对话中的用户确认，无需重复索要：
> - `Bookoutput2.xls` 第四列是漏极电流 ID。
> - 独立 1T 与 1T1M 晶体管同类型、同尺寸，端子和 body/well 连接方式相同。
> - body/well 的具体电位仍未知，但实测查表已包含其影响。
> - Au–Ti 顶电极扫描，W 接 Drain，Source 接地、Gate 恒压。

`analysis/V12_PHYSICS_EVIDENCE_2026-09-20.md` 第57–61行再次明确同样的确认，并指出 JSON 中“待确认”标签早于用户确认。

这是一份明确的**用户确认交接记载**。本轮未重新获得原始用户发言逐字文本：交接链接为[原始分享对话](https://chatgpt.com/s/cx_6aaf9fc547a08191a876eaaad2629503)，本次网页读取未返回可读正文，浏览器读取也未完成。不能把交接原文伪称为本轮重新读到的用户原话；也不能因本轮未取得原话，就把已有确认一律退回未知。

证据层级应改成：

| 项目 | 当前应采用的状态 |
|---|---|
| 反向原始第四列为 ID | 交接记录明确记载用户已确认；原始工作簿未随当前仓库归档，未重新审计转换 |
| 独立与串联晶体管同类型、同尺寸 | 交接记录明确记载用户已确认 |
| 端子与 body/well 连接方式一致 | 交接记录明确记载用户已确认 |
| Source 接地，Gate 对原物理 Source 恒压 | 用户交接确认，且 Nature 正文直接支持 |
| body/well 的具体电位 | 交接仍明确未知；同接法实测端口响应包含其影响 |
| 正向、反向、目标阵列为同一 physical cell | 未获这项确认；同类型同尺寸同接法不等于同一 cell |
| c 的真实数值及器件间分布 | 没有独立标定；不能用材料拟合倒推后称为已测事实 |

## 2. 当前实际调用的曲线与偏置

`analysis/spectral_dynamic_2026-11-03/joint_problem.py:82` 构造 `LookupSpectralCircuit(mapping, 1.1)`；默认 mapping 来自 `vulcan2d/transistor.py:283` 的 `measured_transistor_mapping()`。

| 分支 | 实际来源 | gate 范围及处理 | 能说明什么 |
|---|---|---|---|
| 正向 | `vulcan2d/data/fig2b_transistor_lookup.json`；来源 `2b.xlsx`，官方 Zenodo 7607096 | 0.5、1.0、1.5、2.0、2.5、3.0 V；VDS=0→5→0 V | 对应 Nature Fig.2b 的 standalone 输出，不含直接 VG=1.1 V 曲线 |
| 负向 | `vulcan2d/data/bookoutput2_reverse_transistor_lookup.json`；来源 `Bookoutput2.xls`，标签 CHIP-CE4-10，文件日期2026-09-08 | VG=0至5.5 V、步距0.1 V；保留负物理 Vd 的幅值 | 包含直接 VG=1.1 V；这是后来提供的本地数据，不能写成 Nature 2023 已公开的反向源数据 |

两份原工作簿的已记录 SHA256 分别为：

- `2b.xlsx`: `4009913b0d35ae02f1030d25e2fadba5a9fdd2dcf830f050d3e59c41eb51a518`
- `Bookoutput2.xls`: `39f75147d5b19205b390ccb040073b25c52db428bbe647c5aef4d0e6f7ffe7db`

原工作簿未包含在当前 Git HEAD 文件树；本轮审查的是已归档 JSON 与转换源码，不能宣称重新核验原表→JSON 的内容一致性。交接第167、218行也明确这一限制。

`transistor.py:48` 默认先平均去/回扫、设置零原点并取累计最大值；`transistor.py:95` 随漏压及栅压线性插值。因此 VG=1.1 V 的正向电流是 `0.8 I(VG=1.0)+0.2 I(VG=1.5)`，不是直接测量，也不是按 MOS 方程识别出的器件参数。

`transistor.py:229` 明确保持原物理 Source=0；负 VDS 不重命名源漏，不改变原 VGS，而选择实测负向分支。`shared_spectral_ivp.py:206` 按外加电压符号选分支，并以有符号材料电流求 KCL。`shared_gap.py:193–204` 为两个分支使用同一 `current_scale` 和额外串联电阻。没有发现现行研究入口仍使用旧的正向奇对称镜像或只缩放 SET 的问题。

## 3. Nature 与补充信息能进一步约束什么

原文：[Zhu et al., Nature 618, 57–62](https://www.nature.com/articles/s41586-023-05973-1)；[官方补充信息](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-023-05973-1/MediaObjects/41586_2023_5973_MOESM1_ESM.pdf)。本地文本位于 `../verification/2026-09-20/physics-sources/`。

- **Electronic memory 正文，缓存 `zhu2023-nature.txt:894`**：Fig.2b 先测一只 standalone T；1T1M 在顶 Au–Ti 电极施加 RVS，物理 Source 接地，同时施加恒定 VG，Fig.2c 为1.1 V。支持当前端子/电压方向，不支持为了补曲线而随极性重定义 VGS。
- **Methods / Characterization，缓存第925行**：DC 使用 Keithley4200 的 ramped-voltage sweep，三只 SMU 对应 drain、source、gate。没有给出1.1 V 正向细扫、源/体独立偏置日志、Kelvin 接线或逐段时间，不能从这段话推算接触电阻或扫速。
- **SI Fig.1，印刷p.3 / PDF p.4**：图中每列 TE 与 SOURCE 由五个支路共享，每行 GATE 独立。MOS 的第四/体端画向三角参考地符号，提供设计接线层面的支持；图中没有给出实际 body/well 电位日志。未选行的门极设置也没有在所读正文说明。应记录单支路等效所需的选通/漏电前提，但不能据此直接断言实验存在并行漏电，更不能擅自增加并行通道。
- **SI Fig.2，印刷p.4 / PDF p.5**：standalone T 在左侧第3–8列，1T1M阵列在右侧；图标出 Source/Drain/Gate pad。明确是独立参考结构与阵列结构，不能把“同类型同尺寸”升级成逐 cell 配对标定。
- **SI Fig.5，印刷p.7 / PDF p.8；文本第256–258行**：明确沟道长度 **500 nm**。180 nm 是工艺节点，不能作为这只晶体管的沟道长度。图没有给出足以校准 W/L、电迁移率或器件间电流比例的宽度/分布数据。
- **SI Note1，文本第345–348行**：无 hBN 的 Au/W 与 Ag/W 接触呈欧姆响应，Au/Ti/W 因 Ti 层可能含氧而电流较低。支持接触层会影响导电，但没有给出目标 cell 的独立额外串联电阻数值。

本轮实际检查过的 SI 图像保存于 `source_inspection/supp-04.png`、`supp-05.png`、`supp-08.png` 和放大的 `supp1-cell-04.png`。图像支持上面的接线/位置/尺寸读图；没有用图像估算 c。

## 4. VG=1.1 V 插值不能给 c 的标定或置信区间

已有 `analysis/transistor_gate_interpolation_audit.json` 对正向1.1 V给出：在 VDS=1 V，线性电流、sqrt电流、log电流、PCHIP 分别约48.782、46.151、44.021、45.871 µA；这些只是共享原始门极样本的插值选择。在 VDS=5 V 其差异又明显变小，说明不能简单把插值选择等同于全域共同常数 c。

本轮直接核对相关电压：VDS=1.2 V时，VG=1.0/1.1插值/1.5 V分别为37.612/51.055/104.825 µA；VDS=1.4 V时为39.345/53.111/108.174 µA。只有额外假定门极电流单调时，邻近两条实测门极曲线才提供宽包络。这个包络没有识别1.1 V的精确位置，更不能给“c偏差12%–16%发生概率”。

已完成的 `circuit_order_sensitivity.md` 量化的是假设改变电路映射会在何处失去一项材料排序约束。它不能倒过来证明 c应取该值。用户已确认同尺寸，也意味着不应把所需 c 解释成任意 W/L 比例校正。

## 5. 可实际落实的改正与最小文件清单

本轮未修改以下文件，交给主代理协调，避免破坏正在写的报告及旧结果哈希。

| 优先级 | 文件/位置 | 最小改正 |
|---|---|---|
| 当前证据入口 | `CLAUDE.md:169–170` | 将“body/well wiring未确认”拆成：交接称已确认同接线；具体电位/同physical cell/定量映射未识别 |
| 当前说明 | `analysis/UNIFIED_TRANSISTOR_REPAIR_2026-10-29.md:97–105` | 加勘误：第四列ID及类型尺寸接线已有用户确认；不继续要求重复确认；保留同cell与微观归因限制 |
| 重新生成来源说明 | `vulcan2d/precompute_reverse_transistor.py:3–7,70–73` 与 `data/bookoutput2_reverse_transistor_lookup.json:9–10` | 将“provisional ID/wiring”等陈旧标签改为明确交接来源；不改变数值曲线。若需保持快照哈希，可先由独立勘误覆盖 |
| 正向元数据 | `vulcan2d/precompute_transistor.py:46` 与 `data/fig2b_transistor_lookup.json:12` | 区分已知端子/名义匹配、SI给出的500nm，与仍未知的精确1.1V曲线、宽度、同cell与统计变异 |
| 防止下次审计再生成旧说法 | `analysis/audit_unified_transistor.py:279` | 将同cell、body具体电位及已有同接法确认分开表达 |

`transistor.py:232–237` 当前写的是“固定源极、继承body接法、尚非同physical transistor”，与上述区分相容，无需因此改数值实现。历史 `unified_transistor_2026-10-29/*/summary.json` 等冻结产物应保留原记录并链接勘误，不能重写旧数值证据。最新电路敏感性报告只说同physical cell/统计分布未知，也无需撤销其数值结果；可补一条用户名义匹配确认以消除歧义。

现有资料能落实的工作是纠正来源状态、固定正负符号/电压参考、把门极插值及单支路假设明确写出。**没有足够证据进行新的 c、gate-offset、body-bias 或额外串联电阻标定。** 下一份最有判别力的独立资料是既有测量中的正向VG=1.1V输出与阵列选通设置；不能用当前材料曲线本身来冒充这项独立电路校准。
