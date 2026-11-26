# 1T / 1T1R 自洽性修复与证据

> 2026-11-03 用户纠正：要求统一的是 **1R 与 1T1R 的 h-BN 数学模型**。
> 本文晶体管修复仍成立，但不是该要求的完成证明。当前材料模型工作见
> [1R/1T1R 状态说明](SHARED_HBN_STATUS_2026-11-03.md)。

日期：2026-10-29；研究接口 v0.13.1。基线提交：`87b6a154374db96fdd37167a8ad9b516e0122229`。

本轮修复了两项可复现的代码问题，统一了研究接口的晶体管端口定律与电路求解。
**默认研究版曲线原本已经分别使用正、反向独立 1T 测量，本轮没有提高其拟合精度，
也没有证明统一的微观 MOS 或 h-BN 机制。** 桌面演示仍属于旧模型。

## 实际发现了什么

1. v0.13 允许只改 SET 中晶体管的电流比例，RESET 中的同一只晶体管却保持不变。
   将 SET 的 `current_scale` 改为 0.5，示例回线的 SET 电流最大变化约 21.23 µA，
   RESET 逐点变化为零。旧接口没有拒绝这组矛盾参数。
2. 旧 `model.i_tr` 接收双极性实测 mapping 时，没有传递 RESET 极性，落入正向默认分支。
   在 |VDS|=1 V、VG=1.1 V 下，它错误返回 48.7817 µA，反向测量实际为 113.866 µA。
3. 历史入口容易混淆：v7 的正向仿射映射在同一点给 29.0860 µA；v12/v13 的默认
   SET 已给正确的直接插值 48.7817 µA。**不能把旧映射的问题归给现行默认 SET。**
   旧 `model.Params()` 的正/反向经验值为 16.0046 / 5.96718 µA；这些是默认参数
   对照值，并非重新加载桌面标定 NPZ 后的运行结果。

## 两类器件现在共用什么方程

固定原来的物理源极为零，令内部漏极节点为 D：

\[
\mathrm{1T}:\quad I_D=F_T(V_D,V_G),
\]
\[
\mathrm{1T1R}:\quad V_D=V_{\rm app}-V_h,
\qquad I=F_T(V_D,V_G)=F_h(V_h,x).
\]

因此，同一内部漏极电压、同一栅压下，两种接法的晶体管电流必须相同。
器件端口曲线不同可以来自 h-BN 的分压和状态，不能通过悄悄更换晶体管参数解释。
当串联电阻趋近零，1T1R 应连续回到独立 1T；电阻趋近无穷时电流趋零。

`F_T` 是保留物理源极参考的有符号实测端口函数：正偏压查 Fig.2b，负偏压查
Bookoutput2。负偏压不交换物理源漏后继续沿用旧 VGS，也不把正向曲线奇对称镜像。
返回值称为**总漏极电流**；数据没有分辨沟道与体端支路。

实现变化：

- `measured_transistor_mapping()` 提供直接栅压映射的正反向共用对象；
  `drain_current(VDS, VGS)` 按物理电压符号选择测量分支。
- v0.13.1 的 `transistor_mapping=` 一次控制完整循环。历史分段参数仍可输入，
  但电流比例、访问电阻和栅压参考必须相容，否则明确拒绝。
- SET 与 RESET 使用同一个 KCL 求解器；旧显式 proxy 入口为复现历史保留，
  当前 v0.13.1 完整循环不接受它替代实测反向曲线。
- 旧模型显式接入双极性 mapping 时，已修复漏传极性的问题。
  其默认经验参数、桌面仿真及 patch 图没有迁移为新研究模型。

```python
from vulcan2d import measured_transistor_mapping, simulate_calibrated_cycle

transistor = measured_transistor_mapping()
standalone_current = transistor.drain_current(-1.0, 1.1)
result = simulate_calibrated_cycle(
    set_voltage, reset_voltage, gate_voltage=1.1,
    transistor_mapping=transistor,
)
```

## 验证结果

| 检查 | 结果 |
| --- | --- |
| 针对性测试 | 50 项通过，其中新增 10 项自洽性与边界回归 |
| 54 个冻结状态负载点，覆盖 3 个栅压、3 个状态及正反偏压 | 最大 KCL 误差 1.03×10⁻¹⁴ A（向上取整） |
| 与独立 Brent 求根交叉验证 | 最大电流差 8.52×10⁻¹⁵ A（向上取整） |
| 串联欧姆电阻降至 0.01 Ω | 正/反向相对独立 1T 偏差约 1.40×10⁻⁷ / 2.02×10⁻⁶ |
| 第 4～53 圈完整默认轨迹 | 最大 KCL 误差 8.48×10⁻¹⁶ A（向上取整） |
| 与从基线提交单独导出的旧源码比较 | 50 圈电压、电流、内部压降全部逐点相同 |
| 实测电流是否超过同偏压独立 1T 的无源串联上限 | 所检查点零违反；最大比例 SET 0.5323、RESET 0.08674 |

按老师的意见，主要评价使用第 4～53 圈；前三圈保留原始记录，不继续要求模型
解释第 3→4 圈跃迁。“没有 SET 成功”与“SET 后自恢复”仍是两种可能说明。

沿用历史完整组标定参数，正常循环的逐点对数 RMSE 为：

| 分支 | 评分点数 | RMSE / decade |
| --- | ---: | ---: |
| SET | 25,000 | 0.16485 |
| RESET | 8,500 | 0.25885 |

**这只是既有参数在新评价范围上的描述性重评分。** 没有进行第 4～53 圈重拟合，
不是新的留出预测，也不能与旧连续候选模型的后 18 圈预测误差直接排名。
原始 XLSX 在本机是云端占位文件，本轮使用此前保存的 native observed 数组，
没有宣称重新读取或重新核验工作簿。所有正常循环的波形相同，计算缓存只复用
电压输入产生的预测，实测电流仅用于外部评分。

## 尚未闭合的物理证据

2026-11-03 来源勘误：早期 [项目交接](../../PROJECT_HANDOFF.md) 第178–183行已记载用户确认
反向第四列为 ID、独立/串联晶体管同类型同尺寸、端子和 body/well 接线一致。
后续报告遗漏了这些确认；不能继续重复索要。交接仍未确认 body/well 的具体
电位、是否同一 physical cell 或逐器件电流比例。详见
[来源审计](spectral_dynamic_2026-11-03/CIRCUIT_SOURCE_CONSTRAINT_AUDIT.md)。
下述范围据此更正；历史 JSON/转换元数据里的旧“待确认”标签由此勘误覆盖，
不重写冻结曲线或哈希快照。

正向 1T 数据来自 Nature 2023 公开 Fig.2b；反向数据来自 2026-09-08 的
`CHIP-CE4-10`。反向第四列按已确认的 ID 使用；同类型、同尺寸、同接线支持
共用实测端口模型，但逐 cell 的定量对应仍需区分。因此不能声称
两组数据已证明来自同一只晶体管，也不能把反向大电流唯一归因为体二极管。

h-BN 的 SET/RESET 仍采用不同的输运包络和经验交接，它们尚不是同一个已验证
的微观状态。此次修复没有解决这一更深层问题。下一项最有价值的物理约束是
核实逐 cell 的定量对应和正向 VG=1.1 V 直接曲线；在已确认接线下判断剩余的
极性差异应归于 h-BN，还是还需要测量支持的晶体管支路。暂不继续增加隐藏状态
或为前三圈优化机制。

## 复现与文件

- 当前最终数值与来源哈希：[`verified/summary.json`](unified_transistor_2026-10-29/verified/summary.json)。
- 当前逐点输出：[`verified/curves.npz`](unified_transistor_2026-10-29/verified/curves.npz)。
- [50 项测试日志](unified_transistor_2026-10-29/core_tests.log)。
- [审查脚本](audit_unified_transistor.py)。

结果目录顶层的 `summary.json/curves.npz` 同步保存了最终副本；引用以 `verified/` 为准。
包含 pandas 的更广验证入口因依赖文件尚未下载到本机而中断，日志保留在
`interrupted_pandas_import.log`，不计为通过。本轮没有声称全仓库测试全部通过。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest \
  tests.test_transistor_consistency tests.test_model_v6 tests.test_model_v7 \
  tests.test_model_v8 tests.test_model_v11 tests.test_model_v12 tests.test_model_v13 -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  analysis/audit_unified_transistor.py --output-dir /tmp/vulcan-1t-audit
```

第二条命令重算当前自洽性与评分；严格前后相同检查还需传入由基线源码产生的
`--baseline` NPZ，最终 JSON 已记录基线提交、文件哈希及导出说明。
