# VULCAN-2D v0.12 交接说明：闭合 SET 半回路

> **审计说明（2026-09-20）：本文件是已归档的 Claude 中间探索记录，不能再作为
> 当前实现或参数说明。** 后续审计发现其中沿用的 v0.6 仿射栅压映射会把 1.1 V
> 请求映为约 0.896 V，并导致本批 SET 反演回折；pre/connected 共用 `sinh`、逐循环强制 0--1 归一化和把曝光写成
> mV 也不成立。上述问题已经修正，完整自主模型、严格留一验证和现行公式见
> `vulcan2d/README_v0.12.md` 与
> `analysis/full_cycle_autonomous_summary.json`。下文仅保留为开发过程记录。

> 写于 2026-09-16。这是**进行中**的工作状态，不是完成文档。
> 工作目录：VULCAN-2D 仓库根目录。

---

## 0. 一句话任务

把模型从

```
实测SET曲线 + RESET扫描电压  →  预测RESET曲线
```

升级成

```
完整扫描电压序列 + V_G + 初始态  →  自主SET → 自主RESET → 完整回滞环
```

**硬约束：预测某一轮时，不得读取该轮的目标 SET/RESET 电流。**

---

## 1. 已完成

### 1.1 前置事实（已验证，可直接采信）

- 用**正向** Fig. 2b 晶体管 lookup（`load_default_lookup()`）对 `1T1M写入.xlsx` 的 53 条 SET 曲线做去嵌入：**26,659 点全部可反解，0 个失败**。RESET 侧用的是 `load_reverse_lookup()`，两者方向不同，别搞混。
- SET 转变点：前/回支最大分离处 `V_app` 中位 **1.43 V，CV 5.0%**，比 RESET 侧更可复现。
- **关键陷阱**：SET 的 `V_hBN` 在循环内**不单调**（晶体管钳位，升到约 1.5 V 后回折，53/53 循环皆然）。所以状态驱动量**绝不能用瞬时 `E_hBN`**，必须用单调的累积曝光 `exposure = ∫|V_hBN| dV_app`。
- SET 转变发生在**内部场约 2.5 MV/cm**（对应 `V_hBN` ≈ 1.5 V @ 6 nm），比 RESET 的 `reset_e50_mvc = 0.837` 高约 3 倍。两侧阈值本来就该是独立随机变量。

### 1.2 新文件：`vulcan2d/set_identification.py`（**已写完，可运行**）

镜像 `reset_identification.py` 的结构：

| 函数 | 作用 |
|---|---|
| `ReconstructedSet` | 去嵌入后的循环数据类（同 `ReconstructedReset` 布局） |
| `reconstruct_set(v, i, Vg, mapping)` | 逐点反解 `V_DS`，输出 `V_hBN` |
| `_set_branch_mask(cycle, "pre"\|"conn")` | 选 pre 包络（升支）或 connected 包络（降支）的稳定段 |
| `fit_shared_pre_alpha(cycles)` | 共享 `alpha_pre`，profile 掉每循环幅值 |
| `set_exposure(cycle)` | 累积曝光（mV 单位），单调 |
| `_isotonic_increasing` | PAVA，SET 侧是**递增**（RESET 侧是递减，别照抄） |
| `identify_set_state(cycle, alpha_pre)` | 输出 `IdentifiedSetState`，含 `g_pre_a`/`g_conn_a`/`e50_exp` |
| `set_transition_state(exp, e50, w)` | 递增 logistic：`1/(1+exp[-(e-e50)/w])` |
| `set_width_from_levels(e90, e10)` | 宽度换算，注意符号与 RESET 版相反 |

### 1.3 已跑出的标定值（1.1 V，53 循环，全部识别成功）

```
shared alpha_pre = 5.812003      shape MAD = 0.046 decade
e50 exposure   = 1448.14 mV      CV 10.7%
width          = 126.98 mV
g_pre  (HRS)   = 7.494e-10       MAD 0.235 decade
g_conn (LRS)   = 2.658e-07       MAD 0.084 decade
g_conn/g_pre   = 347.7
```

logistic-in-exposure 对识别态 `x` 的拟合：中位 RMS **0.0597**（p90 0.107）。

---

## 2. 卡住的地方（**下一个接手的人从这里开始**）

### 2.1 现象

用「双包络 + 状态插值」重建正向支电流：

```
I(V_h) = x·g_conn·sinh(α·V_h) + (1−x)·g_pre·sinh(α·V_h)
```

得到的**中位 RMSE = 0.585 decade**（p90 0.669），太大，不能用。

### 2.2 根因诊断（已定位）

打印逐点残差可以看到：**低压段（`V_hBN` < 1.0 V）拟合很好（ratio ≈ 0.95–1.0），高压段彻底崩掉**（ratio 一路涨到 62）。

具体地：
- 状态 `x` 在 `V_hBN` ≈ 1.1 V 处就饱和到 1.0；
- 饱和之后 `sinh` 包络继续指数增长，而**实测电流已经平了**（`I_measured` 封顶约 50 µA）；
- 所以是**负载线钳位**在起作用，不是包络形状错了。

验证：把饱和后的 `x=1.0` 代进负载线求解，模型给出的 `V_hBN = 1.048 V`、`I = 58.9 µA`；而实测是 `V_hBN` 涨到 1.76 V、`I` 停在 49.9 µA。

**这说明 `sinh` 形式不适合做 SET 侧的 connected 包络** —— 它在高压外推太猛。

### 2.3 三条可选修法（未验证，按推荐度排序）

1. **给 connected 包络换成钳位形状**。RESET 侧 v0.11 已经把 HRS 从 `sinh` 换成了有效 PF 形式 `A·V·exp(β√V)` 并收益 64%；SET 侧大概率也要走同一条路。最自然的候选是**饱和型**：`I_conn = I_sat·tanh(V/V_k)`，或者直接把晶体管饱和电流 `I_cc` 作为期望上限喂进负载线求解，让模型自己解出 `V_hBN`。

2. **扩大包络拟合窗口**。当前 `_set_branch_mask` 的 `conn` 支上限是 `high_v=1.20`，可能把过渡区一起圈进去了。先检查 `high_v` 取 0.8–1.0 是否更干净。

3. **检查 `alpha_pre` 是否被高压点污染**。`pre` 支上限 `0.35` 看着合理，但值得扫一下 0.25/0.35/0.5 看 MAD 怎么变。

> 建议顺序：先做 3（最便宜），再做 2，最后做 1。

---

## 3. 已定架构决策（不要推翻，除非有强理由）

用户已拍板的两条：

1. **SET 状态表示 = 混合方案**：**标量 `x` 为核**（与 RESET 的 `s` 对称、可从数据直接识别、验证方法可完全复用），**K-patch `φ_k` 为装饰**（只服务 3D UI 和 CAFM 先验，不参与电气验证）。K-patch 必须**均值钉死**：`Σ w_k·θ(φ_k) ≡ x`，钉法通过解一个公共平移量实现，**不要硬设 `φ_k = x`**（那样会丢掉异质性）。
2. **首里程碑只做 1.1 V 闭环**。跑通验证后再横向扩 0.9 / 1.3 V。

---

## 4. 剩余待办

| # | 任务 | 状态 |
|---|---|---|
| 1 | `set_identification.py` | ✅ 已写完，参数已标定 |
| 2 | 修掉 2.1 的包络问题 | ❌ **卡在这里** |
| 3 | `model_v12.py`：自主 SET 标量核 + K-patch 钉子 + 负载线 | ⬜ 未开始 |
| 4 | `run_full_cycle`：SET 终态 → RESET 初态直接交接（移除 `transfer_law` 默认路径） | ⬜ 未开始 |
| 5 | `validate_v12.py`：留一循环 SET RMSE + **按电压分箱** + 闭环连续性 | ⬜ 未开始 |
| 6 | `tests/test_model_v12.py` + 跑通全量测试 | ⬜ 未开始 |
| 7 | `README_v0.12.md` + 更新 `__init__.py` 导出 | ⬜ 未开始 |

---

## 5. 复用清单（别重复造轮子）

- `vulcan2d/transistor.py` — `load_default_lookup()`（正向） / `load_reverse_lookup()`（反向），`TransistorMapping.voltage_for_current()` 是去嵌入的核心。
- `vulcan2d/model_v7.py` — `reset_hbn_current()`、`divider_reset()`（负载线二分），以及 `reset_hrs_basis()` 里那个 **`transport` 分支写法**，v0.12 可以直接复制这个模式来切换 SET 包络形式。
- `vulcan2d/validate_v5.py` — `_median_branch()`、`_loop_rmse()`、`_as_curves()`、`_load_cycles()`。
- `vulcan2d/validate_v11.py` — `_branch_diagnostics()`（**分箱 RMSE，直接复用**）、`_split_half_floor()`（实验噪声下限）。
- `vulcan2d/validate_full_cycle.py` — `_load_paired_cycles()`、`_fit_reset_parameters()`、`GATE_VOLTAGE`。
- `vulcan2d/cycle_link.py` — `pair_cycle_sheets()` 用于 S_i/R_i 配对。

---

## 6. 必须守住的诚实边界

这些是项目一贯的自我约束，新代码不能破：

- `x` 是「可导电连接比例」，**不是**空位浓度、细丝半径或 patch 占据数。
- SET / RESET 阈值保持**独立随机变量**（数据不支持同一原子势垒）。
- **不引入真实时间常数** —— 只有 DC 数据，给不出开关时间、脉冲宽度、等待时间。
- `exposure` 是协议相关的代理量，**不是**能量或电荷（没记录扫描速率）。
- 第一阶段只在**已测栅压窗口 + DC 协议内**成立，不外推到新器件、新温度、新扫描幅值。

---

## 7. 其他提醒

- **v0.7–v0.11 至今全部未 commit**（`git status` 里一堆 untracked）。动手加 v0.12 之前建议先提交，否则新改动会和旧改动混在一个 diff 里没法审查。
- 注意输入文件：`1T1M写入.xlsx` 是 SET（53 个 S_i sheet），`1T1M擦除.xlsx` 是 RESET（53 个 R_i sheet）。`0.9V栅压*.xlsx` / `1.3V栅压*.xlsx` 是另外两组栅压。
- Python 环境：`/usr/bin/python3`，numpy 2.0.2 + scipy 1.13.1 + pandas + openpyxl 都在。
- RESET 侧（v0.11）**已经验证好，一行都不要动**。它的指标是 RESET 正向 0.176 / 回程 0.124 decade。

---

## 8. 复现我这轮结论的最小脚本

```bash
cd /path/to/VULCAN-2D
python3 -c "
import numpy as np, warnings; warnings.filterwarnings('ignore')
from vulcan2d.transistor import load_default_lookup, TransistorMapping
from vulcan2d.reset_identification import load_reset_workbook
from vulcan2d import set_identification as SI
pos = load_default_lookup()
m = TransistorMapping(lookup=pos, gate_gain=1.0953794419765472,
                      gate_offset_v=-0.3090066462755203,
                      current_scale=1.0, series_resistance_ohm=0.0)
raw = load_reset_workbook('1T1M写入.xlsx')
cycles = [SI.reconstruct_set(v, i, 1.1, m) for _, v, i in raw]
a, mad = SI.fit_shared_pre_alpha(cycles)
states = [SI.identify_set_state(c, a) for c in cycles]
print('alpha_pre', a, 'mad', mad)
print('e50', np.nanmedian([s.e50_exp for s in states]))
print('g_conn/g_pre', np.median([s.g_conn_a/s.g_pre_a for s in states]))
"
```

---

## 9. 给接手者的第一句话

> 先跑第 8 节的脚本确认能复现，然后去第 2.3 节，从**方案 3**（扫 `pre` 支的 `high_v`）开始。目标是把第 2.1 节那个 0.585 decade 的重建误差压下去 —— 那个数字不降下来，后面 `model_v12.py` 建在沙子上。
