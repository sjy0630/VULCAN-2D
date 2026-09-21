# VULCAN-2D 理论基础、文献证据与论文写作边界

日期：2026-07-14

2026-09-20 核查提示：本文保留历史建模思路，不能作为 v0.12/v0.13 的执行方程说明。当前原文证据、竞争机制和参数检查以 [物理判别记录](PHYSICS_DISCRIMINATION_2026-09-20.md) 为准；文中不可逆损伤、具体缺陷桥等为待检验解释。

配套文献库：[vulcan2d_theory.bib](../references/vulcan2d_theory.bib)

导师提出的 1R/1T1R、二维限域、CAFM、XTEM 与金属离子/硼空位组合逻辑，见
[文章证据链路线图](PAPER_EVIDENCE_CHAIN_2026-07-14.md)。

朱凯晨博士论文及其与 Mario Lanza 团队 h-BN 工作的专项核查、工作区统一
解释和新增预测，见
[朱凯晨/Lanza h-BN 物理机制综述](ZHU_LANZA_HBN_PHYSICS_REVIEW_2026-07-15.md)。

## 1. 核心结论

VULCAN-2D v0.3 可以形成一条自洽的物理解释链：

> h-BN 中空间分布的局部弱区，在电场作用下逐步转入高传输构型；这些构型
> 可以是本征缺陷桥，也可以包含沿晶界或硼空位进入的金属离子。多个横向
> 微区的电流并联叠加；CMOS 晶体管通过串联负载线限制电流过冲，使路径停留
> 在部分连接、可逆的软退化区。若限流和瞬态能量继续增大，局域路径可能发展
> 为一根或少数完整导电纳米细丝；循环中的阈值和阻态漂移可能与不可逆缺陷
> 积累或可逆历史效应有关，当前数据尚未区分。

这条解释与目标 1T1M 器件、h-BN 软击穿、原子缺陷桥和一般跳跃输运理论
相容，但不是所有环节都已由当前数据唯一验证。最稳妥的论文定位是：

**这是一个由物理机制约束、由实验统计标定的降阶模型，而不是已经完成微观
机理鉴定的第一性原理模型。器件端的渐进转换也不能单独证明微观上绝对
“非丝状”；局部金属通道和本征缺陷桥都应保留为候选。**

证据分为三级：

- **A级，直接证据**：相同 6 nm h-BN/CMOS 1T1M 器件或本课题数据。
- **B级，材料级证据**：h-BN 的 CAFM、TEM、DFT 或隧穿实验，但器件结构不同。
- **C级，理论/类比证据**：一般跳跃理论、其他非丝状 RRAM 或可靠性模型。

## 2. 建议在论文中使用的总模型

为避免把电流前因子误称为小信号电导，论文中建议用
`A_s(phi)` 表示状态相关的电流尺度：

```text
A_s(phi) = I_s^(s) sum_k w_k [exp(xi_H,k)(1-Theta(phi_k))
                               + G_on,eff Theta(phi_k)]
I_hBN = A_s(phi) sinh(alpha_s V_h)
```

其中 `s` 表示 SET/RESET 极性，`sum_k w_k=1`，`phi_k` 是第 `k` 个面积微区
处于高传输缺陷构型的占据率，范围为 `[phi_min,1]`。当前默认
`Theta(phi)=phi`；这代表并联面积混合，不应称为已经验证的渗流临界律。

状态动力学可统一写成两态主方程：

```text
dphi_k/dt = k_+,k(V,T)(1-phi_k) - k_-,k(V,T)(phi_k-phi_min)
```

在正向 SET 时 `k_+ >> k_-`，在反向 RESET 时 `k_- >> k_+`，便分别得到
代码当前使用的两个极限：

```text
SET:   dphi_k/dt = g_k(1-phi_k)
RESET: dphi_k/dt = r_k(phi_min-phi_k)
```

晶体管与 h-BN 由基尔霍夫关系自洽耦合：

```text
V_app = V_h + V_tr
I_hBN(V_h,phi) = I_tr(V_tr,V_G)
```

这三组方程分别描述输运、内部状态和 1T1M 外部约束，构成完整的忆阻系统。

## 3. `sinh(alpha V_h)` 的理论推导

### 3.1 详细平衡

考虑局域态之间的热激活跳跃。无场势垒为 `E_a`，局部电化学势降为
`Delta mu`。对称势垒近似下，正、反向跃迁率为

```text
Gamma_+ = nu exp(-E_a/k_B T) exp(+Delta mu/2k_B T)
Gamma_- = nu exp(-E_a/k_B T) exp(-Delta mu/2k_B T)
```

净通量因此为

```text
Gamma_+ - Gamma_-
  = 2 nu exp(-E_a/k_B T) sinh(Delta mu/2k_B T)
```

若局部势降近似为 `Delta mu = q a_eff E = q a_eff V_h/d`，则

```text
I = I_0(T,phi) sinh[q a_eff V_h/(2 k_B T d)]
alpha(T) = q a_eff/(2 k_B T d)
```

Riess 和 Maier 从局域热平衡出发给出了相同形式的对称跳跃电流方程，且允许
驱动力超出线性响应范围。因此，`sinh` 是本模型中理论依据最强的函数之一，
不是纯粹为了拟合而引入的经验式 [@Riess2008Hopping]。

### 3.2 拟合参数的物理量级

目标器件 h-BN 厚度约 `d=6 nm`。在 300 K 下 `V_T=k_B T/q=25.85 mV`，
由 `a_eff=2 alpha V_T d` 得到：

| 极性 | 拟合 `alpha` | `a_eff` |
|---|---:|---:|
| SET | 6.747 V^-1 | 2.09 nm |
| RESET | 7.666 V^-1 | 2.38 nm |

这个长度明显大于 h-BN 单层间距约 0.33 nm。因此 `a_eff` 更适合解释为
“若干局域态组成的有效电势降区段”或粗粒化关联长度，不能解释为一个电子的
最近邻原子跳距。若强行采用单层跳距，理论 `alpha` 约为 1.06 V^-1，与拟合值
不符；这提示实际电势可能集中在部分层、缺陷团簇或界面附近。

### 3.3 能证明什么，不能证明什么

- 能证明：`sinh` 与满足详细平衡的场辅助跳跃/缺陷辅助输运相容。
- 不能证明：输运一定是 TAT、Poole-Frenkel、单一声子跳跃或某一种具体缺陷。
- 当前 HRS 数据中 `sinh` 的对数拟合优度约 0.998；Schottky 型拟合也较接近，
  因此必须依靠变温数据比较 `alpha proportional to 1/T`、激活能和势垒降低规律。

## 4. `phi_k` 的物理含义与并联面积模型

### 4.1 状态变量不是“第 k 根细丝”

建议把 `phi_k` 定义为：

> 第 `k` 个粗粒化面积微区内，能够显著提高面外传输的活性缺陷构型、缺陷桥
> 或缺陷团簇的归一化占据率。

这一定义容纳三类材料证据：

1. h-BN 在反复电应力下逐步产生缺陷、SILC 和软击穿，电导连续上升
   [@Ranjan2020Localized; @Ranjan2023Molecular]。
2. DFT/NEGF 表明，多空位可以重构为可电控的层间桥，单个桥可使传输提高
   两个数量级以上 [@Ducry2022AbInitio]。
3. h-BN 隧穿结中可直接看到纳米尺度缺陷充电造成的库仑阻塞和电流台阶，
   说明禁带内局域态确实参与面外输运 [@Chandni2015DefectTunneling]。

原子尺度局域桥的存在并不等于整个器件只能由一根宏观导电细丝控制。多个
局域桥或缺陷团簇若依次参与，宏观上仍可表现为渐进的器件级软退化响应。
反过来，平滑 I-V 也不能证明原子尺度完全没有局域路径。当前 patch 模型正是
连接原子尺度局域性与器件尺度渐进性的粗粒化假设；完整 CNF/QPC 只应在出现
量子化电导、欧姆极低阻或突变断裂等证据后升级为主模型
[@Roldan2023Quantization; @Pazos2023Temporal]。

### 4.2 并联求和的来源

上下电极间不同横向位置承受近似相同的端电压，因此总电流为面积积分：

```text
I = integral_A J(r) dA
  approximately sum_k Delta A_k J_k(t_eff,k,c_k,phi_k)
```

Chen 等对 h-BN forming 的分析明确指出，局部电流同时由缺陷区域面积 `A_k`
和该位置的有效绝缘厚度 `t_eff,k` 决定，可写成 `I=J(t_eff)A`
[@Chen2020WaferScale]。因此当前 `w_k` 应称为**有效贡献权重**，近似包含
`A_k g_k`：`A_k` 是几何面积，`g_k` 汇总缺陷深度、局部势垒和电极接入条件。

当前 `K=10` 是对面积积分和阈值分布的数值离散，不代表器件里恰好有 10 根
细丝。新增 CAFM 的表观斑块密度若只作数量级外推，`0.053 um2` 面积对应约
10.3 个斑块，与 `K=10` 同量级；但因样品和成膜方法不同，这只能交叉检查
粗粒化阶数，不能当作真实通道计数。`w_k` 的 Dirichlet 分布仍是统计假设。

### 4.3 线性混合为何是当前最保守选择

当微区电流并联时，活性面积分数的一阶混合自然给出
`Theta(phi)=phi`。只有在获得器件面积、横向连通性或临界标度数据后，才适合
引入 `phi^p`、有效介质或渗流临界指数。当前 `p_perc=1`，论文中应称为
“面积混合指数”，不要把它包装成已验证的 Bruggeman 或渗流模型。

## 5. 状态动力学的物理来源

### 5.1 两态主方程与有界性

若一个微区包含未激活态 `0` 和高传输态 `1`，占据率为 `phi`，则标准两态
主方程就是

```text
dphi/dt = k_+(1-phi) - k_- phi
```

引入不可恢复残余态 `phi_min` 后，恢复项改为
`-k_-(phi-phi_min)`。这一写法自动给出饱和和有界性，解释了为何 SET 在
`phi -> 1` 时减慢、RESET 在 `phi -> phi_min` 时减慢。代码采用指数积分，
因此数值上对任意步长都保持状态范围。

### 5.2 场加速速率

对于受电场偏置的热激活构型跃迁，同样由正反向跃迁率可得

```text
k_net proportional to exp(-E_a/k_B T) sinh(gamma q a E/k_B T)
```

Ducry 等人的 DFT 计算显示，电场和缺陷电荷态会改变 h-BN 多空位重构及层间
桥形成的能垒，给场控制的构型速率提供了材料级依据。更直接地，Chen 等在
h-BN 阵列补充材料中采用 `Delta G approximately G^gamma sinh(beta V)` 描述
离子型非易失存储器的指数电压动力学 [@Chen2020WaferScale]。其他非丝状
忆阻器模型也有相同形式 [@Wang2015Nonfilamentary]。

当前代码使用

```text
g_k,r_k proportional to sinh[beta_s(|V_app|-V_th,k)]_+
```

它是把复杂的局部场、能垒和观测时间窗压缩成阈值后的降阶形式。这里有两个
必须写清的边界：

- `beta_set/reset` 是有效场加速系数，不是从第一性原理提取的原子跳距。
- 状态目前由端电压 `V_app` 驱动，而输运由内部电压 `V_h` 驱动。这样可避免
  软击穿后 `V_h` 因晶体管分压下降而让状态冻结，但它仍是经验近似。获得独立
  晶体管输出曲线后，应改为局部 h-BN 场或其时间积分驱动。

### 5.3 当前激活能不可辨识

代码中的 `E_a,set=1.00 eV`、`E_a,reset=0.92 eV` 只在偏离 300 K 时参与
热加速；在当前单温数据中它们与速率前因子完全不可分离。因此论文不能把这
两个值写成“实验提取的激活能”，只能写为待变温实验标定的占位参数。

## 6. Weibull 阈值与随机性

h-BN 击穿实验多次报告 Weibull 电压统计、层间短接和缺陷优先位置
[@Hattori2015Layer; @Ranjan2021Breakdown]。这支持把局部弱点/能垒分布映射为
随机阈值。目标数据也显示 `V_SET` 和 `V_RESET` 的 CV 分别约 25.0% 和
24.1%，说明单一确定阈值不够。

但 Weibull 不是唯一分布，也不能仅凭 53 个循环证明最弱链统计。RRAM 文献
中还存在缺陷聚类模型，并指出普通 Weibull 有时不能完整描述 SET 统计
[@Raghavan2016Clustering]。因此当前 Weibull 应表述为“与 h-BN 击穿统计相容
且能复现本数据的低参数分布”，而不是已确认的失效物理。

当前最合理的层次结构是：

```text
固定器件：局部阈值偏移 dtheta_k、面积权重 w_k、HRS 缺陷景观 xi_H,k
逐循环：中心阈值 V_th,c、LRS 增益波动 xi_on、晶体管/测量噪声
```

它把 D2D 的静态空间结构与 C2C 随机性分开。由于只有一个器件，模型目前只
标定了固定器件内的 C2C；不能据此预测跨器件良率。

## 7. 晶体管负载线为何是机制的一部分

目标 Nature 论文直接指出，约 90% 的独立 `0.053 um2` h-BN 器件表现为
不稳定波动且没有稳定阻变；少数器件在不低于 1 mA 的限流下进入丝状双极
阻变，但耐久约百次。串联 CMOS 晶体管能够精确限制 h-BN 电流、避免过冲并
显著提高耐久性 [@Zhu2023Hybrid]。因此

```text
V_app = V_h + V_tr,  I_hBN = I_tr
```

不是外围电路细节，而是目标器件工作机理的一部分。它解释了：

- LRS 电流在晶体管负载线上钳位，`I_cc` 的 CV 只有约 1%；
- h-BN 进入高传输态后内部电压下降，限制进一步硬击穿；
- 阻态变化很大时，端口电流变化仍可保持受控。

朱凯晨博士论文、Lanza 团队的低/高限流和热分析进一步表明，栅压、限流及
瞬态能量能够把器件从部分连接的软路径推向完整 CNF/QPC 区
[@Zhu2023Thesis; @Zhu2019Tristate; @Lanza2022Temperature]。因此晶体管的
物理作用应写成“选择并稳定工作区”，而不是“证明器件绝对没有细丝”。

代码中的 `tanh` 晶体管公式只是无独立输出曲线时的平滑经验负载线，不是
MOSFET 物理模型。论文可以强调串联自洽求解，但不应声称该 `tanh` 公式由
BSIM 或器件几何推导得到。

## 8. 循环损伤项的解释与边界

反复电应力导致 h-BN 中电荷俘获、SILC、缺陷累积和逐层软击穿，这为一个
缓慢不可逆损伤变量 `D` 提供材料依据 [@Ranjan2020Localized;
@Ranjan2023Molecular]。当前模型采用

```text
D_(n+1) = D_n + kappa(|Delta phi_SET|+|Delta phi_RESET|)
V_th,SET = V_th,SET,0 + c_E D
phi_min = phi_floor + c_f D
G_on,eff = G_on exp(xi_on-c_g D)
```

三项分别对应：更难激活、RESET 后残余泄漏增加、LRS 高传输增益下降。它们的
符号与 53 轮数据中的 `V_SET` 上升、`R_HRS` 下降和 `R_LRS` 上升一致。

但是线性阈值漂移和指数电导衰减是最低阶、保持正值的经验律，不是文献唯一
推出的形式。尤其 `G_on` 随损伤下降可能同时包含缺陷重构、接触变化和晶体管
工作点漂移。论文应称其为“经验耐久性状态”，并用更长循环序列验证函数形式。

## 9. 逐项证据矩阵

| 模型项 | 主要物理解释 | 证据 | 等级 | 结论强度 |
|---|---|---|---|---|
| 1T1M 串联负载线 | 晶体管限流、抑制过冲 | Zhu 2023 | A | 强 |
| 渐进软转变 | 分布式缺陷逐步激活 | Zhu 2023；Ranjan 2020/2023 | A/B | 强 |
| `sinh(alpha V_h)` | 正反向跳跃的详细平衡 | Riess 2008；Chen 2020 h-BN 动力学；Wang 2015 | B/C | 方程强，具体缺陷弱 |
| 缺陷辅助输运 | 禁带局域态参与面外电流 | Chandni 2015 | B | 中强 |
| 高/低传输缺陷构型 | 多空位层间桥重构 | Ducry 2022；Ranjan 2023 | B | 中强 |
| 并联 patch | 局部弱区电流的离散积分 | 新增溅射 h-BN CAFM；Chen 2020；Shen 2021 | A/B | 中强 |
| `w_k` | 局部面积乘局部导电能力 | Chen 2020 的 `I=J(t_eff)A`；CAFM 面积分布 | B | 中强 |
| 软路径到 CNF 的工作区转变 | 限流/能量控制路径完整程度 | Zhu 2019/2023；Lanza 2022；Pazos 2023 | A/B | 强方向性 |
| `Theta(phi)=phi` | 活性面积线性混合 | 面积并联的一阶近似 | C | 中 |
| 两态状态方程 | 缺陷构型生成/恢复主方程 | 统计动力学；Ducry 2022 | B/C | 中强 |
| Weibull 阈值 | 弱点/能垒分布 | Hattori 2015；Ranjan 2021 | B | 中 |
| log-normal 阻态 | 多个乘性随机势垒/增益 | 本数据统计 | A/C | 中 |
| 耐久损伤 `D` | 不可逆缺陷和 SILC 累积 | Ranjan 2020/2023；本数据漂移 | A/B | 方向强、函数弱 |
| SET/RESET 非对称 | 电极、带阶和带电缺陷非对称 | Zhu 2023；Ducry 2022 | A/B | 中 |
| 热反馈 | 焦耳热改变跃迁率 | SIM2RRAM 文献 | C | 当前数据不可辨识 |

## 10. 核心文献阅读笔记

### 10.1 与目标器件直接相关

1. **Zhu et al., Nature 2023** [@Zhu2023Hybrid]：约 6 nm CVD h-BN、最大
   0.053 um^2 的 1T1M；晶体管限制过冲；HRS 约 200 Mohm、LRS 约
   200 kohm；渐进非线性转换被作者用于支持非丝状工作；耐久可达百万级。
   这是串联负载线、工作区间和器件尺度的最直接依据。
2. **Zhu 博士论文 2023** [@Zhu2023Thesis]：把 CAFM 热点、XTEM 缺陷、
   晶界辅助金属进入、硼空位、限流控制 S-LRS/LRS，以及 1T1M 抑制过冲放在
   同一研究链中。论文对部分较高栅压工作区也讨论 QPC/CNF，因此与最终
   Nature 文章合读后更支持“工作区随栅压/能量变化”，而不是绝对二分。
3. **Chen et al., Nature Electronics 2020** [@Chen2020WaferScale]：CAFM、
   XTEM、面积和限流实验显示 forming 从最弱原生缺陷开始；弱区宽度、深度及
   面积共同决定电流；1 uA 与 1 mA 分别对应部分和较完整的 CNF。补充材料的
   `Delta G approximately G^gamma sinh(beta V)` 直接支持本模型状态动力学。
4. **Shen et al., Advanced Materials 2021** [@Shen2021Defects]：h-BN 面外
   电流由最导电位置优先承担，说明 CAFM 热点和原子缺陷比皱褶等宏观形貌更
   直接地决定阻变位置。

### 10.2 h-BN 退化与缺陷统计

5. **Ranjan et al., ACS AEM 2023** [@Ranjan2023Molecular]：CAFM、TEM、
   TAT 模拟和 DFT 联合表明 h-BN 经历电荷俘获、SILC、软击穿；仅提高陷阱
   密度不足以拟合，需同时减小有效电学厚度；相邻层空位可形成分子桥。
6. **Ranjan et al., ACS AEM 2021** [@Ranjan2021Breakdown]：3 nm 单晶 h-BN
   的击穿电压呈紧密单峰 Weibull；重复应力增加电导，拟合的有效厚度常以
   约 0.3 nm 单层为单位变化；缺陷倾向在特定位置形成。
7. **Ranjan et al., ACS AMI 2020** [@Ranjan2020Localized]：2--5 nm 多层
   h-BN 的局域测量显示 progressive breakdown、连续软击穿和缺陷渐生；RTN
   指向渗流路径附近的离散陷阱；厚度不均匀时统计会偏离简单 Weibull。
8. **Hattori et al., ACS Nano 2015** [@Hattori2015Layer]：直接观察到 h-BN
   逐层击穿，Weibull 统计提示各向异性的缺陷形成。
9. **Wen et al., Advanced Materials 2021** [@Wen2021Encryption]：CVD 多层
   h-BN 中少数原子宽的无序天然缺陷产生稳定的面外 RTN 电流；文章的 CAFM
   图也是新增 PPT 中 CVD 对照图的来源。它支持“晶体区域包围并限域局部活性
   缺陷”，但与溅射样品的偏压和扫描条件不同，不能直接比较缺陷密度。
10. **Ranjan et al., Scientific Reports 2018** [@Ranjan2018CAFM]：CAFM 在
   h-BN 局部区域观察到双极性和阈值型阻变及自恢复，证明局部缺陷路径可以
   被电场开启和恢复。
11. **Xiao et al., Advanced Functional Materials 2017** [@Xiao2017CAFM]：
   用 CAFM 电流图和局部 I-V 区分突变丝状分量与分布式电导变化。该文研究
   TiO2/SiOx，只用于说明 CAFM 证据应如何解读，不作为 h-BN 微观机制证据。

### 10.3 原子缺陷与载流子输运

12. **Ducry et al., npj 2D Materials and Applications 2022**
   [@Ducry2022AbInitio]：DFT+NEGF 给出多空位的可控层间桥；电场和电荷改变
   构型能垒；桥接后电导可提高 100 倍以上。支持“高传输缺陷构型”，但研究
   结构与本器件不同，且其单缺陷转换较突变。
13. **Chandni et al., Nano Letters 2015** [@Chandni2015DefectTunneling]：
   h-BN 隧穿电阻随厚度近似指数增长，较厚结出现纳米缺陷单电子充电特征；
   退火后缺陷特征消失。支持局域态参与输运。
14. **Weston et al., Physical Review B 2018** [@Weston2018Defects]：混合泛函
   DFT 表明平衡生长条件下许多本征空位形成能较高，实际缺陷化学可能受 C、
   O、H 杂质主导。因此论文不能未经化学表征就把所有活性中心指定为 `V_B`
   或 `V_N`。

### 10.4 方程与建模方法

15. **Riess and Maier, PRL 2008** [@Riess2008Hopping]：从局域热平衡推导
   对称一般跳跃电流，给出电流正比于局部非平衡电导乘
   `sinh(Delta mu/2kBT)`，是本模型输运方程的核心理论支撑。
16. **Wang et al., Scientific Reports 2015** [@Wang2015Nonfilamentary]：
    另一材料体系中的均匀非丝状势垒调制模型；离子跳跃速度含 `sinh` 场项，
    连续性方程与状态相关势垒可解释渐进、饱和的 DC/脉冲响应。用于证明模型
    形式有物理建模先例，不能用来证明 h-BN 中是氧离子迁移。
17. **Raghavan, Microelectronics Reliability 2016**
    [@Raghavan2016Clustering]：用缺陷聚类描述 forming、SET 和 RESET 统计，
    提醒 Weibull 并非 RRAM 阈值统计的唯一方案。该文以丝状/渗流器件为主，
    在本课题中只作为统计对照。

### 10.5 老师提供的本地文献

- [2017 SIM2RRAM](../SIM2RRAM/2017%20SIM2RRAM.pdf)：支持把电子输运、缺陷
  状态和温度场耦合求解的总体框架，但主要实例采用 Poole-Frenkel/QPC 和细丝。
- [2013 RESET 模拟](../SIM2RRAM/2013%20%E6%A8%A1%E6%8B%9Freset.pdf)：支持
  RESET 中电热自洽和缺陷演化的建模方法，材料与几何并非目标 h-BN 1T1M。
- [2019 h-BN/SIM2RRAM](../SIM2RRAM/2019.pdf)：支持 h-BN 器件的随机性模拟，
  但其导电细丝假设不宜直接移植为本模型主机制。
- [2017 Xiao AFM](../SIM2RRAM/2017%20Xiao%20AFM.pdf)：展示 CAFM 如何通过
  局部 I-V、电流图和形貌图区分丝状突变与分布式电导变化；材料为 TiO2/SiOx，
  仅作表征方法参照。
- [2024 Roldan](../SIM2RRAM/2024%20J.%20Roldan.pdf)：支持 RRAM 变异性、
  hopping/TAT 候选机制和统计分析方法。
- `SIM2RRAM/QPC` 中的文献用于构造“丝状瓶颈模型”对照组；本课题没有量子化
  电导台阶或单通道证据，因此 QPC 不应成为当前主方程。

## 11. 建议的模型修正路线

### 11.1 现在即可采用的表述修正

1. 把 `G_ens` 在论文中改称 `A_s(phi)` 或“状态相关电流前因子”，因为其
   单位是 A，不是严格的小信号电导。
2. 把 `phi_k` 称为“活性高传输缺陷构型占据率”，不称为细丝半径。
3. 把 `p_perc=1` 称为线性面积混合，不宣称渗流临界行为。
4. 把 `sinh` 归因于详细平衡下的有效场辅助跳跃；TAT、PF、Schottky 保留为
   待变温实验排除的竞争机制。
5. 把 `D` 称为经验耐久性状态，不把 `c_E/c_f/c_g` 解释为材料常数。
6. 把 `w_k` 称为有效贡献权重，说明它近似包含弱区面积与局部导电能力。
7. 把“非细丝”限定为当前器件端口的高阻渐进工作区；保留低能软路径向
   CNF/QPC 演化的可能性。

### 11.2 有新数据后再改代码

1. **变温模型**：强制 `alpha(T)=alpha_0 T_0/T`，同时拟合
   `I_0(T) proportional to exp(-E_a/kBT)`；与 PF、Schottky、TAT 做全局
   模型选择，而不是逐温度单独拟合。
2. **局部场驱动**：获得独立晶体管 `I_D-V_DS-V_G` 后，用
   `E_h=V_h/d_eff(phi)` 或累计应力 `integral f(E_h,T)dt` 驱动状态。
3. **统一正反速率**：使用完整两态主方程，让 SET/RESET 两个速率在两种极性
   下都非零，可自然描述保持、回退和亚阈值扰动。
4. **层次随机模型**：多器件数据到位后，将固定 patch 参数建模为 D2D 随机
   层，将循环阈值和增益建模为条件 C2C 层。
5. **统计分布检验**：比较 Weibull、log-normal、generalized gamma 与缺陷
   聚类模型，用 AIC/BIC、Q-Q 图和留一器件验证选择分布。

## 12. 可直接用于论文的物理解释段落

> 本文将 h-BN 有效面积离散为若干并联微区，并以状态变量 `phi_k` 表征第
> `k` 个微区中高传输缺陷构型的占据率。该状态不等同于宏观导电细丝，而是
> 对局域缺陷态、缺陷团簇及可能的层间缺陷桥进行粗粒化描述。h-BN 的 CAFM、
> TEM 与第一性原理研究表明，电应力可逐步生成缺陷并诱导相邻层之间的高传输
> 构型，从而降低有效面外输运势垒。不同微区的阈值和权重不均匀，使其依次
> 激活并在器件尺度形成渐进电阻转变。

> 在给定状态下，载流子电流采用对称场辅助跳跃形式
> `I=A(phi)sinh(alpha V_h)`。该形式可由正、反向热激活跳跃速率的详细平衡
> 推出，其中 `alpha` 表征局部电化学势降相对于热能的有效耦合。拟合得到的
> 有效长度大于单层间距，说明 `alpha` 应理解为多局域态区段或非均匀电势降的
> 粗粒化参数，而不是单一原子跳距。由于现有数据仅包含室温准静态扫描，本文
> 不对 TAT、Poole-Frenkel、Schottky 或其他缺陷辅助机制做唯一归因。

> CMOS 晶体管与 h-BN 串联后共同满足电压分配和电流连续条件。晶体管在软
> 击穿发生后限制电流过冲，并通过负载线降低 h-BN 内部电压，从而抑制不可逆
> 硬击穿，使局部缺陷或金属辅助路径停留在部分连接的高阻区。提高栅压、
> 限流或瞬态能量时，这些路径仍可能继续发展为完整 CNF/QPC。因此，该机制
> 解释了实验中稳定的电流钳位和较低的限流波动，也把渐进软退化与高电流
> 丝状阻变统一为由外部约束选择的不同工作区。

## 13. 论文中应避免的过度表述

不要写：

- “实验已经证明电流由 TAT 主导。”
- “器件内有 10 个导电细丝/渗流通道。”
- “Weibull 证明了最弱链击穿机制。”
- “1.00 eV 和 0.92 eV 是拟合得到的 SET/RESET 激活能。”
- “热效应对当前回线起决定作用。”
- “模型能够预测任意温度、脉宽、面积和器件。”

建议写：

- “数据与分布式缺陷辅助、场辅助输运和渐进软退化相容。”
- “patch 是面积积分的粗粒化离散，不预设宏观细丝几何。”
- “器件处于晶体管控制的高阻渐进工作区；这不排除纳米尺度局域桥。”
- “Weibull 是与 h-BN 击穿统计相容的低参数阈值模型。”
- “微观输运和真实激活能仍待变温与动态实验辨识。”

## 14. 最小实验闭环

要把上述解释从“文献相容”提升为“本器件已验证”，建议按优先级补充：

1. **多温度 HRS/LRS I-V**：检验 `alpha proportional to 1/T`，提取激活能，
   区分 hopping、PF、Schottky 与 TAT。
2. **多扫描速率/多脉宽**：识别真实时间常数和 `k_+/-`，检验主方程。
3. **独立晶体管输出曲线**：获得真实内部 `V_h` 和局部场，减少参数补偿。
4. **多面积、多器件**：检验电流面积标度、阈值面积标度并分离 D2D/C2C。
5. **不同限流/栅压**：验证晶体管负载线对软击穿、窗口和耐久的因果作用。
6. **更长耐久序列**：验证 `D` 的线性/指数经验律是否需要饱和或多时间尺度。
7. **局域表征**：若条件允许，用 CAFM、TEM/EELS 或噪声谱判断活性区是分布式
   缺陷团簇还是少数局域通道。

## 参考链接

- [Zhu et al., Nature 618, 57--62 (2023)](https://www.nature.com/articles/s41586-023-05973-1)
- [Kaichen Zhu, PhD Thesis (2023)](https://hdl.handle.net/2445/193137)
- [Chen et al., Nature Electronics 3, 638--645 (2020)](https://www.nature.com/articles/s41928-020-00473-w)
- [Shen et al., Advanced Materials 33, 2103656 (2021)](https://doi.org/10.1002/adma.202103656)
- [Pan et al., Advanced Functional Materials 27, 1604811 (2017)](https://doi.org/10.1002/adfm.201604811)
- [Lanza et al., Advanced Electronic Materials 8, 2100580 (2022)](https://doi.org/10.1002/aelm.202100580)
- [Pazos et al., Advanced Functional Materials, 2213816 (2023)](https://doi.org/10.1002/adfm.202213816)
- [Roldan et al., Applied Physics Letters 122, 203502 (2023)](https://doi.org/10.1063/5.0147403)
- [Riess and Maier, Physical Review Letters 100, 205901 (2008)](https://pubmed.ncbi.nlm.nih.gov/18518555/)
- [Ranjan et al., ACS Applied Electronic Materials 5, 1262--1276 (2023)](https://pubs.acs.org/doi/10.1021/acsaelm.2c01736)
- [Ranjan et al., ACS Applied Electronic Materials 3, 3547--3554 (2021)](https://pubs.acs.org/doi/10.1021/acsaelm.1c00469)
- [Ranjan et al., ACS Applied Materials & Interfaces 12, 55000--55010 (2020)](https://pubs.acs.org/doi/10.1021/acsami.0c17107)
- [Hattori et al., ACS Nano 9, 916--921 (2015)](https://pubmed.ncbi.nlm.nih.gov/25549251/)
- [Ducry et al., npj 2D Materials and Applications 6, 58 (2022)](https://www.nature.com/articles/s41699-022-00340-6)
- [Chandni et al., Nano Letters 15, 7329--7333 (2015)](https://pubmed.ncbi.nlm.nih.gov/26509431/)
- [Weston et al., Physical Review B 97, 214104 (2018)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.214104)
- [Wang et al., Scientific Reports 5, 10150 (2015)](https://www.nature.com/articles/srep10150)
- [Raghavan, Microelectronics Reliability 64, 54--58 (2016)](https://www.sciencedirect.com/science/article/pii/S0026271416302839)
- [Wen et al., Advanced Materials 33, 2100185 (2021)](https://doi.org/10.1002/adma.202100185)
- [Ranjan et al., Scientific Reports 8, 2854 (2018)](https://www.nature.com/articles/s41598-018-21138-x)
- [Xiao et al., Advanced Functional Materials 27, 1700384 (2017)](https://doi.org/10.1002/adfm.201700384)
