# VULCAN-2D 统一局部连通状态模型

日期：2026-09-22

## 1. 本轮解决的核心问题

此前电气模型能够在固定直流协议下分别预测 SET 和 RESET，但 SET 使用包络坐标
`x`，RESET 使用另一套包络坐标 `p`。两者都能拟合电流，却不是同一个跨极性保存
的状态，因此“数学模型”和“局域弱区逐步连接/断开”的物理解释没有完全打通。

本轮模型改为只保留一组局部状态

\[
z_k\in[0,1],\qquad \bar z=\sum_{k=1}^{K}w_kz_k,\qquad \sum_k w_k=1.
\]

`z_k` 表示第 `k` 个粗粒化局部区域的**有效导电连通程度**。它不是直接测得的
硼空位浓度、金属离子浓度或细丝半径。SET 增大同一组 `z_k`，RESET 减小同一组
`z_k`；电流也由这组状态直接计算。因此，状态、输运、空间解释和完整循环现在
属于同一个数学模型。

## 2. 电路层：晶体管和 h-BN 必须自洽分压

对每一个外加电压点求解

\[
|V_{\mathrm{app}}|=|V_{\mathrm{hBN}}|+|V_{\mathrm{DS}}|,
\qquad
I_{\mathrm{hBN}}(V_{\mathrm{hBN}},\mathbf z)
=I_T(V_{\mathrm{DS}},V_G).
\]

SET 使用 Nature Fig. 2b 的正向晶体管查表，RESET 使用新增独立晶体管的负向
查表。这样 h-BN 的内部电压不是人为指定，而是由当前 h-BN 状态和晶体管负载线
共同决定。Zhu 等人的 1T1M 结构和限流工作区为这种串联求解提供器件依据
[[1]](#参考文献)。

## 3. 状态如何同时生成 SET 和 RESET 电流

### 3.1 SET

正向的两个有效输运包络为

\[
I_{\mathrm{pre}}^+(v)=A_{\mathrm{pre}}v
\exp(\beta_{\mathrm{pre}}\sqrt v),\qquad
I_{\mathrm{conn}}^+(v)=A_{\mathrm{conn}}v^n,
\]

其中 `v=|V_hBN|`。每个局部区域的电流为

\[
I_k^+(v,z_k)=(1-z_k)I_{\mathrm{pre}}^+(v)
+z_kI_{\mathrm{conn}}^+(v),
\]

器件总电流是 `w_k` 加权并联和。

SET 的准静态扫描驱动量仍采用

\[
\Xi=\int |V_{\mathrm{hBN}}|\,\mathrm d|V_{\mathrm{app}}|,
\qquad
\bar z_{\mathrm{eq}}^+(\Xi)=
\left[1+\exp\left(-\frac{\Xi-\Xi_{50}}{w_\Xi}\right)\right]^{-1}.
\]

升压支只允许 `z_k` 增加，回扫支冻结。这里 `Xi` 的单位是 V²，只是固定扫描
协议下的电压域应力坐标，不是时间、能量或离子迁移速率。

### 3.2 RESET

负向两个有效包络为

\[
I_{\mathrm{LRS}}^-(v)=G_L\sinh(\alpha_Lv),\qquad
I_{\mathrm{HRS}}^-(v)=A_Hv\exp(\beta_H\sqrt v),
\]

而局部电流使用**同一个** `z_k`：

\[
I_k^-(v,z_k)=z_kI_{\mathrm{LRS}}^-(v)
+(1-z_k)I_{\mathrm{HRS}}^-(v).
\]

平均内部场

\[
E_{\mathrm{hBN}}=10|V_{\mathrm{hBN}}|/d_{\mathrm{nm}}
\quad(\mathrm{MV/cm})
\]

驱动 RESET 存活比例

\[
\bar z_{\mathrm{eq}}^-(E)=
\left[1+\exp\left(\frac{E-E_{50}}{w_E}\right)\right]^{-1}.
\]

负向升压支只允许 `z_k` 减小，回扫支冻结。SET 最后一个局部状态向量被逐元素
传给 RESET，第一个 RESET 点不重新初始化为 1。

## 4. 为什么分布模型与端口计算不再矛盾

由于局部电流对状态采用同一组两包络线性混合，任意极性都满足

\[
\sum_kw_k[(1-z_k)I_0+z_kI_1]
=(1-\bar z)I_0+\bar zI_1.
\]

因此：

- `K=1` 是端口平均场模型；
- `K>1` 是同一模型的空间粗粒化展开；
- 两者在给定 `bar z` 时给出完全相同的端口电流；
- 局部可视化不是另一套“只负责讲故事”的模型；
- 现有端口 I–V 只能约束 `bar z`，不能确定局部区域的真实先后顺序。

当前 `K=10` 的 `w_k` 由 CAFM 投影导电斑块面积分布构造：把斑块按面积排序，
分成十个近似等数量分组，再以各组总投影面积归一化。这个先验只约束几何
异质性；CAFM 与目标器件并非同批样品，不能据此宣称目标器件恰有十根细丝，
也不能把面积直接等同于局部导电能力。局域导电和弱区的文献依据可参见 Chen、
Ranjan 和 Ducry 等工作 [[2–4]](#参考文献)。

## 5. 跨极性幅值因子怎样解释

现有配对数据仍需要把正向连接包络幅值换算为负向初始 LRS 幅值：

\[
G_L=\eta\,
\frac{A_{\mathrm{conn}}V_{\mathrm{ref}}^n}
{\sinh(\alpha_LV_{\mathrm{ref}})}.
\]

统一状态后，`eta` **不再也不应该被叫作状态保留率**。状态本身通过 `z_k`
连续传递；`eta` 只表示极性、电极、晶体管工作象限和读出公式不同所造成的
输运幅值换算。它不能被解释成金属离子保留比例或硼空位迁移效率。

## 6. 严格验证结果

验证规则与原 53 对全循环审计完全一致：预测目标循环时，目标 SET 和 RESET
电流均不进入该折参数拟合；它们只用于最终评分。运行时输入仍只有 SET 电压
波形、RESET 电压波形和栅压。

| 评估集合 | SET 逐点 pooled RMSE | RESET 逐点 pooled RMSE | SET 中位回线（升/回） | RESET 中位回线（升/回） |
| --- | ---: | ---: | ---: | ---: |
| 全部 53 对，同协议 RESET | 0.167 decade | 0.370 decade | 0.078 / 0.085 | 0.183 / 0.125 |
| 原训练规则可用的 51 对 | 0.166 decade | 0.293 decade | 0.078 / 0.085 | 0.192 / 0.124 |
| 原规则拒绝的 R2/R3 两对 | 0.180 decade | 1.200 decade | 0.083 / 0.085 | 1.569 / 0.368 |

空间与连续性数值审计结果：

- `K=10` 与 `K=1` 的最大端口电流相对差：`1.11e-13`；
- `bar z=sum(w_k z_k)` 最大恒等误差：`0`；
- SET 末态到 RESET 初态的局部向量最大差：`0`；
- SET 升支单调增加、RESET 升支单调减少均通过；
- 新增与相关回归测试共 28 项通过。

这说明本轮的收益是**物理—数学结构统一，而且不牺牲已有预测精度**；它没有
凭空解决 R2/R3 的异常 RESET。全部 53 对的 0.370 decade 比 51 对的 0.293
decade 更诚实地反映当前模型面对所有原始循环时的误差。

## 7. 现在可以声称什么、不能声称什么

可以声称：

1. 在已验证的 1.1 V 栅压、+5 V SET、−1.7 V RESET 直流协议内，模型可从
   电压序列自主给出完整 1T1M I–V 回线。
2. 同一组局部连通状态生成端口电流并贯穿 SET/RESET，不再使用两套互不相干
   的状态解释。
3. 分布式空间表示与端口紧凑模型是严格的平均场关系，而非两套矛盾的模型。
4. 该解释与 h-BN 局域弱区、缺陷辅助传输、限流控制工作区等文献相容。

不能声称：

1. 已经证明 `z_k` 是硼空位、Ti/Ag 离子或某种具体原子细丝的数量；
2. 单温直流 I–V 已唯一证明 Poole–Frenkel、hopping 或其他输运机制；
3. 已经预测脉宽、驻留时间、扫描速率或真实开关时间；
4. CAFM 面积权重已经标定目标器件中每个局部通道的电流和阈值；
5. 部分 SET 后的 RESET 已经得到实验验证。模型在数学上支持该路径，但现有
   标定循环都接近完成 SET。

## 8. 对论文和下一步工作的意义

本轮不是为了增加一个更复杂的拟合器，而是形成可写入论文的主模型：

> 晶体管与 h-BN 自洽分压决定内部应力；异质 h-BN 中若干局域弱区由一个连续
> 连通状态向量描述；正向扫描使局部连接增长，反向扫描使同一连接衰减；局部
> 电流的加权并联和严格给出端口电流。缺陷桥或金属辅助路径是与文献相容的
> 候选微观实现，但现有电学数据不唯一识别原子物种。

接下来最重要的不是再引入一套独立物理故事，而是：

1. 在论文中给出上述统一方程、目标排除验证和异常循环；
2. 把 `K=1` 与 `K=10` 的严格等价、CAFM 权重边界和 SET→RESET 状态连续性
   做成主图/补充图；
3. 用消融对照判断 `eta` 幅值桥、SET 电压曝光和 RESET PF 类回支各自带来多少
   可验证收益；
4. 若以后取得部分 SET→RESET 或脉冲数据，再标定局部顺序和真实时间动力学。

## 参考文献

1. Zhu et al., “Memristive hardware accelerators for artificial intelligence,”
   *Nature* 618, 57–62 (2023). https://doi.org/10.1038/s41586-023-05973-1
2. Chen et al., “Wafer-scale integration of two-dimensional materials in
   high-density memristive crossbar arrays,” *Nature Electronics* 3, 638–645
   (2020). https://doi.org/10.1038/s41928-020-00473-w
3. Ranjan et al., “Conductive atomic force microscope study of bipolar and
   threshold resistive switching in 2D hexagonal boron nitride films,”
   *Scientific Reports* 8, 2854 (2018).
   https://doi.org/10.1038/s41598-018-21138-x
4. Ducry et al., “The role of defects in the resistive switching mechanism of
   hexagonal boron nitride,” *npj 2D Materials and Applications* 6, 58 (2022).
   https://doi.org/10.1038/s41699-022-00340-6
5. Shi et al., “Electronic synapses made of layered two-dimensional materials,”
   *Nature Electronics* 1, 458–465 (2018).
   https://doi.org/10.1038/s41928-018-0118-9

## 可追溯产物

- 实现：`vulcan2d/model_unified.py`
- 自动测试：`tests/test_model_unified.py`
- 验证程序：`vulcan2d/validate_unified.py`
- 全部审计：`analysis/unified_local_connectivity_validation.json`
- 逐循环记录：`analysis/unified_local_connectivity_validation_cycles.csv`
- 曲线档案：`analysis/unified_local_connectivity_validation_curves.npz`
- 验证图：`analysis/figures/24_unified_local_connectivity_validation.png`
