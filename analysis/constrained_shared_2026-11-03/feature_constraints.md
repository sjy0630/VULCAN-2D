# 统一1R/1T1R模型：下一轮拟合前的实测约束

仅使用固定训练集：裸1R前14个原编号记录；1T1R第4–35圈。未读测试数组，未读旧模型预测或拟合参数。所有量都是端口曲线特征；局部斜率不等同于冻结状态的本构导数。

## 建议先冻结的验收口径

1. 两接法共用材料电流方程、状态方程及参数；晶体管和仪器限流只进入外电路。实际hBN压降必须参与演化。
2. 1T1R连续传播状态，不能每圈重置。裸R只按已声明的条件SET/RESET配对处理，不假定表间时间或同一cell。
3. 以下主特征的模型中位数应同时落入预先固定的验收区间：通常为训练经验Q10–Q90；平台波动幅度只设经验Q90上限，允许平滑模型更平。逐项列失败，不能以总RMSE抵消。该范围不是置信区间。
4. 同时报告真实逐记录RMSE、各分支及裸SET非限流/限流分数；保留记录散布。两圈替代目标不能称为完整顺序损失。
5. 电学守恒和数值收敛通过后再谈精度。通过电学约束仍不等于证明微观机制。

## 关键联合形状

1T1R SET返扫的端口指数从低压超线性转为高压次线性：0.1–0.2V指数中位1.428，1–2V为1.210，2–4V为0.580；后一区间Q10–Q90仅0.542–0.618。2V与4V电流中位分别29.555µA、44.127µA，必须与0.2V的0.753µA同时复现。
RESET外扫1.0–1.4V窗口的端口电流近似平台：窗口几何平均电流中位2.679µA，端点指数中位0.00147；但每条曲线有波动，不能把它解释成已经证明的静态恒流元件。整个RESET峰中位3.713µA。
以上都是不依赖旧材料参数的端口观测。正向SET的VG=1.1V晶体管曲线来自1.0/1.5V之间插值；反向表含1.1V节点，但该事实不证明是同一只晶体管。内部Rh随电流上升与否受相应映射影响，不能作为无条件材料否决条件。正向尺度0.8或有效栅压1.03V只能作为有记录的假设敏感性情景，不能称为已知±20%校准误差。

## 主特征训练范围

| 特征 | 中位数 | 验收区间 | 单位 |
|---|---:|---:|---|
| `1r.set.outward.I_2V` | 5.45175e-08 | 1.88541e-08 – 1.87685e-07 | A |
| `1r.reset.return.I_0p2V` | 7.67438e-11 | 2.13399e-11 – 4.77003e-10 | A |
| `1t1r.set.return.I_0p2V` | 7.5301e-07 | 5.42976e-07 – 1.10537e-06 | A |
| `1t1r.set.return.I_2V` | 2.9555e-05 | 2.8515e-05 – 3.02394e-05 | A |
| `1t1r.set.return.I_4V` | 4.4127e-05 | 4.35902e-05 – 4.44812e-05 | A |
| `1t1r.reset.outward.I_0p2V` | 4.3708e-07 | 2.50626e-07 – 6.61772e-07 | A |
| `1t1r.reset.return.I_0p2V` | 7.89955e-10 | 3.00877e-10 – 3.5152e-09 | A |
| `1t1r.set.return.alpha_0p1_0p2V` | 1.42755 | 1.17903 – 1.68692 | dimensionless |
| `1t1r.set.return.alpha_1_2V` | 1.2101 | 1.07865 – 1.46756 | dimensionless |
| `1t1r.set.return.alpha_2_4V` | 0.579567 | 0.542395 – 0.618344 | dimensionless |
| `1t1r.reset.outward.alpha_1_1p4V` | 0.00146948 | -0.867601 – 0.664196 | dimensionless |
| `1t1r.reset.outward.geomean_current_1_1p4V` | 2.67851e-06 | 1.9035e-06 – 4.02668e-06 | A |
| `1t1r.reset.outward.log_current_q90_q10_span_1_1p4V` | 0.13718 | 0 – 0.16922 | decade |
| `1r.reset.outward.peak_current` | 0.00504561 | 0.00310618 – 0.00990941 | A |
| `1r.reset.outward.peak_abs_voltage` | 0.5875 | 0.475 – 0.795 | V |
| `1t1r.reset.outward.peak_current` | 3.71315e-06 | 2.65342e-06 – 5.98616e-06 | A |
| `1t1r.reset.outward.peak_abs_voltage` | 1.12 | 0.706 – 1.66 | V |
| `1r.set.transition_midpoint_at_0p995mA` | 2.675 | 2.59 – 2.775 | V |
| `1r.set.return.R_0p05V` | 91.0853 | 69.1557 – 187.68 | ohm |
| `1r.reset.outward.R_fit_0p05_0p2V` | 93.4801 | 71.2007 – 207.751 | ohm |

电流单位A。完整逐记录值、跨越区间、原始索引、被floor筛出的数量及全部辅助特征均在同名JSON中。

## 曲线散布参考

| 数据/相位 | 经验逐点log散布RMS | 共同确定曲线总RMSE参考√2×散布 |
|---|---:|---:|
| 1r set | 0.414340 | 0.585965 |
| 1r reset | 0.724623 | 1.024771 |
| 1t1r set | 0.110612 | 0.156429 |
| 1t1r reset | 0.204020 | 0.288528 |

上述参考来自训练记录变化，不是不可约测量噪声。循环状态模型可能解释其中一部分；应使用偏差、观测散布、预测散布和协方差的精确分解。平台点极小散布不应被当作材料精度硬指标。

## 数据处理边界

裸RESET最初−25mV点没有删掉；主低阻读数用−0.05至−0.2V，同时保存包含−25mV的拟合敏感性。零偏及低于既有floor的电流仍在原始记录；特征统计显示筛选数。没有把floor当成实测电流，也没有插值生成点。

裸SET的阈值给出最后低于阈值与首次达到阈值的电压区间；不能把区间中点解释成精确的原子势垒或材料演化停止阈值。SET限流平台不能用于直接推断本征硬态电阻。

## 假设登记

下列条目记录依据、使用需求与失效后的处理；没有替研究者确认其“必要/简化”属性。

### A1

Use recovered Git workbooks as current experimental records; native cache values are unchanged.

依据：Recovered workbook/native-cache equality audit; current training-array digest；用途：Data provenance。证据：No interpolation, clipping or resampling in this script。若不成立：Rebuild constraints from corrected original source.

### A2

Treat the regular voltage column as nominal terminal command for circuit modeling; separately expose actual hBN voltage under compliance.

依据：Native regular ramps and1mA plateau; voltage header AV；用途：Avoid driving material with5V while reporting1mA through~100ohm。证据：Strong circuit consistency, not independent instrument readback documentation。若不成立：Use verified readback voltage and measurement model; preserve both columns.

### A3

Bare SET has1mA compliance; bare RESET is not clipped at1mA.

依据：Prior confirmed protocol and raw RESET peaks above1mA；用途：Correct external circuit closure。证据：Raw peak/current distributions in this report。若不成立：Replace with documented instrument settings; do not change intrinsic law by topology.

### A4

Measured branch slopes are trajectory features; they do not establish a frozen hidden state or a constitutive derivative.

依据：No state-resolved measurements accompany these sweeps；用途：Prevent overinterpreting sublinear slopes/plateau。证据：Same-polarity hysteresis is explicitly measured。若不成立：Use independent retention/pulse constraints before asserting frozen-state laws.

### A5

1T1R numbered SET/RESET pairs4-35 are treated as an ordered sequence; first3 remain outside primary normal-operation targets.

依据：Existing confirmed S/R pairing and advisor exclusion；用途：State continuity and fixed training scope。证据：Native labels retained。若不成立：Revise sequence metadata; never infer missing elapsed time.

### A6

Bare same-label SET-to-RESET pairing is conditional; chronology between bare sheets and physical cell identity are not asserted.

依据：Workbook labels; absent independent chronology/cell map；用途：Avoid invented cross-record state history。证据：Low read consistency is evidence of compatibility only。若不成立：Use verified cell/sequence metadata, retaining raw labels.

### A7

Same nominal stack/geometry motivates a shared material equation; individual microscopic states need not be identical.

依据：Zhu Nature Methods and SI chip layout, previously verified；用途：Define the unification contract。证据：Common nominal Au-Ti/hBN/W geometry, distinct physical cells。若不成立：Document measured structural differences before allowing corresponding parameters.

### A8

No physical dwell time, sweep speed, thermal transient or cell identity is inferred from voltage-point spacing.

依据：Available records contain voltage/current only；用途：Bound interpretation of any dynamics。证据：No time columns used。若不成立：Add measured time/protocol data before calibrating rates in seconds.

### A9

Historical phase current floors are reporting/censor thresholds, not certified instrument detection limits.

依据：Existing1e-11A SET/1e-12A RESET scoring convention；用途：Comparable finite log scores。证据：Censored raw values remain present and counts are reported。若不成立：Repeat with documented noise/offset model and preserve threshold sensitivity.

### A10

Empirical Q10-Q90 feature bands and within-index scatter are descriptive training variability, not confidence intervals or irreducible noise.

依据：Fixed repeated-record feature definitions before next fit；用途：Data-scaled prospective acceptance proposal。证据：No iid/population or new-device coverage claim。若不成立：Use validated stochastic/state model and independent experiments for population claims.

### A11

The positive SET transistor mapping at nominalVG=1.1V is interpolated between1.0V/1.5V gate curves. The reverse lookup contains a1.1V node; this does not independently establish same-transistor identity. Internal hBN resistance trends are conditional on the appropriate mapping.

依据：Independent direct lookup inventory: SET gates0.5,1.0,1.5,...; reverse table0.1V gate spacing includes1.1V；用途：Separate raw terminal-shape evidence from deembedded-material inference。证据：Reviewer reports SET mapping scale~0.8 or effectivegate~1.03V can change the inferred internal resistance trend; these are assumption tests, not established calibration uncertainty bounds。若不成立：Keep raw terminal feature gates; report polarity-specific mapping-conditioned inference and profile justified gate interpolation before introducing new material physics.

