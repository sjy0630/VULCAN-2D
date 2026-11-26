# 固定载流子供给因子：先验必要预算计划

计划先于运行保存。候选在共同 SET/RESET 速率前都乘 |I|/I*，I*=1 mA
固定为单位，不拟合指数；beta=1、A_RESET=0，谱形和材料参数族不变。
这是“构型尝试频率受载流子供给限制”的有效假设，尚非已验证机制。

- gamma_c={0.2,0.4,1.0} eV，alpha={20,50}，共6情景。
- JS/JR使用beta1筛查中的既有有限规则；JS保留实际初态2.8V启动及公共低阻
  目标的stop barrier上界，允许JS>Icc。初态各只锚定一次。
- 不运行ODE或优化，不扩大alpha/gamma_c/谱形域，120秒上限。
- 复用beta阶段softSET、bareRESET、softRESET及同记录慢段谱矩阵；只新算
  一次14条裸跃迁的128分片谱电流矩阵，供全部gamma_c/JS共享。

## SET 必要界

每条裸窄括区，用V_after与gamma∈[g_pre,upper,g_after,lower]分128片：

    I_upper=min(Icc,F(V_after,g_hi))
    u_upper=ln(1+gamma_c/g_lo)
    h_upper=(I_upper/I*)[ln(I_upper/JS)+u_upper]_+
    k_jump,lower=Σ(delta_u/h_upper)/delta_commandV

某个非零状态片的h_upper=0则为stop barrier。不能把低电流阶段跳过，或把
整个跃迁都用Icc驱动。主预算只采用14条最宽容的跃迁下界。2→2.8V到达
和旧低读预算仍作为辅助下界，不能只以正启动力代替达到限流。

softSET高压返扫的超驱动左积分乘缓存I_lower/I*，给k_series上界，需与
bare参考速率及同一alpha相容。同记录5%缓变/跃迁条件另外记录为诊断，
不自动加入完整曲线的硬验收门槛。

## RESET 必要界

按 `conditional_clock_2026-11-03/SAME_RECORD_BOUND_REVIEW.md` 第10节：
soft最大驱动使用max[(I_upper/I*)[ln(I_upper/JR)+u_upper]_+]；bare低读
左积分使用ΣdeltaV(I_lower/I*)[ln(I_lower/JR)]_+。这里bare RESET没有
1mA限流，I*仅为归一化。SET/RESET共用同一alpha。

无穷速率上界表示必要预算未约束，必须单列，不能当完美通过，也不是数学
失败；有限下界配无穷上界允许有限速率。按本轮预定选择规则，仅将两极性的
正有限上下界同时相容者列为已选条件种子；无穷上界情景另记“必要区间相容
但上界欠约束、未选择种子”，不能写成“无交集”。仍不作轨迹/机制验证。

## 停止和解释边界

六情景全部记录，若无种子即停止。保留共同路径率/实际电压/理想SET限流
等前提，以及公共Q90读预算不等于每条实测点的范围说明。
正的供给因子不改变正向态单调增加或同态电压速率排序，因此既有Fermi窗口
和跨拓扑时间比较不会自动消失。旧产物保持不变。
