# 两个预选 soft SET 候选：完整单圈条件前向检查

固定共同谱、Ron/gc/JS与第4圈外扫0.2V锚。先用同一有符号物理ODE从0.2V反积到0V求初态，再完整0→5→0；返扫沿累计路径继续演化，没有冻结低压段。每候选只做指定名义与加密精度，无拟合、额外seed、RESET或跨圈历史。

| 候选 | 精度 | 状态 | 全500点RMSE/dec | 外扫低压10点 | 两支低压20点 | 其余480点 |
|---|---|---|---:|---:|---:|---:|
| beta4 | nominal | failed_preserved | — | — | — | — |
| beta4 | refined | completed_conditional_single_cycle_forward | 0.7747359 | 2.975207 | 2.104612 | 0.6638274 |
| beta6 | nominal | completed_conditional_single_cycle_forward | 0.698574 | 2.189857 | 1.549471 | 0.6389862 |
| beta6 | refined | failed_preserved | — | — | — | — |
| beta4 | nominal | completed_conditional_single_cycle_forward | 0.7747358 | 2.975205 | 2.104611 | 0.6638273 |
| beta6 | refined | completed_conditional_single_cycle_forward | 0.6985739 | 2.189857 | 1.54947 | 0.6389862 |

完成4个完整运行，失败2个；含首次接口检查的总计算3.505s，预算120s。状态：completed_two_seed_precision_comparison_with_preserved_numerical_failures。

首次接口检查因exp(log5)向下舍入而误拒5V（0.0356s、0个ODE）；原源码/计划/结果完整保留于pre_circuit_domain_fix。修正为原生log域比较，并独立约束晶体管压降及夹根。随后四个运行完整保留于pre_inactive_endpoint_fix，其中两个完成，两个仅因返扫末端A≈1e−13V超出表低压分辨率停止。

只恢复两次失败精度，未重跑两条已完成轨迹。恢复时用I_T(A,u)≤F(vmin,u)证明低于表电压分辨率的正部驱动严格为0，未给域外电流赋0、未夹电压或状态；上界为正则仍停止。详细触发次数、上界及状态范围见JSON。累计六个求解尝试仍只有两个物理seed，全部耗时从120s预算扣除。

低压组定义为0<|V命令|≤0.2V；原始非零电压、|I|≥1e−11A及有限观测的mask保持500点。没有减零点，没有替换或删去低偏残差；条件锚也纳入评分，故不是独立测试误差。

JSON保存反向/外扫/返扫求解器状态、初末态、0.2V锚重合误差、KCL旧新法比较、加密差异和逐点低压残差；NPZ保存所有完整原生轨迹及已完成求解段。越界或求解失败原样保留，不投影或继续扩域。

六个原SET返扫特征仅将此单圈值与原32圈聚合参考带并列，不声明聚合门槛通过。Backward初态来自本圈已测0.2V，因此仅检查指定锚附近存在的完整条件轨迹，不是识别真实初态，也不是前向因果初始化验证。

KCL预检36点；新旧电流最大log差1.58e-13dec，局部电压最大差7.98e-13V，KCL残差5.69e-19A；选择scalar_brent。

候选beta4：β=4，JS=5.08422985651 A，k=2.06644998893 V⁻¹。JS是log驱动参照值，有效阈值为JS exp(−βu)，不能当作实际限流。

候选beta6：β=6，JS=1938.54612161 A，k=0.890074437104 V⁻¹。JS是log驱动参照值，有效阈值为JS exp(−βu)，不能当作实际限流。

beta4加密：原生电流最大改变5.45e-06dec，状态最大改变3.16e-05，末态差-1.71e-07。

beta6加密：原生电流最大改变1.95e-06dec，状态最大改变9.5e-06，末态差4.46e-07。
