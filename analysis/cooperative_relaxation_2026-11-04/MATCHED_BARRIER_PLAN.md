# 唯一匹配m1的保存角点屏障：执行前计划

状态：**只准备，尚未执行**。`matched_linear_barrier.py` 已做纯语法compile，未导入，也未将新λ/J代入保存数组。脚本SHA256：`ee61e9687d1e9d1ae37e243c1028bfd9fa79fe9048772a6171c5f55acf687d05`。由root与shared只读复核后，root才决定执行；本准备不自动启动计算。

## 唯一问题与不变条件

实际匹配m1的两次forward已在有限Γ表域外停止，原失败完整保留，不重试或扩表。本检查只问：旧initial_barrier保存的固定qbar电路角点，能否为**这一个实际映射**提供早期不可上穿前缀及原500点必要误差下界？结果不替代已失败的完整轨迹，也不把数值域失败改写成物理反证。

参数必须直接读取 `matched_linear_control.plan.json:mapped_m1_parameters`，并与实际control报告及两次尝试中的parameters逐项相同；不得手填、重映射、重新选qref、调λ/J/κ或扫描。qbar只取旧initial_barrier对应case，没有新反演。

四个冻结输入在准备时的SHA256：

| 输入 | SHA256 |
|---|---|
| 旧 `initial_barrier.json` | `fda98bde2b03ca5849341b3b87860c7a94ac45a9edd9143fb2ffa12e60257139` |
| 旧 `initial_barrier.npz` | `d44323a203b1b803c8b6d05583fe7c7d5e3a5f568d032100093e2522905a0bda` |
| 本阶段 `matched_linear_control.plan.json` | `bfbe5cae79977db0a5fdeab6d826e29ff402a12e102c4d879cfd4081ba5e2fea` |
| 本阶段 `matched_linear_control.json` | `7fcc3b7f244e03443d72ac3eeb065bd80335ad3f674cfeb2cd41c1cd45ea78b5` |

执行前核查两套已保存来源链的当前哈希、共同谱表/晶体管/观测缓存哈希、control已通过的谱/电路/锚/掩码逐数组一致性、相同Ron/Γc，以及control真实.2V条件锚与旧名义qbar的一致性。0.3%包络qbar须仍上包同一条件入口；该包络不是实验噪声或新精度认证。

## 精确输入字段与四个条件

仅原 `eta0_nominal`、`eta0_refined`、`eta003_nominal`、`eta003_refined` 四个前缀，原0.01/0.005 V网格、原0/0.003材料包络，不新增网格或节点。

每格从旧NPZ读取：

- `*_strong_node_current_upper_a[1:]` 作为Imax；
- `*_weak_node_voltage_upper_v[1:]` 作为vmax；
- `*_strong_node_voltage_lower_v[:-1]` 作为vmin；
- `*_command_nodes` 作为原命令格边界；
- `*_certified_native_mask`、`*_optimistic_current_upper_a` 作为已保存原生评分域与电流上界；
- 通用 `command_voltage_v`、`observed_current_a`、`original_scoring_mask`、`outward_branch_mask`。

准备时仅查看压缩包成员名，所需28个成员均存在；没有加载它们执行新参数运算。角点只依赖相同固定qbar、谱/Ron和电路，不依赖旧动力学λ/J，因而可重新用于这一映射。

## 唯一算术与理论

对于m1，`q_prime=kappa1*R`，其中

`R=v*((I/J1)*(1-q)-exp(lambda1*v)*q)`。

λ1>=0、J1>0、κ1>=0时，每格在q=qbar处使用

`cap=vmax*Imax*(1-qbar)/J1`

`loss=vmin*exp(lambda1*vmin)*qbar`

`guard=1e-12*(cap+loss)+1e-18`

`U=cap-loss+guard`。

沿从.2V开始的连续格，仅在U有限且严格<0时认证；第一个未知、非有限或非负上界立即停止。此时连续、局部唯一且可接受的解从q<=qbar不能向上穿过边界。这个条件证明只使用边界处的已保存电路角点，不构造新轨迹，不宣称域外延拓已验证。

将新的认证前缀与**旧认证前缀和旧原生mask取交集**。即使新U在更远的已有格为负，也不扩展原生电流上界保存域。只在交集内复用Iceil并计算：

`floor=sqrt(sum(max(log10(abs(Iobs)/Iceil),0)**2)/500)`。

所有其它原500评分点的误差乐观设0；不改分母、不删低压点、不替换电流底噪。若没有正长度认证前缀，floor写null，明确“无结论”，不以边界单点或0数值制造结论，不扩域补救。

只与已经存在的m2同条件成绩和旧sinh同条件成绩比较，不增加新验收门槛。不预判前缀、下界或比较符号。本检查只约束该固定映射与读出/初态条件，不排除整个m1家族，不证明微观协作。

## 执行预算、产物与停止

- 保存输入检查与JSON计划后，四条件算术总预算10秒，信号超时；无自动重试。
- 新KCL求根=0，反演=0，ODE=0，拟合=0，参数扫描=0，谱积分=0。
- 输出 `matched_linear_barrier.plan.json/.json/.npz/.md`；保存新cap/loss/guard/U、使用mask、残差下界数组及来源哈希。
- 全部旧文件保持冻结；既有mapped control失败记录不更改。
- 结果交shared独立保存数组复核。没有前缀就到此停止，有界证据完成后也不继续拓展m1搜索，转向已设计的RESET4→SET5 signed连续闭环。
