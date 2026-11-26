# 固定 m=2 协作恢复候选：有界积分初始化

状态：**方案及脚本待 root 与 reviewer 确认，尚未执行。** 日期：2026-11-04。旧阶段全部只读。本阶段不改变旧结论：此前 m=1 可逆候选的有限三参数条件拟合 RMSE 为 0.345118456 decade，未优于同条件 sinh 基线的 0.182230640 decade。

## 固定函数假设及唯一数学改动

本次明确指定新的固定恢复函数，不再在结果出来后选择指数：

\[
\frac{dq}{ds}=\kappa v\left[\frac{I}{J_+}(1-q)-e^{\lambda v}q^2\right],
\quad q=e^{-u}=\frac{\Gamma}{\Gamma+\Gamma_c},\quad v=|V_h|.
\]

m=2 固定，不是第四个拟合参数；后续候选的有效动力学参数仍只有 λ、κ、J+。这是一项可被检验的函数假设，`cooperative` 仅为候选名称，**不能据此声称已证明双粒子过程、微观协作机制或实际陷阱占据率**。

由 u=−log q 直接得到

\[
\frac{du}{ds}=\kappa v e^{\lambda v-u}-\frac{\kappa}{J_+}vI\operatorname{expm1}(u).
\]

严格复用上一阶段积分初始化方法，只将发射/恢复积分由 E=∫v exp(λv)ds 改成

\[
E_i(\lambda)=\int_i v e^{\lambda v-u}\,ds.
\]

捕获积分和系数映射完全不变：

\[
C_i=\int_i vI\operatorname{expm1}(u)\,ds,
\qquad\alpha=\kappa\ge0,\quad\beta=\kappa/J_+\ge0.
\]

不改固定材料谱、晶体管表、初态条件、观测子集、权重、λ 域或 seed 规则，不增加平行导电项。不把任何动态点指定成平衡点。

## 唯一观测来源与时间顺序

输入为冻结的 `analysis/soft_drive_structure_2026-11-03/DRIVE_STRUCTURE_INVERSION.json`，SHA256：
`f56c1405b503179233db1131675732a5abed76191806990020121778dafa9fe5`。

仅用其中 26 条既有 `observations` 的 branch、native_index、command_v、material_voltage_v、current_a、u、gamma_ev 和固定 Ron/Gamma_c 身份信息。它们是同 cycle 4 SET、固定读出谱与固定晶体管下的条件反演状态，不是直接测得的内部状态，也不是本候选预测。新增脚本还将记录上一阶段初始化源码的哈希作为方法来源，但不导入或执行该源码。

按每个 branch 的 `native_index` 排序。外扫 0.2→5 V、返扫 5→0.2 V，各含 13 点和 12 个非零区间，共 24 区间。s 为分支内累计 |ΔA|，返扫时同样递增，不额外翻转速率符号。κ 的单位 V⁻²；扫描/停留时间未知，不能换算成真实秒钟速率。

两个 5 V 原观测分开保留在各自分支端点：不合并、不强制同 u、不对零 Δs 作除法，也不视为可允许的瞬时状态跳变。实际 forward 后续仍需单一连续峰态。本初始化不验证峰值连接、0→0.2 V 历史、RESET、裸 1R、跨圈传播或缺失圈。

## 与上一阶段相同的替代路径、损失和回归

每个区间对 u、v、log I 随归一化位置 t 作线性插值，再 I=exp(log I)。插值保证 u>0、I>0，但中间三者一般不满足同一个 KCL/材料读出。它是条件替代路径，低积分残差不证明实际动力学可达；求积加密只检验此插值的积分准确性，不能证明插值真实。

令 y_i=Δu_i，保持 24 区间平均速率错配等权：

\[
L(\lambda)=\frac1{24}\sum_{i=1}^{24}
\left[\frac{\alpha E_i(\lambda)-\beta C_i-y_i}{\Delta s_i}\right]^2.
\]

报告 sqrt(L)，单位 V⁻¹，及全部积分残差、各分支 RMS/最大绝对值/残差总和。它不是原 500 点 log I RMSE、路径长度加权损失、噪声似然或实际末态误差。不得拿它与 0.345 或 0.182 decade 直接作数值比较。

每 λ 用同一双列非负回归：列范数归一化，无约束双列最小二乘可行解，加两条轴边界与原点的有限活动集枚举。数值秩相对容差仍为 1e−12；保存归一奇异值、条件数、列相关、KKT 残差、未约束解及选中的活动集，最优解若落零边界也原样保留。仅 α>0、β>0、数值秩为 2 才映成有限 κ=α、J+=α/β。零边界不加 epsilon，不伪造有限正参数。

## 冻结范围与上限

- λ 固定为 0、0.05、…、2.00 V⁻¹，共 41 点。
- 这是有限有效探索域。在被动电路 0≤v≤5 V 时 exp(λv)≤exp(10)；新 E 额外含 q=e⁻ᵘ≤1，不增加该数值上界。该范围不是微观能垒、实验置信区间或完整物理可行域。端点最优不触发扩域或细化。
- 固定 Gauss–Legendre 8 点与 16 点，16 点为主结果，8 点只检查求积敏感性。总计 82 次二系数线性回归、23616 个 λ 相关求积节点，不追加插值方法或第三套求积。
- 所有 E/C 的 8→16 相对差最大值须≤1e−6，才把数值上可用 seed 标为 `ready_for_forward_initialization=true`；否则仍保存候选并停止等待 root。该门槛只是求积自检，不是实验误差或模型准确性标准。
- 输入校验、来源哈希和声明计划落盘后计时，计算上限 30 秒。超时/失败保留记录，不自动重启、加点或换 seed。
- 脚本无 `--run-authorized` 即拒绝执行；父代理及 reviewer 确认前只作源码语法/差异审查。

## 至多三项初始化及公平比较接口

完全沿用三个固定 λ 分段 `[0,2/3)`、`[2/3,4/3)`、`[4/3,2]`。每段取 16 点主求积 L 最小的 profile 点；分数完全相同时选较小 λ。如果该段最优点是零边界、退化或非有限映射，则该段不产 seed，不用次优替补。完全相同的 `(λ,κ,J+)` 去重，重复不补位。Gauss 8 不另选 seed，只报告分段赢家及主 seed 参数/损失变化。所有 41×2 profiles 都保存。

`seeds[]` 与前阶段保持同一接口：`seed_id`、`band/bin`、`selection`、`profile_index`、`lambda_per_v`、`kappa_per_v2`、`J_positive_a`、`initial_u_at_command_0p2_v`、`initial_q_at_command_0p2_v`、`average_rate_mismatch_rms_per_v`、`ready_for_forward_initialization`。新增标识 `fixed_recovery_exponent_m=2`，seed_id 前缀为 `cooperative_`。条件初态仍取同一个已保存外扫 0.2 V 状态，不拟合或按圈重设。

本次仅准备/授权后运行积分初始化；未来至多三个原 500 点 forward、一次有限三参数 fit 和精度验证仍由 root 单独安排。此脚本没有 ODE、KCL、反演、非线性拟合或保留集读取入口。

## 产物与结论边界

准备产物：本文件及 `cooperative_initialization.py`。授权后仅在本 stage 生成 `cooperative_initialization.plan.json/.json/.npz/.md`，拒绝覆盖任意已有输出。保存完整 26 点、24 区间、两套积分、回归/seed 记录和来源哈希，旧阶段不修改。

积分初始化成绩只约束固定 m=2、固定谱、指定插值/权重/λ 域的替代路径。不能据此宣称新候选优于基线、统一 1R/1T1R 已完成、实际保持性成立或微观机制已经识别，也不能凭有限域错配普遍排除所有协作恢复形式。所有新增函数和插值假设明确属于本次方法选择，未登记为用户已经证实的实验事实。
