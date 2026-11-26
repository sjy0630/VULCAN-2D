# 固定 m=2 恢复项 forward kernel：准备记录

状态：仅准备和语法检查；未导入、加载缓存、求根、积分、拟合或扫描参数。有限校准计划由 root 后续单独预声明。此候选尚未得到数值或物理验证。

## 来源与唯一动力学变更

复制自冻结的 `analysis/reversible_family_2026-11-03/forward_kernel.py`，来源 SHA256：`4382a56449b0b736a963356b1257873f49e0cf3125f74d6f2ec6f9aa865f0f17`。

新 `forward_kernel.py` SHA256：`052971a4ee8e927769118a3bccac323c7d5ad9a81e08d0e5caf1fe7a38181302`。

唯一数值动力学变更为：

- 原式：`u_prime = kappa*v*(exp(lambda*v) - I*expm1(u)/J)`。
- 新式：`u_prime = kappa*v*(exp(lambda*v-u) - I*expm1(u)/J)`。
- 等价 q 式：`q_prime = kappa*v*((I/J)*(1-q) - exp(lambda*v)*q**2)`。

m固定为2，表示 q 方程恢复项的二次阶数，未加入 Parameters，不按圈改变，也不是本阶段可优化的第四参数。λ、J、κ仍为全部三个显式动力学参数；参数域基本约束保持 λ>=0、J>0、κ>=0，后续有限搜索域另由 root 决定。任何机制解释仍是待验证假设；二次项不证明两个具体粒子参与反应。

q=exp(−u)、Gamma=gamma_c/expm1(u) 与原谱读出原样保留。q是有效谱宽坐标，不改称实测电子占据率。所有电路、电流、原生数据、参数结构、积分方法和评分代码均沿用冻结父核；新增内容仅是候选名称、固定m/方程元数据与近零界说明。

## 保持的接口与条件轨迹

- `Parameters(lambda_per_v, J_positive_a, kappa_per_v2)`：无默认种子。
- `Precision(name, rtol, atol, max_step)`：显式精度。
- `ForwardKernel.load()`：仅在后续调用时读取相同cache、冻结Ron/Gamma_c和原.2V锚；导入不执行加载。
- `validate_circuit(...)`：显式36点fast/严格42步二分与原legacy比较；准备期间未调用。
- `evaluate(..., circuit_mode=..., check_budget=...)`：输出summary/arrays/原500残差；root负责每次及全阶段硬预算、每次失败落盘和停止。
- `audit_native_legacy(...)`：显式后验审计，准备期间未调用。

轨迹仍从固定外扫.2V锚积分到5V，真实顶点末态连续传给返扫；同一ODE另由.2V逆向到0补齐低压外扫评分。原503个原生点、500个评分点、250/250分支、低压点、三实际端点、六参考、首次跨越、连续性和KCL诊断全部保留，无隐藏重置。异常仍通过KernelFailure携带参数、根失败、最后RHS及已有段数组；caller_budget_exhausted需由外层停止相应预算。

## 近零界与数值域

保持原保守局部界：

`B = 0.5*kappa*vmin**2*(exp(lambda*vmin) + pi*gamma_c/(Ron*J))`。

因为允许域内u>0，新恢复项 `exp(lambda*v-u)<=exp(lambda*v)`，原界仍适用，无需放宽或换界。只有B<1e−12且状态/谱宽合法时才允许0<A<vmin的有界RHS遗漏；这不是精确不活动或全轨迹误差界。A=0仍在验证状态后取真实I=u_prime=0。

0<u<=80、Gamma在原有限表内、正局部电压在表内的限制保持。不夹状态/电流/电压、不外推、不使用域外电流。fast和严格二分保持合法根括区，legacy仅在其原路径合法后作显式参考。

## 实际检查

`compile(source, path, 'exec')`纯语法检查通过；没有执行模块。独立只读源代码差异审查 `/root/shared_kernel_review` 已通过：唯一动力学变更及 q² 等价关系正确，m未进入Parameters，原近零界保守，电路/评分/连续初态未变，脚本哈希一致。该审查没有执行模块，也不等同数值或微观机制验证。全部旧阶段、旧runner和生产默认保持冻结。本准备不宣称统一1R/1T1R模型、RESET、跨圈历史或聚合验收通过。
