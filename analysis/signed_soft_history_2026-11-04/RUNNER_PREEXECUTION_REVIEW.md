# Signed 连续历史 runner 执行前审查

结论：**通过准备源码审查，可由 root 按既定有限计划执行唯一一次**。这不是实际数值通过或物理有效性结论。本 reviewer 没有 import/执行 runner 或 kernel，没有新 KCL、反演、ODE、b 扫描或拟合。

已审源码：

- `signed_forward_kernel.py`：`30b3df1eeb9aa22e2f8d3e8827b2950b97a38e14a4700d829edcbae6b4c57d6e`。
- `bounded_signed_trial.py`：`fef8ca0b145a765fc7705bfee4d9766773167f68292901dd5f516ec06482c03f`。
- 理论审查文档已冻结：`5aaa76faf1184ab9bf372eb5f6c2ab6ff15cce8a72c318ad47d90ccbf2f50d7e`。

## 范围与参数选择

只有 RESET4 原 170 点残差的 MSE 进入目标函数与 best 候选；SET5 的观测虽被预先加载作校验与最终评分，但不进入 objective、候选选择或 b 搜索。先运行 b=0、ln10 两个固定端点，再进行唯一一次最多18目标调用的 bounded 搜索；同时有20次总搜索调用、25个相位ODE尝试的硬限制。没有新的谱、参数域扩展、重启或反演。

正场参数和状态接口保持固定。唯一正向回归以冻结 SET4 的 derived_initial_u 起步；正式 RESET4 始终用冻结的 SET4 terminal_u=3.655082141628295，不替换成回归的新末态。最终两个精度各自运行 RESET4，再把本精度实际 RESET4 终态传给 SET5；SET5 没有 0.2 V 观测锚或自由初态。

## 数据和数值门槛

原 SET4/SET5 各503原生记录、500评分点，RESET4为173/170。runner 将两条新记录与已审小缓存的命令、电流和原掩码逐项相等核验，SET4也与冻结参考数组核对。不因预测小、失配大或低压而删除点；单圈特征只是诊断，不冒充原20项聚合合同。

12静态点及24个已解点RHS代数值均在计划内。静态通过包含 signed 符号、被动性、KVL/KCL、电流及局部电压与旧参考之差；RHS从统一 q 方程独立变换为 u，与实现逐点比较，正场要求 b 不变性。一次正向回归通过电流/u门槛和完整原生状态检查后才进入 b 搜索。

最终逐相位检查原生数组/掩码身份、符号和分压方向、两端被动性、KVL/KCL、有限 Γ 域、q=exp(−u)、重复顶点/零点状态、入口和终态、段间及相位连接。每个最终相位均有完整原生 legacy 核验；nominal/refined 的 RESET4 与 SET5 分别比较全原掩码电流与全部 u。只有全部门槛及来源哈希通过，才生成 verified_scores。

## 预算和失败路径

NumericalBudget 只在显式数值调用内累计时间：总240 s、RESET搜索120 s，每条搜索相位12 s、回归及最终相位30 s。核通过回调在积分/求根前后检查预算；调用返回后的再次检查使超预算完成结果也不能成为拟合候选。序列化、压缩、字节哈希、读写盘和来源哈希另计，不让慢I/O触发物理求解超时。

搜索内的 PhaseFailure(caller_budget_exhausted) 和外层 TimeoutError/BudgetExceeded 均在保存当前记录后转 StopSearch，终止整段搜索；只允许已有完整 best 进入预先保留的最终验证。无完整候选则停止、不造分数。域/分辨率失败作为无有效评分保存，优化器可接收 infinity，但 JSON 使用 null 加失败状态，不伪造成大有限残差。

PhaseFailure保存可用部分数组、段账本、最后RHS和根失败；若数值调用返回后才被预算拒绝，holder仍保留完整返回数组。所有attempt独立输出NPZ/JSON并记录实际序列化字节的SHA256；已存在attempt目录或计划会拒绝重启/覆盖。最终来源变化会撤销numerical_validation和verified_scores。

## 最终非数值补强

初审版本 `afb4fe329eaccf8de2918be5a2fee045278b73d3d7ab5e91fe0209b2f239aea4` 后，按 root 建议将每条 trial 的成功、PhaseFailure和调用后预算失败 summary 改成 deepcopy 快照，防止后续 legacy/诊断修改早落盘历史字典。每条phase保存后增加 flush 的简短trial/b/status/RMSE/attempt日志。已逐项复核最终diff；它们不改变模型、评分、参数选择或预算。

本次审查至此结束，不追加执行前计算或测试范围。实际运行后的数值验证、曲线解释与失败分类需基于新保存结果另行审查。
