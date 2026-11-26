# Access 训练初态剖面后的补充数值核查

本补充只检查新选中的裸 R 初态：6.181778507799914 → 5.881778507799914。
材料参数、1T1R 初态、全部其他参数、原始波形、晶体管曲线与核心文件哈希均
与此前 `access_final_independent_audit.json` 精确一致。没有重新优化或重跑
B=0 消融。

较严 Radau 配置再次出现 `Required step size is less than spacing between
numbers`，已保存在 `access_profile_initial_strict_failure.json`。随后仅使用
实际训练采用的生产容差（rtol=1e−6、atol=1e−8、current_rtol=5e−4）与
严格独立 DOP853 参考交叉检查；原验收门槛没有放宽。

结果：

- 全部存储观察点的电流、u 以及末态在浮点输出上相同；这不意味着未采样
  快变区间也逐点精确相同。
- 21 个独立电流空间负载根最大相对差为 2.68×10⁻¹³；电压闭合、被动性
  和功耗分解通过。
- 裸 SET 末态仍精确为 u=0，与旧审计进入裸 RESET 的状态相同。
  因而裸 RESET 和两对 1T1R 的五个分支可继承旧审计，不必重复计算。
- 50 mV 读阻和零偏微分电阻仍为 93.4801 Ω。1T1R 的轨迹、电阻趋势及
  峰值结论完全不变。

裸 SET 原始训练记录 RMSE 为 0.64879 decade，非限流区为 0.79925 decade；
这些是校准数据的描述性评分。此补充仅证明当前初态的数值计算与旧审计的
继承条件，并不将该候选升级为物理验证通过。

完整证据：`access_profile_initial_audit.json`；复现：
`audit_access_initial_supplement.py`。报告包含材料参数的规范化 SHA256、前次
审计哈希与各文件哈希，运行期间输入未变。
