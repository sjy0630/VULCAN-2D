# 同一裸器件的已知循环历史

先读 `../BARE_HISTORY_RESULTS_2026-11-03.md`；来源记录为
`identity_confirmation.json`。真实顺序已确认，4/23为测过但未提供的循环。

`segmented_problem.py` 的训练接口保留原14条裸训练记录和5583项残差，仅允许
两个明确的初态输入：第1圈的初态与缺失第4圈之后、第5圈之前的未知边界状态。
每段内不重置；一个材料对象用于所有扫描。它没有把缺失圈补成默认波形，
也尚不是完成联合标定的新模型。

```python
from segmented_problem import make_problem
problem = make_problem()
# q = [log10_gamma_c, log10_JS, log10_JR, log10_kS, log10_kR, u1, u5]
# result = problem.evaluate(q)
```

路径时钟仍是有效电压路径，没有秒单位或等待动力学。第15圈RESET终态保留在
`result['predictions']['1r', 15, 'reset']['final_u']`，将来的第16圈必须继承它。
第24圈前缺第23圈，不能通过重新选择初态将其包装成普通留出预测。

产物：

- `RAW_BOUNDARY_AUDIT.md/.json`：原训练低压读数、限流与符号约定。
- `TRANSPORT_CEILING_AUDIT.md/.json`：固定Ron输运上限的必要条件检查。
- `BOUNDARY_SCOPE_REVIEW.md`：反演与状态交接的适用条件。
- `repeat_protocol_diagnostic.md/.json/.npz`：原4候选各3对连续重复，无优化。
- `segmented_validation.json`：9项接口检查，使用合成积分器。
- `segmented_smoke.json/.npz`：原首候选的真实14对接口复算；u5=u1仅为检查情景。
- `SEGMENTED_ADAPTER_REVIEW.md`：独立代码审核及已修小项。
