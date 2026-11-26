# 原21个数值未知点：一次有上限的积分复核

原47点结果保持冻结，没有新增形状、修改core、运行ODE或更改验收门槛。只使用已有64/128阶规则和既有独立Z积分，120秒上限。

状态：completed_original21_recheck；已尝试21点；数值可用19点；仍未知2点。
原结果哈希保持不变：True。

| 点 | 状态 | 使用已有积分 | 所选必要条件未否决情景 |
|---|---|---|---|
| global_stratum_00 | resolved_existing_integration | adaptive_Z | [] |
| global_stratum_01 | numerical_unknown_after_recheck | 未完成 | 数值未知 |
| global_stratum_03 | resolved_existing_integration | 64 | [] |
| global_stratum_04 | resolved_existing_integration | 64 | [] |
| global_stratum_05 | resolved_existing_integration | adaptive_Z | [] |
| global_stratum_07 | resolved_existing_integration | 128 | [] |
| global_stratum_11 | resolved_existing_integration | adaptive_Z | [] |
| global_stratum_12 | resolved_existing_integration | 128 | [] |
| global_stratum_14 | resolved_existing_integration | 128 | [] |
| global_stratum_16 | resolved_existing_integration | 128 | [] |
| global_stratum_18 | resolved_existing_integration | 128 | [] |
| global_stratum_19 | resolved_existing_integration | 64 | [] |
| global_stratum_20 | resolved_existing_integration | 128 | [] |
| global_stratum_21 | resolved_existing_integration | 64 | [] |
| global_stratum_22 | numerical_unknown_after_recheck | 未完成 | 数值未知 |
| global_stratum_23 | resolved_existing_integration | adaptive_Z | [] |
| global_stratum_24 | resolved_existing_integration | 64 | [] |
| global_stratum_25 | resolved_existing_integration | 128 | [] |
| global_stratum_27 | resolved_existing_integration | 128 | [] |
| global_stratum_28 | resolved_existing_integration | adaptive_Z | [] |
| global_stratum_31 | resolved_existing_integration | 64 | [] |

通过数值复核只恢复了这些既有形状点的可计算性。未知点不计物理失败；未被必要条件否决也不等于通过四相动态验证。
adaptive_Z后端的证据是已有自适应积分误差估计及已核对的单能级公式；它不自动认证生产全状态插值域，后续若要实际模型仍须单独核验。
此次复核到此停止，不扩张形状域或引入新积分器。详细误差、失败和原始结果哈希在JSON。
