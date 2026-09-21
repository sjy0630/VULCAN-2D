"""Generate the auditable counterfactual result, including every tested case."""
from pathlib import Path
import json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def run():
    r = json.loads((ROOT/'analysis/contact_memory_counterfactual.json').read_text())
    audit = json.loads((ROOT/'analysis/contact_memory_refinement.json').read_text())
    assert len(audit['candidates']) == 8 and audit['all_refinements_pass']
    table = []
    for name, row in r['candidates'].items():
        split = '全53参数' if name.startswith('all_descriptive') else '前35参数'
        label = {'fixed_positive': '固定+g', 'fixed_negative': '固定−g',
                 'negative_to_positive': '−g→+g', 'positive_to_negative': '+g→−g'}[row['contact_label']]
        f = row['full_scores']; selectivity = row['RESET_outward_4_over_3']/row['SET_return_4_over_3']
        table.append(f'| {split} / {label} | {f["rmse_decade"]:.6f} | {f["first3_rmse"]:.6f} | '
                     f'{f["later50_rmse"]:.6f} | {row["last18_scores"]["rmse_decade"]:.6f} | '
                     f'{row["SET_return_4_over_3"]:.6g} | {row["RESET_outward_4_over_3"]:.6g} | {selectivity:.6g} |')
    refinements = [x for candidate in audit['candidates'].values() for x in candidate['refinements']]
    maximum = max(x['maximum_log_change'] for x in refinements)
    weighted = max(x['weighted_log_change'] for x in refinements)
    roots = sum(x['independent_frozen_root_checks'] for x in audit['candidates'].values())
    root_error = max(x['maximum_root_voltage_error_V'] for x in audit['candidates'].values())
    observed_c = r['observed_RESET_outward_4_over_3']/r['observed_SET_return_4_over_3']
    content = f'''# 接触耦合记忆反事实：能产生跳变，但未解释极性选择性

日期：2026-09-21。状态：本轮无重拟合反事实及数值核验完成；完整物理模型、
随机路径及新颖性仍未解决。原研究目标继续进行，生产M13未改变。

## 结论与直接证据

已有两组保留电荷模型参数下，在第3对结束的零偏处交换有效接触耦合，确实
能产生跨圈电流跳变；但SET与RESET同时改变，未复现实验要求的选择性。
这项结果不排除重新估计参数后的接触机制，更不是全部物理模型族的否证。

在完全相同的0.10–0.30 V原生低压窗口，实测S4/S3={r['observed_SET_return_4_over_3']:.9f}，
R4/R3={r['observed_RESET_outward_4_over_3']:.9f}。比值均为对应支路电流的几何平均比。
定义便于判别的选择性对比 C=(R4/R3)/(S4/S3)，实验C={observed_c:.6f}。
全53参数的−g→+g反事实为S约46.61倍、R约42.39倍，C约0.909；前35参数
为S约349.50倍、R约215.23倍，C约0.616。只看第二组R的跳变大小会掩盖SET失败。

前35参数的该反事实在第3圈预测SET回程几何平均1.259 nA，实测651.37 nA；
第4圈预测439.84 nA，实测657.59 nA。它用过低的第3圈SET换取了跳变，
不能作为两条支路的联合物理解释。C只是一项诊断，不替代全曲线和绝对电流。
同一循环两个端口电流若乘相同的纯比例因子，C不变；此代数性质不能推广成
调整内部B并重算非线性晶体管/状态反馈后仍不变的定理。

## 受控物理假设与实现

沿用 `CONTACT_ASYMMETRY_MEMORY_DESIGN_2026-09-21.md`，不重新拟合。
g=ln(ΓL/ΓR)，冻结条件电流只依赖B=A·expit(g)。交换g的符号时，以
log10A'=log10A+[log expit(g)−log expit(g')]/ln(10)补偿，保持B不变。
条件电子占据则为n=expit(g)fL+expit(−g)fR，能够改变后续q、m跃迁率。
零偏fL=fR，因此条件占据和慢率在切换点不变。

g的幅度沿用两个来源拟合的约12。补偿后的A可能超出旧拟合界限；完整值与
越界标记保存在JSON，没有裁剪。A是eNΓR的参数表示，不能直接解释为测得的
原子尝试率；物理时间仍是电压扫描路径单位，没有新增秒级保持寿命。

新增 `retained_contact_memory.py`，复用原来的原生正概率传播器和负载求解器，
不修改已归档的物理模块。切点为2028点之后；S4起点与R3终点两个原生零点
均保留，完整P(q,m;E)连续传递，无每圈或RESET重置、无实测电流驱动。
续传接口只接受零偏起点与同能量节点基底的合法全状态，未提供非零偏多根交接。

四种设置×两个来源参数，共八条完整53对/35,828点轨迹。34,662个计分点、
每对八个等质量窗口及787个低信号标记保留。两种方向与固定正负对照全部展示：

| 来源 / 接触设置 | 全53误差 | 首3误差 | 后50误差 | 后18误差 | S4/S3 | R4/R3 | C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

误差单位decade。全53参数是共同描述；前35参数来自原模型的顺序训练。
所有切换边界均人为指定，后18列只是在该假设下延续状态的条件误差，
不能称为随机接触模型预测了第4圈的事件，也不是新独立实验验证。
固定正接触逐点恢复此前归档结果；现有真实连续模型的后18基准仍为0.273370。

## 数值与来源核验

- 同参数分段和完整53对计算对电流、全状态、内部电压逐点完全相同。
- 随机相关初态、零时长重复点、非法输入、补偿恒等式及独立自适应ODE检查通过。
- 15项相关测试通过；均未放宽旧的物理加密门槛。
- 全部八个候选各两级加密：64→128→192能量节点，表格0.005→0.0025→0.00125 V，
  路径0.01→0.005→0.0025 V。最大计分电流差{maximum:.8g} decade，
  最大加权差{weighted:.8g} decade，均小于原先0.005/0.001门槛。
- {roots}个独立冻结负载根核对通过，最大内部电压差{root_error:.4g} V。
  电路KCL、状态概率非负/归一、完整重复零点与切换处状态连续性均核验。

## 对下一步物理研究的约束

我们获得了可被具体模型检验的联合目标：正向读取几乎不变、反向响应增加
两个多数量级。通道数量、共同导电幅度或这组参数下的接触控制变化，不能
仅凭“产生了大跳变”宣称解释成功。本次是后者的明确失败记录。

此前循环顺序检验p=1/11713继续说明可交换循环基线不足，但不排除相关随机
路径或采集条件改变；本次不增加其统计独立性，也没有确定离子物种。
四域支持误差0.17516仍是逐点利用实测电流选状态的乐观下限，不能拿来
替代轨迹验证。有限域共用慢状态的能谱反作用还需热力学推导，不能直接
把逐能级平均速率移植成全局随机构型率。

下一步优先审查仍继承自旧瞬时平衡模型的五个输运先验（能谱中心/宽度、
两位置、相互作用比例），评估它们是否限制了保留电荷模型的极性响应。
先在同一连续固定接触基准中做受物理边界约束的敏感性及可辨识性检查，
再决定是否需要新增接触状态。保持SET/RESET完整历史和相同训练划分，
同时检查选择性、首三圈绝对电流、普通回线、后期漂移及后18预测；
不得用只改善RESET的一条反事实升级为随机机制。尚不声称新颖性得到证实。

证据图：`figures/42_contact_memory_counterfactual.png`、同名PDF。
运行与完整输出：`check_contact_memory_counterfactual.py`、
`contact_memory_counterfactual.json/.npz`、八份完整状态文件；
数值核验：`contact_memory_refinement.json/.npz`。
'''
    (ROOT/'analysis/CONTACT_MEMORY_RESULTS_2026-09-21.md').write_text(content)


if __name__ == '__main__': run()
