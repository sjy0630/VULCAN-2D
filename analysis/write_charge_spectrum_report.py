"""Build the stage report from finished, preserved result files."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
read = lambda name: json.loads((ROOT/'analysis'/name).read_text())
main = read('charge_spectrum_comparison.json')
wide = read('charge_spectrum_expanded_prior.json')
groups = read('charge_spectrum_expanded_groups.json')
validation = read('charge_spectrum_validation.json')
old = read('coupled_charge_comparison.json')
assert main['status'] == groups['status'] == 'complete'

def metric(result, group):
    return result['SET_only_five_fold_scores'][group]['negative_rmse_decade']

rows = []
for name, result, fit in [('单能级四态反馈', old, old['splits']['all_descriptive']['fit']),
                          ('有限谱：原先验', main, main['splits']['all_descriptive']['fit']),
                          ('有限谱：放宽先验', groups, wide['fit'])]:
    chrono = result['splits']['chronological']['targets']['SET_only']['scores']['all']['negative_rmse_decade']
    rows.append(f"| {name} | {fit['rmse_decade']:.5f} | {metric(result,'all'):.5f} | {metric(result,'first3'):.5f} | {metric(result,'later'):.5f} | {chrono:.5f} |")
failed = sum(not r['converged'] for s in groups['splits'].values() for r in s['fit']['multistart'])
pf, wf = main['splits']['all_descriptive']['fit'], wide['fit']
text = f'''# 有限能谱、电荷状态传播与条件记忆上界

## 结论

有限导电能谱改善了四态电荷反馈的共同描述，但没有完成SET→RESET预测，尤其未解释前三圈。放宽能量范围后，高压斜率从约0.068改善到约0.712，实测中位0.594；这说明原先验参与了形状失配，不能把一次受界限约束的拟合当成整个机制的否证。放宽后完整分组敏感性也已完成，不能仅用全数据改善宣称预测成功。

更有物理意义的新增结果是一个条件动力学上界：在各位点总尝试率不随偏压和另一占据改变的模型中，已选参数不能靠电荷相互作用产生很强的“零偏慢、加偏快”的慢模态分离。全数据窄谱与宽谱解的上界均约1.23倍。这个结果限定到选中参数和明确速率假设，不是对目标器件保持时间的测量，也不排除带额外激活势垒、能量相关供给或结构演变的机制。

203项测试通过；全部原始共同数据、初始三圈和缺失点标记保留。主目标仍active且未达到，生产M13保持不变。

## 1. 模型与对照

导电能量在[E_d−w,E_d+w]均匀分布，各通道各有自己的旁侧电荷，位置、库仑作用和接触比共享。每圈仍只有一个共同正负电流尺度和保留结构能量E_t。平均的是各能量下的完整电流，不是平均势垒的电流。均匀谱不是测得的DOS；每通道独立侧电荷也不是多个导电位点共享一个陷阱。

共享参数E_d、x_d、x_t、u、ln(Gamma_L/Gamma_R)、w。E_t没有逐扫描演化律，因此当前候选仍是状态条件下的读取模型，不能冒充完整记忆器件。零谱宽严格回到前轮五参数核。旁侧交换速率不在稳态拟合内；其影响在新增动态传播器中明确保留。

原先验E_d∈[−0.5,0.5] eV、w∈[0,0.6] eV；其余参数沿用前轮。全数据与六个训练折的中心触及+0.5 eV上界（留出1–11的训练折除外），因此进行事后扩展至E_d∈[−1,1]、w∈[0,1] eV的检验。

原先验全数据解为`{pf['shared_parameters']}`；宽先验解为`{wf['shared_parameters']}`。谱最低能量分别为{pf['shared_parameters'][0]-pf['shared_parameters'][5]:.6f}和{wf['shared_parameters'][0]-wf['shared_parameters'][5]:.6f} eV。宽先验最优解未触及共享参数边界，但不等于唯一识别。电流尺度仍不能分离通道数和绝对跃迁率，位置/相互作用仍受条件几何先验约束。

## 2. 相同全部53对的结果

仍用原5,072有效内部网格点、16个高压缺失位置、三个窗口权重0.25/0.25/0.5。全数据共同描述读取两种极性；主预测中共享参数仅由训练对确定，目标SET回程识别局部尺度/E_t，然后预测目标RESET。内部电压由测量去嵌入，这些分数不是自主外加端口完整周期预测。本阶段未新增原生端口电路验证。

| 模型 | 全数据共同描述 | 五折RESET全部 | 五折RESET首3圈 | 五折RESET后50圈 | 顺序后18圈RESET |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

单位均为decade，但共同描述与RESET预测是不同任务。前轮平衡WKB电荷候选的同一内部RESET五折误差为0.81066，当前两种能谱先验仍须按这个相同任务参照判断，而非只和单能级比较。

原先验7个共享拟合/35起点、12次目标识别均收敛。宽先验全数据4起点均收敛；另外6个共享训练折/30起点及12次目标识别完成，选中解与目标解均收敛，未选中起点有{failed}次未收敛，保留全部记录。扩展折只使用该训练集合已有窄谱解、固定起点和同训练集合的无谱宽解初始化，不使用全数据共享参数。扩展属于开发后的敏感性，不是从未参与模型选择的外部验证。

宽先验的首三圈共同描述加权误差仍{wide['scores']['first3']['weighted_rmse_decade']:.5f}，其中正向低压{wide['scores']['first3']['positive_low_rmse_decade']:.5f} decade；它与主SET初始化后的巨大负向误差共同表明交接尚未解决。不能以高压斜率接近实测替代这个要求。

## 3. 精确状态传播与可检验的记忆约束

已实现`charge_moments.py`，沿分段恒定电压传播两个平均占据，跨正负电压不重置状态。与完整四态生成矩阵的独立指数传播比较：100个随机参数/初态的平均占据最大差{validation['maximum_mean_occupation_propagation_difference']:.3g}，并通过包含SET样式正向→零→负向→零的多步单元检查。时间仍为Gamma_R*t，未标定为秒；此模块尚未接入晶体管和位移电流，是精确状态传播器，不是瞬态端口模型。

设a=Gamma_L+Gamma_R、b=Gamma_t、P=(F0−F1)(c0−c1)。两个平均占据的慢衰减率为

`lambda_slow = [a+b−sqrt((a−b)^2+4abP)]/2`。

利用`lambda_slow/min(a,b) >= 1−sqrt(P)`，以及零偏下`P<=f(E_min)`，得到

`lambda_slow(V)/lambda_slow(0) <= 1/[1−sqrt(f(E_min))]`。

完整证明在`CHARGE_RELAXATION_BOUND_2026-09-21.md`。原先验各训练解的上界约1.21–1.31，全数据为1.22676；宽先验全数据为1.22700。上界对这些参数下的任意局部E_t、任意正旁侧速率及谱内所有独立通道成立。因此仅把一个偏压无关旁侧速率调得很小，不会自动形成相互作用导致的强慢模态速度分离。

这不是任意初态的电流开关时间上界，也不包括慢结构能量演变、偏压相关隧穿矩阵元、额外迁移势垒或不同的多位点网络。不能据此宣称h-BN器件没有电荷记忆；准确结论是当前选中读取参数尚未提供所需保留机制。

## 4. 数值与文献核验

首次64点积分测试在极宽谱先验例子中有约9.62e-7 decade差异，未通过2e-7门槛；在任何数据拟合前改为96点分段积分并保留失败日志。主拟合全部19组训练/目标状态的96→192点最大差{max(r['maximum_96_192_difference_decade'] for r in validation['quadrature_refinements']):.3g} decade，另加密至384点。42个正负/高低电压案例与独立自适应积分的最大差{max(r['adaptive_vs_384_difference_decade'] for r in validation['independent_adaptive_integrals']):.3g} decade。宽先验使用192点，并在综合审查中核对全部训练/目标状态的384点值。

完整203项测试通过（45.003 s），包括零谱宽、梯度、谱积分、正耗散、无相互作用极限、独立四态动态传播和慢模态性质。综合来源、整对分组、预测和误差重算见`charge_spectrum_validation_audit.json`；文件身份见`charge_spectrum_artifact_manifest.json`。图为`figures/38_charge_spectrum_evidence.png/.pdf`。此前六份阶段清单所列文件保持原身份。

原文核对见`CHARGE_DYNAMICS_LITERATURE_NOTE_2026-09-21.md`。Michalek与Bulka的arXiv:0907.2879已经讨论电容耦合、动态库仑阻塞、亚稳极化态及不同弛豫时间。但其连续DOS正统隧穿率不同于当前固定总尝试率的离散能级，不能把我们的精确闭合或上界直接推广过去。该文也不是目标h-BN的物种或保持证据。两个新推导的文献新颖性仍未确认。

## 5. 下一项：从读取模型进入保留状态演化

下一步不再仅扩展静态谱以压低平均误差。需要明确哪个保留坐标改变E_t/势垒，并由完整SET、零偏交接和RESET历史推进它；目标RESET不得每圈自由重新初始化。可检验候选是具有独立激活势垒的结构/陷阱构型，电荷读取保持快速守恒，构型转移受共同能量景观及正反速率关系约束。这只是下一候选，不是已经确定的离子物种。

必须共同面对初始三圈、普通循环、正负读取、负向恢复及后期漂移，并对随机构型切换给出适当的概率预测而非承诺每次跳变的确定时刻。无扫描时间元数据时只能识别速率与扫描时标的组合，不能伪造绝对保持时间。现有同协议数据仍可推进；新增实验和UI不是此阶段前提。整体物理解释、自主状态交接和新颖性尚未完成。
'''
(ROOT/'analysis/CHARGE_SPECTRUM_RESULTS_2026-09-21.md').write_text(text)
print(ROOT/'analysis/CHARGE_SPECTRUM_RESULTS_2026-09-21.md')
