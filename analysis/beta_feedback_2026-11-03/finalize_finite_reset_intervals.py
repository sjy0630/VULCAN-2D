"""Clarify finite RESET intervals using saved arrays only; no spectrum rerun."""
from pathlib import Path
import json,math,hashlib
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]


def main():
    source=OUT/'finite_beta_budget_screen.json';d=json.loads(source.read_text())
    with np.load(OUT/'fixed_spectrum_budget_arrays.npz') as z:current=z['bare_reset_current_lower_a'].copy()
    b=json.loads((ROOT/'analysis/spectral_dynamic_2026-11-03/seed_screen.json').read_text())
    anchors=b['anchors'];greset=b['finite_read_bounds']['gamma_reset_low_min_ev']
    gbhi=anchors['soft_reset_1p4V_high']['gamma_ev'];galo=anchors['soft_reset_0p3V_low']['gamma_ev']
    for case in d['cases']:
        gc=case['gamma_c_ev'];alpha=case['alpha'];jrmax=case['JR_maximum_soft_drive_a']
        jrbase=np.geomspace(1e-6,.99*jrmax,181) if .99*jrmax>1e-6 else np.array([])
        jr=np.unique(np.r_[jrbase,max(current)*np.linspace(.8,.999,121)]);jr=jr[(jr>1e-6)&(jr<.99*jrmax)]
        lower=alpha*(math.log1p(gc/gbhi)-math.log1p(gc/galo))/(1.1*np.log(jrmax/jr))
        integral=np.sum(np.maximum(np.log(np.maximum(current[:,None],1e-300)/jr[None,:]),0)*.002,axis=0)
        upper=np.divide(math.log1p(gc/greset),integral,out=np.full(len(jr),np.inf),where=integral>0)
        finite=np.isfinite(upper);compatible=finite&(lower<=upper)
        ratio=np.divide(lower,upper,out=np.full(len(jr),np.inf),where=finite)
        best=int(np.argmin(ratio)) if len(ratio) else None
        case['RESET_compatible_threshold_count']=int(sum(compatible))
        case['RESET_unbounded_upper_threshold_count']=int(sum(~finite))
        case['minimum_RESET_required_over_allowed']=float(ratio[best]) if best is not None and math.isfinite(ratio[best]) else None
        case['best_RESET_JR_a']=float(jr[best]) if best is not None else None
        assert case['SET_compatible_threshold_count']==0 and case['status']=='no_joint_finite_main_budget_seed'
    d['finite_RESET_interval_review']={
        'method':'Pure algebra from saved bareRESET currents; infinite upper-rate bounds are counted separately and excluded from finite seed compatibility/minimum ratio.',
        'candidate_seed_count_changed':False,'spectrum_rerun':False,
        'finalizer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    d['script_sha256']=hashlib.sha256((OUT/'finite_beta_budget_screen.py').read_bytes()).hexdigest()
    source.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    lines=['# 有限beta反馈必要预算筛','',
           '24个预先声明情景；固定谱、同beta作用SET/RESET、同alpha作用两极性。无ODE、无优化。公共Q90读预算不冒充每条实测点。','',
           '| gamma_c/eV | beta | alpha | SET有限区间数 | RESET有限区间数 | SET最小下/上比 | RESET有限域最小下/上比 |','|---:|---:|---:|---:|---:|---:|---:|']
    for r in d['cases']:lines.append(f'| {r["gamma_c_ev"]} | {r["beta"]} | {r["alpha"]:g} | {r["SET_compatible_threshold_count"]} | {r["RESET_compatible_threshold_count"]} | {r["minimum_SET_required_over_allowed"]:.6g} | {r["minimum_RESET_required_over_allowed"]:.6g} |')
    lines+=['','**24情景均无有限主预算种子，按计划停止。**',
            'RESET积分可能为0，表示此宽松预算不给速率上限，不能解读成拟合完美。此类阈值在JSON另计，表中已排除无限上界。该澄清只使用缓存数组重算代数，没有重复谱积分。',
            'JS上界为实际初态2.8V激活上界与Icc*exp(beta*uL)低阻可达上界的较小者；允许JS>Icc并检查stop barrier。RESET按同beta重算。',
            '逐条5%缓变/跃迁诊断与c0全阈值界另存在JSON，未自动升级为全曲线验收gate。真实非匀速/停留协议需要时间戳，不能由电压点距推断。',
            '改变beta没有自动解除跨拓扑同态电压比较或单向态冻结谱的Fermi窗口界。结论仅限当前有限域、声明预算与原模型类。',
            f'主计算耗时{d["elapsed_s"]:.3f}s；旧源产物哈希保持不变：{d["source_files_unchanged"]}。']
    (OUT/'finite_beta_budget_screen.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'candidate_seed_count':len(d['candidate_seeds']),
                      'finite_RESET_counts':[r['RESET_compatible_threshold_count'] for r in d['cases']]},indent=2))


if __name__=='__main__':main()
