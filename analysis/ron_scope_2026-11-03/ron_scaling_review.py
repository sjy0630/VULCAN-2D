"""Cached transport scaling and three fixed-Ron anchor checks, no ODE/fit.

Material table alone scales by R0/R. Every series current is separately
re-solved against the unchanged measured transistor; anchors are re-inverted.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from vulcan2d.transistor import measured_transistor_mapping


class CachedRonTransport:
    def __init__(self):
        self.table_path=ROOT/'analysis/spectral_dynamic_2026-11-03/transport_g010_v0025.npz'
        with np.load(self.table_path) as z:
            self.lg=z['log_gamma'].copy();self.lv=z['log_voltage'].copy();self.li=z['log_current'].copy()
        shape=json.loads((ROOT/'analysis/constrained_shared_2026-11-03/lognormal_spectral_refinement.json').read_text())
        self.r0=shape.get('parameters',shape)['ron_ohm']
        self.mapping=measured_transistor_mapping();self.knots={}
        for phase in ('set','reset'):
            lookup=self.mapping._active_lookup(phase)
            x=np.unique(np.concatenate([a['drain_voltage_v'] for a in lookup.curves()]))
            self.knots[phase]=(x,np.array([self.mapping.current(float(v),1.1,phase) for v in x]))

    def material(self,v,gamma,ron,phase='set'):
        v,gamma=np.broadcast_arrays(np.asarray(v,float),np.asarray(gamma,float))
        positive=v>0;x=np.log(gamma);y=np.log(np.maximum(v,np.exp(self.lv[0])))
        assert np.all((x>=self.lg[0])&(x<=self.lg[-1]))
        assert np.all(y<=self.lv[-1]+1e-12)
        ix=np.clip(np.searchsorted(self.lg,x)-1,0,len(self.lg)-2)
        iy=np.clip(np.searchsorted(self.lv,y)-1,0,len(self.lv)-2)
        fx=(x-self.lg[ix])/(self.lg[ix+1]-self.lg[ix]);fy=(y-self.lv[iy])/(self.lv[iy+1]-self.lv[iy])
        p=self.li[0 if phase=='set' else 1]
        log_i=(1-fx)*((1-fy)*p[ix,iy]+fy*p[ix,iy+1])+fx*((1-fy)*p[ix+1,iy]+fy*p[ix+1,iy+1])
        return np.where(positive,np.exp(log_i)*self.r0/ron,0.)

    def inverse(self,local_v,current,ron,phase='set'):
        y=math.log(abs(local_v));j=int(np.clip(np.searchsorted(self.lv,y)-1,0,len(self.lv)-2))
        f=(y-self.lv[j])/(self.lv[j+1]-self.lv[j])
        profile=(1-f)*self.li[0 if phase=='set' else 1,:,j]+f*self.li[0 if phase=='set' else 1,:,j+1]
        target=math.log(current*ron/self.r0)
        assert profile[0]<target<profile[-1] and np.all(np.diff(profile)>=0)
        return math.exp(float(np.interp(target,profile,self.lg)))

    def series(self,command,gamma,ron,phase='set'):
        command,gamma=np.broadcast_arrays(np.asarray(command,float),np.asarray(gamma,float))
        lo=np.zeros_like(command);hi=command.copy()
        for _ in range(42):
            mid=(lo+hi)/2
            residual=self.material(mid,gamma,ron,phase)-np.interp(command-mid,*self.knots[phase])
            high=residual>0;hi=np.where(high,mid,hi);lo=np.where(high,lo,mid)
        current=self.material(lo,gamma,ron,phase)
        return current,lo


def main():
    started=time.monotonic();transport=CachedRonTransport()
    calibration_path=HERE/'RON_CALIBRATION_SCOPE.json';calibration=json.loads(calibration_path.read_text())
    rons=[x['Ron_ohm'] for x in calibration['predeclared_Ron_sensitivity_points']]
    source_path=ROOT/'analysis/parameter_transfer_2026-11-03/SERIES_ENTRY_REVIEW.json'
    source=json.loads(source_path.read_text())
    recheck_path=ROOT/'analysis/parameter_transfer_2026-11-03/SERIES_THRESHOLD_RECHECK.json'
    recheck=json.loads(recheck_path.read_text())
    base_path=ROOT/'analysis/spectral_dynamic_2026-11-03/seed_screen.json';base=json.loads(base_path.read_text())
    sensitivity_path=ROOT/'analysis/spectral_dynamic_2026-11-03/finite_rate_order_sensitivity.json'
    sens=json.loads(sensitivity_path.read_text())
    g5row=next(x for c in sens['sensitivity_cases'] if c['fractional_symmetric_current_expansion']==.05
               for x in c['endpoint_bands'] if x['command_v']==5 and x['side']=='lower')
    cache_path=ROOT/'analysis/beta_feedback_2026-11-03/fixed_spectrum_budget_arrays.npz'
    with np.load(cache_path) as z:
        commands=z['soft_set_command_left_v'].copy();old_exact_lower=z['soft_set_current_lower_a'].copy()
    initial=source['anchors']['set_out_0p2']['cycle4']
    out_i=source['anchors']['set_out_2']['expanded_current_band_a'][0]
    out_local=2.-transport.mapping.voltage_for_current(out_i,1.1,'set')
    ret2=base['anchors']['soft_set_return_2V_high']
    anchor_defs={
        'initial_0p2':dict(local_v=initial['local_voltage_v'],current_a=initial['current_a'],reference_gamma_ev=source['initial_gamma_cycle4_from_original_exact_anchor_ev']),
        'outward_2V_lower':dict(local_v=out_local,current_a=out_i,reference_gamma_ev=source['SET_target_2V_gamma_lower_ev']),
        'return_5V_lower':dict(local_v=g5row['local_v'],current_a=g5row['current_a'],reference_gamma_ev=g5row['gamma_ev']),
        'return_2V_upper':dict(local_v=ret2['local_v'],current_a=ret2['current_a'],reference_gamma_ev=ret2['gamma_ev'])}
    baseline_gamma={k:transport.inverse(v['local_v'],v['current_a'],transport.r0) for k,v in anchor_defs.items()}
    original_gc=recheck['best_fixed_gamma_c_ev']
    original_row=next(r for r in source['rows'] if r['gamma_c_ev']==original_gc)
    fixed_js=original_row['best_JS_a']
    cap=source['transistor_current_upper_at_command2V_a']
    rows=[];arrays={}
    for ri,ron in enumerate(rons):
        a=ron/transport.r0;gc=a*original_gc
        gamma={k:transport.inverse(v['local_v'],v['current_a'],ron) for k,v in anchor_defs.items()}
        anchor_rows={}
        for name,definition in anchor_defs.items():
            gbase=baseline_gamma[name]
            scaled_current=float(transport.material(definition['local_v'],a*gbase,ron))
            u_before=math.log1p(original_gc/gbase);u_after=math.log1p(gc/gamma[name])
            anchor_rows[name]=dict(**definition,baseline_table_gamma_ev=gbase,rederived_gamma_ev=gamma[name],
                scaling_gamma_a_times_baseline_ev=a*gbase,
                rederived_gamma_over_a_baseline_minus_one=gamma[name]/(a*gbase)-1,
                forward_material_current_at_scaled_gamma_over_observed_minus_one=scaled_current/definition['current_a']-1,
                u_at_scaled_gc=u_after,u_baseline=u_before,u_shift=u_after-u_before)
        high_current,high_local=transport.series(commands,gamma['return_5V_lower'],ron)
        initial_i2,initial_local2=transport.series(2.,gamma['initial_0p2'],ron)
        B=math.log1p(gc/gamma['initial_0p2']);A=math.log1p(gc/gamma['outward_2V_lower'])
        u5=math.log1p(gc/gamma['return_5V_lower']);u2=math.log1p(gc/gamma['return_2V_upper'])
        du=u5-u2
        jsmax=min(.001*(1-1e-8),float(initial_i2)*math.exp(B),cap*math.exp(A))

        def budget(js):
            c=math.log(cap/js);lower=math.log((B+c)/(A+c))/2.
            integral=float(.05*np.maximum(np.log(high_current/js)+u2,0).sum())
            upper=du/integral if integral>0 else math.inf
            return dict(JS_a=js,k_lower_per_v=lower,k_upper_per_v=upper if math.isfinite(upper) else None,
                        k_upper_unbounded=not math.isfinite(upper),lower_over_upper=lower/upper,
                        integral_lower=integral)

        probes=[]
        for command,gn in ((.2,'initial_0p2'),(2.,'initial_0p2'),(2.,'return_5V_lower'),(5.,'return_5V_lower')):
            old_i,old_v=transport.series(command,baseline_gamma[gn],transport.r0)
            new_i,new_v=transport.series(command,a*baseline_gamma[gn],ron)
            probes.append(dict(command_v=command,state_anchor=gn,reference_current_a=float(old_i),
                               scaled_state_current_a=float(new_i),terminal_relative_change=float(new_i/old_i-1),
                               reference_local_v=float(old_v),scaled_state_local_v=float(new_v)))
        broad=[]
        for voltage,g in ((.05,1.),(.2,1.),(2.,.1),(5.,.1),(2.,10.)):
            f0=float(transport.material(voltage,g,transport.r0));f1=float(transport.material(voltage,a*g,ron))
            broad.append(dict(local_v=voltage,gamma_baseline_ev=g,scaled_gamma_ev=a*g,scaled_current_over_reference=f1/f0))
        row=dict(Ron_ohm=ron,a=a,scaled_gc_ev=gc,anchor_reinversions=anchor_rows,
                 narrow_series_KCL_probes=probes,broad_regime_counterexamples=broad,
                 initial_terminal_at_2V_a=float(initial_i2),initial_local_at_2V_v=float(initial_local2),
                 SET_return_allowed_du=du,JS_necessary_upper_a=jsmax,
                 budget_original_fixed_JS=budget(fixed_js),budget_0p98_own_ceiling=budget(.98*jsmax),
                 budget_upper_boundary_limit=budget(float(np.nextafter(jsmax,0.))),
                 fixed_JS_inside_activation_domain=bool(fixed_js<jsmax),
                 high_return_current_60node_array_min_a=float(min(high_current)),
                 high_return_current_60node_array_max_a=float(max(high_current)),
                 max_KCL_abs_residual_a=float(np.max(abs(high_current-np.interp(commands-high_local,*transport.knots['set'])))))
        rows.append(row)
        arrays[f'ron_{ri}_soft_set_current_lower_a']=high_current
        arrays[f'ron_{ri}_soft_set_local_voltage_v']=high_local
        arrays[f'ron_{ri}_anchor_gamma_ev']=np.array(list(gamma.values()))
        arrays[f'ron_{ri}_Ron_ohm']=np.array(ron)
    baseline_current=arrays['ron_0_soft_set_current_lower_a']
    for ri,row in enumerate(rows):
        row['max_rederived_high_return_current_change_from_table_baseline']=float(np.max(abs(arrays[f'ron_{ri}_soft_set_current_lower_a']/baseline_current-1)))
        row['boundary_limit_ratio_relative_change_from_table_baseline']=row['budget_upper_boundary_limit']['lower_over_upper']/rows[0]['budget_upper_boundary_limit']['lower_over_upper']-1
    baseline_offsets=dict(max_table_recomputed_60node_relative_difference_from_old_exact_cache=float(np.max(abs(baseline_current/old_exact_lower-1))),
                          original_exact_cached_best_ratio=recheck['lowest_ratio_all_sampled_gamma_c'],
                          new_same_Ron_table_reinverted_boundary_ratio=rows[0]['budget_upper_boundary_limit']['lower_over_upper'],
                          anchor_inverse_relative_offsets={k:baseline_gamma[k]/v['reference_gamma_ev']-1 for k,v in anchor_defs.items()})
    report=dict(status='completed_three_fixed_Ron_scaling_checks_no_ODE_no_fit_no_grid_screen',
                exact_identity='F_R(V,Gamma)=(R0/R)*F_R0(V,Gamma); terminal series current must be re-solved.',
                scaling=dict(transform='R->aR, Gamma->aGamma, gamma_c->a gamma_c',
                             exact_state_invariance='u=log(1+gamma_c/Gamma) is unchanged exactly',
                             approximate_current_invariance='Write F=(Gamma/R)*H(V,Gamma); Fnew/Fold=H(V,aGamma)/H(V,Gamma)->1 as Gamma->0 at fixed finite T and V.',
                             asymptotic_H='pi*integral rho(E)*[f(E-(1-eta)V)-f(E+eta V)] dE',
                             dynamics='If terminal I and u match, original beta1 A0 drives log(|I|/J)+u match with J,k unchanged. This is approximate transport invariance, not a new mechanism.'),
                fixed_original_gc=original_gc,fixed_original_JS_a=fixed_js,rows=rows,baseline_numerical_offsets=baseline_offsets,
                recommendation='A single predeclared 19gc x3Ron cached bounded screen is worthwhile as a calibration/scope audit, not as an expected physical rescue. Re-invert anchors and re-solve KCL at each Ron; mapped small-Gamma checks show whether changes are mostly parameter re-labeling. Do not run ODE/fit before a compatible necessary interval appears.',
                limitations=['Only three predeclared Ron values and one mapped old gc were probed; this report performs no19x3screen or threshold optimization.',
                    'The old best ratio is not a global continuous-gc minimum. Fixed-gc grids at a changed Ron probe shifted effective gc/R coordinates, so an apparent improvement can be grid coverage rather than new physics.',
                    'Finite Gamma, off-resonant Gamma-squared contributions, approach to broad ohmic regime, and fixed spectral/thermal scales break simple scaling.',
                    'Ron->aRon changes the exact broad limit |I|<=|V|/Ron. Lower Ron can correct the bare amplitude ceiling yet leave the series early-fast/late-slow conflict.',
                    'Original gamma_c algorithmic box is not scale invariant; transformed coordinates near its boundary may leave it. This is a search-box effect.',
                    'New numbers use the existing interpolated material table; no new quadrature, ODE, fit or holdout. Table-baseline shifts are separately reported; no new interval-arithmetic certification.',
                    'The 0.98 boundary and upper-limit probes are deterministic diagnostics; activation equality is only a limit, not an admissible seed.'],
                elapsed_s=time.monotonic()-started,
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (calibration_path,source_path,recheck_path,base_path,sensitivity_path,cache_path,transport.table_path,
                     ROOT/'vulcan2d/shared_spectral.py',ROOT/'vulcan2d/transistor.py',Path(__file__))})
    (HERE/'RON_SCALING_REVIEW.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    arrays['soft_set_command_left_v']=commands
    np.savez_compressed(HERE/'RON_SCALING_REVIEW.npz',**arrays)
    lines=['# Ron缩放与SET必要界范围审查','',
        '固定谱下材料电流严格满足F_R=(R0/R)F_R0；串联端电流不能直接按此倍数缩放。本次对每个Ron重新反演原训练端点，再重新解测得晶体管的KCL。',
        '', '写F=(Γ/Ron)H(V,Γ)。在固定有限温度和电压、Γ→0时，H趋于π∫ρ(E)[f(E−(1−η)V)−f(E+ηV)]dE。因而Ron→aRon、Γ→aΓ、gc→agc使u=ln(1+gc/Γ)精确不变，且F近似不变。终端I和u近似相同时，原电流驱动的SET/RESET动力学也近似不变，J、k不需要因该变换改变。',
        '', '该关系在有限Γ修正、非共振Γ²贡献、接近宽谱欧姆极限，以及固定温度/能谱尺度处失效。宽谱极限电流变为原来的1/a，因此降低Ron确能改变裸器件的电流上限，但这不自动解决串联曲线的先快后慢矛盾。',
        '',f'只检验原最低比值点gc={original_gc:.9g}eV的映射gc→agc，未开展19×3扫描、阈值优化或新候选挑选。',
        '', '| Ron/Ω | a | 映射gc/eV | 锚Γ相对aΓ最大偏差 | 高返扫KCL电流最大变化 | 上界极限速率下/上比 |',
        '|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        maximum=max(abs(v['rederived_gamma_over_a_baseline_minus_one']) for v in row['anchor_reinversions'].values())
        lines.append(f'| {row["Ron_ohm"]:.9g} | {row["a"]:.9g} | {row["scaled_gc_ev"]:.9g} | {maximum:.6g} | {row["max_rederived_high_return_current_change_from_table_baseline"]:.6g} | {row["budget_upper_boundary_limit"]["lower_over_upper"]:.9g} |')
    lines += ['',f'原精确缓存比值{recheck["lowest_ratio_all_sampled_gamma_c"]:.9g}；相同Ron重新用表反演/解KCL后的对照值{rows[0]["budget_upper_boundary_limit"]["lower_over_upper"]:.9g}。两者偏移需与真正Ron效应分开，不能混称模型改进。',
              '', '建议允许一次预先锁定19gc×3Ron的缓存有界筛查，用来核验校准和参数范围；不宜把它当作已找到物理出路。每个Ron必须重算状态锚、KCL电流数组、阈值激活上限。若出现必要区间相容，也仅获得进一步轨迹检查的资格。',
              '原19点不构成连续gc全局最优；不同Ron下固定gc网格对应不同gc/R，改善可能来自采样位置变化。缩放后的gc越出算法盒只说明参数盒不具缩放不变性。',
              f'本次纯缓存计算{report["elapsed_s"]:.3f}s；只读既有锚/缓存，不读取保留集，不做新谱积分、ODE或fit。详细小Γ/宽Γ反例、KCL残差及来源hash见JSON。']
    (HERE/'RON_SCALING_REVIEW.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(elapsed_s=report['elapsed_s'],baseline_offsets=baseline_offsets,
        rows=[dict(Ron_ohm=r['Ron_ohm'],a=r['a'],scaled_gc=r['scaled_gc_ev'],
                   anchor_relative_offsets={k:v['rederived_gamma_over_a_baseline_minus_one'] for k,v in r['anchor_reinversions'].items()},
                   max_high_return_current_change=r['max_rederived_high_return_current_change_from_table_baseline'],
                   fixed_JS_ratio=r['budget_original_fixed_JS']['lower_over_upper'],
                   ceiling_ratio=r['budget_upper_boundary_limit']['lower_over_upper'],
                   ceiling_ratio_change=r['boundary_limit_ratio_relative_change_from_table_baseline'],
                   broad_examples=r['broad_regime_counterexamples']) for r in rows]),indent=2))


if __name__=='__main__':main()
