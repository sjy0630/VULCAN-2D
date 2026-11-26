"""Series-only entry diagnostics; cached fixed spectrum, no ODE or optimizer."""
from pathlib import Path
import sys,json,math,hashlib,time
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from vulcan2d.transistor import measured_transistor_mapping


def main():
    started=time.monotonic();old=ROOT/'analysis/spectral_dynamic_2026-11-03'
    # Retain the originally issued report before correcting its restart bound
    # and separating physical RESET bounds from the numerical parameter box.
    for suffix in ('json','md'):
        source=OUT/f'SERIES_ENTRY_REVIEW.{suffix}'
        snapshot=OUT/f'SERIES_ENTRY_REVIEW.pre_review.{suffix}'
        if source.exists() and not snapshot.exists():
            snapshot.write_bytes(source.read_bytes())
    original_json=OUT/'SERIES_ENTRY_REVIEW.pre_review.json'
    original_hash=hashlib.sha256(original_json.read_bytes()).hexdigest()
    base=json.loads((old/'seed_screen.json').read_text())
    sens=json.loads((old/'finite_rate_order_sensitivity.json').read_text())
    highcase=next(c for c in sens['sensitivity_cases'] if c['fractional_symmetric_current_expansion']==.05)
    with np.load(old/'transport_g010_v0025.npz') as z:
        lg=z['log_gamma'].copy();lv=z['log_voltage'].copy();li=z['log_current'].copy()
    mapping=measured_transistor_mapping();knots={}
    for phase in ('set','reset'):
        look=mapping._active_lookup(phase);x=np.unique(np.concatenate([a['drain_voltage_v'] for a in look.curves()]))
        knots[phase]=(x,np.array([mapping.current(float(v),1.1,phase) for v in x]))

    def current(v,gamma,phase):
        v,gamma=np.broadcast_arrays(np.asarray(v,float),np.asarray(gamma,float));positive=v>0
        x=np.log(gamma);y=np.log(np.maximum(v,np.exp(lv[0])))
        assert np.all((x>=lg[0])&(x<=lg[-1])) and np.all(y<=lv[-1]+1e-12)
        ix=np.clip(np.searchsorted(lg,x)-1,0,len(lg)-2);iy=np.clip(np.searchsorted(lv,y)-1,0,len(lv)-2)
        fx=(x-lg[ix])/(lg[ix+1]-lg[ix]);fy=(y-lv[iy])/(lv[iy+1]-lv[iy])
        p=li[0 if phase=='set' else 1]
        result=np.exp((1-fx)*((1-fy)*p[ix,iy]+fy*p[ix,iy+1])+fx*((1-fy)*p[ix+1,iy]+fy*p[ix+1,iy+1]))
        return np.where(positive,result,0.)

    def terminal(v,gamma,phase):
        v,gamma=np.broadcast_arrays(np.asarray(v,float),np.asarray(gamma,float));low=np.zeros_like(v);high=v.copy()
        for _ in range(38):
            mid=(low+high)/2;res=current(mid,gamma,phase)-np.interp(v-mid,*knots[phase])
            yes=res>0;high=np.where(yes,mid,high);low=np.where(yes,low,mid)
        return current((low+high)/2,gamma,phase)

    def inverse(command,observed,phase):
        vt=float(mapping.voltage_for_current(float(observed),1.1,phase));vh=abs(command)-vt
        assert vh>0 and observed>0
        y=math.log(vh);j=int(np.clip(np.searchsorted(lv,y)-1,0,len(lv)-2));f=(y-lv[j])/(lv[j+1]-lv[j])
        profile=(1-f)*li[0 if phase=='set' else 1,:,j]+f*li[0 if phase=='set' else 1,:,j+1]
        assert profile[0]<math.log(observed)<profile[-1]
        return {'current_a':float(observed),'local_voltage_v':vh,'gamma_ev':math.exp(float(np.interp(math.log(observed),profile,lg)))}

    raw={};loaded=hashlib.sha256()
    with np.load(ROOT/'analysis/shared_hbn_2026-11-03/data/native_observations.npz') as z:
        for phase in ('set','reset'):
            ys=[]
            for cid in range(4,36):
                p=f'1t1r_cycle_{cid}_{phase}';v=z[p+'_voltage'].copy();i=abs(z[p+'_current'].copy())
                loaded.update(p.encode());loaded.update(v.tobytes());loaded.update(i.tobytes());ys.append(i)
            raw[phase]=(v,np.array(ys))

    def anchor(phase,branch,target):
        v,i=raw[phase];ap=int(np.argmax(abs(v)));indices=np.arange(len(v));mask=indices<=ap if branch=='outward' else indices>ap
        j=int(np.argmin(np.where(mask,abs(abs(v)-target),np.inf)));values=i[:,j]
        low=float(np.quantile(values,.1)/1.05);high=float(np.quantile(values,.9)*1.05)
        return {'phase':phase,'branch':branch,'command_v':float(v[j]),'index':j,
                'training_q10_q90_a':np.quantile(values,[.1,.9]).tolist(),'expanded_current_band_a':[low,high],
                'gamma_band_ev':[inverse(v[j],low,phase)['gamma_ev'],inverse(v[j],high,phase)['gamma_ev']],
                'cycle4':inverse(v[j],float(values[0]),phase)}

    a={
        'set_out_0p2':anchor('set','outward',.2),'set_out_2':anchor('set','outward',2.),
        'set_return_5':anchor('set','return',5.),'set_return_4':anchor('set','return',4.),
        'set_return_2':anchor('set','return',2.),'set_return_0p2':anchor('set','return',.2),
        'reset_out_0p2':anchor('reset','outward',.2),'reset_out_0p3':anchor('reset','outward',.3),
        'reset_out_1p4':anchor('reset','outward',1.4),'reset_return_0p2':anchor('reset','return',.2)}
    g0=base['finite_read_bounds']['cycle4_set_initial_0p2V_gamma_ev']
    initial_i2=float(terminal(2.,g0,'set'));initial_i5=float(terminal(5.,g0,'set'))
    icap2=float(mapping.current(2.,1.1,'set'))
    g_target2=a['set_out_2']['gamma_band_ev'][0]
    g5=next(e['gamma_ev'] for e in highcase['endpoint_bands'] if e['command_v']==5 and e['side']=='lower')
    g2=base['anchors']['soft_set_return_2V_high']['gamma_ev']
    with np.load(ROOT/'analysis/beta_feedback_2026-11-03/fixed_spectrum_budget_arrays.npz') as z:
        high_current=z['soft_set_current_lower_a'].copy()
        rgamma=z['soft_reset_gamma_edges_ev'].copy();reset_current_upper=z['soft_reset_current_upper_a'].copy()
    # Readout-cycle coupling is diagnostic; never assign independent polarity states.
    gLlo,gLhi=a['set_return_0p2']['gamma_band_ev'];gHlo,gHhi=a['reset_return_0p2']['gamma_band_ev']
    rpeak=float(max(abs(raw['reset'][0])));reset_path=2*rpeak-.2
    gamma_reset=np.geomspace(gHlo,gLhi,129);reset_commands=np.linspace(0,rpeak,35)
    reset_current=terminal(reset_commands[:,None],gamma_reset[None,:],'reset')
    positive_restart_gamma=np.geomspace(min(gHlo,g0),max(gHhi,g0),65)
    restart_current=terminal(5.,positive_restart_gamma,'set')
    set_tail_gamma=np.geomspace(gLlo,gLhi,65);reset_tail_gamma=np.geomspace(gHlo,gHhi,65)
    set_tail_current=terminal(.2,set_tail_gamma[1:],'set')
    reset_tail_current=terminal(.2,reset_tail_gamma[1:],'reset')
    initial_i0p2=float(terminal(.2,g0,'set'))
    gcs=np.unique(np.r_[np.logspace(-4,0,17),.03,.05]);rows=[];candidates=[]
    for gc in gcs:
        u0=math.log1p(gc/g0);u2out=math.log1p(gc/g_target2)
        u5=math.log1p(gc/g5);u2return=math.log1p(gc/g2);duhigh=u5-u2return
        activation_max=initial_i2*math.exp(u0)
        jsmax=min(.001*(1-1e-8),activation_max,icap2*math.exp(u2out))
        js=np.geomspace(1e-7,.98*jsmax,181) if .98*jsmax>1e-7 else np.array([])
        c=np.log(icap2/js);barrier=c+u2out<=0
        with np.errstate(divide='ignore',invalid='ignore'):
            lower=np.where(barrier,np.inf,np.log((u0+c)/(u2out+c))/2.)
        integral=np.sum(.05*np.maximum(np.log(high_current[:,None]/js[None,:])+u2return,0),axis=0)
        upper=np.divide(duhigh,integral,out=np.full(len(js),np.inf),where=integral>0)
        lo=np.maximum(1e-4,lower);hi=np.minimum(10**1.5,upper);ok=(lo<=hi)&np.isfinite(lo)
        ratio=np.divide(lower,upper,out=np.full(len(js),np.inf),where=upper>0)
        best=int(np.argmin(ratio)) if len(ratio) else None
        c0=math.log(icap2/jsmax);D=float(np.sum(.05*(u2return+np.log(high_current/icap2))))
        applicable=3*c0+D>0 and c0+u0>0 and u0>u2out and duhigh>0
        alljs=(u0-u2out)/(2*duhigh)*min(3.,(3*c0+D)/(u0+c0)) if applicable else None
        # RESET has the same common-state interval from SET return through HRS.
        uLmin=math.log1p(gc/gLhi);uLmax=math.log1p(gc/gLlo)
        uHmin=math.log1p(gc/gHhi);uHmax=math.log1p(gc/gHlo)
        duRrequired=max(0.,uHmin-uLmax);duRallowed=max(0.,uHmax-uLmin)
        Jupper=np.max(reset_current[:,1:]*(1+gc/gamma_reset[None,:-1]),axis=1)
        Jlower=np.min(reset_current[:,:-1]*(1+gc/gamma_reset[None,1:]),axis=1)
        jRmax=float(max(Jupper));jr=np.geomspace(1e-8,min(.01,.98*jRmax),181)
        hmax=np.maximum(np.log(jRmax/jr),0);rlo=np.divide(duRrequired,reset_path*hmax,out=np.full(len(jr),np.inf),where=hmax>0)
        dv=float(reset_commands[1]-reset_commands[0]);weights=np.full(34,2*dv)
        # Outward0..peak plus returnpeak..0.2; drop lowest0.2return bins,
        # favoring feasibility if grid boundaries do not coincide exactly.
        weights[reset_commands[:-1]<.2]=dv
        rIntegral=np.sum(weights[:,None]*np.maximum(np.log(np.maximum(Jlower[:-1,None],1e-300)/jr[None,:]),0),axis=0)
        rup=np.divide(duRallowed,rIntegral,out=np.full(len(jr),np.inf),where=rIntegral>0)
        rok=np.maximum(1e-4,rlo)<=np.minimum(10**1.5,rup)
        # I increases with gamma while exp(u)=1+gc/gamma decreases. Node
        # minima therefore do not bound their continuous product from below.
        restart_J_min=float(np.min(restart_current[:-1]*(1+gc/positive_restart_gamma[1:])))
        set_tail_J_upper=float(np.max(set_tail_current*(1+gc/set_tail_gamma[:-1])))
        reset_tail_J_upper=float(np.max(reset_tail_current*(1+gc/reset_tail_gamma[:-1])))
        initial_J0p2=initial_i0p2*math.exp(u0)
        # Only these sufficient inactive-tail conditions permit using the read
        # bands as unchanged phase-boundary bands without integrating the tails.
        tail_sok=ok&(js>max(set_tail_J_upper,initial_J0p2))&(js<restart_J_min)
        tail_rok=rok&(jr>reset_tail_J_upper)
        row={'gamma_c_ev':float(gc),'u0_cycle4_once':u0,'JS_activation_by_2V_upper_a':activation_max,
             'JS_finite_domain_upper_a':jsmax,'SET_finite_rate_interval_count':int(sum(ok)),
             'minimum_SET_outward_required_over_return_allowed':float(ratio[best]) if best is not None else None,
             'best_JS_a':float(js[best]) if best is not None else None,
             'best_SET_k_lower':float(lower[best]) if best is not None else None,
             'best_SET_k_upper':float(upper[best]) if best is not None and math.isfinite(upper[best]) else None,
             'all_JS_under_activation_domain_ratio_lower_bound':alljs,
             'RESET_finite_rate_interval_count':int(sum(rok&np.isfinite(rup))),
             'RESET_physical_upper_unbounded_but_box_compatible_count':int(sum(rok&~np.isfinite(rup))),
             'RESET_box_compatible_interval_count':int(sum(rok)),
             'RESET_rate_count_definition':'finite count requires a finite physical upper bound; unbounded physical upper bounds remain compatible with finite box rates and are counted separately, not rejected.',
             'RESET_required_opening':duRrequired,'RESET_allowed_opening':duRallowed,
             'RESET_reachable_state_band_u':[uLmin,uHmax],'SET_restart_J_min_over_HRS_band_a':restart_J_min,
             'SET_tail_0p2_to_zero_J_upper_a':set_tail_J_upper,'RESET_tail_0p2_to_zero_J_upper_a':reset_tail_J_upper,
             'initial_0_to_0p2_J_upper_a':initial_J0p2,
             'SET_rate_and_tail_link_count':int(sum(tail_sok)),'RESET_rate_and_tail_link_count':int(sum(tail_rok)),
             'RESET_budget_scope':'Read-band starting/ending states are linked to true boundaries only if inactive-tail bounds hold; otherwise tail evolution remains unresolved, not reset.'}
        if np.any(tail_sok) and np.any(tail_rok):
            ids=np.flatnonzero(tail_sok)
            if len(ids):
                si=int(ids[len(ids)//2]);ri=int(np.flatnonzero(tail_rok)[len(np.flatnonzero(tail_rok))//2])
                ks=math.sqrt(lo[si]*hi[si]);kr=math.sqrt(max(1e-4,rlo[ri])*min(10**1.5,rup[ri]))
                q=[math.log10(gc),math.log10(js[si]),math.log10(jr[ri]),math.log10(ks),math.log10(kr),u0]
                candidate={'q6':q,'gamma_c_ev':float(gc),'JS_a':float(js[si]),'JR_a':float(jr[ri]),'kS_per_v':ks,'kR_per_v':kr,
                           'SET_rate_interval':[float(lo[si]),float(hi[si])],
                           'RESET_rate_interval':[float(max(1e-4,rlo[ri])),float(min(10**1.5,rup[ri]))],
                           'initial_SET_drive_at_2V':float(math.log(initial_i2/js[si])+u0),
                           'restart_drive_lower_over_observed_HRS_band':float(math.log(restart_J_min/js[si])),
                           'scope':'Entry-budget diagnostic only, same state propagated across polarities; no trajectory has been integrated.'}
                candidates.append(candidate);row['candidate']=candidate
        rows.append(row)
    report={'status':'series_only_entry_budget_review_no_ODE_no_optimization','training_ids':list(range(4,36)),
            'law':'fixed original SpectralShape,beta1,A_RESET0; separate group parameters only for localization',
            'q6_order':['log10_gamma_c','log10_JS','log10_JR','log10_kS','log10_kR','u0'],
            'q6_bounds':{'lower':[-4,-7,-8,-4,-4,0],'upper':[0,math.log10(.001*(1-1e-8)),-2,1.5,1.5,30]},
            'anchors':a,'initial_gamma_cycle4_from_original_exact_anchor_ev':g0,
            'frozen_initial_series_current_a':{'command2V':initial_i2,'command5V':initial_i5},
            'transistor_current_upper_at_command2V_a':icap2,'SET_target_2V_gamma_lower_ev':g_target2,
            'RESET_peak_command_v':rpeak,'RESET_to_return_0p2_path_v':reset_path,
            'rows':rows,'selected_diagnostic_seeds':candidates[:2],
            'erratum':{'pre_review_json':'SERIES_ENTRY_REVIEW.pre_review.json','pre_review_sha256':original_hash,
                       'corrections':['Restart lower bound now uses min over adjacent gamma intervals of I(5V,gamma_lo)*(1+gc/gamma_hi); sampled product minima were not continuous lower bounds.',
                                      'RESET physical upper unbounded but numerical-box compatible counts are separate from finite-physical-upper counts. Infinite physical upper bounds are not incompatibility.'],
                       'unchanged':'All SET outward/return budget fields and no-SET-overlap conclusions are unchanged; cached spectral interpolation only, no new spectral integration or ODE.'},
            'scope':['Only1T1R4..35arrays read; no bare or holdout observations in this entry calculation.',
                     'Cycle4 initial state is anchored once; no independent RESET initial state or per-cycle resets.',
                     'EndpointQ10/Q90 plus5%are declared generous diagnostics, not exact per-record scoring gates.',
                     'SET2Varrival uses entire0..2V path and maximum transistor current at2V to favor feasibility.',
                     'SETreturn5..2 uses previously cached conservative current lower array and existing5%high endpoint sensitivity.',
                     'RESET budget links SETreturn0.2 toRESETreturn0.2 conditionally; only verified inactive-tail inequalities permit transporting these read bands to actual phase boundaries. No boundary state is reset.',
                     'This is a cached-table diagnostic; historical509point table audit maxerror0.000989decade is not new exact certification.',
                     'Within-record constant command-path rates and current measured transistor mapping are assumed; real timing may need review.',
                     'Separate group parameters are for locating a problem; they do not complete a unified model.'],
            'elapsed_s':time.monotonic()-started,'loaded_training_arrays_sha256':loaded.hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'SERIES_ENTRY_REVIEW.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# 1T1R单组入口诊断：原谱、beta1、A=0','',
           '只读4–35训练数组，固定原谱与T映射，不ODE、不优化。组别独立参数只是定位，不能称完成统一模型。','',
           '第4圈初态沿原0.2V锚固定一次。外扫2V已进入高电流，不能只验证5V“能启动”：本次同时检查0→2V闭合速率下界与5→2V返扫慢变化上界。','',
           '| gamma_c/eV | SET相容有限速率区间 | SET最小下/上界比 | 全JS保守下界 | RESET物理上界有限且盒内相容 | RESET物理上界无限但盒内相容 |','|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["gamma_c_ev"]:.6g} | {r["SET_finite_rate_interval_count"]} | {r["minimum_SET_outward_required_over_return_allowed"]:.6g} | {r["all_JS_under_activation_domain_ratio_lower_bound"]} | {r["RESET_finite_rate_interval_count"]} | {r["RESET_physical_upper_unbounded_but_box_compatible_count"]} |')
    lines+=['',f'选入q6诊断seed：{len(report["selected_diagnostic_seeds"])}。'+('当前声明的单组端点预算仍有SET速率冲突；没有给出伪可行seed。' if not candidates else '仅入场预算，完整周期/32圈轨迹尚未验证。'),
            'RESET预算以SET返扫0.2V读带为条件起态，末端0.2V的HRS带用于下次SET激活检查。只有J阈值高于各0.2→0尾段的驱动上界、保证尾段冻结，才能把读带沿用到真实相位边界；这些充分条件已保存并进入seed选择。否则尾段状态未定，不能重置。',
            '离散gamma_c及有限阈值网格/声明分位预算的失败不能自动替代原全曲线训练评分。真实时序和表插值近似仍须保留范围说明。',
            '', '勘误：已保留原JSON/MD为pre_review快照。SET重启驱动下界改用相邻Γ区间的I(Γ_lo)·(1+gc/Γ_hi)交叉端点界；原节点乘积最小值不能保证连续下界。RESET物理上界为无穷而盒内仍有有限速率的情形单列，不能判为不相容。SET外扫/返扫比值字段未变。',
            f'原JSON SHA256：{original_hash}',
            f'用时{report["elapsed_s"]:.3f}s；详细锚、速率区间与q6（若有）见JSON。']
    (OUT/'SERIES_ENTRY_REVIEW.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'elapsed_s':report['elapsed_s'],'selected_seeds':report['selected_diagnostic_seeds'],
                      'summary':[{k:r[k] for k in ['gamma_c_ev','SET_finite_rate_interval_count','minimum_SET_outward_required_over_return_allowed','all_JS_under_activation_domain_ratio_lower_bound','RESET_finite_rate_interval_count']} for r in rows]},indent=2))


if __name__=='__main__':main()
