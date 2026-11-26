"""Bare-only aggregate necessary budgets from saved spectrum nodes; no solver."""
from pathlib import Path
import ast, hashlib, json, math
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
OLD=ROOT/'analysis/spectral_dynamic_2026-11-03'

def main():
    paths=[OLD/'fit_initiation.py',OLD/'seed_screen.json',OLD/'finite_rate_order_audit.json',
           OLD/'transport_g010_v0025.npz',ROOT/'analysis/beta_feedback_2026-11-03/fixed_spectrum_budget_arrays.npz',
           ROOT/'analysis/shared_hbn_2026-11-03/data/native_observations.npz',
           ROOT/'analysis/conditional_clock_2026-11-03/budget_seeds.json',
           ROOT/'analysis/constrained_shared_2026-11-03/feature_constraints.json']
    seed=json.loads(paths[1].read_text()); prior=json.loads(paths[2].read_text())
    # Read the actual declared algorithmic box, without importing/running a fit.
    assignments={n.targets[0].id:n.value for n in ast.parse(paths[0].read_text()).body
                 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    box={k:eval(compile(ast.Expression(assignments[k]),str(paths[0]),'eval'),
                {'np':np,'__builtins__':{}}) for k in ('LOWER','UPPER')}
    lower,upper=box['LOWER'],box['UPPER']
    with np.load(paths[3]) as z:
        gamma=np.exp(z['log_gamma']); volt=np.exp(z['log_voltage']); cur=np.exp(z['log_current'])
        metadata=json.loads(str(z['metadata']))
    assert np.all(np.diff(gamma)>0) and np.all(np.diff(volt)>0)
    def vi(v,side):
        idx=np.searchsorted(volt,v,side='left' if side=='upper' else 'right')
        return min(len(volt)-1,idx) if side=='upper' else max(0,idx-1)
    def gamma_bound(v,current,branch,side):
        # Upper-voltage row yields gamma lower bound, and conversely.
        k=vi(v,'upper' if side=='lower' else 'lower'); f=cur[branch,:,k]
        if side=='lower': ids=np.flatnonzero(f<=current); j=int(ids[-1])
        else: ids=np.flatnonzero(f>=current); j=int(ids[0])
        return float(gamma[j]),j
    train=seed['training_ids']['1r']; arrays=[]
    with np.load(paths[5]) as z:
        for cid in train:
            prefix=f'1r_cycle_{cid}_set'; v=z[prefix+'_voltage']; i=np.abs(z[prefix+'_current'])
            if arrays: assert np.array_equal(v,arrays[0][0])
            arrays.append((v.copy(),i.copy()))
        reset_wave=z[f'1r_cycle_{train[0]}_reset_voltage'].copy()
        reset_curr=np.array([np.abs(z[f'1r_cycle_{cid}_reset_current']) for cid in train])
    v=arrays[0][0]; y=np.array([a[1] for a in arrays]);ap=int(np.argmax(v))
    j2=int(np.argmin(abs(v[:ap+1]-2)));jp=int(np.argmin(abs(v[:ap+1]-2.55)))
    q2=np.quantile(y[:,j2],[.1,.9]);qp=np.quantile(y[:,jp],[.1,.9])
    assert np.all(y[:,jp]<.995e-3)
    g2lo,jg2lo=gamma_bound(v[j2],q2[0],0,'lower')
    g2hi,jg2hi=gamma_bound(v[j2],q2[1],0,'upper')
    gprehi,jgprehi=gamma_bound(v[jp],qp[1],0,'upper')
    ga,jga=gamma_bound(2.8,.995e-3,0,'lower')
    mask=np.arange(len(reset_wave))>int(np.argmax(abs(reset_wave)))
    jrread=int(np.argmin(np.where(mask,abs(abs(reset_wave)-.2),np.inf)))
    reset_tail_upper=float(np.quantile(reset_curr[:,jrread],.9))
    gt,jgt=gamma_bound(abs(reset_wave[jrread]),reset_tail_upper,1,'upper')
    slow_v=v[j2:jp+1];dv=np.diff(slow_v)
    fi=cur[0,jg2lo:jgprehi,:][:,[vi(x,'lower') for x in slow_v[:-1]]].T
    gs=gamma[jg2lo:jgprehi+1]
    with np.load(paths[4]) as z: reset_lower_i=z['bare_reset_current_lower_a'].copy()
    reset_vmax=float(np.max(abs(reset_wave))); rpath=2*reset_vmax-abs(float(reset_wave[jrread]))
    # Every RESET trajectory from a SET low-read state to its tail must cross
    # this state band. Both endpoints use existing conservative read budgets.
    gset=seed['finite_read_bounds']['gamma_set_low_min_ev']
    greset=seed['finite_read_bounds']['gamma_reset_low_min_ev']
    gR0=seed['anchors']['bare_set_0p2V_geomean']['gamma_ev']
    init_i28=json.loads(paths[6].read_text())['inputs']['initial_bare_current_at_2p8V_a']
    features=json.loads(paths[7].read_text())['features']
    peak_i=features['1r.reset.outward.peak_current']['proposed_empirical_acceptance_interval'][0]
    peak_vlo,peak_vhi=features['1r.reset.outward.peak_abs_voltage']['proposed_empirical_acceptance_interval']
    gp,jgp=gamma_bound(peak_vhi,peak_i,1,'lower')
    r_apex=int(np.argmax(abs(reset_wave)))
    peak_nodes=np.r_[np.abs(reset_wave[:r_apex+1])[np.abs(reset_wave[:r_apex+1])<peak_vlo],peak_vlo]
    peak_i_lower=np.array([0. if x==0 else cur[1,jgp,vi(x,'lower')] for x in peak_nodes[:-1]])
    jtop=int(np.searchsorted(gamma,gset,side='left'))
    # Exact common low-read bound used at the last edge; current at the larger
    # saved gamma node bounds that final partial cell from above.
    rg=np.r_[gamma[jgt:jtop],gset]
    assert np.all(np.diff(rg)>0)
    ri=np.array([cur[1,min(len(gamma)-1,int(np.searchsorted(gamma,x,side='left'))),vi(reset_vmax,'upper')] for x in rg[1:]])
    gc_values=[r['gamma_c_ev'] for r in prior['profiled_rate_and_clock_rows']
               if lower[0]-1e-12<=math.log10(r['gamma_c_ev'])<=upper[0]+1e-12]
    cases=[]; candidates=[]
    jrgrid=np.geomspace(10**lower[2],10**upper[2],181)
    def interval(klo,kup,kmin,kmax):
        physical_finite=np.isfinite(kup)
        lo=np.maximum(klo,kmin);hi=np.minimum(kup,kmax)
        ok=np.isfinite(klo)&(klo>0)&(lo<hi)
        return lo,hi,ok,physical_finite
    for gc in gc_values:
        u0=math.log1p(gc/gR0);ul=math.log1p(gc/gset);ur=math.log1p(gc/greset)
        if not lower[5]<=u0<=upper[5]:continue
        jsmax=min(10**upper[1],init_i28*(1+gc/gR0),.001*math.exp(ul))
        if .98*jsmax<=10**lower[1]:continue
        js=np.geomspace(10**lower[1],.98*jsmax,181);c=np.log(.001/js)
        B=math.log1p(gc/gprehi);A=math.log1p(gc/ga);B2=math.log1p(gc/g2hi)
        jump=np.maximum(0,np.log((B+c)/(A+c))/(2.8-float(v[jp])))
        arrival=np.maximum(0,np.log((B2+c)/(A+c))/.8)
        read=np.maximum(0,np.log((B2+c)/(ul+c))/7.95)
        kslo=np.maximum.reduce([jump,arrival,read])
        jlow=np.min(fi*(1+gc/gs[None,1:]),axis=1)
        sint=np.sum(dv[:,None]*np.maximum(np.log(jlow[:,None]/js[None,:]),0),axis=0)
        du=math.log1p(gc/g2lo)-B
        ksup=np.divide(du,sint,out=np.full(len(js),np.inf),where=sint>0)
        sl,sh,so,sfinite=interval(kslo,ksup,10**lower[3],10**upper[3])
        rdu=np.log1p(gc/rg[:-1])-np.log1p(gc/rg[1:])
        rh=np.maximum(np.log(ri[:,None]/jrgrid[None,:])+np.log1p(gc/rg[:-1,None]),0)
        times=np.divide(rdu[:,None],rh,out=np.full_like(rh,np.inf),where=rh>0)
        krlo=np.sum(times,axis=0)/rpath
        rint=np.sum(.002*np.maximum(np.log(np.maximum(reset_lower_i[:,None],1e-300)/jrgrid[None,:]),0),axis=0)
        krup_without_peak=np.divide(ur,rint,out=np.full(len(jrgrid),np.inf),where=rint>0)
        # If the required peak occurs at V>=Vlo and I>=Imin by Vhi,
        # monotone RESET must remain u<=u_peak_max throughout0..Vlo.
        peak_int=np.sum(np.diff(peak_nodes)[:,None]*np.maximum(np.log(np.maximum(peak_i_lower[:,None],1e-300)/jrgrid[None,:]),0),axis=0)
        peak_umax=math.log1p(gc/gp)
        krup_peak=np.divide(peak_umax,peak_int,out=np.full(len(jrgrid),np.inf),where=peak_int>0)
        krup=np.minimum(krup_without_peak,krup_peak)
        rl,rhbox,ro,rfinite=interval(krlo,krup,10**lower[4],10**upper[4])
        _,_,ro0,rfinite0=interval(krlo,krup_without_peak,10**lower[4],10**upper[4])
        sf=so&sfinite;rf=ro&rfinite
        row={'gamma_c_ev':gc,'SET_bounded_count':int(sum(sf)),'SET_unbounded_upper_count':int(sum(so&~sfinite)),
             'RESET_bounded_count':int(sum(rf)),'RESET_unbounded_upper_count':int(sum(ro&~rfinite)),
             'RESET_bounded_count_without_peak_gate':int(sum(ro0&rfinite0)),
             'JS_ceiling_a':jsmax,'initial_u_bare':u0}
        if np.any(sf) and np.any(rf):
            ss=np.where(sf,sh/sl,0);rr=np.where(rf,rhbox/rl,0)
            a=int(np.argmax(ss));b=int(np.argmax(rr));ks=math.sqrt(sl[a]*sh[a]);kr=math.sqrt(rl[b]*rhbox[b])
            candidate={'gamma_c_ev':gc,'JS_a':float(js[a]),'JR_a':float(jrgrid[b]),'kS_per_command_V':ks,'kR_per_command_V':kr,
                       'initial_u_bare':u0,'SET_allowed_interval_within_original_box':[float(sl[a]),float(sh[a])],
                       'RESET_allowed_interval_within_original_box':[float(rl[b]),float(rhbox[b])],
                       'SET_physical_upper':float(ksup[a]),'RESET_physical_upper':float(krup[b]),
                       'minimum_interval_ratio':float(min(ss[a],rr[b])),
                       'bare_six_coordinates':[math.log10(gc),math.log10(js[a]),math.log10(jrgrid[b]),math.log10(ks),math.log10(kr),u0]}
            row['candidate']=candidate;candidates.append(candidate)
        cases.append(row)
    selected=[]
    for x in sorted(candidates,key=lambda x:x['minimum_interval_ratio'],reverse=True):
        if not selected or all(abs(math.log10(x['gamma_c_ev']/a['gamma_c_ev']))>=.5 for a in selected):selected.append(x)
        if len(selected)==2:break
    report={'status':'bare_only_aggregate_cached_node_budget_diagnostic_no_ODE','fixed_beta':1.,'fixed_A_RESET':0.,
            'algorithmic_bounds_source':'analysis/spectral_dynamic_2026-11-03/fit_initiation.py',
            'original_lower':lower.tolist(),'original_upper':upper.tolist(),'loaded_spectrum_metadata':metadata,
            'aggregate_constraints':{'training_ids':train,'I_2V_q10_q90_a':q2.tolist(),'I_2p55V_q10_q90_a':qp.tolist(),
              'gamma_2V_conservative_band_ev':[g2lo,g2hi],'gamma_2p55V_upper_ev':gprehi,'gamma_2p8V_arrival_lower_ev':ga,
              'RESET_return_0p2V_current_Q90_a':reset_tail_upper,'gamma_RESET_tail_upper_ev':gt,
              'SET_read_gamma_lower_ev':gset,'RESET_early_read_gamma_lower_ev':greset,'RESET_available_path_to_tail_v':rpath,
              'RESET_peak_current_lower_a':peak_i,'RESET_peak_voltage_interval_v':[peak_vlo,peak_vhi],
              'RESET_peak_gamma_lower_ev':gp},
            'cases':cases,'selected_diagnostic_starts':selected,
            'scope':['Only bare training data and bare material budgets constrain rates; no series rate or alpha constraint.',
              '2V/current/low-read/tail are existing aggregate targets;2.55V aggregate Q10-Q90 is a new declared bare-entry diagnostic, not a silently added fitting gate.',
              'Cached spectrum nodes provide conservative monotone rectangles; no new spectral quadrature, interpolation evaluation, inversion solver, ODE or optimization.',
              'Fixed spectrum and beta1 A0 law, original algorithmic box; original discrete gamma_c values inside that box only.',
              'Finite budget overlap is necessary only and does not validate a trajectory, fit, peak, mechanism or parameter transfer.',
              'Infinite necessary upper bounds are separately counted as uninformative; not used as a successful or failed physical bound.',
              'Ideal SET compliance, continuous monotone state and constant command-path rates remain conditional; numerical cache is not interval arithmetic.'],
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'bare_entry_budget_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'n_gamma':len(cases),'any_bounded_both':len(candidates),'selected':selected,
                     'SET_bounded_gamma':sum(x['SET_bounded_count']>0 for x in cases),
                     'RESET_bounded_gamma':sum(x['RESET_bounded_count']>0 for x in cases),
                     'aggregate':report['aggregate_constraints']},indent=2))

if __name__=='__main__':main()
