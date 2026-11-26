"""Conditional frozen-envelope fit followed by a necessary reachability bound.

This is not a trajectory fit. State freezing in bare pre-SET and series SET
return is an explicit screening assumption; failure does not exclude every
finite-rate member of the family. Uses training observations only.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.optimize import least_squares,minimize_scalar
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from vulcan2d.transistor import measured_transistor_mapping

OUT=Path(__file__).parent
DATA=ROOT/'analysis/shared_hbn_2026-11-03/data/native_observations.npz'
REF=ROOT/'analysis/shared_gap_refinement_2026-11-03/block_fit.json'

def inverse(v,a,d,k,ron):
    lo=np.full_like(v,-40.);hi=np.log(np.maximum(v/ron,1e-300))
    for _ in range(52):
        m=(lo+hi)/2;i=np.exp(m)
        voltage=ron*i+d*np.arcsinh(i/a)+k*i**3
        hi=np.where(voltage>v,m,hi);lo=np.where(voltage>v,lo,m)
    return np.exp((lo+hi)/2)

def main():
    ref=json.loads(REF.read_text());ron=ref['shared_ron_ohm'];mapping=measured_transistor_mapping()
    fits={}; data={}
    with np.load(DATA) as raw:
        for top in ['1r','1t1r']:
            ids=ref['split'][top]['train']; v=raw[f'{top}_cycle_{ids[0]}_set_voltage']
            currents=np.array([abs(raw[f'{top}_cycle_{cid}_set_current']) for cid in ids])
            gm=10**np.mean(np.log10(np.maximum(currents,1e-300)),axis=0)
            index=np.arange(len(v));apex=np.argmax(v)
            mask=(v>=.1)&(v<=2.25)&(index<=apex) if top=='1r' else (v>=.06)&(index>apex)
            vv=v[mask];ii=gm[mask]
            vh=vv.copy() if top=='1r' else vv-np.array([mapping.voltage_for_current(i,1.1,'set') for i in ii])
            data[top]={'command':vv,'hbn':vh,'observed':ii}
            def residual(q):
                a,d=10**q[0],10**q[1];k=0. if top=='1r' else 10**q[2]
                return np.log10(inverse(vh,a,d,k,ron)/ii)/np.sqrt(len(ii))
            rows=[]
            for a0,d0,k0 in [(1e-9,.3,1e13),(1e-7,.1,1e12),(1e-8,.2,1e14)]:
                seed=np.log10([a0,d0] if top=='1r' else [a0,d0,k0])
                lower=[-14.,-3.] if top=='1r' else [-14.,-3.,5.]
                upper=[-2.,1.] if top=='1r' else [-2.,1.,18.]
                opt=least_squares(residual,seed,bounds=(lower,upper),max_nfev=150,
                                  diff_step=1e-4,ftol=1e-10,xtol=1e-10,gtol=1e-8)
                rows.append({'q':opt.x.tolist(),'rmse_decade':float(np.linalg.norm(residual(opt.x))),
                             'success':bool(opt.success),'optimality':float(opt.optimality)})
            chosen=min(rows,key=lambda x:x['rmse_decade']);a,d=10**np.array(chosen['q'][:2]);k=0. if top=='1r' else 10**chosen['q'][2]
            fits[top]={'fits':rows,'a_A':a,'d_V':d,'k_V_per_A3':k,'selected':chosen}
    ar,dr=fits['1r']['a_A'],fits['1r']['d_V'];at,dt,kt=fits['1t1r']['a_A'],fits['1t1r']['d_V'],fits['1t1r']['k_V_per_A3']
    delta=np.log(at/ar); result={'state_inversion_nondegenerate':bool(delta!=0 and dr!=dt)}
    if delta!=0 and dr!=dt:
        v0=(dr-dt)/delta;ut=dt/v0;ur=dr/v0
        result.update(v0_V=v0,u_soft=ut,u_bare_pre=ur,delta_u=delta)
        if v0>0 and ut>0 and ur>ut:
            i0=at*np.exp(ut);b=kt*1e-9/ut;ipk=float(max(data['1t1r']['observed']));jc=ipk*np.exp(ut)
            es=v0*np.arcsinh(jc/i0);c=b*(jc/.001)**3
            vc=lambda u:ron*jc*np.exp(-u)+u*es+u*c*np.exp(-3*u)
            nodes=np.linspace(0,ur,10001);voltages=vc(nodes);im=int(np.argmax(voltages));up=float(nodes[im])
            local=minimize_scalar(lambda u:-vc(u),bounds=(max(0,up-ur/1000),min(ur,up+ur/1000)),method='bounded')
            peak=max(float(voltages[im]),float(-local.fun))
            kr=kt*ur/ut
            result.update(i0_A=i0,access_B_V=b,minimum_Jcrit_A=jc,minimum_eth_set_V=es,
                C_V=c,threshold_barrier_max_V=peak,threshold_barrier_u=float(local.x),
                cubic_bound_at_one_third_V=float(c/(3*np.e)) if ur>=1/3 else None,
                bare_prefit_cubic_fraction_max=float(max(kr*data['1r']['observed']**3/data['1r']['hbn'])),
                frozen_family_can_cross_to_zero_with_5V=bool(peak<=5),
                frozen_family_can_cross_to_zero_with_2p8V=bool(peak<=2.8),
                compliance_can_cross_to_zero=bool(jc<=.001))
    report={'scope':'training-only frozen-envelope and threshold reachability diagnostic, not full dynamic fit',
        'fits':fits,'derived_shared_parameters_and_necessary_gates':result,
        'assumptions':['BareHRS0.1–2.25V and seriesSETreturn>=.06V states are treated frozen.',
            'Barepre cubic term neglected at fitting, then explicitly checked.',
            'CommonRon,V0,I0,B; oldnonzeroSETkinetics; closedbare endpoint must traverse overlappingu range.',
            'Observed mean-current deembedding is diagnostic, not prediction.',
            'Reachabilityfailure rejects this frozen-envelope/threshold subcase, not all finite-rate trajectories.'],
        'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),DATA,REF]}}
    (OUT/'actual_access_reachability_screen.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
