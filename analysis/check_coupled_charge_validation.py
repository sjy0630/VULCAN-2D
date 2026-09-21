"""High-precision master equation, shape checks and kinetic-limit audit."""
from decimal import Decimal, localcontext
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'analysis'))
from compare_parallel_paths import assemble,VOLTAGE
from vulcan2d.coupled_charge import stationary,log_rates,EDGES,charging
from vulcan2d.contact_network import log10_shape as legacy_shape
from vulcan2d.transport_candidates import TransportAssumptions


def decimal_stationary(voltage,shared,energy):
    """Independent 160-digit Gaussian elimination using elementary rates."""
    with localcontext() as ctx:
        ctx.prec=160
        rates=[Decimal(str(float(x))).exp() for x in log_rates(voltage,shared,energy)[0]]
        q=[[Decimal(0) for _ in range(4)] for _ in range(4)]
        for rate,(a,b) in zip(rates,EDGES):q[b][a]+=rate;q[a][a]-=rate
        q[3]=[Decimal(1)]*4
        matrix=[row+[Decimal(1 if i==3 else 0)] for i,row in enumerate(q)]
        for col in range(4):
            pivot=max(range(col,4),key=lambda i:abs(matrix[i][col]))
            matrix[col],matrix[pivot]=matrix[pivot],matrix[col]
            divisor=matrix[col][col]
            matrix[col]=[x/divisor for x in matrix[col]]
            for i in range(4):
                if i!=col:
                    factor=matrix[i][col]
                    matrix[i]=[a-factor*b for a,b in zip(matrix[i],matrix[col])]
        p=[r[-1] for r in matrix]
        il=rates[0]*p[0]-rates[1]*p[1]+rates[4]*p[2]-rates[5]*p[3]
        return np.array([float(x) for x in p]),float(il.copy_abs().log10())


def run():
    path=ROOT/'analysis/coupled_charge_comparison.json';main=json.loads(path.read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for p,h in main['input_sha256'].items():
        if digest(ROOT/p)!=h:raise RuntimeError('Input changed: '+p)
    ids,y,w,_=assemble();checks=[];direct=[];slopes=[];rates=[]
    for label,split in main['splits'].items():
        fit=split['fit'];shared=fit['shared_parameters'];states=np.array(fit['states'])
        out=stationary(VOLTAGE[None,:],shared,states[:,1,None])
        p=out['probabilities'];scale=np.maximum(np.abs(out['current_scale']),1e-300)
        checks.append({'split':label,'maximum_probability_sum_error':float(np.max(np.abs(p.sum(axis=0)-1))),
                       'minimum_probability':float(np.min(p)),
                       'maximum_cycle_affinity_residual':float(np.max(np.abs(out['forward_reverse_cycle_log_ratio_residual']))),
                       'maximum_gross_left_right_relative_mismatch':float(np.max(np.abs(out['gross_left_current']+out['gross_right_current'])/scale)),
                       'maximum_side_current_relative_to_net':float(np.max(np.abs(out['gross_side_current'])/scale))})
        # Initial states plus median and final training state, both polarities and wide voltage.
        selected=np.unique(np.r_[np.arange(min(3,len(states))),len(states)//2,len(states)-1])
        for j in selected:
            for voltage in (-.3,-.1,.1,.3,2.,3.):
                prob,logi=decimal_stationary(voltage,shared,states[j,1])
                own=stationary(voltage,shared,states[j,1])
                direct.append({'split':label,'training_cycle_id':split['training_ids'][j],'voltage_V':voltage,
                               'maximum_probability_difference':float(np.max(np.abs(prob-own['probabilities']))),
                               'log10_current_difference_decade':float(abs(logi-own['log10_abs_current_scale']))})
        if label=='all_descriptive':
            prediction=states[:,0,None]+out['log10_abs_current_scale']
            for cid,observed,pred,weight in zip(ids,y,prediction,w):
                mask=(VOLTAGE>=2)&(VOLTAGE<=3)&(weight>0);x=np.log10(VOLTAGE[mask]);x-=x.mean()
                slopes.append({'cycle_id':int(cid),'n_points':int(sum(mask)),
                               'observed_slope':float(observed[mask]@x/(x@x)),
                               'predicted_slope':float(pred[mask]@x/(x@x))})
            for logt in (-8.,-4.,0.,4.,8.):
                other=np.array(shared);other[5]=logt
                shifted=stationary(VOLTAGE[None,:],other,states[:,1,None])['log10_abs_current_scale']
                rates.append({'ln_Gamma_t_over_R':logt,'maximum_current_change_decade':float(np.max(np.abs(shifted-out['log10_abs_current_scale'])))})
    fit=main['splits']['all_descriptive']['fit'];shared=np.array(fit['shared_parameters']);states=np.array(fit['states'])
    assumptions=TransportAssumptions();vr=assumptions.reference_voltage_v
    slow=shared.copy();slow[4]=-30.;slow[5]=0.
    grid=np.r_[VOLTAGE,vr]
    raw=stationary(grid[None,:],slow,states[:,1,None])['log10_abs_current_scale']
    normalized=raw[:,:-1]-raw[:,-1,None]
    default=stationary(grid[None,:],shared,states[:,1,None])['log10_abs_current_scale']
    default=default[:,:-1]-default[:,-1,None]
    u=charging(shared)[0];ed,xd,xt=shared[:3];t=(xt-xd)/(1-xd)
    if not .01<=t<=.99:raise RuntimeError('Selected geometry cannot be mapped to the archived ordered-site limit')
    legacy=np.array([legacy_shape(VOLTAGE,'finite_left',[ed,e,xd,t,u],assumptions) for e in states[:,1]])
    limits={'interpretation':'Rate-separated four-state model tends to the already tested R-equilibrated slow-left-contact kernel; not a new independent mechanism.',
            'maximum_slow_limit_vs_archived_shape_decade':float(np.max(np.abs(normalized-legacy))),
            'maximum_selected_shape_vs_archived_limit_decade':float(np.max(np.abs(default-legacy))),
            'selected_Gamma_L_over_R':float(np.exp(shared[4])), 'conditional_U_ev':float(u)}
    files=[Path(__file__),path,ROOT/'vulcan2d/contact_network.py',ROOT/'vulcan2d/transport_candidates.py']+[ROOT/p for p in main['input_sha256']]
    result={'scope':'Independent 160-digit master-equation validation and shape/kinetic-limit checks; no new experimental data',
            'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in dict.fromkeys(files)},
            'training_numerics':checks,'independent_decimal_checks':direct,'positive_high_voltage_slopes':slopes,
            'side_rate_fixed_parameter_sensitivity':rates,'slow_contact_limit':limits}
    (ROOT/'analysis/coupled_charge_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Decimal checks',len(direct),'max log error',max(r['log10_current_difference_decade'] for r in direct),flush=True)
    print('Slow contact limit',limits,flush=True);print('Rate profile',rates,flush=True)


if __name__=='__main__':run()
