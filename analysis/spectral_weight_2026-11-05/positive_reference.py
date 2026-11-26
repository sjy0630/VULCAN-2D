"""Independent exponential midpoint check of the first three bare cycles.

Each frozen-rate update is a convex birth/death transition; no clipping,
state floor, or trial of an unphysical population is needed.
"""
import json
import math
import time
from dataclasses import fields
import numpy as np
import check_weight as w


class LowGammaContinuation(w.WeightMaterial):
    """Continuous small-Gamma asymptote, checked against direct integration."""
    def validate_state(self,state,*,table=True):
        q,a=super().validate_state(state,table=False)
        if not 0<q<1: raise ValueError('Strict finite q required')
        gamma=self.readout.gamma_c_ev*q/(1-q)
        if gamma<=0 or math.log(gamma)>self.readout.lg[-1]: raise ValueError('Invalid gamma')
        return q,a

    def _table_abs(self,magnitude,q,sign,*,reference=False):
        gamma=self.readout.gamma_c_ev*q/(1-q)
        anchor=math.exp(self.readout.lg[0])*(1+1e-12)
        if gamma<anchor:
            qa=anchor/(anchor+self.readout.gamma_c_ev)
            return super()._table_abs(magnitude,qa,sign,reference=reference)*(gamma/anchor)
        return super()._table_abs(magnitude,q,sign,reference=reference)


def advance(old, rates, dt):
    out=[]
    for value,(birth,death) in zip(old,rates):
        rate=birth+death
        out.append(value*math.exp(-rate*dt)+(birth/rate)*(-math.expm1(-rate*dt)) if rate else value)
    return np.array(out)


def run(material,problem,initial,h):
    p=material.parameters
    state=np.array([initial[0],initial[1]/p.a_max_s])
    results={}
    def rates(v,y):
        physical=(float(y[0]),float(y[1]*p.a_max_s))
        point=circuit.solve(v,material,physical)
        vh=point['local_voltage_v']; mag=abs(vh)
        if mag==0: return ((0.,0.),(0.,0.))
        sign=1 if vh>0 else -1; chi=(1-sign)*p.b
        drive=abs(point['current_a'])
        if material.drive_mode=='per_reference_weight': drive/=material.weight(physical)
        return ((p.kappa_positive_per_v2*mag*drive/p.J_positive_a*math.exp(-chi),
                 p.kappa_positive_per_v2*mag*math.exp(chi+p.lambda_per_v*mag)),
                (p.kappa_a_per_v2*mag*math.exp(sign*p.b)*y[0],
                 p.kappa_a_per_v2*mag*p.rho*math.exp(-sign*p.b)*(1-y[0])))
    for cid in (1,2,3):
        for phase in ('set','reset'):
            key=('1r',cid,phase)
            circuit=w.k.BareCircuit(.001 if phase=='set' else None)
            voltage=problem.data[key]['command_voltage_v']; start=0.
            states=[]; currents=[]
            for stop in voltage:
                distance=abs(stop-start)
                n=int(math.ceil(distance/h))
                if n:
                    dt=distance/n; direction=(stop-start)/distance
                    for j in range(n):
                        v=start+direction*j*dt
                        half=advance(state,rates(v,state),dt/2)
                        state=advance(state,rates(v+direction*dt/2,half),dt)
                assert np.all((state>=0)&(state<=1))
                physical=(float(state[0]),float(state[1]*p.a_max_s))
                currents.append(circuit.solve(float(stop),material,physical)['current_a'])
                states.append(state.copy()); start=stop
            results['__'.join(map(str,key))]=dict(states=np.array(states),current=np.array(currents))
    return results


def compare(a,b):
    di=du=dx=0.
    for key in a:
        aa,bb=a[key],b[key]; mask=aa['current']!=0
        di=max(di,float(np.max(abs(np.log10(abs(aa['current'][mask]/bb['current'][mask]))))))
        du=max(du,float(np.max(abs(np.log(aa['states'][:,0]/bb['states'][:,0])))))
        dx=max(dx,float(np.max(abs(aa['states'][:,1]-bb['states'][:,1]))))
    return dict(max_log_current_decade=di,max_log_q=du,max_x=dx)


def main():
    H=w.HERE; result=json.loads((H/'RESULT.json').read_text()); problem=w.tp.load_training_problem()
    base=w.q1.load_q1_material(w.k.Parameters(**result['parameters'],enable_a=True),
        candidate_source_id='positive_exponential_midpoint',candidate_source_sha256=w.sha(__file__))
    values={f.name:getattr(base,f.name) for f in fields(w.k.CoupledMaterial)}
    out=dict(scope='Only bare cycles 1-3, same two unrefitted candidates; no full-task score',cases={},
             integrator='Second-order exponential midpoint, convex transitions; no clipping/floor',
             original_failure_preserved=True)
    out['sources']={str(H/'BOUNDARY_DIAGNOSTIC_PLAN.md'):w.sha(H/'BOUNDARY_DIAGNOSTIC_PLAN.md'),str(H/'positive_reference.py'):w.sha(__file__),str(H/'check_weight.py'):w.sha(H/'check_weight.py')}
    started=time.monotonic()
    for label,mode in [('weighted_total','total'),('weighted_per_reference','per_reference_weight')]:
        model=LowGammaContinuation(**values,drive_mode=mode); runs=[]
        continuation=[]
        for gamma in [1e-26,1e-30]:
            q=gamma/(gamma+model.readout.gamma_c_ev)
            for v in [-2.,-.2,-.001,.001,.2,2.]:
                ref=w.direct(model,v,q)[0]
                continuation.append(abs(math.log10(model.current(v,(q,0))/ref)))
        assert max(continuation)<.01
        for h in [.01,.005,.0025]:
            r=run(model,problem,result['initial_states']['bare1'],h); runs.append(r)
            np.savez_compressed(H/f'{label}_positive_h{h}.npz',**{key+'__'+name:a for key,row in r.items() for name,a in row.items()})
        changes=[compare(a,b) for a,b in zip(runs[:-1],runs[1:])]
        radau={}
        with np.load(H/(label+'_refined.npz')) as z:
            for key in runs[-1]:
                if key=='1r__3__reset': continue
                radau[key]=dict(current=z[key+'__current_a'],states=np.column_stack((z[key+'__state_q'],z[key+'__state_x'])))
        agreement=compare(radau,{key:runs[-1][key] for key in radau})
        spectrum=[]
        for key,row in runs[-1].items():
            cid,phase=key.split('__')[1:]; vv=problem.data['1r',int(cid),phase]['command_voltage_v']
            for j in np.linspace(1,len(vv)-2,3,dtype=int):
                q,x=row['states'][j]; physical=(float(q),float(x*model.parameters.a_max_s))
                circuit=w.k.BareCircuit(.001 if phase=='set' else None)
                v=circuit.solve(float(vv[j]),model,physical)['local_voltage_v']
                exact=w.direct(model,v,float(q))[0]*model.weight(physical)
                spectrum.append(abs(math.log10(exact/row['current'][j])))
        out['cases'][label]=dict(completed_phases=6,step_changes=changes,agreement_with_completed_Radau=agreement,
            low_gamma_continuation_error_decade=max(continuation),continuation_samples=len(continuation),
            max_spectrum_error_decade=max(spectrum),n_spectrum_samples=len(spectrum),
            final_q=float(runs[-1]['1r__3__reset']['states'][-1,0]),
            final_x=float(runs[-1]['1r__3__reset']['states'][-1,1]),
            passed=changes[-1]['max_log_current_decade']<.001 and agreement['max_log_current_decade']<.001 and max(spectrum)<.01)
        w.save('POSITIVE_REFERENCE.json',out)
        print(json.dumps(dict(case=label,**out['cases'][label])),flush=True)
    out.update(elapsed_s=time.monotonic()-started,status='independent_boundary_diagnostic_complete')
    w.save('POSITIVE_REFERENCE.json',out)


if __name__=='__main__': main()
