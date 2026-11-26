"""Finite, unfitted spectral-weight controls; separate physics and numerics."""
from dataclasses import dataclass, fields
from pathlib import Path
import hashlib
import json
import math
import sys
import time
import numpy as np
from scipy.integrate import quad
from scipy.special import digamma, roots_hermitenorm, expit

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
sys.path.insert(0, str(ANALYSIS/'coupled_linear_recovery_2026-11-04'))
import q1_material as q1
tp, k = q1.training, q1.kernel
import feature_audit as fa


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, data):
    def numeric(value):
        if isinstance(value,np.generic): return value.item()
        raise TypeError(type(value).__name__)
    (HERE/name).write_text(json.dumps(data, indent=2, allow_nan=False,default=numeric)+'\n')


@dataclass(frozen=True)
class WeightMaterial(q1.Q1Material):
    drive_mode: str = 'total'

    def weight(self, state):
        # a is only the inherited solver's scaled coordinate A*x, not a branch.
        return 1 + self.readout.ron_ohm * state[1]

    def components(self, voltage_hbn_v, state, *, reference=False):
        base = super().components(voltage_hbn_v, state, reference=reference)
        base['hard_current_a'] = (self.weight(state)-1)*base['soft_current_a']
        base['current_a'] = base['soft_current_a'] + base['hard_current_a']
        return base

    def rate_components(self, state, point):
        terms = super().rate_components(state, point)
        if self.drive_mode == 'per_reference_weight':
            terms['capture_q_per_v'] /= self.weight(state)
        elif self.drive_mode != 'total':
            raise ValueError('Unknown drive assumption')
        return terms

    def metadata(self):
        out = super().metadata()
        out.update(material_variant='spectral_weight_'+self.drive_mode,
                   state_definition='q=Gamma/(Gamma+Gamma_c), x in [0,1]; inherited a slot stores A*x only',
                   weight_law='W=1+beta*x', beta=self.parameters.a_max_s*self.readout.ron_ohm,
                   branch_alias_note='hard_current_a=(W-1)*F; no independent Ohmic branch',
                   total_current_drives_capture=self.drive_mode=='total',
                   q_drive='I' if self.drive_mode=='total' else 'I/W',
                   microscopic_closure='Common network coupling q; population-conditional q would need birth mixing',
                   parameters_recalibrated=False)
        return out


def direct(material, voltage, q, order=None):
    """Fermi energy integral analytically evaluated, ensemble integrated in z."""
    r = material.readout
    shape = r.shape_metadata
    gamma = r.gamma_c_ev*q/(1-q)
    thermal = 8.617333262145e-5*shape['temperature_k']
    v = np.asarray(voltage)

    def at(z):
        energy = shape['median_energy_ev']*np.exp(shape['sigma_log_energy']*z)
        a = .5+(gamma+1j*(energy+shape['eta']*v))/(2*np.pi*thermal)
        b = .5+(gamma+1j*(energy-(1-shape['eta'])*v))/(2*np.pi*thermal)
        return gamma/r.ron_ohm*(digamma(a)-digamma(b)).imag

    if order is not None:
        z, w = roots_hermitenorm(order)
        return sum(wi*at(zi) for zi, wi in zip(z,w))/sum(w)
    # Normalize by gamma so very small states do not pass on absolute tolerance.
    value, error = quad(lambda z: float(at(z))/gamma*np.exp(-z*z/2)/np.sqrt(2*np.pi),
                        -10, 10, epsabs=1e-13, epsrel=1e-10, limit=400)
    return value*gamma, error*gamma


def diagnostics(material):
    r=material.readout
    spectral=[]
    for q in [1e-6,1e-3,.1,.8]:
        for v in [-2.,-.2,.2,2.]:
            tab=material.components(v,(q,0))['current_a']
            ref,err=direct(material,v,q)
            high=float(direct(material,v,q,512))
            spectral.append(dict(q=q,v=v,table=tab,adaptive=ref,estimated_abs_error=err,
                table_error_decade=abs(math.log10(tab/ref)),
                gh512_error_decade=abs(math.log10(high/ref))))
    # Independent integration over electron energy at one moderately broad level.
    thermal=8.617333262145e-5*r.shape_metadata['temperature_k']
    energy=.3; gamma=.02; eta=r.shape_metadata['eta']; v=.7
    f=lambda e: gamma**2/((e-energy)**2+gamma**2)*(expit(-(e-(1-eta)*v)/thermal)-expit(-(e+eta*v)/thermal))
    raw,err=quad(f,-3,3,epsabs=1e-12,epsrel=1e-11,points=[energy],limit=300)
    a=.5+(gamma+1j*(energy+eta*v))/(2*np.pi*thermal)
    b=.5+(gamma+1j*(energy-(1-eta)*v))/(2*np.pi*thermal)
    analytic=gamma*(digamma(a)-digamma(b)).imag
    jac=[]
    for q in [1e-6,1e-3,.1,.8]:
        for label,vv in [('low',np.geomspace(1e-5,1e-3,12)),('wide',np.geomspace(.01,4,36))]:
            # Derivatives of log I with respect to log Gamma and log W.
            g=r.gamma_c_ev*q/(1-q); h=1e-4
            qp=g*np.exp(h)/(g*np.exp(h)+r.gamma_c_ev)
            qm=g*np.exp(-h)/(g*np.exp(-h)+r.gamma_c_ev)
            plus=np.array([direct(material,float(v),qp)[0] for v in vv])
            minus=np.array([direct(material,float(v),qm)[0] for v in vv])
            d=(np.log(plus)-np.log(minus))/(2*h)
            s=np.linalg.svd(np.column_stack((d,np.ones_like(d))),compute_uv=False)
            jac.append(dict(q=q,window=label,sensitivity_min=float(min(d)),sensitivity_max=float(max(d)),
                            singular_ratio=float(s[-1]/s[0])))
    checks=[]
    weighted=WeightMaterial(**{f.name:getattr(material,f.name) for f in fields(k.CoupledMaterial)})
    for q in [1e-6,.1,.8]:
        for x in [0.,.5,1.]:
            state=(q,material.parameters.a_max_s*x)
            for v in [-4.,-.2,0.,.2,4.]:
                c=weighted.current(v,state); base=material.current(v,(q,0))
                assert c*v>=0 and abs(c-weighted.weight(state)*base)<1e-14
                if v==0: assert c==0
                # Per-channel Lorentzian <=1, ensemble average <=1.
                assert abs(c)<=weighted.weight(state)*abs(v)/r.ron_ohm*(1+1e-10)
                checks.append((q,x,v))
    result=dict(status='structural_and_readout_diagnostics_complete',n_algebra_points=len(checks),
        beta=material.parameters.a_max_s*r.ron_ohm,reference_Ron_ohm=r.ron_ohm,
        G0_S=7.748091729e-5,reference_effective_modes=1/(r.ron_ohm*7.748091729e-5),
        spectral_checks=spectral,jacobian=jac,
        max_table_error_decade=max(x['table_error_decade'] for x in spectral),
        direct_energy_relative_error=abs(raw/analytic-1),
        direct_energy_error_estimate=err,
        table_threshold_decade=.01,
        inherited_table_passed=max(x['table_error_decade'] for x in spectral)<.01,
        invariants_passed=True,baseline_and_weighted_initial_readout_equal=True,
        scientific_scope='No absolute channel count identified; Ron and free W scale degenerate; no fitted physical-time kinetics')
    save('DIAGNOSTICS.json',result)
    return result


def serialize_run(label, result, material):
    arrays={}
    for key,phase in result['predictions'].items():
        prefix='__'.join(map(str,key))
        for name,a in phase.get('arrays',{}).items():
            arrays[prefix+'__'+name]=a
        if 'state_a_s' in phase.get('arrays',{}):
            arrays[prefix+'__state_x']=phase['arrays']['state_a_s']/material.parameters.a_max_s
    np.savez_compressed(HERE/(label+'.npz'),**arrays)
    summary=dict(summary=result['summary'],score=result['score'],
        segments=[{name:value for name,value in s.items() if name!='result'} for s in result['segments']],
        phases={'__'.join(map(str,key)):p['summary'] for key,p in result['predictions'].items()})
    save(label+'.json',summary)


def roots(result, material):
    max_i=max_v=max_kcl=max_kvl=0.; count=0
    for (top,cid,phase),p in result['predictions'].items():
        a=p['arrays']; circuit=k.SeriesCircuit() if top=='1t1r' else k.BareCircuit(.001 if phase=='set' else None)
        inds=np.unique(np.linspace(1,len(a['current_a'])-2,5,dtype=int))
        for j in inds:
            state=(float(a['state_q'][j]),float(a['state_a_s'][j]))
            point=circuit.solve(float(a['command_voltage_v'][j]),material,state,reference=True)
            max_i=max(max_i,abs(point['current_a']-a['current_a'][j])/max(abs(a['current_a'][j]),1e-30))
            max_v=max(max_v,abs(point['local_voltage_v']-a['local_voltage_v'][j]))
            count+=1
        max_kcl=max(max_kcl,float(np.max(abs(a['KCL_residual_a']))))
        max_kvl=max(max_kvl,float(np.max(abs(a['KVL_residual_v']))))
        assert np.all((a['state_q']>0)&(a['state_q']<1))
        assert np.all((a['state_a_s']>=0)&(a['state_a_s']<=material.parameters.a_max_s))
    return dict(n=count,max_relative_current_difference=max_i,max_local_voltage_difference=max_v,
                max_KCL_A=max_kcl,max_KVL_V=max_kvl,passed=max_i<1e-6,
                scope='Independent root variable/bisection; same frozen spectral table')


def compare(a,b):
    delta=u=x=0.
    for key in a['predictions']:
        aa,bb=a['predictions'][key]['arrays'],b['predictions'][key]['arrays']
        mask=(aa['current_a']!=0)&(bb['current_a']!=0)
        delta=max(delta,float(np.max(abs(np.log10(abs(aa['current_a'][mask]/bb['current_a'][mask]))))))
        u=max(u,float(np.max(abs(aa['state_u']-bb['state_u']))))
        x=max(x,float(np.max(abs(aa['state_a_s']-bb['state_a_s']))))
    return dict(max_log_current_decade=delta,max_u=u,max_scaled_state=x,passed=delta<.001)


def main():
    declaration=ANALYSIS/'coupled_linear_recovery_2026-11-04/FORWARD_DECLARATION.json'
    d=json.loads(declaration.read_text())
    params=d['candidates'][1]['parameters']
    entries=d['initial_states']
    material=q1.load_q1_material(k.Parameters(**params,enable_a=True),
        candidate_source_id='spectral_weight_finite_unfitted_control',candidate_source_sha256=sha(__file__))
    sources=dict(material.source_sha256)
    sources.update({str(p):sha(p) for p in [Path(__file__),HERE/'PLAN.md',declaration]})
    if (HERE/'SOURCE_FREEZE.json').exists() and not (HERE/'SOURCE_FREEZE_initial.json').exists():
        (HERE/'SOURCE_FREEZE_initial.json').write_bytes((HERE/'SOURCE_FREEZE.json').read_bytes())
    save('SOURCE_FREEZE.json',sources)
    diag=diagnostics(material)
    print(json.dumps({'stage':'diagnostics','max_table_error_decade':diag['max_table_error_decade']}),flush=True)
    problem=tp.load_training_problem()
    sources.update(problem.source_sha256)
    sources[str(Path(fa.__file__))]=sha(fa.__file__)
    save('SOURCE_FREEZE.json',sources)
    values={f.name:getattr(material,f.name) for f in fields(k.CoupledMaterial)}
    models=[('additive',material),('weighted_total',WeightMaterial(**values,drive_mode='total')),
            ('weighted_per_reference',WeightMaterial(**values,drive_mode='per_reference_weight'))]
    precisions=[k.Precision('nominal',1e-7,1e-9,1e-11,.02),k.Precision('refined',1e-9,1e-11,1e-13,.01)]
    out=dict(scope='Finite unrefitted controls, original entries, training only',cases={},
        scientific_success=False,inherited_table_passed=diag['inherited_table_passed'],
        initial_states=entries,parameters=params)
    started=time.monotonic()
    def budget():
        if time.monotonic()-started>900: raise TimeoutError('900 second forward budget')
    for label,model in models:
        runs=[]; row={}; out['cases'][label]=row
        for precision in precisions:
            runlabel=label+'_'+precision.name
            try:
                existing=HERE/(runlabel+'.json')
                if existing.exists() and json.loads(existing.read_text())['summary']['status']=='complete_original_training':
                    saved=json.loads(existing.read_text())
                    result=dict(summary=saved['summary'],score=saved['score'],predictions={})
                    with np.load(HERE/(runlabel+'.npz')) as archive:
                        for key in problem.phase_keys:
                            prefix='__'.join(map(str,key))+'__'
                            result['predictions'][key]=dict(arrays={n[len(prefix):]:archive[n].copy() for n in archive.files if n.startswith(prefix)})
                else:
                    result=problem.evaluate(model,entries,precision,'Radau',budget,collect_rhs=False)
                    serialize_run(runlabel,result,model)
                features=fa.evaluate_features(problem,result['predictions'])
                save(runlabel+'_features.json',features)
                row[precision.name]=dict(status='complete',score=result['score'],
                    features_passed=features['n_passed'],roots=roots(result,model))
                runs.append(result)
            except tp.TrainingFailure as exc:
                serialize_run(runlabel,exc.result,model)
                row[precision.name]=dict(status='incomplete',error=str(exc),summary=exc.result['summary'])
            save('RESULT.json',out)
            print(json.dumps(dict(case=runlabel,status=row[precision.name]['status'],
                features_passed=row[precision.name].get('features_passed'),elapsed_s=time.monotonic()-started)),flush=True)
        if len(runs)==2: row['convergence']=compare(*runs)
        save('RESULT.json',out)
    out.update(elapsed_forward_s=time.monotonic()-started,
        sources_unchanged=all(sha(p)==digest for p,digest in sources.items()),
        status='finite_controls_complete_no_fit_or_holdout')
    save('RESULT.json',out)


if __name__=='__main__': main()
