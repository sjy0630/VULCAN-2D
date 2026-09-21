"""Research steady-state contact/localized-state networks, not M13 replacement.

Electrochemical potentials differ by the FULL applied h-BN bias. The two-site
channel uses a prescribed linear electrostatic profile and at most one electron;
this is not a Poisson solution or evidence for observed Coulomb blockade.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import least_squares
from .transport_candidates import TransportAssumptions, LN10

NAMES=('fixed_level','contact_rate','two_contact_rates','two_site_chain',
       'equilibrium_left','equilibrium_right','finite_left','finite_right',
       'coulomb_left','coulomb_right','empirical_branches')


@dataclass(frozen=True)
class ChargingAssumptions(TransportAssumptions):
    dielectric_r:float=3.1
    minimum_site_separation_nm:float=.25

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.dielectric_r) or self.dielectric_r<=0:raise ValueError('Invalid dielectric assumption')
        if not np.isfinite(self.minimum_site_separation_nm) or self.minimum_site_separation_nm<=0:raise ValueError('Invalid distance assumption')


def domain(name):
    bounds={
        'fixed_level':([0.,.01],[1.,.99]),
        'contact_rate':([0.,.01,-12.,-12.],[1.,.99,12.,12.]),
        'two_contact_rates':([0.,.01,-12.,-12.,-12.],[1.,.99,12.,12.,12.]),
        'two_site_chain':([-.5,-.5,.01,.01,-12.,-12.],[.5,.5,.95,.99,12.,12.]),
        'equilibrium_left':([-.5,-.5,.01,.01],[.5,.5,.95,.99]),
        'equilibrium_right':([-.5,-.5,.01,.01],[.5,.5,.95,.99]),
        'empirical_branches':([1.,.05,.15,.2,-5.],[5.,1.,4.,8.,5.])}
    if name.startswith(('finite_','coulomb_')):
        return np.array([-.5,-.5,.01,.01,0.]),np.array([.5,.5,.95,.99,3. if name.startswith('finite_') else 1.])
    if name not in bounds:raise ValueError(name)
    return tuple(np.array(x) for x in bounds[name])


def initial_points(name):
    values={
        'fixed_level':[(.05,.1),(.1,.3),(.2,.7),(.3,.9)],
        'contact_rate':[(.1,.1,-2.,2.),(.1,.5,-6.,4.),(.2,.8,2.,-2.),(.05,.3,6.,-4.)],
        'two_contact_rates':[(.1,.1,-2.,2.,0.),(.1,.5,-6.,4.,0.),(.2,.8,2.,0.,-2.),(.05,.3,6.,-4.,2.)],
        'two_site_chain':[(.1,-.1,.1,.2,0.,0.),(.2,.05,.3,.5,-3.,3.),
                          (-.1,.1,.2,.6,3.,-3.),(.1,-.2,.65,.4,-4.,-4.)],
        'empirical_branches':[(2.,.5,1.5,1.8,-.3),(2.,.2,2.,2.,-.5),
                              (3.,.8,1.,1.2,0.),(1.5,.7,3.,2.5,-1.)]}
    if name.startswith(('finite_','coulomb_')):
        uvals=(.4,1.,2.,.8) if name.startswith('finite_') else (.3,.7,.95,.5)
        return [np.array([*v[:4],u],dtype=float) for v,u in zip(values['two_site_chain'],uvals)]
    if name.startswith('equilibrium_'):return [np.array(v[:4],dtype=float) for v in values['two_site_chain']]
    return [np.array(v,dtype=float) for v in values[name]]


def _validate(name,parameters):
    p=np.asarray(parameters,dtype=float);lo,hi=domain(name)
    if p.shape!=lo.shape or np.any(~np.isfinite(p)) or np.any(p<lo-1e-12) or np.any(p>hi+1e-12):
        raise ValueError('Invalid network parameters')
    return p


def network_log_rates(voltage,parameters,assumptions=TransportAssumptions()):
    """a,b,c,d,e,f for 0→1→2→0 and the reverse ring, in k0 units."""
    p=_validate('two_site_chain',parameters);v=np.asarray(voltage,dtype=float)
    if np.any(~np.isfinite(v)):raise ValueError('Voltage must be finite')
    e1,e2,x1,t,gl,gr=p;x2=x1+(1-x1)*t;vt=assumptions.thermal_voltage_v
    eps1=e1+(1-x1)*v;eps2=e2+(1-x2)*v
    left=(eps1-v)/vt;right=eps2/vt;delta=(eps2-eps1)/vt
    a=gl-np.logaddexp(0.,left);d=gl-np.logaddexp(0.,-left)
    c=gr-np.logaddexp(0.,-right);f=gr-np.logaddexp(0.,right)
    b=-np.maximum(delta,0.);e=-np.maximum(-delta,0.)
    return np.array([a,b,c,d,e,f])


def _network_trees(logrates):
    a,b,c,d,e,f=logrates
    tau0=logsumexp(np.array([d+c,b+c,e+d]),axis=0)
    tau1=logsumexp(np.array([a+e,f+e,c+a]),axis=0)
    tau2=logsumexp(np.array([f+b,a+b,d+f]),axis=0)
    tau=np.array([tau0,tau1,tau2])
    return tau,logsumexp(tau,axis=0)


def stationary_network(voltage,parameters,assumptions=TransportAssumptions()):
    """Stationary probabilities and signed edge currents in k0 units."""
    logs=network_log_rates(voltage,parameters,assumptions);tau,z=_network_trees(logs)
    probabilities=np.exp(tau-z)
    a,b,c,d,e,f=np.exp(logs);p0,p1,p2=probabilities
    currents=np.array([a*p0-d*p1,b*p1-e*p2,c*p2-f*p0])
    return probabilities,currents


def _raw_log_abs(v,name,p,assumptions):
    vt=assumptions.thermal_voltage_v
    if name.startswith(('finite_','coulomb_')):
        e1,e2,x1,t,_=p;x2=x1+(1-x1)*t
        eps1=e1+(1-x1)*v;eps2=e2+(1-x2)*v
        u=charging_energy(name,p,assumptions)
        if name.endswith('left'):
            z=logsumexp(np.array([np.zeros_like(v),-eps1/vt,-eps2/vt,-(eps1+eps2+u)/vt]),axis=0)
            numerator=np.logaddexp(-np.logaddexp(0.,(eps1-v)/vt),
                                  -eps2/vt-np.logaddexp(0.,(eps1+u-v)/vt))
            drive=np.maximum(-v,0.)/vt
        else:
            z=logsumexp(np.array([np.zeros_like(v),-(eps1-v)/vt,-(eps2-v)/vt,-(eps1+eps2+u-2*v)/vt]),axis=0)
            numerator=np.logaddexp(-np.logaddexp(0.,eps2/vt),
                                  -(eps1-v)/vt-np.logaddexp(0.,(eps2+u)/vt))
            drive=np.maximum(v,0.)/vt
        return numerator+drive+np.log(-np.expm1(-np.abs(v)/vt))-z
    if name.startswith('equilibrium_'):
        e1,e2,x1,t=p;x2=x1+(1-x1)*t
        eps1=e1+(1-x1)*v;eps2=e2+(1-x2)*v
        if name=='equilibrium_left':
            z=logsumexp(np.array([np.zeros_like(v),-eps1/vt,-eps2/vt]),axis=0)
            logfermi=-np.logaddexp(0.,(eps1-v)/vt)
            drive=np.maximum(-v,0.)/vt
        else:
            z=logsumexp(np.array([np.zeros_like(v),-(eps1-v)/vt,-(eps2-v)/vt]),axis=0)
            logfermi=-np.logaddexp(0.,eps2/vt)
            drive=np.maximum(v,0.)/vt
        return logfermi+drive+np.log(-np.expm1(-np.abs(v)/vt))-z
    if name=='two_site_chain':
        logs=network_log_rates(v,p,assumptions);_,z=_network_trees(logs)
        a,b,c,d,e,f=logs
        product=np.where(v>0,a+b+c,d+e+f)
        return product+np.log(-np.expm1(-np.abs(v)/vt))-z
    if name=='empirical_branches':
        nlo,nhi,vc,nneg,offset=p;u=np.abs(v);vr=assumptions.reference_voltage_v
        def roll(u):return nlo*np.log(u)-.5*(nlo-nhi)*np.logaddexp(0.,2*np.log(u/vc))
        return np.where(v>0,roll(u)-roll(vr),offset*LN10+nneg*np.log(u/vr))
    energy,eta=p[:2];a=(energy-eta*v)/vt;b=(energy+(1-eta)*v)/vt
    out=np.maximum(a,b)+np.log(-np.expm1(-np.abs(v)/vt))-np.logaddexp(0.,a)-np.logaddexp(0.,b)
    if name!='fixed_level':
        logleft=p[2]+p[3]*v
        logright=p[4]*v if name=='two_contact_rates' else np.zeros_like(v)
        out+=logleft+logright-np.logaddexp(logleft,logright)
    return out


def charging_energy(name,parameters,assumptions=ChargingAssumptions()):
    if not name.startswith(('finite_','coulomb_')):raise ValueError('No finite charging energy in this kernel')
    p=_validate(name,parameters)
    if name.startswith('finite_'):return float(p[4])
    eps=getattr(assumptions,'dielectric_r',3.1)
    rmin=getattr(assumptions,'minimum_site_separation_nm',.25)
    vertical=assumptions.thickness_nm*(1-p[2])*p[3]
    return float(p[4]*1.439964/(eps*max(vertical,rmin)))


def log10_shape(signed_voltage,name,parameters,assumptions=TransportAssumptions()):
    """Log magnitude, referenced to POSITIVE vref; reverse rates remain shared."""
    p=_validate(name,parameters);v=np.asarray(signed_voltage,dtype=float)
    if np.any(~np.isfinite(v)) or np.any(v==0):raise ValueError('Finite nonzero voltage required')
    return (_raw_log_abs(v,name,p,assumptions)
            -_raw_log_abs(np.array(assumptions.reference_voltage_v),name,p,assumptions))/LN10


def current(voltage,log10_reference_current_a,name,parameters,assumptions=TransportAssumptions()):
    v=np.asarray(voltage,dtype=float)
    if np.any(~np.isfinite(v)) or not np.isfinite(log10_reference_current_a):raise ValueError('Invalid voltage/amplitude')
    result=np.zeros_like(v);mask=v!=0
    logs=log10_reference_current_a+log10_shape(v[mask],name,parameters,assumptions)
    if np.any(logs>np.log10(np.finfo(float).max)):raise OverflowError('No silent current clipping')
    result[mask]=np.sign(v[mask])*10**logs
    return float(result) if result.ndim==0 else result


@dataclass(frozen=True)
class Pair:
    cycle_id:int
    positive_voltage_v:np.ndarray
    positive_log10_current_a:np.ndarray
    negative_voltage_v:np.ndarray
    negative_log10_current_a:np.ndarray

    def __post_init__(self):
        for v,y,sign in [(self.positive_voltage_v,self.positive_log10_current_a,1),
                         (self.negative_voltage_v,self.negative_log10_current_a,-1)]:
            v,y=np.asarray(v),np.asarray(y)
            if v.ndim!=1 or v.shape!=y.shape or len(v)<3:raise ValueError('Each branch needs at least 3 points')
            if np.any(~np.isfinite(v)) or np.any(~np.isfinite(y)) or np.any(sign*v<=0):raise ValueError('Invalid paired curve')


def profile_amplitude(pair,name,parameters,assumptions=TransportAssumptions(),positive_anchor=None):
    if positive_anchor is not None:
        mask=pair.positive_voltage_v<=positive_anchor
        if sum(mask)<3:raise ValueError('Insufficient positive low anchor')
        return float(np.mean(pair.positive_log10_current_a[mask]-log10_shape(pair.positive_voltage_v[mask],name,parameters,assumptions)))
    pos=np.mean(pair.positive_log10_current_a-log10_shape(pair.positive_voltage_v,name,parameters,assumptions))
    neg=np.mean(pair.negative_log10_current_a-log10_shape(pair.negative_voltage_v,name,parameters,assumptions))
    return float((pos+neg)/2)


def fit_shared_shape(pairs,name,assumptions=TransportAssumptions(),separate_amplitudes=False):
    pairs=tuple(pairs)
    if len(pairs)<2 or len({p.cycle_id for p in pairs})!=len(pairs):raise ValueError('At least two distinct pairs required')
    volts=[];ys=[];groups=[];weights=[]
    for k,p in enumerate(pairs):
        for branch,(v,y) in enumerate([(p.positive_voltage_v,p.positive_log10_current_a),
                                      (p.negative_voltage_v,p.negative_log10_current_a)]):
            volts.append(v);ys.append(y)
            groups.extend([2*k+branch if separate_amplitudes else k]*len(v))
            weights.extend([1/np.sqrt(2*len(pairs)*len(v))]*len(v))
    v=np.concatenate(volts);y=np.concatenate(ys);g=np.array(groups);w=np.array(weights);w2=w*w
    denom=np.bincount(g,weights=w2)
    def residual(p):
        r=y-log10_shape(v,name,p,assumptions)
        means=np.bincount(g,weights=r*w2)/denom
        return (r-means[g])*w
    lo,hi=domain(name)
    attempts=[least_squares(residual,p,bounds=(lo,hi),x_scale=hi-lo,
        max_nfev=500,ftol=2e-8,xtol=2e-8,gtol=2e-8) for p in initial_points(name)]
    good=[f for f in attempts if f.success and np.isfinite(f.cost)]
    if not good:raise RuntimeError('No converged fit: '+name)
    best=min(good,key=lambda f:f.cost);p=best.x
    singular=np.linalg.svd(best.jac*(hi-lo)[None,:],compute_uv=False)
    amps=[profile_amplitude(pair,name,p,assumptions) for pair in pairs]
    grid=np.r_[-np.geomspace(.001,3,400)[::-1],np.geomspace(.001,3,400)]
    currents=current(grid,0.,name,p,assumptions)
    # Report negative slopes; don't silently clip them for a solver.
    scale=np.maximum(np.maximum(np.abs(currents[:-1]),np.abs(currents[1:])),1e-250)
    negsteps=np.where(np.diff(currents)<-1e-9*scale)[0]
    return {'name':name,'parameters':p.tolist(),'bounds':[lo.tolist(),hi.tolist()],
        'n_training_pairs':len(pairs),'training_cycle_ids':[x.cycle_id for x in pairs],
        'separate_branch_amplitudes':separate_amplitudes,
        'profiled_pair_branch_equal_rmse_decade':float(np.sqrt(2*best.cost)),
        'log10_reference_amplitude_center':float(np.median(amps)),
        'log10_pair_amplitudes':dict(zip([x.cycle_id for x in pairs],amps)),
        'parameter_at_boundary':bool(np.any(np.minimum(p-lo,hi-p)/(hi-lo)<1e-5)),
        'conditional_physics':conditional_physics(name,p,assumptions),
        'scaled_jacobian_singular_values':singular.tolist(),
        'scaled_jacobian_condition':float(singular[0]/singular[-1]) if singular[-1]>0 else None,
        'negative_differential_segments_on_diagnostic_grid':len(negsteps),
        'ndr_voltage_intervals_V':[[float(grid[k]),float(grid[k+1])] for k in negsteps],
        'multistart':[{'parameters':f.x.tolist(),'rmse_decade':float(np.sqrt(2*f.cost)),
                       'success':bool(f.success),'nfev':int(f.nfev)} for f in attempts]}


def conditional_physics(name,parameters,assumptions=ChargingAssumptions()):
    p=_validate(name,parameters)
    out={'temperature_K':assumptions.temperature_k,
         'interpretation':'Conditional effective parameters; not identification of microscopic sites or species'}
    if name=='two_site_chain' or name.startswith(('equilibrium_','finite_','coulomb_')):
        x1=p[2];x2=x1+(1-x1)*p[3]
        distance=assumptions.thickness_nm*(x2-x1)
        eps=getattr(assumptions,'dielectric_r',3.1);minimum=getattr(assumptions,'minimum_site_separation_nm',.25)
        upper=1.439964/(eps*max(distance,minimum))
        out.update(E1_eV=float(p[0]),E2_eV=float(p[1]),x1_fraction=float(x1),x2_fraction=float(x2),
                   conditional_vertical_separation_nm=float(distance),assumed_dielectric_r=float(eps),
                   conditional_unscreened_mutual_energy_upper_eV=float(upper))
        if name.startswith(('finite_','coulomb_')):
            out['mutual_energy_eV']=charging_energy(name,p,assumptions)
            out['U_over_conditional_geometric_upper']=out['mutual_energy_eV']/upper
    return out
