"""Exact five-parameter steady current of the archived four-state network.

Side exchange rate affects correlations/dynamics but cancels from steady mean
current when each site's total attempt rate is occupation independent.
"""
import numpy as np
from .coupled_charge import charging,VT,LN10


def log_fermi_pair(z):
    return -np.logaddexp(0.,z),-np.logaddexp(0.,-z)


def stationary_current(voltage,shared,trap_energy):
    shared=np.asarray(shared,float)
    if shared.shape!=(5,) or np.any(~np.isfinite(shared)):
        raise ValueError('Five finite parameters: E_d,x_d,x_t,U fraction,ln Gamma_L/R')
    ed,xd,xt,_,gl=shared;u,_=charging(np.r_[shared,0.])
    v,et=np.broadcast_arrays(np.asarray(voltage,float),np.asarray(trap_energy,float))
    if np.any(~np.isfinite(v)) or np.any(~np.isfinite(et)):raise ValueError('Finite inputs required')
    level=ed+(1-xd)*v;side=et+(1-xt)*v;norm=np.logaddexp(gl,0.)
    l0,lc0=log_fermi_pair((level-v)/VT);r0,rc0=log_fermi_pair(level/VT)
    l1,lc1=log_fermi_pair((level+u-v)/VT);r1,rc1=log_fermi_pair((level+u)/VT)
    a0=np.logaddexp(gl+l0,r0)-norm;a0c=np.logaddexp(gl+lc0,rc0)-norm
    a1=np.logaddexp(gl+l1,r1)-norm;a1c=np.logaddexp(gl+lc1,rc1)-norm
    c0,c0c=log_fermi_pair(side/VT);c1,c1c=log_fermi_pair((side+u)/VT)
    with np.errstate(divide='ignore'):
        shift=np.log(-np.expm1(-u/VT))
        drive=np.log(-np.expm1(-np.abs(v)/VT))
    delta_a=np.logaddexp(gl+l0+lc1,r0+rc1)+shift-norm
    denominator=np.logaddexp(np.logaddexp(a0c,a1),delta_a+np.logaddexp(c0c,c1))
    logm=np.logaddexp(c0+a0c,c1+a0)-denominator
    logmc=np.logaddexp(c0c+a1c,c1c+a1)-denominator
    logn=np.logaddexp(a0+c0c,a1+c0)-denominator
    lognc=np.logaddexp(a0c+c1c,a1c+c1)-denominator
    delta0=np.where(v>=0,l0+rc0,r0+lc0)+drive
    delta1=np.where(v>=0,l1+rc1,r1+lc1)+drive
    logi=gl-norm+np.logaddexp(logmc+delta0,logm+delta1)
    return {'log10_abs_current_scale':logi/LN10,'current_scale':np.sign(v)*np.exp(logi),
            'mean_conductor_occupation':np.exp(logn),'mean_side_occupation':np.exp(logm),
            'conductor_normalization_error':np.exp(logn)+np.exp(lognc)-1,
            'side_normalization_error':np.exp(logm)+np.exp(logmc)-1}


def predict(voltage,shared,states):
    v=np.asarray(voltage,float);states=np.atleast_2d(states)
    if v.ndim!=1 or np.any(v==0) or states.shape[1]!=2:raise ValueError('Voltage vector and two-coordinate states required')
    return states[:,0,None]+stationary_current(v[None,:],shared,states[:,1,None])['log10_abs_current_scale']
