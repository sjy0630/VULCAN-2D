"""Conditional spectral readout screen, not a complete switching model.

Energy coordinates are in eV (numerically volts per elementary charge).
Lorentzian transmission averaged over a normalized Gaussian level ensemble.
The same signed passive law applies to either external circuit. The state
coupling gamma(u)=gamma_c/expm1(u) has a precise Ohmic u=0 limit, but u is
an effective coordinate, not a measured distance or unique microscopic state.
"""
from dataclasses import dataclass
import math
from functools import lru_cache
import numpy as np
from scipy.special import digamma, roots_hermitenorm

KB_EV_PER_K = 8.617333262145e-5

@lru_cache(maxsize=16)
def gaussian_rule(order):
    if int(order)!=order or order<8:raise ValueError('At least eight quadrature nodes required')
    nodes,weights=roots_hermitenorm(int(order))
    weights=weights/np.sum(weights)
    return nodes,weights

def spectral_current(voltage, gamma_ev, epsilon_ev, width_ev, eta, ron_ohm,
                     temperature_k=300., order=96):
    """Landauer integral at fixed gamma, with a common positive G=1/Ron.

    No fitted polarity scale, effective temperature or observed current enters
    this readout. Gaussian broadening is a declared candidate assumption.
    """
    values=[gamma_ev,epsilon_ev,width_ev,eta,ron_ohm,temperature_k]
    if not np.all(np.isfinite(values)) or gamma_ev<=0 or width_ev<0 or not 0<=eta<=1 or ron_ohm<=0 or temperature_k<=0:
        raise ValueError('Invalid spectral parameters')
    v=np.asarray(voltage,dtype=float)
    if not np.all(np.isfinite(v)):raise ValueError('Voltage must be finite')
    shape=v.shape; flat=v.reshape(-1); thermal=KB_EV_PER_K*temperature_k
    # Broad-level expansion avoids loss of precision from subtracting two
    # nearly equal digammas. Switch only when omitted relative terms are tiny.
    energy_scale=abs(epsilon_ev)+8*width_ev+np.max(abs(flat),initial=0)+10*thermal
    if gamma_ev>1e4*max(energy_scale,1e-12):
        m=epsilon_ev+(eta-.5)*flat
        correction=(m*m+width_ev**2+flat*flat/12+math.pi**2*thermal**2/3)/gamma_ev**2
        out=flat/ron_ohm*(1-correction)
    else:
        if width_ev==0:nodes,weights=np.array([0.]),np.array([1.])
        else:nodes,weights=gaussian_rule(order)
        energies=epsilon_ev+width_ev*nodes
        left=energies[None,:]+eta*flat[:,None]
        right=energies[None,:]-(1-eta)*flat[:,None]
        a=.5+(gamma_ev+1j*left)/(2*math.pi*thermal)
        b=.5+(gamma_ev+1j*right)/(2*math.pi*thermal)
        out=gamma_ev/ron_ohm*((digamma(a)-digamma(b)).imag@weights)
    out[flat==0]=0.
    if not np.all(np.isfinite(out)):raise FloatingPointError('Invalid spectral current')
    return float(out[0]) if shape==() else out.reshape(shape)

@dataclass(frozen=True)
class SpectralReadout:
    ron_ohm:float
    gamma_c_ev:float
    epsilon_ev:float
    width_ev:float
    eta:float
    temperature_k:float=300.
    order:int=96

    def __post_init__(self):
        spectral_current(0.,self.gamma_c_ev,self.epsilon_ev,self.width_ev,self.eta,
                         self.ron_ohm,self.temperature_k,self.order)

    def current(self,voltage,u):
        u=float(u)
        if not math.isfinite(u) or u<0:raise ValueError('State must be finite and nonnegative')
        if u==0:
            v=np.asarray(voltage,dtype=float)
            if not np.all(np.isfinite(v)):raise ValueError('Voltage must be finite')
            return float(v/self.ron_ohm) if v.ndim==0 else v/self.ron_ohm
        gamma=self.gamma_c_ev/math.expm1(u)
        return spectral_current(voltage,gamma,self.epsilon_ev,self.width_ev,self.eta,
                                self.ron_ohm,self.temperature_k,self.order)
