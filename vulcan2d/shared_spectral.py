"""Experimental common signed spectral material law; no circuit or cycle labels.

The fixed lognormal energy ensemble is a fitted readout hypothesis, not a
measured defect DOS. A full-tail energy quadrature and a bounded broad-gamma
expansion cover state-dependent coupling. Dynamics use command-path volts,
not identified physical seconds. This module does not replace desktop defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import ClassVar

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma, ndtr, ndtri, roots_legendre

KB_EV_PER_K = 8.617333262145e-5
DEFAULT_SHAPE_PATH = (Path(__file__).resolve().parents[1] / 'analysis' /
    'constrained_shared_2026-11-03/lognormal_spectral_refinement.json')


class SpectralDomainError(ValueError):
    """An inverse/state/voltage query lies outside the declared numerical domain."""


@dataclass(frozen=True)
class SpectralShape:
    median_energy_ev: float = 1.8009424069670275
    sigma_log_energy: float = 1.123747674000319
    eta: float = 0.40149318809101636
    ron_ohm: float = 93.4800875524324
    temperature_k: float = 300.
    voltage_limit_v: float = 5.

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
            if name == 'eta':
                if not 0. < value < 1.:
                    raise ValueError('This inverse interface requires 0 < eta < 1')
            elif value <= 0.:
                raise ValueError(f'{name} must be positive')
            object.__setattr__(self, name, value)
        # These are domain guards, not physical fitted bounds. The fixed shape
        # used by the project lies comfortably within them.
        if self.sigma_log_energy > 2. or self.voltage_limit_v > 10.:
            raise SpectralDomainError('Quadrature is defined for sigma<=2 and |V|<=10 V')

    @classmethod
    def from_json(cls, path=None):
        payload = json.loads(Path(DEFAULT_SHAPE_PATH if path is None else path).read_text())
        values = payload.get('parameters', payload)
        return cls(**{name: values[name] for name in
            ('median_energy_ev', 'sigma_log_energy', 'eta', 'ron_ohm', 'temperature_k')})

    @property
    def thermal_ev(self):
        return KB_EV_PER_K*self.temperature_k

    def _validate_voltage(self, voltage):
        v = np.asarray(voltage, dtype=float)
        if not np.all(np.isfinite(v)) or np.any(abs(v) > self.voltage_limit_v):
            raise SpectralDomainError(f'Voltage must be finite and |V|<={self.voltage_limit_v} V')
        return v

    @lru_cache(maxsize=24)
    def _energy_rule(self, order):
        if int(order) != order or order < 16:
            raise ValueError('Energy quadrature needs an integer order >=16')
        # Explicit lognormal probability tail, not the old 100-eV cutoff.
        upper = self.median_energy_ev*math.exp(-self.sigma_log_energy*float(ndtri(1e-24)))
        upper = max(upper, 2.*self.voltage_limit_v, 10.)
        panels = list(np.linspace(0., self.voltage_limit_v,
                                  int(math.ceil(self.voltage_limit_v/.25))+1))
        while panels[-1] < upper:
            panels.append(min(upper, 2.*panels[-1]))
        nodes, weights = roots_legendre(int(order))
        energy = np.concatenate([.5*(a+b)+.5*(b-a)*nodes for a, b in zip(panels[:-1], panels[1:])])
        integration = np.concatenate([.5*(b-a)*weights for a, b in zip(panels[:-1], panels[1:])])
        z = np.log(energy/self.median_energy_ev)/self.sigma_log_energy
        probability = integration*np.exp(-.5*z*z)/(energy*self.sigma_log_energy*np.sqrt(2*np.pi))
        missing = float(ndtr(-math.log(upper/self.median_energy_ev)/self.sigma_log_energy))
        mass_error = float(np.sum(probability)-(1.-missing))
        if abs(mass_error) > 2e-10:
            raise FloatingPointError('Energy rule does not resolve the declared lognormal shape')
        energy.flags.writeable = probability.flags.writeable = False
        return energy, probability, upper, missing, mass_error

    def quadrature_metadata(self, order=32):
        energy, weights, upper, tail, error = self._energy_rule(order)
        return {'n_nodes': len(energy), 'energy_upper_ev': upper,
                'omitted_probability': tail, 'probability_mass_error': error,
                'quadrature_mass': float(np.sum(weights)), 'order_per_panel': order}

    def _moments(self, voltage):
        v = np.asarray(voltage)
        c = (.5-self.eta)*v
        m = [self.median_energy_ev**k*math.exp(.5*k*k*self.sigma_log_energy**2)
             for k in range(1, 5)]
        second_shift = m[1]-2*c*m[0]+c*c
        fourth_shift = m[3]-4*c*m[2]+6*c*c*m[1]-4*c**3*m[0]+c**4
        t = self.thermal_ev
        variance = v*v/12+math.pi**2*t*t/3
        fourth_noise = v**4/80+math.pi**2*v*v*t*t/6+7*math.pi**4*t**4/15
        return second_shift+variance, fourth_shift+6*second_shift*variance+fourth_noise

    def broad_expansion_bounds(self, voltage, log_gamma_ev):
        """Bounds relative to |V|/Ron, using the exact second/fourth moments.

        1/(1+x^2)=1-x^2+x^4/(1+x^2). The omitted positive term is at most
        E[Delta^4]/gamma^4. Values at zero voltage use the tangent limit.
        """
        v, lg = np.broadcast_arrays(self._validate_voltage(voltage), np.asarray(log_gamma_ev, dtype=float))
        second, fourth = self._moments(v)
        with np.errstate(over='ignore', under='ignore'):
            inverse2 = np.exp(-2*lg)
            return second*inverse2, fourth*inverse2**2

    def _direct(self, voltage, log_gamma, order, chunk_size):
        energy, weight, _, _, _ = self._energy_rule(order)
        answer = np.empty_like(voltage)
        for start in range(0, len(voltage), chunk_size):
            sl = slice(start, start+chunk_size)
            v = voltage[sl, None]
            gamma = np.exp(log_gamma[sl, None])
            a = .5+(gamma+1j*(energy[None, :]+self.eta*v))/(2*math.pi*self.thermal_ev)
            b = .5+(gamma+1j*(energy[None, :]-(1-self.eta)*v))/(2*math.pi*self.thermal_ev)
            answer[sl] = gamma[:, 0]/self.ron_ohm*((digamma(a)-digamma(b)).imag@weight)
        return answer

    def current_at_log_gamma(self, voltage, log_gamma_ev, *, order=32, chunk_size=128):
        """Broadcast signed current at natural ln(gamma/eV), for table creation.

        +inf gamma is the exact Ohmic limit. Tiny voltage uses a symmetric
        Richardson polynomial to avoid digamma subtraction cancellation.
        No observed current, topology, gate voltage or phase label is an input.
        """
        v, lg = np.broadcast_arrays(self._validate_voltage(voltage), np.asarray(log_gamma_ev, dtype=float))
        if np.any(np.isnan(lg)) or np.any(lg < -700.) or int(chunk_size) != chunk_size or chunk_size <= 0:
            raise SpectralDomainError('ln(gamma) must be >=-700 (or +inf); chunk_size must be positive')
        shape = v.shape
        vf, gf = v.reshape(-1), lg.reshape(-1)
        out = np.zeros_like(vf)
        correction, remainder = self.broad_expansion_bounds(vf, gf)
        broad = (correction < .01) & (remainder <= 1e-11)
        out[broad] = vf[broad]/self.ron_ohm*(1.-correction[broad])
        active = (~broad) & (vf != 0.)
        small = active & (abs(vf) < 1e-5)
        normal = active & (~small)
        out[normal] = self._direct(vf[normal], gf[normal], order, int(chunk_size))
        if np.any(small):
            unique, inverse = np.unique(gf[small], return_inverse=True)
            h = 1e-4
            probes = np.tile(np.array([h, -h, h/2, -h/2]), len(unique))
            widths = np.repeat(unique, 4)
            y = self._direct(probes, widths, order, int(chunk_size)).reshape(-1, 4)
            g1, g2 = (y[:, 0]-y[:, 1])/(2*h), (y[:, 2]-y[:, 3])/h
            a1, a2 = (y[:, 0]+y[:, 1])/(2*h*h), 2*(y[:, 2]+y[:, 3])/(h*h)
            g0, a0 = (4*g2-g1)/3, (4*a2-a1)/3
            cubic, quartic = 4*(g1-g2)/(3*h*h), 4*(a1-a2)/(3*h*h)
            x = vf[small]
            out[small] = x*(g0[inverse]+x*(a0[inverse]+x*(cubic[inverse]+x*quartic[inverse])))
        if not np.all(np.isfinite(out)) or np.any(out*vf < 0.):
            raise FloatingPointError('Invalid spectral current in the declared domain')
        return float(out[0]) if shape == () else out.reshape(shape)

    def current_at_gamma(self, voltage, gamma_ev, *, order=32):
        gamma = np.asarray(gamma_ev, dtype=float)
        if np.any(np.isnan(gamma)) or np.any(gamma <= 0.):
            raise ValueError('gamma must be positive (or +inf)')
        return self.current_at_log_gamma(voltage, np.log(gamma), order=order)


@dataclass(frozen=True)
class SpectralLaw:
    shape: SpectralShape
    gamma_c_ev: float
    jset_a: float
    jreset_a: float
    kset_per_v: float
    kreset_per_v: float
    state_limit: float = 80.
    order: int = 32
    reset_initiation_log: float = 0.
    beta: ClassVar[float] = 1.

    def __post_init__(self):
        if not isinstance(self.shape, SpectralShape):
            raise TypeError('shape must be SpectralShape')
        for name in ('gamma_c_ev', 'jset_a', 'jreset_a', 'kset_per_v', 'kreset_per_v',
                     'state_limit', 'reset_initiation_log'):
            value = float(getattr(self, name))
            nonnegative = name in ('kset_per_v', 'kreset_per_v', 'reset_initiation_log')
            if not math.isfinite(value) or value < 0. or (not nonnegative and value == 0.):
                raise ValueError(f'Invalid {name}')
            object.__setattr__(self, name, value)
        if self.state_limit > 600.:
            raise SpectralDomainError('state_limit above the declared finite-state domain')
        self.shape._energy_rule(self.order)

    @property
    def ron_ohm(self):
        return self.shape.ron_ohm

    def validate_state(self, u):
        value = float(u)
        if not math.isfinite(value) or not 0. <= value <= self.state_limit:
            raise SpectralDomainError('State crossed the numerical domain; not an absorbing RESET bound')
        return value

    def log_gamma(self, u):
        state = self.validate_state(u)
        return math.inf if state == 0. else math.log(self.gamma_c_ev)-math.log(math.expm1(state))

    def current(self, voltage_hbn_v, u):
        state = self.validate_state(u)
        voltage = self.shape._validate_voltage(voltage_hbn_v)
        if state == 0.:
            result = voltage/self.ron_ohm
            return float(result) if result.ndim == 0 else result
        return self.shape.current_at_log_gamma(voltage, self.log_gamma(state), order=self.order)

    def saturation_current(self, u):
        """Full-window magnitude for the supported 0<eta<1 family."""
        log_value = math.log(math.pi/self.ron_ohm)+self.log_gamma(u)
        return math.inf if log_value > 709. else math.exp(log_value)

    def voltage(self, current_a, u, *, max_voltage_v=None):
        """Signed inverse, rejecting unavailable current or voltage-domain exits.

        Negative current is inverted through the negative branch, never by
        assuming an odd material law. Finite-gamma saturation is not clipped.
        """
        state = self.validate_state(u)
        values = np.asarray(current_a, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError('Current must be finite')
        limit = self.shape.voltage_limit_v if max_voltage_v is None else float(max_voltage_v)
        if not math.isfinite(limit) or not 0. < limit <= self.shape.voltage_limit_v:
            raise SpectralDomainError('Invalid inverse voltage limit')
        def one(target):
            if target == 0.:
                return 0.
            if state == 0.:
                result = target*self.ron_ohm
                if abs(result) > limit:
                    raise SpectralDomainError('Ohmic inverse exceeds voltage domain')
                return result
            sign, magnitude = math.copysign(1., target), abs(target)
            if magnitude >= self.saturation_current(state):
                raise SpectralDomainError('Specified current reaches/exceeds the finite spectral saturation')
            available = sign*self.current(sign*limit, state)
            # Scalar versus batched BLAS reductions can differ by several ulps
            # at the exact domain endpoint; this is not current clipping.
            tolerance = 32*np.finfo(float).eps*max(available, magnitude)
            if magnitude > available+tolerance:
                raise SpectralDomainError('Specified current requires voltage outside the declared domain')
            if abs(magnitude-available) <= tolerance:
                return sign*limit
            root = brentq(lambda x: sign*self.current(sign*x, state)-magnitude,
                          0., limit, xtol=1e-300, rtol=8*np.finfo(float).eps)
            return sign*root
        result = np.array([one(float(x)) for x in values.reshape(-1)]).reshape(values.shape)
        return float(result) if result.ndim == 0 else result

    def derivative(self, u, current_a):
        """Common log overdrive, with an optional near-closed RESET increment.

        RESET uses log(|I|/JR)+u-A*exp(-u), with dimensionless A>=0.
        At u=0 its current threshold has log value log(JR)+A; exp(A)
        is never evaluated. A is an effective hypothesis, not a measured
        activation energy. At fixed supply, its overdrive derivative is
        1+A*exp(-u)-chi_external: raising the initiation threshold does not
        automatically provide stable RESET feedback. A=0 preserves the
        original floating-point operation path exactly.
        """
        state, current = self.validate_state(u), float(current_a)
        if not math.isfinite(current):
            raise ValueError('Current must be finite')
        if current == 0. or (current > 0. and state == 0.):
            return 0.
        threshold = self.jset_a if current > 0. else self.jreset_a
        rate = self.kset_per_v if current > 0. else self.kreset_per_v
        drive = math.log(abs(current))-math.log(threshold)+self.beta*state
        if current < 0. and self.reset_initiation_log > 0.:
            drive -= self.reset_initiation_log*math.exp(-state)
        overdrive = max(drive, 0.)
        result = rate*overdrive
        if not math.isfinite(result):
            raise FloatingPointError('Path rate overflow')
        return -result if current > 0. else result

    def set_stopping_bound(self, maximum_current_a):
        """Necessary no-crossing SET bound; never projects the initial state.

        Valid only when this same current bound holds over the whole segment.
        +inf for Imax=0 means the complete segment is inactive.
        """
        maximum = float(maximum_current_a)
        if not math.isfinite(maximum) or maximum < 0.:
            raise ValueError('Maximum current must be finite and nonnegative')
        return math.inf if maximum == 0. else max(0., math.log(self.jset_a)-math.log(maximum))
