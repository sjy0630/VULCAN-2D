"""Identifiable current amplitude coordinates for joint transport/history fits.

This changes the parameterization only, not the archived physical equations.
The constant saddle-polarizability term is zero in this 14-parameter branch.
"""
import numpy as np
from scipy.special import log_expit

NAMES = ('log10B', 'Ed_center', 'halfwidth', 'xd', 'xt', 'U_fraction',
         'ln_GL_over_GR', 'DeltaEd', 'Et0', 'Et1', 'ellq', 'G01', 'dipole', 'ellm')
LO = np.array([-10., -.5, 0., .01, .01, 0., -12., 0., -1.2, -1.2, -40., -.6, -.3, -40.])
HI = np.array([-2., 1.5, 1.5, .99, .99, 1., 12., 1.2, 1.2, 1.2, 5., .6, .3, 20.])
SCALE = np.array([1., .2, .2, .1, .1, .1, 3., .2, .2, .2, 5., .2, .1, 5.])
INHERITED = np.array([1, 2, 3, 4, 5])
PREVIOUS_ACTIVE = np.array([0, 6, 7, 8, 9, 10, 11, 12, 13])


def pack(shared, parameters):
    s, p = np.asarray(shared, dtype=float), np.asarray(parameters, dtype=float)
    if s.shape != (6,) or p.shape != (9,) or np.any(~np.isfinite(np.r_[s, p])) or p[7] != 0:
        raise ValueError('Finite six/nine parameters and zero saddle polarizability required')
    b = p[0]+log_expit(s[4])/np.log(10.)
    return np.r_[b, s[0], s[5], s[1:5], p[1:7], p[8]]


def unpack(coordinates):
    x = np.asarray(coordinates, dtype=float)
    if x.shape != (14,) or np.any(~np.isfinite(x)):
        raise ValueError('Fourteen finite coordinates required')
    s = np.array([x[1], x[3], x[4], x[5], x[6], x[2]])
    p = np.r_[x[0]-log_expit(x[6])/np.log(10.), x[7:13], 0., x[13]]
    return s, p
