"""Apply the saved persistent-step rule identically to data and model paths."""
import numpy as np


def batch_events(vapp, currents, zero_reference):
    va = np.abs(np.asarray(vapp)); current = np.atleast_2d(np.abs(currents)); apex = int(np.argmax(va))
    if current.shape[1] != len(va): raise ValueError('Matching path length required')
    zero = np.broadcast_to(np.asarray(zero_reference), (len(current),))
    rise = np.zeros_like(current, dtype=bool); drop = np.zeros_like(current, dtype=bool)
    for j in range(3, apex-1):
        if not .2-1e-10 <= va[j-1] < va[j] or va[j]-va[j-1] > .05000001: continue
        following = np.median(current[:, j:j+3], axis=1)
        preceding = np.median(current[:, j-3:j], axis=1)
        for target, numerator, denominator in [
            (rise, np.minimum(current[:, j], following)-zero, np.maximum(current[:, j-1], preceding)+zero),
            (drop, np.minimum(current[:, j-1], preceding)-zero, np.maximum(current[:, j], following)+zero)]:
            target[:, j] = (numerator > 0) & (denominator > 0) & (numerator > 10*denominator)
    before_drop = np.maximum.accumulate(drop, axis=1)
    return {'rise': rise, 'drop': drop, 'any_rise': rise.any(axis=1),
            'drop_then_rise': (rise & before_drop).any(axis=1)}


def log_readouts(vapp, currents):
    """Native equal-voltage points only, at the previously used read windows."""
    va = np.abs(np.asarray(vapp)); current = np.atleast_2d(np.asarray(currents)); apex = int(np.argmax(va))
    def indices(low, high):
        out = [j for j in range(apex+1) if low-1e-10 <= va[j] <= high+1e-10]
        pairs = [(j, next(k for k in range(apex+1, len(va)) if abs(va[k]-va[j]) <= 1e-10)) for j in out]
        return np.array(pairs).T
    ou, re = indices(.5, .8); low_ou, low_re = indices(.1, .3)
    if np.any(current[:, np.r_[ou, re, low_re]] <= 0): raise ValueError('Positive read currents required')
    return {'log10_return_outward_ratio_0p5_0p8': np.mean(np.log10(current[:, re]/current[:, ou]), axis=1),
            'log10_return_current_A_0p1_0p3': np.mean(np.log10(current[:, low_re]), axis=1)}
