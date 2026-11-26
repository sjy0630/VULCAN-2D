"""Shared gap with one passive voltage term driven by actual current.

Unlike the archived normalized-current access candidate, the added term has
no exp(3u) factor. This module changes only frozen-state transport and its
inverse. The original gap drive and state kinetics are inherited unchanged.
It is a testable effective readout, not an identified microscopic mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq

from .shared_gap import GapLaw, _asinh_exp

ACCESS_REFERENCE_CURRENT_A = 1e-3
_LOG_REFERENCE_CURRENT = math.log(ACCESS_REFERENCE_CURRENT_A)
_LOG2 = math.log(2.)


def _log_asinh_exp(logx):
    return logx if logx < -20. else math.log(_asinh_exp(logx))


@dataclass(frozen=True)
class ActualAccessGapLaw(GapLaw):
    """Vh = Ron*I + u*V0*asinh(I*exp(u)/I0) + u*B*(I/Iref)^3.

    B=access_voltage_v is nonnegative and measured in volts; Iref=1mA is a
    fixed unit convention, not a fitted current threshold. Both circuit
    topologies use the same instance and parameters. For positive current,
    dVh/dI = Ron + u*V0/hypot(I,I0*exp(-u)) + 3*u*B*I^2/Iref^3 > 0.
    Thus the signed odd law is passive and has a unique intrinsic inverse.
    u=0 is exactly Ohmic; B=0 delegates to the old kernel bitwise.
    """

    access_voltage_v: float = 0.

    def voltage(self, current_a, u):
        state = self.validate_state(u)
        if self.access_voltage_v == 0. or state == 0.:
            return super().voltage(current_a, state)
        base = super().voltage(current_a, state)
        currents = np.asarray(current_a, dtype=float)
        # Actual current: deliberately no +3*state in this coefficient.
        log_coefficient = math.log(state) + math.log(self.access_voltage_v) - 3.*_LOG_REFERENCE_CURRENT
        if currents.ndim == 0:
            current = float(currents)
            if current == 0.:
                return base
            try:
                extra = math.exp(log_coefficient + 3.*math.log(abs(current)))
            except OverflowError as exc:
                raise FloatingPointError('Actual-current access voltage overflow') from exc
            result = base + math.copysign(extra, current)
            if not math.isfinite(result):
                raise FloatingPointError('Actual-current access voltage overflow')
            return result
        nonzero = currents != 0.
        extra = np.zeros_like(currents)
        with np.errstate(over='raise', invalid='raise'):
            extra[nonzero] = np.exp(log_coefficient + 3.*np.log(abs(currents[nonzero])))
            result = base + np.copysign(extra, currents)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError('Actual-current access voltage overflow')
        return result

    def _current_with_series(self, magnitude_v, u, extra_resistance_ohm=0.):
        if self.access_voltage_v == 0. or u == 0. or magnitude_v == 0.:
            return super()._current_with_series(magnitude_v, u, extra_resistance_ohm)
        resistance = self.ron_ohm + extra_resistance_ohm
        logv, logr = math.log(magnitude_v), math.log(resistance)
        logi0 = math.log(self.i0_a)
        log_gap_scale = math.log(self.v0_v) + math.log(u)
        log_cubic = math.log(u) + math.log(self.access_voltage_v) - 3.*_LOG_REFERENCE_CURRENT
        # asinh(x)<=x bounds linear+gap voltage by its zero-current tangent.
        # At lower, the whole tangent and the cubic each contribute <=V/2.
        # Positive extra drops give the upper bound I<=V/Rseries_total.
        log_linear = float(np.logaddexp(logr, log_gap_scale + u - logi0))
        lower = min(logv-log_linear-_LOG2, (logv-log_cubic-_LOG2)/3.)
        upper = logv-logr

        def residual(logi):
            gap_log_voltage = log_gap_scale + _log_asinh_exp(logi+u-logi0)
            log_voltage = float(np.logaddexp(np.logaddexp(logr+logi, gap_log_voltage),
                                             log_cubic+3.*logi))
            return log_voltage-logv

        root = brentq(residual, lower-2e-12, upper+2e-12,
                      xtol=2e-12, rtol=9e-15)
        result = math.exp(root)
        if result == 0.:
            raise FloatingPointError('Intrinsic current underflow outside numerical domain')
        return result
