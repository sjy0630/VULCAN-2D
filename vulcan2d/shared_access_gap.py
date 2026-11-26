"""One shared effective-gap law with a current-dependent access voltage.

The extra voltage is an effective internal series drop, not an identified
thermal or contact mechanism. It adds no state, topology or device label.
Only the original gap voltage drives the inherited migration dynamics.
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
    # asinh(x)/x differs from unity by less than float precision here.
    return logx if logx < -20. else math.log(_asinh_exp(logx))


@dataclass(frozen=True)
class AccessGapLaw(GapLaw):
    """V = Ron I + u V0 asinh(I exp(u)/I0) + u B (I exp(u)/Iref)^3.

    B=access_voltage_v >=0 and Iref=1mA is a fixed unit convention. The
    inherited drive remains V0*asinh(|I|*exp(u)/I0): this additional passive
    series drop does not become migration field. Both circuit topologies
    use the same instance and parameters.
    """

    access_voltage_v: float = 0.

    def voltage(self, current_a, u):
        state = self.validate_state(u)
        if self.access_voltage_v == 0. or state == 0.:
            return super().voltage(current_a, state)
        base = super().voltage(current_a, state)
        currents = np.asarray(current_a, dtype=float)
        log_coefficient = math.log(state) + math.log(self.access_voltage_v) + 3.*state - 3.*_LOG_REFERENCE_CURRENT
        if currents.ndim == 0:
            current = float(currents)
            if current == 0.:
                return base
            try:
                extra = math.exp(log_coefficient + 3.*math.log(abs(current)))
            except OverflowError as exc:
                raise FloatingPointError("Access voltage overflow outside numerical domain") from exc
            result = base + math.copysign(extra, current)
            if not math.isfinite(result):
                raise FloatingPointError("Access voltage overflow outside numerical domain")
            return result
        nonzero = currents != 0.
        extra = np.zeros_like(currents)
        with np.errstate(over="raise", invalid="raise"):
            extra[nonzero] = np.exp(log_coefficient + 3.*np.log(abs(currents[nonzero])))
            result = base + np.copysign(extra, currents)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Access voltage overflow outside numerical domain")
        return result

    def _current_with_series(self, magnitude_v, u, extra_resistance_ohm=0.):
        if self.access_voltage_v == 0. or u == 0. or magnitude_v == 0.:
            return super()._current_with_series(magnitude_v, u, extra_resistance_ohm)
        resistance = self.ron_ohm + extra_resistance_ohm
        logv, logr = math.log(magnitude_v), math.log(resistance)
        logi0 = math.log(self.i0_a)
        log_gap_scale = math.log(self.v0_v) + math.log(u)
        log_cubic = math.log(u) + math.log(self.access_voltage_v) + 3.*u - 3.*_LOG_REFERENCE_CURRENT
        # The original zero-current tangent is no longer a lower current
        # bound once the cubic drop is present. At this new lower bound the
        # entire linear tangent contributes <=V/2 and the cubic <=V/2.
        log_linear = float(np.logaddexp(logr, log_gap_scale + u - logi0))
        lower = min(logv-log_linear-_LOG2, (logv-log_cubic-_LOG2)/3.)
        upper = logv-logr  # positive extra drops imply I <= V/Rseries_total

        def residual(logi):
            gap_log_voltage = log_gap_scale + _log_asinh_exp(logi+u-logi0)
            log_voltage = float(np.logaddexp(np.logaddexp(logr+logi, gap_log_voltage),
                                             log_cubic+3.*logi))
            return log_voltage-logv

        root = brentq(residual, lower-2e-12, upper+2e-12,
                      xtol=2e-12, rtol=9e-15)
        result = math.exp(root)
        if result == 0.:
            raise FloatingPointError("Intrinsic current underflow outside numerical domain")
        return result
