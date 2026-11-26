"""Conditional protocol clock; one unchanged seven-parameter spectral law.

alpha = r_series / r_bare is an externally fixed scenario, not a fitted
material parameter or an experimentally measured rate. With cumulative
absolute command-voltage path s, bare du/ds=H and series du/ds=H/alpha.
The same factor acts on SET and RESET. No physical seconds are assigned.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPECTRAL_DIR = ROOT / "analysis/spectral_dynamic_2026-11-03"
for directory in (ROOT, SPECTRAL_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from joint_problem import SpectralJointProblem, PARAMETER_NAMES
from vulcan2d.shared_spectral import SpectralShape, SpectralLaw
from vulcan2d.shared_spectral_ivp import (
    BareSpectralCircuit, LookupSpectralCircuit, SpectralTable,
    TabulatedSpectralLaw, simulate_spectral_ivp,
)
from vulcan2d.transistor import measured_transistor_mapping

TABLE_PATH = SPECTRAL_DIR / "transport_g010_v0025.npz"


def _validate_alpha(alpha):
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must be a positive finite rate ratio, not boolean")
    value = float(alpha)
    if not math.isfinite(value) or value <= 0. or not math.isfinite(1. / value):
        raise ValueError("alpha and its reciprocal must be positive and finite")
    return value


@dataclass(frozen=True)
class ProtocolClock:
    alpha: float

    def __post_init__(self):
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))

    @property
    def series_multiplier(self):
        return 1. / self.alpha

    def metadata(self):
        return {
            "alpha_r_series_over_r_bare": self.alpha,
            "bare_path_derivative_multiplier": 1.,
            "series_path_derivative_multiplier": self.series_multiplier,
            "both_polarities_use_same_multiplier": True,
            "alpha_is_external_fixed_scenario_not_fit_parameter": True,
            "physical_seconds_assigned": False,
            "sweep_rate_measured": False,
            "clock_coordinate": "cumulative_absolute_command_voltage_path_v",
            "interpretation": "Conditional constant relative protocol speed; no measured timing, dwell or wait inferred.",
        }


@dataclass(frozen=True)
class _ClockedDerivativeView:
    """Delegate all material operations except conversion of du/ds units.

    base_law remains the exact same material object used by the bare path.
    Current, inverse voltage, thresholds, shape and boundaries are delegated.
    """
    base_law: object
    multiplier: float

    def __getattr__(self, name):
        return getattr(self.base_law, name)

    def derivative(self, u, current_a):
        return self.multiplier * self.base_law.derivative(u, current_a)


class ProtocolClockSimulator:
    """Apply a clock conversion at the integrator boundary, by circuit identity."""

    def __init__(self, alpha, circuits, backend=simulate_spectral_ivp):
        self.clock = ProtocolClock(alpha)
        self.backend = backend
        self.bare_circuits = tuple(circuits["1r"])
        self.series_circuits = tuple(circuits["1t1r"])
        if any(a is b for a in self.bare_circuits for b in self.series_circuits):
            raise ValueError("Bare and series circuits need distinct identities for clock routing")

    def __call__(self, command, initial_u, law, circuit, **options):
        if any(circuit is candidate for candidate in self.bare_circuits):
            adapted = law
        elif any(circuit is candidate for candidate in self.series_circuits):
            adapted = (law if self.clock.alpha == 1. else
                       _ClockedDerivativeView(law, self.clock.series_multiplier))
        else:
            raise ValueError("Circuit does not belong to this protocol-clock problem")
        return self.backend(command, initial_u, adapted, circuit, **options)


class ConditionalClockProblem(SpectralJointProblem):
    """Inherited training masks, continuous 32-record propagation and scoring."""

    def checkpoint_metadata(self):
        result = super().checkpoint_metadata()
        result["conditional_protocol_clock"] = self.protocol_clock.metadata()
        result["frozen_spectral_shape"] = asdict(self.frozen_shape)
        result["frozen_table_path"] = str(TABLE_PATH)
        result["reset_initiation_log_fixed"] = 0.
        result["timing_policy"] = self.protocol_clock.metadata()["interpretation"]
        return result


def make_problem(alpha, integrator_options=None, mode="periodic") -> SpectralJointProblem:
    """Build the training-only conditional clock adapter, without evaluating it.

    q order is unchanged: five log10 coordinates (gamma_c, J_SET, J_RESET,
    k_SET, k_RESET), then linear initial_u_1r and initial_u_1t1r_cycle4.
    A_RESET is exactly zero and is not an eighth parameter. At alpha=1 the
    original simulator function is injected directly, with no derivative view.
    """
    clock = ProtocolClock(alpha)
    shape = SpectralShape.from_json()
    table = SpectralTable.load(TABLE_PATH, shape)

    def factory(q, ron):
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("Seven finite coordinates required; alpha is external")
        if abs(ron - shape.ron_ohm) > 1e-8:
            raise ValueError("Shared training Ron differs from the frozen spectral shape")
        exact = SpectralLaw(shape, *10. ** q[:5], reset_initiation_log=0.)
        return TabulatedSpectralLaw(exact, table)

    series = LookupSpectralCircuit(measured_transistor_mapping(), 1.1)
    circuits = {"1r": (BareSpectralCircuit(.001), BareSpectralCircuit()),
                "1t1r": (series, series)}
    simulator = (simulate_spectral_ivp if clock.alpha == 1. else
                 ProtocolClockSimulator(clock.alpha, circuits))
    sources = [Path(__file__), TABLE_PATH,
               ROOT / "analysis/constrained_shared_2026-11-03/lognormal_spectral_refinement.json",
               ROOT / "vulcan2d/shared_spectral.py",
               ROOT / "vulcan2d/shared_spectral_ivp.py",
               ROOT / "vulcan2d/shared_gap.py",
               ROOT / "vulcan2d/shared_gap_ivp.py",
               ROOT / "vulcan2d/data/fig2b_transistor_lookup.json",
               ROOT / "vulcan2d/data/bookoutput2_reverse_transistor_lookup.json"]
    problem = ConditionalClockProblem(
        factory, simulate_fn=simulator, circuits=circuits, mode=mode,
        integrator_options=integrator_options, additional_source_paths=sources,
        include_reset_initiation=False,
    )
    problem.protocol_clock, problem.frozen_shape = clock, shape
    assert tuple(problem.parameter_names) == PARAMETER_NAMES
    return problem
