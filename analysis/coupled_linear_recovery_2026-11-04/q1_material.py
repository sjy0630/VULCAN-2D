"""Explicit q-linear recovery control; no experiment or data load on import.

Only q recovery differs from the frozen q-squared material. Readout,
capture, a kinetics, shared b, external circuits and training adapter are
inherited unchanged. A q2 parameter set is not an equivalent q1 rate fit.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import importlib
import math
from pathlib import Path
import sys
from types import MappingProxyType
from typing import ClassVar

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KERNEL_PATH = ROOT / 'analysis/coupled_native_training_2026-11-04/coupled_forward_v2.py'
KERNEL_SHA256 = 'c6873e8cec07965a3637bb695c942fab2fd75d699f0413ae1a2b0332097c2d2e'
TRAINING_PATH = KERNEL_PATH.with_name('training_problem_v2.py')
TRAINING_SHA256 = 'de4539d573e3b0c13fbfbf8a9d89b28d29a2baa8d3458ef3145bb0e9f32480f7'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    """Record this new implementation and require exact frozen dependencies."""
    result = {}
    for path, expected in ((KERNEL_PATH, KERNEL_SHA256),
                           (TRAINING_PATH, TRAINING_SHA256)):
        if sha(path) != expected:
            raise ValueError(f'Frozen dependency changed: {path}')
        result[str(path.resolve())] = expected
    result[str(Path(__file__).resolve())] = sha(__file__)
    return result


# The adapter already exposes the exact kernel module identity used in its
# isinstance checks. Importing it loads neither observations nor lookup tables.
source_hashes()
if str(TRAINING_PATH.parent) not in sys.path:
    sys.path.insert(0, str(TRAINING_PATH.parent))
training = importlib.import_module('training_problem_v2')
if Path(training.__file__).resolve() != TRAINING_PATH.resolve():
    raise RuntimeError('training_problem_v2 module name occupied by another source')
kernel = training.kernel


@dataclass(frozen=True)
class Q1Material(kernel.CoupledMaterial):
    """Same material/circuit interface, with fixed first-order q recovery."""

    fixed_recovery_order: ClassVar[int] = 1

    def __post_init__(self):
        for path, digest in source_hashes().items():
            if self.source_sha256.get(path) != digest:
                raise ValueError(f'Q1 material lacks exact implementation provenance: {path}')

    def rate_components(self, state, point):
        # Preserve all original state, material identity, solved-current and
        # passivity validation, plus the three unchanged rate components.
        terms = super().rate_components(state, point)
        vh = float(point['local_voltage_v'])
        if vh == 0:
            return terms
        q = float(state[0])
        p = self.parameters
        sign, v = (1 if vh > 0 else -1), abs(vh)
        chi = (1 - sign) * p.b
        # Direct multiplication is valid also at q=0. Never calculate q2/q.
        recovery = p.kappa_positive_per_v2 * v * math.exp(chi + p.lambda_per_v * v) * q
        if not math.isfinite(recovery) or recovery < 0:
            raise kernel.DomainFailure('Q1 positive recovery rate overflow/domain failure')
        terms['recovery_q_per_v'] = recovery
        return terms

    def metadata(self):
        result = super().metadata()
        result.update(
            fixed_recovery_order=self.fixed_recovery_order,
            material_variant='q1_a_linear_recovery_control',
            recovery_law='kappa_positive * abs(Vh) * exp(chi + lambda*abs(Vh)) * q',
            parent_fixed_recovery_order=2,
            rate_parameters_cross_order_equivalent=False,
            rate_clock='command-voltage path; not identified physical-time kinetics',
        )
        return result


def as_q1_material(parent):
    """Promote an explicit frozen q2 material; choose no parameters or states.

    This preserves numeric parameters exactly for a structural comparison.
    It does NOT assert that q2 and q1 kinetic scales are equivalent. The caller
    must separately predeclare any finite q1 rate comparison before execution.
    """
    if type(parent) is not kernel.CoupledMaterial:
        raise TypeError('Require the exact frozen q2 CoupledMaterial as parent')
    sources = dict(parent.source_sha256)
    if sources.get(str(KERNEL_PATH.resolve())) != KERNEL_SHA256:
        raise ValueError('Parent provenance differs from the frozen q2 kernel')
    for path, digest in source_hashes().items():
        if path in sources and sources[path] != digest:
            raise ValueError(f'Conflicting implementation provenance: {path}')
        sources[path] = digest
    values = {field.name: getattr(parent, field.name) for field in fields(kernel.CoupledMaterial)}
    values['source_sha256'] = MappingProxyType(sources)
    return Q1Material(**values)


def load_q1_material(parameters, *, candidate_source_id, candidate_source_sha256,
                     nearzero_policy='linear_endpoint', nearzero_anchor_v=None):
    """Explicitly load inherited frozen readout only; no roots or ODE runs."""
    parent = kernel.load_material(
        parameters, candidate_source_id=candidate_source_id,
        candidate_source_sha256=candidate_source_sha256,
        nearzero_policy=nearzero_policy, nearzero_anchor_v=nearzero_anchor_v,
    )
    return as_q1_material(parent)
