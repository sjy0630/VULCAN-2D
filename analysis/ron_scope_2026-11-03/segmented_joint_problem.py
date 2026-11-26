"""Shared beta1/A0 spectral law with explicit, full training histories.

Bare cycles1..3 and5..15 are two observed segments; measured-but-unavailable
cycle4 is never silently bridged. Series cycles4..35 form one continuous
segment. One material object is used for all92 phases in each evaluation.
Ron is an external fixed scenario, not a ninth optimization coordinate.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPECTRAL_DIR = ROOT / 'analysis/spectral_dynamic_2026-11-03'
for directory in (ROOT, SPECTRAL_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from joint_problem import SpectralJointProblem, TRAIN
from vulcan2d.shared_spectral import SpectralLaw, SpectralShape
from vulcan2d.shared_spectral_ivp import SpectralTable, TabulatedSpectralLaw

TABLE_PATH = SPECTRAL_DIR / 'transport_g010_v0025.npz'
PARAMETER_NAMES = (
    'log10_gamma_c_ev', 'log10_J_SET_A', 'log10_J_RESET_A',
    'log10_k_SET_per_V', 'log10_k_RESET_per_V',
    'initial_u_1r_cycle1', 'initial_u_1r_cycle5', 'initial_u_1t1r_cycle4',
)
SEGMENTS = (
    ('bare_cycles1_to3', '1r', (1, 2, 3), 5),
    ('bare_cycles5_to15', '1r', tuple(range(5, 16)), 6),
    ('series_cycles4_to35', '1t1r', tuple(range(4, 36)), 7),
)


class SegmentedSpectralJointProblem(SpectralJointProblem):
    """Reuse the frozen pair simulator and scorer, replacing history only."""

    def __init__(self, law_factory, *, ron_ohm, simulate_fn=None, circuits=None,
                 integrator_options=None, additional_source_paths=(), mode='full'):
        if mode != 'full':
            raise ValueError('Segmented history supports full evaluation only')
        if not np.isfinite(ron_ohm) or ron_ohm <= 0:
            raise ValueError('Scenario Ron must be finite and positive')
        super().__init__(law_factory, simulate_fn=simulate_fn, circuits=circuits,
                         mode='full', integrator_options=integrator_options,
                         additional_source_paths=additional_source_paths,
                         include_reset_initiation=False)
        self.training_median_ron_ohm = float(self.ron)
        self.transport_ron_ohm = float(ron_ohm)
        self.ron = self.transport_ron_ohm
        self.parameter_names = PARAMETER_NAMES
        self.n_parameters = 8
        self.initial_indices = {'1r_cycle1': 5, '1r_cycle5': 6, '1t1r_cycle4': 7}
        self.fallback_full = False
        assert self.train_ids['1r'] == list(SEGMENTS[0][2] + SEGMENTS[1][2])
        assert self.train_ids['1t1r'] == list(SEGMENTS[2][2])
        if any(command[0] != 0. or command[-1] != 0. for command in self.waveforms.values()):
            raise ValueError('Native phase waveforms must start/end at zero for the inherited pair interface')
        self.spectral_adapter_source_sha256 = self.objective_source_sha256
        self.objective_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def validate_q(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (8,) or not np.all(np.isfinite(q)):
            raise ValueError('Segmented q requires five finite log10 material coordinates and three initial states')
        if np.any(q[5:] < 0.):
            raise ValueError('All three observed-segment initial states must be nonnegative')
        return q

    def _validate_material(self, law):
        exact = law.exact_law if isinstance(law, TabulatedSpectralLaw) else law
        if not isinstance(exact, SpectralLaw):
            raise TypeError('The common material must be SpectralLaw or its tabulated adapter')
        if exact.beta != 1. or exact.reset_initiation_log != 0.:
            raise ValueError('This interface freezes beta=1 and A_RESET=0')
        if exact.shape.ron_ohm != self.transport_ron_ohm:
            raise ValueError('Actual common material Ron differs from the declared external scenario')
        return exact

    def evaluate(self, q, *, mode=None):
        if mode not in (None, 'full'):
            raise ValueError('No periodic/reused-pair mode exists for segmented history')
        q = self.validate_q(q)
        law = self.law_factory(q, self.transport_ron_ohm)
        exact = self._validate_material(law)
        predictions, history = {}, []
        for name, topology, labels, initial_index in SEGMENTS:
            state = float(q[initial_index])
            exact.validate_state(state)
            transitions = []
            for label in labels:
                initial_state = state
                pair = self._pair(law, state, topology)
                for phase, trace in zip(('set', 'reset'), pair):
                    predictions[topology, label, phase] = trace
                state = float(pair[-1]['final_u'])
                transitions.append({'record_id': label, 'initial_u': initial_state,
                                    'after_SET_u': float(pair[0]['final_u']), 'after_RESET_u': state})
            history.append({'name': name, 'topology': topology, 'record_ids': list(labels),
                            'initial_parameter_index': initial_index,
                            'initial_u': float(q[initial_index]), 'final_u': state,
                            'record_state_links': transitions})
        residual, blocks = self.score_predictions(predictions)
        details = {
            'status': 'full_segmented_training_evaluation_shared_material',
            'requested_mode': 'full', 'used_mode': 'full_segmented_46_pairs',
            'evaluation_split': 'training_only', 'periodicity': None,
            'periodic_shortcut_used': False, 'n_integrated_pairs': 46,
            'n_integrated_phase_calls': 92,
            'n_integrated_bare_pairs': 14, 'n_integrated_series_pairs': 32,
            'shared_material_objects_created': 1,
            'five_material_parameters_shared_across_all_segments': True,
            'actual_transport_Ron_ohm': self.transport_ron_ohm,
            'legacy_training_median_Ron_ohm': self.training_median_ron_ohm,
            'Ron_is_external_fixed_scenario_not_q_coordinate': True,
            'beta_fixed': exact.beta, 'A_RESET_fixed': exact.reset_initiation_log,
            'parameter_names': list(self.parameter_names), 'n_parameters': 8,
            'parameter_coordinate_types': ['log10']*5 + ['linear_state']*3,
            'initial_states_are_segment_boundary_conditions_not_new_material_laws': True,
            'history_segments': history,
            'missing_cycle4_bridged': False,
            'objective_is_exact_original_record_score': True,
            'objective_scope': 'Unchanged original27023-record residual with independently propagated predictions for each observed record',
            'n_residuals': self.n_residuals,
            'verified_cost': float(np.dot(residual, residual)/2.),
            'cost_includes_record_scatter': True,
            'legacy_observation_scatter_statistic': self.scatter_constant_cost,
            'scatter_statistic_subtracted_from_cost': False,
            'scatter_scope': 'Across-record mean/scatter decomposition is not an irreducible error floor when each record has its own continuously propagated prediction.',
            'blocks': blocks,
            'objective_source_sha256': self.objective_source_sha256,
        }
        return {'residual': residual, 'details': details, 'predictions': predictions}

    def compare_modes(self, q):
        raise ValueError('No periodic comparison is defined: segmented history is full only')

    def _periodicity(self, pairs):
        raise ValueError('Periodic prediction reuse is disabled for segmented history')

    def export_training_evaluation(self, q, output_json, *, mode='full',
                                   compare_modes=False, export_traces=True):
        if mode != 'full' or compare_modes:
            raise ValueError('Export supports full segmented histories only, without periodic comparison')
        return super().export_training_evaluation(q, output_json, mode='full',
                                                 compare_modes=False, export_traces=export_traces)

    def checkpoint_metadata(self):
        metadata = super().checkpoint_metadata()
        for key in ('periodic_current_rtol', 'periodic_state_atol', 'observation_scatter_constant_cost'):
            metadata.pop(key, None)
        metadata.update(
            status='shared_spectral_full_segmented_history_interface',
            ron_from_training_ohm=self.training_median_ron_ohm,
            legacy_training_median_Ron_ohm=self.training_median_ron_ohm,
            actual_transport_Ron_ohm=self.transport_ron_ohm,
            Ron_is_external_fixed_scenario_not_q_coordinate=True,
            parameter_coordinate_types=['log10']*5 + ['linear_state']*3,
            material_parameter_scope='Exactly five shared numerical material coordinates within each evaluation; no per-topology, per-cycle or per-segment material parameters',
            initial_state_scope='Three initial states only: observed bare segment1, observed bare segment5, series segment4; they are boundary conditions, not extra material laws',
            history_segments=[{'name': n, 'topology': t, 'record_ids': list(ids),
                               'initial_parameter_index': k} for n, t, ids, k in SEGMENTS],
            requested_mode='full', fallback_full=False, periodic_shortcut_available=False,
            reset_initiation_coordinate='Absent and fixed0; q[7] is series cycle4 initial state, not RESET initiation',
            bare_record_chronology_policy='Continuous1..3 and5..15; missing measured cycle4 is not invented or bridged',
            series_record_order=list(range(4, 36)),
            series_initial_state_used_once_at_record=4,
            legacy_observation_scatter_statistic=self.scatter_constant_cost,
            scatter_statistic_subtracted_from_cost=False,
            timing_policy='Native voltage-path history only; no unmeasured waits, timing, or missing-cycle waveform inserted',
            spectral_adapter_source_sha256=self.spectral_adapter_source_sha256,
        )
        if hasattr(self, 'scenario_spectral_shape'):
            metadata.update(base_spectral_shape=self.base_spectral_shape,
                            actual_scenario_spectral_shape=self.scenario_spectral_shape,
                            beta_fixed=1., A_RESET_fixed=0.,
                            table_amplitude_transformation=self.table_amplitude_transformation,
                            transport_table_metadata=self.transport_table.metadata)
        return metadata


def make_problem(ron_ohm, *, integrator_options=None, simulate_fn=None, mode='full'):
    """Construct one fixed-Ron scenario; do not integrate any trajectory."""
    if mode != 'full':
        raise ValueError('The segmented interface is full only')
    base_shape = SpectralShape.from_json()
    scenario_shape = replace(base_shape, ron_ohm=float(ron_ohm))
    base_table = SpectralTable.load(TABLE_PATH, base_shape)
    shift = math.log(base_shape.ron_ohm/scenario_shape.ron_ohm)
    transform = {
        'source_table_path': str(TABLE_PATH),
        'source_table_sha256': hashlib.sha256(TABLE_PATH.read_bytes()).hexdigest(),
        'original_table_Ron_ohm': base_shape.ron_ohm,
        'actual_transport_Ron_ohm': scenario_shape.ron_ohm,
        'natural_log_current_shift': shift,
        'identity': 'log_I_new = log_I_original + ln(Ron_original/Ron_new), both signed-magnitude branches',
        'changed_shape_fields': ['ron_ohm'] if shift != 0. else [],
        'new_spectral_quadrature_performed': False,
        'precision_scope': 'Exact amplitude identity at fixed shape/Gamma; inherits original numerical quadrature/interpolation error and does not certify new trajectory accuracy',
    }
    metadata = {'shape': asdict(scenario_shape),
                'source_table_metadata_unmodified': dict(base_table.metadata),
                'amplitude_transformation': transform,
                'scope': 'In-memory amplitude-transformed old table; no frozen file overwritten'}
    table = SpectralTable(scenario_shape, base_table.log_gamma.copy(),
                          base_table.log_voltage.copy(), base_table.log_current + shift, metadata)

    def factory(q, actual_ron):
        if actual_ron != scenario_shape.ron_ohm:
            raise ValueError('Caller Ron differs from the fixed scenario')
        with np.errstate(over='raise', invalid='raise'):
            parameters = 10.**np.asarray(q[:5], float)
        exact = SpectralLaw(scenario_shape, *parameters, reset_initiation_log=0.)
        return TabulatedSpectralLaw(exact, table)

    sources = [Path(__file__), TABLE_PATH,
               ROOT/'analysis/bare_history_2026-11-03/identity_confirmation.json',
               ROOT/'analysis/constrained_shared_2026-11-03/lognormal_spectral_refinement.json',
               ROOT/'vulcan2d/shared_spectral.py', ROOT/'vulcan2d/shared_spectral_ivp.py',
               ROOT/'vulcan2d/shared_gap.py', ROOT/'vulcan2d/shared_gap_ivp.py',
               ROOT/'vulcan2d/data/fig2b_transistor_lookup.json',
               ROOT/'vulcan2d/data/bookoutput2_reverse_transistor_lookup.json']
    problem = SegmentedSpectralJointProblem(factory, ron_ohm=scenario_shape.ron_ohm,
                                            simulate_fn=simulate_fn, mode=mode,
                                            integrator_options=integrator_options,
                                            additional_source_paths=sources)
    if abs(problem.training_median_ron_ohm-base_shape.ron_ohm) > 1e-8:
        raise ValueError('Frozen table reference Ron no longer matches the legacy training calibration')
    problem.base_spectral_shape = asdict(base_shape)
    problem.scenario_spectral_shape = asdict(scenario_shape)
    problem.table_amplitude_transformation = transform
    problem.transport_table = table
    return problem
