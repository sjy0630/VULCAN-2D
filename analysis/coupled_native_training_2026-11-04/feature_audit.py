"""Original twenty terminal-feature gates on full native record predictions.

Pure postprocessing only. No model import, ODE, inverse state, periodic surrogate,
or data archive load occurs here. Eligibility and empirical bands are frozen in
the original training-only contract; failed model features are never dropped.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT/'analysis/constrained_shared_2026-11-03/feature_constraints.json'
CONTRACT_SOURCE = CONTRACT_PATH.with_suffix('.py')
TRAIN = {'1r':(1,2,3,5,6,7,8,9,10,11,12,13,14,15),
         '1t1r':tuple(range(4,36))}
PRIMARY = (
    '1r.set.outward.I_2V','1r.reset.return.I_0p2V',
    '1t1r.set.return.I_0p2V','1t1r.set.return.I_2V','1t1r.set.return.I_4V',
    '1t1r.reset.outward.I_0p2V','1t1r.reset.return.I_0p2V',
    '1t1r.set.return.alpha_0p1_0p2V','1t1r.set.return.alpha_1_2V',
    '1t1r.set.return.alpha_2_4V','1t1r.reset.outward.alpha_1_1p4V',
    '1t1r.reset.outward.geomean_current_1_1p4V',
    '1t1r.reset.outward.log_current_q90_q10_span_1_1p4V',
    '1r.reset.outward.peak_current','1r.reset.outward.peak_abs_voltage',
    '1t1r.reset.outward.peak_current','1t1r.reset.outward.peak_abs_voltage',
    '1r.set.transition_midpoint_at_0p995mA',
    '1r.set.return.R_0p05V','1r.reset.outward.R_fit_0p05_0p2V',
)


class UndefinedFeature(ValueError):
    """A finite native model curve does not define this requested feature."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def load_feature_contract():
    """Read the old training feature JSON only; no raw archive is opened."""
    contract = json.loads(CONTRACT_PATH.read_text())
    if contract['script_sha256'] != digest(CONTRACT_SOURCE):
        raise ValueError('Original feature definition source no longer matches saved contract')
    if tuple(contract['proposed_success_criteria_before_next_fit']['primary_feature_ids']) != PRIMARY:
        raise ValueError('Original twenty-feature contract changed')
    if {key:tuple(value) for key,value in contract['training_ids'].items()} != TRAIN:
        raise ValueError('Original feature training scope changed')
    if contract['training_arrays_sha256'] != 'ead6c5db908bab23b572b8959bfc40553e72a0dce966481f1af9d8a23cceab17':
        raise ValueError('Original training array digest changed')
    return contract


def _branch_indices(voltage, branch):
    apex = int(np.argmax(abs(voltage)))
    if branch == 'outward':
        return np.arange(apex+1)
    if branch == 'return':
        return np.arange(apex+1,len(voltage))
    raise ValueError('Feature branch must be outward or return')


def _current(prediction):
    if isinstance(prediction, Mapping):
        prediction = prediction['arrays'] if 'arrays' in prediction else prediction
        prediction = prediction['current_a']
    return np.asarray(prediction,dtype=float)


def _positive(values):
    if not np.all(np.isfinite(values)):
        raise ValueError('Nonfinite arithmetic in a required predicted feature')
    if np.any(np.asarray(values)<=0):
        raise UndefinedFeature('A required predicted current magnitude is nonpositive')


def _exact_index(voltage, index, magnitude, branch):
    index = int(index)
    if index not in _branch_indices(voltage,branch):
        raise ValueError('Frozen native feature index is on the wrong branch')
    if abs(abs(voltage[index])-magnitude)>3e-7:
        raise ValueError('Frozen native voltage sample differs; no interpolation allowed')
    return index


def _record_value(identifier, feature, observed_row, voltage, current):
    """Extract one trajectory feature; feature-level failures return undefined."""
    outward = _branch_indices(voltage,'outward')
    if '.I_' in identifier:
        j = _exact_index(voltage,observed_row['native_index'],feature['magnitude_V'],feature['branch'])
        _positive(current[j])
        return float(current[j]),dict(native_index=j)
    if '.alpha_' in identifier:
        a,b = feature['window_abs_V']
        ja,jb = observed_row['indices']
        _exact_index(voltage,ja,a,feature['branch'])
        _exact_index(voltage,jb,b,feature['branch'])
        _positive(current[[ja,jb]])
        value = np.log(current[jb]/current[ja])/np.log(abs(voltage[jb]/voltage[ja]))
        return float(value),dict(indices=[ja,jb])
    if '.geomean_current_' in identifier or '.log_current_q90_q10_span_' in identifier:
        a,b = feature['window_abs_V']
        indices = outward[(abs(voltage[outward])>=a-1e-9)&(abs(voltage[outward])<=b+1e-9)]
        if len(indices)!=observed_row['n_points']:
            raise ValueError('Frozen plateau native window changed')
        # All originally eligible records contain every required observed point.
        # Never select model points by a model-dependent floor or current range.
        _positive(current[indices])
        logs = np.log10(current[indices])
        value = (10.**logs.mean() if '.geomean_current_' in identifier
                 else np.quantile(logs,.9)-np.quantile(logs,.1))
        return float(value),dict(native_indices=indices.tolist())
    if identifier.endswith('.peak_current') or identifier.endswith('.peak_abs_voltage'):
        if not np.all(np.isfinite(current[outward])):
            raise ValueError('Nonfinite predicted outward current')
        j = int(outward[np.argmax(current[outward])])
        _positive(current[j])
        return (float(current[j]) if identifier.endswith('.peak_current') else float(abs(voltage[j]))),dict(
            native_index=j,endpoint_maximum=j==int(outward[-1]))
    if '.transition_midpoint_at_' in identifier:
        hits = np.flatnonzero(current[outward]>=feature['current_threshold_A'])
        if not len(hits):
            raise UndefinedFeature('Predicted outward curve never reaches fixed 0.995 mA threshold')
        j = int(outward[hits[0]])
        if j==0:
            raise UndefinedFeature('Predicted threshold has no native last-below bracket')
        return float((voltage[j-1]+voltage[j])/2.),dict(native_index=j,
            last_below_voltage_V=float(voltage[j-1]),first_crossing_voltage_V=float(voltage[j]),
            bracket_width_V=float(voltage[j]-voltage[j-1]))
    if identifier=='1r.set.return.R_0p05V':
        j = _exact_index(voltage,observed_row['native_index'],.05,'return')
        _positive(current[j])
        return float(abs(voltage[j])/current[j]),dict(native_index=j)
    if identifier=='1r.reset.outward.R_fit_0p05_0p2V':
        a,b = observed_row['window_abs_V']
        indices = outward[(abs(voltage[outward])>=a-3e-7)&(abs(voltage[outward])<=b+3e-7)]
        if len(indices)!=observed_row['n_points']:
            raise ValueError('Frozen low-voltage fit window changed')
        denominator = float(np.dot(abs(voltage[indices]),current[indices]))
        _positive(denominator)
        return float(np.dot(voltage[indices],voltage[indices])/denominator),dict(native_indices=indices.tolist())
    raise ValueError('Unsupported feature in the fixed original contract: '+identifier)


def evaluate_features(problem, predictions, contract=None):
    """Evaluate record medians against all original fixed bands, individually.

    Predictions must contain exactly the full 92 training phases. Nonfinite
    predictions are execution failures. A missing crossing or zero required
    model current is retained as an undefined feature and fails that gate; no
    finite penalty or exclusion is substituted into optimization objectives.
    """
    contract = load_feature_contract() if contract is None else contract
    expected = {(top,record,phase) for top,ids in TRAIN.items() for record in ids for phase in ('set','reset')}
    if set(problem.data)!=expected or set(predictions)!=expected:
        raise ValueError('Feature evaluation requires exactly all 46 training pairs / 92 native phases')
    curves = {}
    for key in expected:
        data = problem.data[key]
        voltage = np.asarray(data['command_voltage_v'],dtype=float)
        signed = _current(predictions[key])
        if signed.shape!=voltage.shape or signed.ndim!=1 or not np.all(np.isfinite(signed)):
            raise ValueError('Prediction shape/nonfinite failure in phase '+str(key))
        curves[key] = (voltage,abs(signed))
    result = {}
    for identifier in PRIMARY:
        feature = contract['features'][identifier]
        rows,undefined,values = [],[],[]
        for observed_row in feature['records']:
            record = int(observed_row['record'])
            row = dict(record=record,eligible=bool(observed_row['eligible']),
                       observed_value=float(observed_row['value']))
            if row['eligible']:
                key = feature['topology'],record,feature['phase']
                try:
                    value,detail = _record_value(identifier,feature,observed_row,*curves[key])
                    if not math.isfinite(value):
                        raise ValueError('Predicted trajectory feature is nonfinite')
                    values.append(value)
                    row.update(model_value=value,defined=True,**detail)
                except UndefinedFeature as exc:
                    row.update(model_value=None,defined=False,reason=str(exc))
                    undefined.append(record)
            else:
                row.update(model_value=None,defined=None,reason='Excluded by fixed original observed eligibility')
            rows.append(row)
        eligible_count = sum(row['eligible'] for row in rows)
        if eligible_count!=feature['distribution']['n']:
            raise ValueError('Original feature eligible record count changed')
        median = float(np.median(values)) if not undefined and len(values)==eligible_count else None
        interval = list(feature['proposed_empirical_acceptance_interval'])
        passed = median is not None and interval[0]<=median<=interval[1]
        result[identifier] = dict(unit=feature['unit'],model_median=median,
            observed_median=feature['distribution']['median'],
            observed_range=[feature['distribution']['min'],feature['distribution']['max']],
            acceptance_interval=interval,acceptance_mode=feature['acceptance_mode'],
            n_eligible=eligible_count,n_defined=len(values),undefined_records=undefined,
            passed=bool(passed),records=rows)
    failed = [name for name,row in result.items() if not row['passed']]
    return dict(n_primary=len(PRIMARY),n_passed=len(PRIMARY)-len(failed),all_passed=not failed,
        failed_ids=failed,features=result,aggregate='ordinary median of per-record trajectory features',
        fixed_observed_eligibility=True,no_model_floor_censoring=True,no_periodic_or_mean_curve_surrogate=True,
        undefined_features_are_penalty_values=False,contract_path=str(CONTRACT_PATH),
        contract_sha256=digest(CONTRACT_PATH),definition_source_sha256=digest(CONTRACT_SOURCE),
        original_training_arrays_sha256=contract['training_arrays_sha256'],
        historical_bare_pairing_assumption_superseded_by_user_confirmed_sequence=True,
        empirical_intervals_are_confidence_intervals=False,
        mechanism_identification_claimed=False)


def audit_observed_contract(problem, contract=None):
    """Pure-arithmetic extraction check for the root's frozen validation runner."""
    contract = load_feature_contract() if contract is None else contract
    predictions = {key:data['observed_current_a'] for key,data in problem.data.items()}
    result = evaluate_features(problem,predictions,contract)
    disagreements=[]
    max_scaled=0.
    for identifier,feature in result['features'].items():
        for row in feature['records']:
            if not row['eligible']:
                continue
            scale = 1e-30+1e-11*abs(row['observed_value'])
            difference = math.inf if row['model_value'] is None else abs(row['model_value']-row['observed_value'])
            max_scaled=max(max_scaled,difference/scale)
            if difference>scale:
                disagreements.append(dict(feature=identifier,record=row['record'],difference=difference,tolerance=scale))
    return dict(passed=not disagreements and result['all_passed'],n_features=len(PRIMARY),
        n_eligible_record_features=sum(f['n_eligible'] for f in result['features'].values()),
        max_scaled_record_difference=max_scaled,disagreements=disagreements,
        original_feature_bands_unchanged=True,observed_medians_in_all_original_bands=result['all_passed'])
