"""Independent saved-only audit; requires root-confirmed terminal identifiers.

This file never imports a model, optimizer, solver or project implementation.
It reads only declared saved outputs and the 184 original training arrays.
No numerical model evaluations or holdout-array decoding are possible here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / 'analysis/coupled_linear_recovery_2026-11-04'
TRAIN = {'1r': [1, 2, 3] + list(range(5, 16)), '1t1r': list(range(4, 36))}
SEGMENTS = [('bare1', '1r', [1, 2, 3]), ('bare5', '1r', list(range(5, 16))),
            ('series4', '1t1r', list(range(4, 36)))]
OBSERVATIONS = Path('/private/tmp/vulcan-runtime-cache/native_observations.npz')
CONTRACT = ROOT / 'analysis/constrained_shared_2026-11-03/feature_constraints.json'
OBS_SHA = '0aedbe51f41b4f4355e1de4d26048c6790a9d454b8b7d56f51bd76a4671465a8'
EXPECTED_COUNTS = [730, 27, 2027, 2799, 16000, 5440]
CACHE_MANIFEST = Path('/private/tmp/vulcan-exact-read-cache-2026-11-04/ENTRY_PROFILE_MANIFEST_2026-11-05.json')
FALLBACK_CACHE_MANIFEST = Path('/private/tmp/vulcan-exact-read-cache-2026-11-04/MANIFEST.json')
READ_METADATA_STABILIZATIONS = set()


def require(condition, label):
    if not condition:
        raise AssertionError(label)


def close(a, b, label, rtol=2e-12, atol=1e-25):
    require(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False), label)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def exact_read(path):
    """Read the fstat size without an extra EOF probe; never change the file."""
    fd=os.open(path,os.O_RDONLY)
    try:
        def snapshot():
            before=os.fstat(fd);offset=0;parts=[]
            while offset<before.st_size:
                part=os.pread(fd,min(1024*1024,before.st_size-offset),offset)
                require(bool(part),'Unexpected short read: '+str(path))
                parts.append(part);offset+=len(part)
            return before,os.fstat(fd),b''.join(parts)
        signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        before,after,blob=snapshot()
        if signature(before)!=signature(after):
            # Cloud hydration can update mtime during a first read. Accept only
            # identical contents and identity, then TWO fully stable rereads.
            require((before.st_dev,before.st_ino,before.st_size)==(after.st_dev,after.st_ino,after.st_size),
                    'File identity/size changed while read: '+str(path))
            for _ in range(2):
                prior,following,repeated=snapshot()
                require(signature(prior)==signature(following)==signature(after),
                        'File metadata not stable on verification reread: '+str(path))
                require(repeated==blob,'File contents changed on verification reread: '+str(path))
            READ_METADATA_STABILIZATIONS.add(str(path))
        return blob
    finally:
        os.close(fd)


class SavedFiles:
    def __init__(self):
        self.blobs = {}
        self.cache_map={}
        self.cache_used=set()
        self.cache_manifest=CACHE_MANIFEST if CACHE_MANIFEST.is_file() else FALLBACK_CACHE_MANIFEST
        if self.cache_manifest.is_file():
            manifest=json.loads(exact_read(self.cache_manifest))
            self.cache_map={Path(row['path']):row for row in manifest['sources']}

    def _read(self,path):
        path=Path(path)
        if path in self.cache_map:
            row=self.cache_map[path];blob=exact_read(row['cache_path'])
            require(len(blob)==row['bytes'] and sha(blob)==row['sha256'],'Exact cached source bytes: '+str(path))
            self.cache_used.add(path)
            return blob
        return exact_read(path)

    def prefetch(self, paths):
        paths = [Path(p) for p in dict.fromkeys(paths) if Path(p) not in self.blobs]
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.blobs.update(zip(paths, pool.map(self._read, paths)))

    def blob(self, path):
        path = Path(path)
        if path not in self.blobs:
            self.blobs[path] = self._read(path)
        return self.blobs[path]

    def js(self, path):
        return json.loads(self.blob(path))

    def npz(self, path):
        # Whole-file memory reads avoid repeated seeks on macOS dataless files.
        with np.load(io.BytesIO(self.blob(path)), allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}


def observation_training(files):
    blob = files.blob(OBSERVATIONS)
    require(sha(blob) == OBS_SHA, 'Original observations hash')
    data = {}
    decoded = []
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        for top, ids in TRAIN.items():
            for phase in ('set', 'reset'):
                for cid in ids:
                    prefix = f'{top}_cycle_{cid}_{phase}_'
                    names = [prefix + 'voltage', prefix + 'current']
                    v, current = (archive[name] for name in names)
                    decoded.extend(names)
                    mask = np.isfinite(current) & (abs(v) > 1e-12)
                    mask &= abs(current) >= (1e-11 if phase == 'set' else 1e-12)
                    data[top, cid, phase] = dict(command_voltage_v=v, observed_current_a=current,
                                                original_scoring_mask=mask)
    require(len(decoded) == 184 and len(data) == 92, 'Only original training arrays decoded')
    return data, decoded


def phase_arrays(saved, key):
    prefix = f'{key[0]}_cycle_{key[1]}_{key[2]}__'
    return {name[len(prefix):]: value for name, value in saved.items() if name.startswith(prefix)}


def native_checks(a, observed, params, label):
    for name in ('command_voltage_v', 'observed_current_a', 'original_scoring_mask'):
        require(np.array_equal(a[name], observed[name]), label + ' original ' + name)
    q, u, hard = a['state_q'], a['state_u'], a['state_a_s']
    require(np.all(np.isfinite(q + u + hard)), label + ' finite states')
    require(np.all((q > 0) & (q < 1) & (u > 0) & (u <= 80)), label + ' q/u domain')
    require(np.all((hard >= 0) & (hard <= params['a_max_s'])), label + ' current A domain')
    close(q, np.exp(-u), label + ' q/u identity', rtol=1e-13, atol=0)
    current = a['current_a']
    scale = 1e-18 + 1e-7 * abs(current)
    require(np.all(abs(a['KCL_residual_a']) <= scale), label + ' KCL')
    require(np.all(abs(a['KVL_residual_v']) <= 1e-9), label + ' KVL')
    require(np.all(abs(current-a['soft_current_a']-a['hard_current_a']) <= scale), label + ' branch sum')
    require(np.all(abs(a['hard_current_a']-hard*a['local_voltage_v']) <= scale), label + ' hard readout')
    for name in ('current_a', 'soft_current_a', 'hard_current_a'):
        require(np.all(np.isfinite(a[name])) and np.all(a[name]*a['local_voltage_v'] >= 0), label + ' passive ' + name)


def precision_difference(a, b, A):
    mask = a['original_scoring_mask']
    require(np.array_equal(mask, b['original_scoring_mask']), 'Precision unchanged mask')
    require(np.array_equal(a['command_voltage_v'], b['command_voltage_v']), 'Precision unchanged waveform')
    di = float(np.max(abs(np.log10(abs(a['current_a'][mask]))-np.log10(abs(b['current_a'][mask])))))
    du = float(np.max(abs(a['state_u']-b['state_u'])))
    dx = float(np.max(abs(a['state_a_s']-b['state_a_s'])) / A)
    return dict(max_logI_decade=di, max_abs_u=du, max_abs_a_over_A=dx,
                passed=di <= 3e-4 and du <= 1e-3 and dx <= 1e-5)


def feature_value(name, definition, observed_row, v, current):
    current = abs(current)
    outward = np.arange(int(np.argmax(abs(v)))+1)
    if '.I_' in name:
        value = current[int(observed_row['native_index'])]
        return float(value) if value > 0 else None
    if '.alpha_' in name:
        a, b = observed_row['indices']
        if min(current[a], current[b]) <= 0:
            return None
        return float(np.log(current[b]/current[a])/np.log(abs(v[b]/v[a])))
    if '.geomean_current_' in name or '.log_current_q90_q10_span_' in name:
        a, b = definition['window_abs_V']
        selected = outward[(abs(v[outward]) >= a-1e-9) & (abs(v[outward]) <= b+1e-9)]
        require(len(selected) == observed_row['n_points'], 'Original plateau count')
        if np.any(current[selected] <= 0):
            return None
        logs = np.log10(current[selected])
        return float(10**logs.mean() if '.geomean_current_' in name else np.quantile(logs,.9)-np.quantile(logs,.1))
    if name.endswith('.peak_current') or name.endswith('.peak_abs_voltage'):
        j = outward[np.argmax(current[outward])]
        if current[j] <= 0:
            return None
        return float(current[j] if name.endswith('.peak_current') else abs(v[j]))
    if '.transition_midpoint_at_' in name:
        hits = np.flatnonzero(current[outward] >= definition['current_threshold_A'])
        if not len(hits) or hits[0] == 0:
            return None
        j = outward[hits[0]]
        return float((v[j-1]+v[j])/2)
    if name == '1r.set.return.R_0p05V':
        j = int(observed_row['native_index'])
        return float(abs(v[j])/current[j]) if current[j] > 0 else None
    if name == '1r.reset.outward.R_fit_0p05_0p2V':
        a,b = observed_row['window_abs_V']
        selected = outward[(abs(v[outward]) >= a-3e-7) & (abs(v[outward]) <= b+3e-7)]
        require(len(selected) == observed_row['n_points'], 'Original resistance window count')
        denominator = np.dot(abs(v[selected]),current[selected])
        return float(np.dot(v[selected],v[selected])/denominator) if denominator > 0 else None
    raise AssertionError('Unrecognized original feature '+name)


def audit_features(contract, saved_json, predictions, observations):
    rows = {}
    n_record = 0
    primary = contract['proposed_success_criteria_before_next_fit']['primary_feature_ids']
    require(len(primary) == 20, 'Twenty original features')
    for name in primary:
        definition = contract['features'][name]
        expected = saved_json['features']['features'][name]
        values, undefined = [], []
        for row, reported in zip(definition['records'], expected['records'], strict=True):
            require(row['record'] == reported['record'] and row['eligible'] == reported['eligible'], 'Fixed feature eligibility')
            if not row['eligible']:
                continue
            n_record += 1
            key = definition['topology'],int(row['record']),definition['phase']
            value = feature_value(name,definition,row,observations[key]['command_voltage_v'],predictions[key]['current_a'])
            if value is None:
                undefined.append(row['record'])
                require(reported['model_value'] is None and not reported['defined'], 'Undefined feature retained')
            else:
                close(value,reported['model_value'],'Record feature '+name,atol=1e-14)
                values.append(value)
        median = float(np.median(values)) if not undefined else None
        interval = definition['proposed_empirical_acceptance_interval']
        passed = median is not None and interval[0] <= median <= interval[1]
        require(passed == expected['passed'], 'Original gate '+name)
        if median is not None:
            close(median,expected['model_median'],'Median feature '+name,atol=1e-14)
        require(list(interval) == expected['acceptance_interval'], 'Original feature bands')
        rows[name] = dict(model_median=median,observed_median=definition['distribution']['median'],
                          acceptance_interval=interval,passed=bool(passed),undefined_records=undefined)
    require(n_record == 514, '514 eligible record features')
    require(sum(row['passed'] for row in rows.values()) == saved_json['features']['n_passed'], 'Gate count')
    return rows


def audit_score(saved_json, saved_npz, observations, parameters):
    weighted, raw, counts, blocks, predictions = [], [], [], {}, {}
    phases = {(row['topology'],row['record_id'],row['phase']):row['summary'] for row in saved_json['phases']}
    require(set(phases) == set(observations), 'Full original 92 phases')
    for key, observed in observations.items():
        a = phase_arrays(saved_npz,key)
        native_checks(a,observed,parameters,str(key))
        summary = phases[key]
        require(summary['parameters'] == dict(parameters,enable_a=True), 'Fixed material parameters')
        require(summary['fixed_recovery_order'] == 1, 'Fixed recovery order')
        require(summary['material_variant'] == 'q1_a_linear_recovery_control', 'Same q1 material')
        require(summary['no_state_reanchoring'] and summary['all_native_points_retained'], 'No phase reanchoring')
        if key[0] == '1r' and key[2] == 'set':
            require(np.all(abs(a['current_a']) <= .001+1e-18+1e-7*abs(a['current_a'])), 'Bare actual compliance')
        predictions[key] = a
    for top, ids in TRAIN.items():
        for phase in ('set','reset'):
            observed = abs(np.array([observations[top,cid,phase]['observed_current_a'] for cid in ids]))
            predicted = abs(np.array([predictions[top,cid,phase]['current_a'] for cid in ids]))
            mask = np.array([observations[top,cid,phase]['original_scoring_mask'] for cid in ids])
            v = observations[top,ids[0],phase]['command_voltage_v']
            outward = np.arange(len(v)) <= np.argmax(abs(v))
            groups = [mask & (observed < .995e-3) & outward, mask & (observed < .995e-3) & ~outward,
                      mask & (observed >= .995e-3)] if (top,phase) == ('1r','set') else [mask]
            mse = 0.
            for group in groups:
                n = int(group.sum()); counts.append(n)
                errors = np.log10(predicted[group])-np.log10(observed[group])
                normalized = errors/np.sqrt(n*len(groups))
                weighted.append(normalized); raw.append(errors)
                mse += float(np.dot(normalized,normalized))
            raw_block = np.log10(predicted[mask])-np.log10(observed[mask])
            key = top+'_'+phase
            close(mse,saved_json['score']['blocks'][key]['weighted_total_mse'],'Weighted phase MSE')
            blocks[key] = dict(n=int(mask.sum()),weighted_RMSE_decade=float(np.sqrt(mse)),
                               raw_RMSE_decade=float(np.sqrt(np.mean(raw_block**2))))
    weighted,raw = np.concatenate(weighted),np.concatenate(raw)
    require(counts == EXPECTED_COUNTS and len(raw) == 27023,'Original groups and residual length')
    require(np.array_equal(weighted,saved_npz['weighted_residual']), 'Bitwise weighted residual rebuild')
    require(np.array_equal(raw,saved_npz['raw_residual']), 'Bitwise raw residual rebuild')
    cost = .5*float(np.dot(weighted,weighted))
    rmse = float(np.sqrt(np.mean(raw**2)))
    require(cost == saved_json['score']['verified_cost'], 'Exact original cost rebuild')
    require(rmse == saved_json['score']['plain_allpoint_RMSE_decade'], 'Exact raw RMSE rebuild')
    return predictions,phases,dict(cost=cost,raw_RMSE_decade=rmse,blocks=blocks,n_residuals=len(raw),
                                  residual_vectors_rebuilt_bitwise=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terminal-session',type=int,required=True)
    parser.add_argument('--terminal-exit',type=int,required=True)
    args=parser.parse_args()
    files=SavedFiles(); run=HERE/'entry_profile_run'
    report=files.js(run/'RESULT.json')
    require(report['status'] in ('bounded_entry_profile_and_full_history_checks_finished','stopped_with_retained_failure'),
            'Only root-confirmed terminal run may be reviewed')
    freeze=files.js(HERE/'SOURCE_FREEZE.json')['files_sha256']
    files.prefetch(freeze)
    current={path:sha(files.blob(path)) for path in freeze}
    require(current == freeze == report['source_sha256_before'] == report['source_sha256_after'],'All frozen inputs unchanged')
    require(sha(files.blob(HERE/'SOURCE_FREEZE.json')) == report['freeze_sha256_before'] == report['freeze_sha256_after'],
            'Freeze digest unchanged')
    parent=files.js(PARENT/'rate_controls_run/RESULT.json')
    params=parent['selected_coarse_candidate']['parameters']; A=params['a_max_s']
    require(report['fixed_material_parameters'] == params and report['fixed_recovery_order'] == 1,'Material frozen to q1 parent')
    require(report['original_entries'] == parent['fixed_segment_entries'],'Original entry provenance')
    observations,decoded=observation_training(files)
    contract=files.js(CONTRACT)
    old_path=Path(report['old_entry_control_artifact_prefix'])
    old_json=files.js(old_path.with_suffix('.json')); old_npz=files.npz(old_path.with_suffix('.npz'))
    calls=report['prefix_calls']; starts=report['starts']
    expected_paths=[]
    for call in calls:
        prefix=Path(call['artifact_prefix']); expected_paths.extend([prefix.with_suffix('.json'),prefix.with_suffix('.npz')])
    files.prefetch(expected_paths)
    require([c['call_id'] for c in calls] == list(range(1,len(calls)+1)), 'Sequential prefix call IDs')
    require(sum(c['model_evaluation_performed'] for c in calls) == report['n_actual_prefix_evaluations'] <= 900,
            'Actual prefix evaluation count')
    call_arrays={}; prefixes={}; prefix_results=[]
    for name,top,ids in SEGMENTS:
        full=observations[top,ids[0],'set']; v=full['command_voltage_v']
        keep=np.flatnonzero((np.arange(len(v)) <= np.argmax(abs(v))) & (v>0) & (v<=.25))
        stop=int(keep[-1])+1
        data={key:value[:stop] for key,value in full.items()}
        mask=data['original_scoring_mask']; first=int(np.flatnonzero(mask)[0])
        gref=float(abs(data['observed_current_a'][first])/data['command_voltage_v'][first])
        segment=report['segments'][name]
        require(segment['n_eligible'] == int(mask.sum()) and segment['gref_s'] == gref,'Original prefix and gref')
        require(segment['prefix_native_indices'] == list(range(stop)),'Native leading prefix only')
        fit=dict(name='prefix_fit',rtol=1e-8,atol_u=1e-10,atol_x=min(1e-14,1e-6*gref/A),max_step=.01)
        refined=dict(name='prefix_refined',rtol=1e-10,atol_u=1e-12,atol_x=fit['atol_x']/100,max_step=.005)
        require(segment['fit_precision'] == fit and segment['refined_precision'] == refined,'Declared prefix precisions')
        prefixes[name]=dict(observed=data,key=(top,ids[0],'set'),gref=gref,stop=stop,fit=fit,refined=refined)
    for call in calls:
        name=call['segment']; prefix=prefixes[name]; path=Path(call['artifact_prefix'])
        meta=files.js(path.with_suffix('.json')); arrays=files.npz(path.with_suffix('.npz'))
        require(meta['call'] == call,'Saved prefix call matches report')
        call_arrays[call['call_id']]=arrays
        if call['status'] != 'complete_valid':
            require(call['no_finite_penalty_returned'],'No fabricated prefix failure residual')
            continue
        phase=meta['phase_summary']
        require(phase['parameters'] == dict(params,enable_a=True) and phase['fixed_recovery_order'] == 1,'Prefix fixed material')
        require([phase['initial_state_q'],phase['initial_state_a_s']] == [call['initial_state_q'],call['initial_state_a_s']],
                'Requested prefix entry')
        native_checks(arrays,prefix['observed'],params,'Prefix '+str(call['call_id']))
        mask=prefix['observed']['original_scoring_mask']
        residual=np.log10(abs(arrays['current_a'][mask]))-np.log10(abs(prefix['observed']['observed_current_a'][mask]))
        require(np.array_equal(residual,arrays['residual']),'Exact prefix residual')
        require(.5*float(np.dot(residual,residual)) == call['cost'],'Exact prefix cost')
        require(meta['numerical_verification']['passed'],'Saved native prefix verification')
        precision=prefix['refined'] if call['role'] in ('old_entry_regression','endpoint_refined') else prefix['fit']
        require(call['precision'] == precision,'Call-specific declared precision')
        roots=2 if call['role'] in ('old_entry_regression','endpoint_fit_precision','endpoint_refined') else 0
        require(meta['numerical_verification']['alternate_root_checks'] == roots,'Saved prefix root count')
        prefix_results.append(dict(call_id=call['call_id'],cost=call['cost'],n_residuals=len(residual)))
    regressions=report['baseline_regressions']
    require(len(regressions) == 3 and [c['role'] for c in calls[:3]] == ['old_entry_regression']*3,'Three regressions precede all fitting')
    for regression in regressions:
        name=regression['segment']; prefix=prefixes[name]; call=calls[regression['call_id']-1]
        require([call['initial_state_q'],call['initial_state_a_s']] == report['original_entries'][name],'Regression old entry')
        control={key:value[:prefix['stop']] for key,value in phase_arrays(old_npz,prefix['key']).items()
                 if key in ('command_voltage_v','original_scoring_mask','current_a','state_u','state_a_s')}
        difference=precision_difference(call_arrays[call['call_id']],control,A)
        require(difference['passed'] == regression['passed'],'Regression independently recomputed')
        for key,value in difference.items():
            if key != 'passed':close(value,regression['comparison'][key],'Regression maxima')
    require(report['all_prefix_regressions_passed'],'Fitting only after all regressions pass')
    require(len(starts) <= 9,'Nine-start maximum')
    start_reviews=[]; segment_reviews={}
    for name,top,ids in SEGMENTS:
        prefix=prefixes[name]; segment=report['segments'][name]
        local=[s for s in starts if s['segment'] == name]
        require([s['fraction'] for s in local] == [0.,.5,.99],'Three fixed starts per segment')
        valid=[]
        for start in local:
            sid=start['start_id']; local_calls=[c for c in calls if c['start_id'] == sid]
            roles=Counter(c['role'] for c in local_calls)
            for key,role in [('objective_calls','optimizer_objective'),('jacobian_perturbation_calls','jacobian_perturbation'),
                             ('jacobian_base_extra_calls','jacobian_base')]:
                require(roles[role] <= start[key],'Attempted/recorded prefix call accounting')
                if start['status'] == 'valid_numerically_checked_ending_point':
                    require(roles[role] == start[key],'Completed-start exact call accounting')
            require(len(start['finite_difference_steps']) == start['jacobian_perturbation_calls'],'Every FD step logged')
            fraction=start['fraction']; old=report['original_entries'][name]
            proposed=[old[0]*(1-fraction),fraction*prefix['gref']]
            close(start['proposed_initial_state'],proposed,'Fixed initial proposal',rtol=0,atol=0)
            z=np.array([math.log(proposed[0]),proposed[1]/prefix['gref']])
            require(np.array_equal(start['proposed_z'],z),'Proposed normalized coordinates')
            lower=np.asarray(start['lower']);upper=np.asarray(start['upper'])
            require(np.array_equal(start['projected_z'],np.minimum(upper,np.maximum(lower,z))),'Initial-only parameter projection')
            perturbations=[c for c in local_calls if c['role']=='jacobian_perturbation']
            rebuilt_columns=[]
            for step,call in zip(start['finite_difference_steps'],perturbations):
                base=np.asarray(step['base_z']);axis=int(step['axis'])
                desired=1e-4*max(1.,abs(float(base[axis])))
                require(desired==step['requested_abs_step'],'Declared finite-difference requested step')
                plus,minus=float(upper[axis]-base[axis]),float(base[axis]-lower[axis])
                delta=(desired if plus>=desired else -desired if minus>=desired else
                       .5*plus if plus>=minus and plus>0 else -.5*minus)
                trial=base.copy();trial[axis]+=delta;actual=float(trial[axis]-base[axis])
                require(actual!=0 and actual==step['actual_step'],'One-sided feasible actual difference step')
                require(np.all(trial>=lower) and np.all(trial<=upper),'Jacobian perturbation parameter bounds')
                require([math.exp(float(trial[0])),float(trial[1])*prefix['gref']] ==
                        [call['initial_state_q'],call['initial_state_a_s']],'Perturbation corresponds to recorded trial')
                base_state=[math.exp(float(base[0])),float(base[1])*prefix['gref']]
                possible=[c for c in local_calls if c['call_id']<call['call_id']
                          and c['role'] in ('optimizer_objective','jacobian_base')
                          and c['status']=='complete_valid'
                          and [c['initial_state_q'],c['initial_state_a_s']]==base_state]
                require(bool(possible),'Exact saved base exists for every difference')
                if call['status']=='complete_valid':
                    difference=(call_arrays[call['call_id']]['residual']-
                                call_arrays[possible[-1]['call_id']]['residual'])/actual
                    rebuilt_columns.append((axis,base.tolist(),difference))
            if 'scipy_nfev' in start:
                require(start['scipy_nfev'] <= 30 and start['scipy_nfev'] == start['objective_calls'],'SciPy nfev vs objective calls')
            review=dict(start_id=sid,status=start['status'],calls_by_role=dict(roles),
                        actual_model_calls=sum(c['model_evaluation_performed'] for c in local_calls))
            if start['eligible_for_entry_selection']:
                require(start['status'] == 'valid_numerically_checked_ending_point','Only qualified normal ending selectable')
                valid.append(start)
                a,b=(call_arrays[x] for x in start['endpoint_call_ids'])
                difference=precision_difference(a,b,A);require(difference['passed'],'Endpoint independent precision comparison')
                last_call=calls[start['endpoint_call_ids'][1]-1]
                require(last_call['cost'] == start['selection_cost'],'Refined endpoint selects entry')
                for cid in start['endpoint_call_ids']:
                    c=calls[cid-1]
                    require([c['initial_state_q'],c['initial_state_a_s']] == start['ending_state'],'Rerun actual optimizer ending')
                jd=start['jacobian_diagnostics'];J=np.asarray(jd['matrix'])
                require(len(rebuilt_columns)>=2 and [x[0] for x in rebuilt_columns[-2:]]==[0,1],
                        'Two returned Jacobian columns have saved finite-difference evidence')
                require(rebuilt_columns[-2][1]==rebuilt_columns[-1][1]==start['ending_z'],
                        'Returned Jacobian is evaluated at optimizer ending')
                require(np.array_equal(np.column_stack([x[2] for x in rebuilt_columns[-2:]]),J),
                        'Returned optimizer Jacobian independently rebuilt bitwise')
                norms=np.linalg.norm(J,axis=0);_,singular,vt=np.linalg.svd(J,full_matrices=False)
                close(norms,jd['column_norms'],'Jacobian norms');close(singular,jd['singular_values'],'Jacobian singular values')
                ranktol=max(J.shape)*np.finfo(float).eps*singular[0]
                require(jd['numerical_rank'] == int(np.sum(singular>ranktol)),'Jacobian rank')
                alignment=float(np.dot(J[:,0]/norms[0],J[:,1]/norms[1])) if min(norms)>0 else None
                if alignment is not None:close(alignment,jd['normalized_column_inner_product'],'Jacobian alignment')
                else:require(jd['normalized_column_inner_product'] is None,'Undefined zero-column alignment')
                close(abs(vt),abs(np.asarray(jd['svd_right_vectors_rows'])),'SVD right vectors up to sign')
                review.update(selection_cost=start['selection_cost'],ending_state=start['ending_state'],
                              endpoint_precision=difference,jacobian_alignment=alignment,jacobian_singular_values=singular.tolist(),
                              numerical_rank=jd['numerical_rank'],entry_identifiability_established=False)
            else:
                require(start.get('best_intermediate_point_not_selected',True),'Failed-start intermediate not selected')
            start_reviews.append(review)
        if valid:
            selected=min(valid,key=lambda s:(s['selection_cost'],s['start_id']))
            require(segment['selected_start_id'] == selected['start_id'] and segment['selected_entry'] == selected['ending_state'],
                    'Minimum refined prefix cost selection')
        else:
            require(segment['selected_entry'] == report['original_entries'][name],'Explicit old-entry fallback')
        pairs=segment['ending_point_identifiability_diagnostics']['pairs'];computed=[]
        for (left,right),pair in zip(itertools.combinations(valid,2),pairs,strict=True):
            delta_cost=right['selection_cost']-left['selection_cost'];dmse=2*delta_cost/segment['n_eligible']
            dq=math.log(right['ending_state'][0])-math.log(left['ending_state'][0])
            da=(right['ending_state'][1]-left['ending_state'][1])/prefix['gref']
            near=abs(dmse)<=1e-4;distinct=abs(dq)>=math.log(2) or abs(da)>=.1
            close(dmse,pair['delta_mean_squared_log_residual_decade2'],'Endpoint cost comparison')
            require(pair['near_equal_cost_descriptive'] == near and pair['distinct_entries_descriptive'] == distinct,'Predeclared endpoint descriptors')
            require(pair['near_equal_cost_distinct_alternative'] == (near and distinct),'Endpoint alternative flag')
            computed.append(dict(left_start_id=left['start_id'],right_start_id=right['start_id'],
                                 delta_MSE_decade2=dmse,delta_ln_q=dq,delta_a_over_gref=da,alternative=bool(near and distinct)))
        require(report['selected_entries'][name] == segment['selected_entry'],'Three selected entries only')
        segment_reviews[name]=dict(n_qualified_endings=len(valid),selected_entry=segment['selected_entry'],
                                  selected_start_id=segment.get('selected_start_id'),pairs=computed,
                                  original_entry=report['original_entries'][name],
                                  entry_a_is_nonzero=segment['selected_entry'][1]!=0,
                                  entry_a_over_current_A=segment['selected_entry'][1]/A,
                                  old_a0_zero_bounds_automatically_reused=False,
                                  identifiability_remains_unresolved=True)
    print('frozen_material_prefix_calls_selection_identifiability_checked',flush=True)
    full_reviews={};full_arrays={};all_differences=[]
    for request in report['final_requests']:
        require(request['material_parameters'] == params and request['initial_states'] == report['selected_entries'],'Final fixed material / entries')
        path=Path(request['artifact_prefix']);js=files.js(path.with_suffix('.json'));npz=files.npz(path.with_suffix('.npz'))
        if request['status'] != 'complete_valid':
            require(not request['eligible'],'Failed full run not eligible')
            full_reviews[request['precision']['name']]=dict(status=request['status'],summary=js['summary'])
            continue
        require(js['summary']['n_completed_phases'] == 92,'Each complete final has 92 phases')
        require(request['alternate_root_checks'] == 184,'184 saved alternate roots per final')
        predictions,phase_summaries,score=audit_score(js,npz,observations,params)
        for name,top,ids in SEGMENTS:
            previous=report['selected_entries'][name]
            for cid in ids:
                for phase in ('set','reset'):
                    s=phase_summaries[top,cid,phase];a=predictions[top,cid,phase]
                    require([s['initial_state_q'],s['initial_state_a_s']] == previous,'Exact full-history state handoff')
                    require(s['initial_command_voltage_v'] == 0 and a['command_voltage_v'][0] == 0,'Restart from original zero command')
                    require(s['precision'] == request['precision'],'Full declared precision')
                    previous=[s['final_state_q'],s['final_state_a_s']]
        features=audit_features(contract,js,predictions,observations)
        score.update(n_gates_passed=sum(x['passed'] for x in features.values()),features=features,
                     exact_segment_handoffs=True,n_record_features=514,alternate_roots_saved=184)
        name=request['precision']['name'];full_reviews[name]=score;full_arrays[name]=predictions
    if set(full_arrays) == {'nominal','refined'}:
        for key in observations:
            difference=precision_difference(full_arrays['nominal'][key],full_arrays['refined'][key],A)
            all_differences.append(difference)
        maxima={key:max(d[key] for d in all_differences) for key in ('max_logI_decade','max_abs_u','max_abs_a_over_A')}
        maxima['passed']=all(d['passed'] for d in all_differences)
        for key,saved_key in [('max_logI_decade','max_logI_decade'),('max_abs_u','max_abs_u'),('max_abs_a_over_A','max_abs_x')]:
            close(maxima[key],report['final_comparison'][saved_key],'Full precision maximum')
        require(maxima['passed'] == report['final_comparison']['passed'],'Full precision decision')
    else:maxima=None
    comparison=None;full_history_entry_effect=[]
    if 'refined' in full_reviews and 'cost' in full_reviews['refined']:
        new=full_reviews['refined'];old=old_json['score']
        comparison=dict(old_cost=old['verified_cost'],new_cost=new['cost'],cost_fraction_reduction=1-new['cost']/old['verified_cost'],
                        old_raw_RMSE=old['plain_allpoint_RMSE_decade'],new_raw_RMSE=new['raw_RMSE_decade'],
                        raw_RMSE_fraction_reduction=1-new['raw_RMSE_decade']/old['plain_allpoint_RMSE_decade'],
                        old_gates=old_json['features']['n_passed'],new_gates=new['n_gates_passed'],
                        six_entry_coordinates_estimated=True,prefix_points_are_calibration_data=True)
        for segment,top,ids in SEGMENTS:
            for cid in ids:
                phase_rows={}
                for phase in ('set','reset'):
                    key=top,cid,phase;now=full_arrays['refined'][key];previous=phase_arrays(old_npz,key)
                    mask=observations[key]['original_scoring_mask']
                    di=np.log10(abs(now['current_a'][mask]))-np.log10(abs(previous['current_a'][mask]))
                    phase_rows[phase]=dict(max_abs_logI_change_decade=float(np.max(abs(di))),
                        RMS_logI_change_decade=float(np.sqrt(np.mean(di**2))),
                        final_u_change=float(now['state_u'][-1]-previous['state_u'][-1]),
                        final_a_over_A_change=float((now['state_a_s'][-1]-previous['state_a_s'][-1])/A))
                full_history_entry_effect.append(dict(segment=segment,record=cid,is_first_record=cid==ids[0],
                    is_last_record=cid==ids[-1],**phase_rows))
    result=dict(status='completed_independent_saved_entry_review',authoritative_terminal_session=vars(args),
        scope='saved-result arithmetic only; no model, solver, roots, optimization, or holdout decoding',
        n_frozen_sources=len(freeze),all_frozen_sources_unchanged=True,fixed_material_parameters=params,
        source_integrity_scope=dict(runtime_before_after_hash_maps_equal=True,
            reviewed_bytes_match_frozen_hashes=True,validated_cache_manifest=str(files.cache_manifest),
            original_paths_served_by_hash_checked_cache=sorted(str(p) for p in files.cache_used),
            cached_original_paths_not_independently_reread_at_review_time=True),
        files_with_first_read_mtime_change_then_two_identical_stable_rereads=sorted(READ_METADATA_STABILIZATIONS),
        original_entries=report['original_entries'],selected_entries=report.get('selected_entries'),
        n_estimated_entry_coordinates=6,n_training_arrays_decoded=len(decoded),
        n_prefix_call_records=len(calls),n_actual_prefix_evaluations=report['n_actual_prefix_evaluations'],
        prefix_calls_by_role=dict(Counter(c['role'] for c in calls)),n_starts=len(starts),start_reviews=start_reviews,
        segment_reviews=segment_reviews,baseline_regressions_independently_recomputed=True,
        full_reviews=full_reviews,precision_recomputed=maxima,vs_old_entry_control=comparison,
        saved_new_vs_old_entry_full_history_changes=full_history_entry_effect,
        new_vs_old_full_history_tolerances_differ_comparison_is_not_a_precision_test=True,
        prefix_fit_numerical_wall_s=report['prefix_fit_numerical_wall_s'],total_numerical_wall_s=report['total_numerical_wall_s'],
        execution_selection_and_saved_numeric_review_passed=True,
        final_numerically_verified=bool(maxima and maxima['passed']),
        scientific_accuracy_contract_passed=bool('refined' in full_reviews and full_reviews['refined'].get('n_gates_passed')==20),
        entry_identifiability_established=False,microscopic_mechanism_identified=False,
        newly_nonzero_a_entries_do_not_inherit_old_a0_zero_bounds=True,
        prefix_calibration_does_not_establish_global_mechanism=True,
        saved_artifact_hashes={str(path):sha(blob) for path,blob in files.blobs.items() if path.is_relative_to(run)})
    output=HERE/'SAVED_ENTRY_REVIEW.json';output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('n_prefix_call_records','n_actual_prefix_evaluations','n_starts',
          'precision_recomputed','vs_old_entry_control','final_numerically_verified','scientific_accuracy_contract_passed')},indent=2),flush=True)


if __name__ == '__main__':
    main()
