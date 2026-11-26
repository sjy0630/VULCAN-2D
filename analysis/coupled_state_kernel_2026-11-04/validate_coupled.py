"""One predeclared, bounded implementation check; no fitting or hidden retries.

Run only after the root reviewer freezes sources. Import is inert. This runner
does not establish scientific accuracy, mechanism identity, or joint fit success.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'coupled_forward.py'
CONTROL = ROOT / 'analysis/constrained_history_convergence_2026-11-04/final_refined_chain.npz'
CANDIDATE = ROOT / 'analysis/constrained_history_convergence_2026-11-04/requests/102_final_refined.json'
SPECTRAL = ROOT / 'vulcan2d/shared_spectral.py'
PARAMETERS = dict(lambda_per_v=.4470398752554136,
                  J_positive_a=.0037505935471625162,
                  kappa_positive_per_v2=57.67798793605963,
                  b=.769297378470313, a_max_s=.001,
                  kappa_a_per_v2=.1, rho=1.)
INITIAL_U = 9.464551672652638
METHOD = 'Radau'
PRECISIONS = (
    dict(name='nominal', rtol=1e-7, atol_u=1e-9, atol_x=1e-11, max_step=.02),
    dict(name='refined', rtol=1e-9, atol_u=1e-11, atol_x=1e-13, max_step=.01),
)
LIMITS = dict(max_log10_current_difference_decade=3e-4,
              max_absolute_u_difference=1e-3,
              max_absolute_x_difference=1e-5,
              root_current_relative=1e-7, root_current_absolute_a=1e-18,
              root_voltage_absolute_v=1e-9,
              lowfield_spectral_relative=.005, spectral_order_relative=1e-5,
              numerical_budget_s=180.)
EXPECTED = dict(histories=6, phases=14, static_regular=54,
                static_independent=54, lowfield_readouts=48)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, payload):
    Path(path).write_text(json.dumps(clean(payload), indent=2,
                                    ensure_ascii=False, allow_nan=False) + '\n')


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Cumulative numerical wall time only; callbacks check within every RHS."""
    def __init__(self, limit):
        self.limit, self.elapsed, self.started, self.callbacks = limit, 0., None, 0

    def used(self):
        return self.elapsed + (time.monotonic()-self.started if self.started is not None else 0.)

    def check(self):
        self.callbacks += 1
        if self.used() > self.limit:
            raise BudgetExceeded(f'Predeclared numerical budget {self.limit:g} s exhausted')

    @contextmanager
    def section(self):
        if self.started is not None:
            raise RuntimeError('Nested numerical budget section')
        self.started = time.monotonic()
        try:
            self.check()
            yield
            self.check()
        finally:
            self.elapsed += time.monotonic()-self.started
            self.started = None


class Checks:
    def __init__(self):
        self.rows = []

    def record(self, name, passed, **detail):
        self.rows.append(dict(name=name, passed=bool(passed), **detail))

    def bound(self, name, value, limit, **detail):
        self.record(name, math.isfinite(float(value)) and float(value) <= limit,
                    value=float(value), limit=limit, **detail)


def maxabs(value):
    values = np.asarray(value, dtype=float)
    return float(np.max(np.abs(values))) if values.size else 0.


def independent_components(readout, anchor, voltage, state):
    """Manual four-corner interpolation, never the candidate material method."""
    q, a = state
    if voltage == 0:
        return 0., 0., 0.
    sign, magnitude = (1 if voltage > 0 else -1), abs(voltage)
    gamma = readout.gamma_c_ev*q/(1.-q)
    gx, vx = math.log(gamma), math.log(max(anchor, magnitude))
    gaxis, vaxis = readout.lg, readout.lv
    if not gaxis[0] <= gx <= gaxis[-1] or not vaxis[0] <= vx <= vaxis[-1]:
        raise ValueError('Independent readout outside table')
    gi = int(np.searchsorted(gaxis, gx, side='right'))-1
    vi = int(np.searchsorted(vaxis, vx, side='right'))-1
    if gi == len(gaxis)-1:
        gi -= 1
    if vi == len(vaxis)-1:
        vi -= 1
    weights_g = ((gaxis[gi+1]-gx)/(gaxis[gi+1]-gaxis[gi]),
                 (gx-gaxis[gi])/(gaxis[gi+1]-gaxis[gi]))
    weights_v = ((vaxis[vi+1]-vx)/(vaxis[vi+1]-vaxis[vi]),
                 (vx-vaxis[vi])/(vaxis[vi+1]-vaxis[vi]))
    table = readout.li_positive if sign > 0 else readout.li_negative
    log_i = math.fsum(weights_g[dg]*weights_v[dv]*float(table[gi+dg,vi+dv])
                      for dg in (0,1) for dv in (0,1)) + readout.log_scale
    soft = sign*math.exp(log_i)*(magnitude/anchor if magnitude < anchor else 1.)
    hard = a*voltage
    return soft+hard, soft, hard


def independent_transistor(voltage, x, y):
    if not 0 <= voltage <= x[-1]:
        raise ValueError('Independent transistor outside measured domain')
    if voltage == x[-1]:
        return float(y[-1])
    j = int(np.searchsorted(x, voltage, side='right'))-1
    return float(y[j]+(voltage-x[j])*(y[j+1]-y[j])/(x[j+1]-x[j]))


def bisect_root(function, low, high, budget):
    left, right = function(low), function(high)
    if not math.isfinite(left+right) or left > 0 or right < 0:
        raise ValueError(f'Independent root is not bracketed: {left}, {right}')
    if left == 0:
        return low
    if right == 0:
        return high
    for _ in range(80):
        budget.check()
        mid = low+(high-low)/2
        if function(mid) < 0:
            low = mid
        else:
            high = mid
    return low+(high-low)/2


def independent_point(material, topology, command, state, budget):
    """Solve in local Vh; production series solver solves transistor voltage."""
    readout, anchor = material.readout, material.nearzero_anchor_v
    sign, v = (1 if command >= 0 else -1), abs(command)
    cap = 1e-5 if topology == 'bare_limited' else None
    def current(magnitude):
        return abs(independent_components(readout, anchor, sign*magnitude, state)[0])
    vh, vt, vc, limited = v, 0., 0., False
    transistor_current = None
    if topology == 'series' and v != 0:
        tx, ti = ((readout.tx_positive, readout.ti_positive) if sign > 0
                  else (readout.tx_negative, readout.ti_negative))
        low, high = max(0., v-float(tx[-1])), min(v, readout.vmax_v)
        vh = bisect_root(lambda h: current(h)-independent_transistor(v-h,tx,ti),
                         low, high, budget)
        vt = v-vh
        transistor_current = sign*independent_transistor(vt,tx,ti)
    elif cap is not None and current(v) > cap:
        vh = bisect_root(lambda h: current(h)-cap, 0., v, budget)
        vc, limited = v-vh, True
    total, soft, hard = independent_components(readout, anchor, sign*vh, state)
    return dict(current_a=total, soft_current_a=soft, hard_current_a=hard,
                local_voltage_v=sign*vh, transistor_voltage_v=sign*vt,
                compliance_voltage_v=sign*vc, supply_voltage_v=sign*(vh+vt),
                compliance_limited=limited,
                KCL_residual_a=0. if transistor_current is None else total-transistor_current,
                KVL_residual_v=command-sign*(vh+vt+vc))


def point_checks(checks, label, point, reference, command, state, cap):
    scale = LIMITS['root_current_absolute_a']+LIMITS['root_current_relative']*abs(reference['current_a'])
    checks.bound(label+'/independent_current_a', abs(point['current_a']-reference['current_a']), scale)
    checks.bound(label+'/independent_voltage_v', max(abs(point[k]-reference[k]) for k in
                 ('local_voltage_v','transistor_voltage_v','compliance_voltage_v','supply_voltage_v')),
                 LIMITS['root_voltage_absolute_v'])
    checks.record(label+'/same_compliance_branch', point['compliance_limited'] == reference['compliance_limited'])
    for key in ('KCL_residual_a','KVL_residual_v'):
        checks.bound(label+'/'+key, max(abs(point[key]),abs(reference[key])),
                     scale if key.endswith('_a') else LIMITS['root_voltage_absolute_v'])
    checks.bound(label+'/total_branch_sum_a',
                 abs(point['current_a']-point['soft_current_a']-point['hard_current_a']),scale)
    checks.bound(label+'/hard_ohmic_a',abs(point['hard_current_a']-state[1]*point['local_voltage_v']),scale)
    checks.record(label+'/passive', all(point[k]*point['local_voltage_v'] >= 0 for k in
                  ('current_a','soft_current_a','hard_current_a')))
    checks.record(label+'/source_reduces_voltage',
                  abs(point['supply_voltage_v']) <= abs(command)+1e-12 and
                  point['compliance_voltage_v']*command >= -1e-18)
    if cap is not None:
        checks.bound(label+'/compliance_excess_a',max(0.,abs(point['current_a'])-cap),scale)
        if point['compliance_limited']:
            checks.record(label+'/limited_local_voltage_reduced',
                          abs(point['local_voltage_v']) < abs(command))


def run_statics(module, material, budget, checks, report):
    q0, amax, delta = math.exp(-INITIAL_U), PARAMETERS['a_max_s'], material.readout.vmin_v
    voltages = [0., .2, -.2, 1., -1., delta/2, -delta/2, 1.5*delta, -1.5*delta]
    circuits = [('bare', module.BareCircuit()),
                ('bare_limited', module.BareCircuit(compliance_a=1e-5)),
                ('series', module.SeriesCircuit())]
    for state_id, state in enumerate(((q0,0.),(.1,amax/2))):
        for voltage in voltages:
            for topology, circuit in circuits:
                label = f'static/{state_id}/{voltage:.17g}/{topology}'
                with budget.section():
                    point = circuit.solve(voltage,material,state)
                    report['counts']['static_regular'] += 1
                    budget.check()
                    reference = independent_point(material,topology,voltage,state,budget)
                    report['counts']['static_independent'] += 1
                point_checks(checks,label,point,reference,voltage,state,circuit.compliance_a)
                report['static_points'].append(dict(label=label,command_voltage_v=voltage,
                    state=state,topology=topology,regular={k:v for k,v in point.items() if not k.startswith('_')},
                    independent=reference))


def run_lowfield(material, second_anchor, shape, budget, checks, report):
    delta = material.readout.vmin_v
    for q in (math.exp(-INITIAL_U), .1):
        gamma = material.readout.gamma_c_ev*q/(1.-q)
        for voltage in (delta/2,-delta/2,delta,-delta,2*delta,-2*delta):
            row = dict(q=q,voltage_v=voltage,gamma_ev=gamma)
            for name, candidate in [('anchor_delta',material),('anchor_2delta',second_anchor)]:
                with budget.section():
                    row[name] = candidate.components(voltage,(q,0.))['soft_current_a']
                    report['counts']['lowfield_readouts'] += 1
            for order in (32,64):
                with budget.section():
                    row[f'direct_order_{order}'] = shape.current_at_log_gamma(voltage,math.log(gamma),order=order)
                    report['counts']['lowfield_readouts'] += 1
            scale = abs(row['direct_order_64'])
            if scale <= 0:
                raise ValueError('Nonzero low-field voltage has zero direct spectral current')
            label = f'lowfield/{q:.17g}/{voltage:.17g}'
            row['order_relative_difference'] = abs(row['direct_order_32']-row['direct_order_64'])/scale
            row['anchor_sensitivity_relative'] = abs(row['anchor_delta']-row['anchor_2delta'])/scale
            checks.bound(label+'/direct_order_agreement',row['order_relative_difference'],LIMITS['spectral_order_relative'])
            for name in ('anchor_delta','anchor_2delta'):
                row[name+'_relative_error'] = abs(row[name]-row['direct_order_64'])/scale
                checks.bound(label+'/'+name+'_vs_direct',row[name+'_relative_error'],LIMITS['lowfield_spectral_relative'])
            report['lowfield_points'].append(row)


def save_history(output, label, result):
    arrays = {'aggregate_residual':result['residual']}
    summaries = []
    for index, phase in enumerate(result['phases']):
        summaries.append(phase['summary'])
        arrays.update({f'phase{index}__{key}':value for key,value in phase['arrays'].items()})
        arrays[f'phase{index}__residual'] = phase['residual']
    np.savez_compressed(output/(label+'.npz'),**arrays)
    write_json(output/(label+'.json'),dict(summary=result['summary'],phases=summaries))


def history_checks(checks, label, result, material, module, expected_initial, circuits):
    amax = material.parameters.a_max_s
    joins = result['summary']['joins']
    checks.record(label+'/whole_history_initial',
                  result['summary']['initial_state_q'] == expected_initial[0] and
                  result['summary']['initial_state_a_s'] == expected_initial[1])
    previous = expected_initial
    for index, phase in enumerate(result['phases']):
        summary, arrays = phase['summary'],phase['arrays']
        prefix = f'{label}/phase{index}'
        incoming = (summary['initial_state_q'],summary['initial_state_a_s'])
        checks.record(prefix+'/exact_both_state_handoff',incoming == previous)
        previous = (summary['final_state_q'],summary['final_state_a_s'])
        checks.record(prefix+'/join_trace_matches',
                      (joins[index]['initial_state_q'],joins[index]['initial_state_a_s']) == incoming and
                      (joins[index]['final_state_q'],joins[index]['final_state_a_s']) == previous)
        q,a,u = arrays['state_q'],arrays['state_a_s'],arrays['state_u']
        checks.record(prefix+'/finite_native_domain',np.all(np.isfinite(q+a+u)) and
                      np.all((q>0)&(q<1)&(a>=0)&(a<=amax)&(u>0)&(u<=80)))
        checks.bound(prefix+'/q_coordinate',maxabs(q-np.exp(-u)),2e-15)
        checks.record(prefix+'/passive_native',all(np.all(arrays[key]*arrays['local_voltage_v']>=0)
                      for key in ('current_a','soft_current_a','hard_current_a')))
        scale = 1e-18+1e-7*abs(arrays['current_a'])
        checks.bound(prefix+'/KCL_tolerance_ratio',float(np.max(abs(arrays['KCL_residual_a'])/scale)),1.)
        checks.bound(prefix+'/KVL_v',maxabs(arrays['KVL_residual_v']),1e-9)
        checks.bound(prefix+'/total_current_tolerance_ratio',float(np.max(abs(arrays['current_a']-
                      arrays['soft_current_a']-arrays['hard_current_a'])/scale)),1.)
        checks.bound(prefix+'/hard_ohmic_tolerance_ratio',float(np.max(abs(arrays['hard_current_a']-
                      a*arrays['local_voltage_v'])/scale)),1.)
        checks.record(prefix+'/disabled_a_exact_zero',material.parameters.enable_a or np.all(a==0))
        checks.record(prefix+'/zero_command_zero_voltage',np.all(arrays['local_voltage_v'][arrays['command_voltage_v']==0]==0))
        trace = arrays['rhs_trace']
        col = {key:index for index,key in enumerate(module.RHS_COLUMNS)}
        checks.record(prefix+'/rhs_trace_domain',len(trace)>0 and np.all(np.isfinite(trace)) and
                      np.all((trace[:,col['state_q']]>0)&(trace[:,col['state_q']]<1)&
                             (trace[:,col['state_a_s']]>=0)&(trace[:,col['state_a_s']]<=amax)))
        # Independent algebra on the retained trace; no extra model evaluation.
        tq,ta,tv,ti = (trace[:,col[key]] for key in
                      ('state_q','state_a_s','local_voltage_v','current_a'))
        sigma, magnitude = np.sign(tv),abs(tv)
        p = material.parameters
        chi = (1.-sigma)*p.b
        capture = p.kappa_positive_per_v2*magnitude*abs(ti)/p.J_positive_a*np.exp(-chi)*(1.-tq)
        recovery = p.kappa_positive_per_v2*magnitude*np.exp(chi+p.lambda_per_v*magnitude)*tq*tq
        formation = p.kappa_a_per_v2*magnitude*np.exp(sigma*p.b)*tq*(amax-ta)
        loss = p.kappa_a_per_v2*magnitude*p.rho*np.exp(-sigma*p.b)*(1.-tq)*ta
        if not p.enable_a:
            formation,loss = np.zeros_like(tq),np.zeros_like(tq)
        expected_rates = dict(capture_q_per_v=capture,recovery_q_per_v=recovery,
            formation_a_s_per_v=formation,loss_a_s_per_v=loss,
            dq_per_v=capture-recovery,da_s_per_v=formation-loss)
        for key,expected in expected_rates.items():
            denominator = 1e-18+1e-10*np.maximum(abs(expected),abs(trace[:,col[key]]))
            checks.bound(prefix+'/independent_rhs/'+key,
                         float(np.max(abs(expected-trace[:,col[key]])/denominator)),1.)
        checks.record(prefix+'/zero_local_voltage_zero_rates',all(
            np.all(trace[tv==0,col[key]]==0) for key in expected_rates))
        checks.record(prefix+'/rhs_passive',all(np.all(trace[:,col[key]]*tv>=0)
                      for key in ('current_a','soft_current_a','hard_current_a')))
        cap = circuits[index].compliance_a
        if cap is not None:
            checks.bound(prefix+'/compliance_excess_ratio',float(np.max(np.maximum(0,abs(arrays['current_a'])-cap)/scale)),1.)
            limited = arrays['compliance_limited']
            checks.record(prefix+'/limited_voltage_reduced',np.all(abs(arrays['local_voltage_v'][limited])<
                          abs(arrays['command_voltage_v'][limited])))
    if material.parameters.enable_a:
        checks.record(label+'/a_nucleates_from_zero',expected_initial[1]==0 and
                      max(float(np.max(p['arrays']['state_a_s'])) for p in result['phases'])>0)
        checks.record(label+'/q_evolves',any(np.any(p['arrays']['state_q']!=expected_initial[0]) for p in result['phases']))
        if circuits[0].compliance_a is not None:
            checks.record(label+'/bare_set_compliance_exercised',np.any(result['phases'][0]['arrays']['compliance_limited']))


def compare_histories(checks, label, left, right, mask_key=None):
    for index,(pa,pb) in enumerate(zip(left['phases'],right['phases'])):
        a,b = pa['arrays'],pb['arrays']
        mask = a[mask_key] if mask_key else a['command_voltage_v']!=0
        if not np.array_equal(a['command_voltage_v'],b['command_voltage_v']):
            raise ValueError('Comparison waveforms differ')
        checks.record(f'{label}/{index}/positive_comparison_current',
                      np.all(abs(a['current_a'][mask])>0) and np.all(abs(b['current_a'][mask])>0))
        delta = np.log10(abs(a['current_a'][mask]))-np.log10(abs(b['current_a'][mask]))
        checks.bound(f'{label}/{index}/max_logI_decade',maxabs(delta),LIMITS['max_log10_current_difference_decade'],
                     n_points=int(np.sum(mask)),interpretation='absolute log10 difference, not experimental error')
        checks.bound(f'{label}/{index}/max_abs_u',maxabs(a['state_u']-b['state_u']),LIMITS['max_absolute_u_difference'])
        checks.bound(f'{label}/{index}/max_abs_x',maxabs(a['state_a_s']-b['state_a_s'])/PARAMETERS['a_max_s'],
                     LIMITS['max_absolute_x_difference'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'validation_run_v1')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    budget, checks = Budget(LIMITS['numerical_budget_s']),Checks()
    report = dict(status='preparing',purpose='bounded numerical implementation validation only',
        not_a_fit=True,no_heldout_access=True,no_new_native_enabled_a_test=True,
        no_failed_trial_retry=True,no_state_reanchoring=True,no_missing_cycle_bridge=True,
        scientific_accuracy_improvement_claimed=False,physical_mechanism_confirmed=False,
        physical_rejection=False,parameters=PARAMETERS,initial_u_control=INITIAL_U,
        enabled_initial_state=[.1,0.],method=METHOD,precisions=PRECISIONS,limits=LIMITS,
        tolerances_are_numerical_not_experimental=True,expected_counts=EXPECTED,
        counts={key:0 for key in EXPECTED},static_points=[],lowfield_points=[],histories=[],
        source_sha256_before={},source_sha256_after={})
    write_json(output/'RESULT.json',report)
    current_label = None
    try:
        module = load_module(SOURCE,'vulcan_coupled_frozen_validation')
        spectral_module = load_module(SPECTRAL,'vulcan_spectral_direct_coupled_validation')
        provenance = dict(candidate_source_id=str(CANDIDATE.relative_to(ROOT)),candidate_source_sha256=sha(CANDIDATE))
        enabled = module.load_material(module.Parameters(**PARAMETERS,enable_a=True),**provenance)
        disabled = module.load_material(module.Parameters(**PARAMETERS,enable_a=False),**provenance)
        second_anchor = module.load_material(module.Parameters(**PARAMETERS,enable_a=True),
            **provenance,nearzero_anchor_v=2*enabled.readout.vmin_v)
        shape_values = dict(enabled.readout.shape_metadata)
        shape_values['ron_ohm'] = enabled.readout.ron_ohm
        shape = spectral_module.SpectralShape(**{key:shape_values[key] for key in
                    spectral_module.SpectralShape.__dataclass_fields__ if key in shape_values})
        paths = {Path(path) for path in enabled.source_sha256}
        paths.update((SOURCE,SPECTRAL,CANDIDATE,CONTROL,Path(__file__),HERE/'VALIDATION_PLAN.md'))
        report['source_sha256_before'] = {str(path):sha(path) for path in sorted(paths)}
        report['materials'] = dict(enabled=enabled.metadata(),disabled=disabled.metadata(),
                                   second_anchor=second_anchor.metadata())
        report['same_enabled_material_object_for_bare_and_series'] = True
        control_phases, control_saved = [],[]
        with np.load(CONTROL,allow_pickle=False) as archive:
            for name,record_id,phase in [('set4',4,'set'),('reset4',4,'reset'),('set5',5,'set')]:
                arrays = {key:archive[f'{name}__{key}'].copy() for key in
                          ('command_voltage_v','observed_current_a','original_scoring_mask','current_a','state_u')}
                arrays['state_a_s'] = np.zeros_like(arrays['state_u'])
                control_saved.append(dict(arrays=arrays))
                control_phases.append(dict(command_voltage_v=arrays['command_voltage_v'],
                    observed_current_a=arrays['observed_current_a'],scoring_mask=arrays['original_scoring_mask'],
                    initial_command_voltage_v=0.,record_id=record_id,phase=phase))
        checks.record('input/original_mask_counts',
                      [int(np.sum(d['scoring_mask'])) for d in control_phases]==[500,170,500])
        wave = np.concatenate((np.linspace(0.,1.,101),np.linspace(1.,0.,101)[1:]))
        synthetic = [dict(command_voltage_v=wave,initial_command_voltage_v=0.,phase='set'),
                     dict(command_voltage_v=-wave,initial_command_voltage_v=0.,phase='reset')]
        report['synthetic_native_nodes_per_phase'] = len(wave)
        report['status'] = 'running'
        write_json(output/'RESULT.json',report)
        run_statics(module,enabled,budget,checks,report)
        run_lowfield(enabled,second_anchor,shape,budget,checks,report)
        history_results = {}
        for family in ('disabled_native','enabled_series','enabled_bare'):
            material = disabled if family=='disabled_native' else enabled
            data = control_phases if family=='disabled_native' else synthetic
            initial = (math.exp(-INITIAL_U),0.) if family=='disabled_native' else (.1,0.)
            circuits = ([module.BareCircuit(compliance_a=1e-5),module.BareCircuit()] if family=='enabled_bare'
                        else [module.SeriesCircuit() for _ in data])
            phases = [dict(data=d,circuit=c) for d,c in zip(data,circuits)]
            for values in PRECISIONS:
                current_label = family+'_'+values['name']
                report['active_history'] = current_label
                write_json(output/'RESULT.json',dict(report,checks=checks.rows))
                report['counts']['histories'] += 1
                with budget.section():
                    result = module.simulate_history(material,phases,initial,
                        module.Precision(**values),METHOD,budget.check,collect_rhs=True)
                report['counts']['phases'] += len(result['phases'])
                save_history(output,current_label,result)
                report['histories'].append(dict(label=current_label,summary=result['summary']))
                history_results[current_label] = result
                history_checks(checks,current_label,result,material,module,initial,circuits)
                if family=='disabled_native':
                    compare_histories(checks,current_label+'/saved_control',result,
                                      dict(phases=control_saved),'original_scoring_mask')
            compare_histories(checks,family+'/two_precision',history_results[family+'_nominal'],
                              history_results[family+'_refined'],
                              'original_scoring_mask' if family=='disabled_native' else None)
        checks.record('scope/exact_completed_counts',report['counts']==EXPECTED,
                      actual=report['counts'].copy(),expected=EXPECTED)
        report['status'] = 'completed'
    except Exception as exc:
        report['status'] = 'unresolved_execution_failure'
        report['failure'] = dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
                                 active_history=current_label,physical_rejection=False,retried=False)
        if hasattr(exc,'result') and current_label is not None:
            result = exc.result
            report['counts']['phases'] += len(result['phases'])
            save_history(output,current_label+'_failed_partial',result)
            report['partial_history_summary'] = result['summary']
    finally:
        for path,digest in report['source_sha256_before'].items():
            actual = sha(path) if Path(path).exists() else None
            report['source_sha256_after'][path] = actual
            checks.record('source_unchanged/'+path,actual==digest)
        report['checks'] = checks.rows
        report['n_checks'] = len(checks.rows)
        report['n_failed_checks'] = sum(not row['passed'] for row in checks.rows)
        report['numerical_wall_s'] = budget.used()
        report['budget_callbacks'] = budget.callbacks
        report['total_wall_s'] = time.monotonic()-started
        report['passed'] = report['status']=='completed' and report['n_failed_checks']==0
        report['interpretation'] = ('Finite-scope numerical implementation verified; not a full joint fit or mechanism proof'
            if report['passed'] else 'Numerical implementation/extension remains unresolved; no physical rejection follows')
        write_json(output/'RESULT.json',report)
        lines = ['# Bounded coupled-kernel implementation validation','',
                 f"Status: {report['status']}; passed: {report['passed']}.",
                 f"Checks: {report['n_checks']}; failed: {report['n_failed_checks']}.",
                 f"Numerical wall time: {report['numerical_wall_s']:.3f} / 180 s.",
                 '',report['interpretation'],'',
                 'All tolerances are numerical implementation thresholds, not experimental uncertainty.',
                 'No fit, hidden-state inversion, held-out evaluation, missing-cycle bridge, or retry was performed.','',
                 '## Failed checks','']
        lines.extend('- '+row['name'] for row in checks.rows if not row['passed'])
        if 'failure' in report:
            lines.append('- Execution: '+report['failure']['type']+': '+report['failure']['message'])
        (output/'RESULT.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps(dict(output=str(output),status=report['status'],passed=report['passed'],
                              counts=report['counts'],failed_checks=report['n_failed_checks'],
                              numerical_wall_s=report['numerical_wall_s'])))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
