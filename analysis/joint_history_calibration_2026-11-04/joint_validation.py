"""Prepared joint-phase audits; run no model calculation on import.

Independent final readout uses local hBN voltage as the root variable and
48 explicit bisections over the original finite signed tables. It does not
use base.transport, the forward root, a state inversion or an ODE.
"""
from __future__ import annotations

import math
import sys
sys.dont_write_bytecode = True
import numpy as np

LIMITS = {'KCL_a': 1e-13, 'KVL_v': 1e-12, 'join_u': 1e-12,
          'independent_log_current_decade': 5e-7, 'independent_local_voltage_v': 1e-9,
          'precision_log_current_decade': 1e-4, 'precision_state_u': 1e-3}


def _close(a, b, *, atol=1e-12, rtol=1e-12):
    return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)


def _domain(kernel, states):
    u = np.asarray(states, dtype=float)
    finite = bool(np.all(np.isfinite(u)))
    if not finite or np.any((u <= 0.) | (u > 80.)):
        return False, None
    gamma = kernel.gc / np.expm1(u)
    logs = np.log(gamma)
    return bool(np.all(np.isfinite(gamma)) and np.all(logs >= kernel.lg[0])
                and np.all(logs <= kernel.lg[-1])), gamma


def _ports(command, current, local, transistor, kcl, kvl):
    a, i, v, t, c, k = [np.asarray(x, dtype=float) for x in (command,current,local,transistor,kcl,kvl)]
    finite = all(np.all(np.isfinite(x)) for x in (a,i,v,t,c,k))
    if not finite:
        return False
    nonzero, zero = a != 0., a == 0.
    signs = (np.all(i[nonzero]*a[nonzero] > 0.) and np.all(v[nonzero]*a[nonzero] > 0.)
             and np.all(t[nonzero]*a[nonzero] >= 0.) and np.all(i[zero] == 0.)
             and np.all(v[zero] == 0.) and np.all(t[zero] == 0.))
    return bool(signs and np.all(i*v >= 0.) and np.all(i*t >= 0.)
        and np.all(abs(c) <= LIMITS['KCL_a']) and np.all(abs(k) <= LIMITS['KVL_v'])
        and np.all(abs(a-v-t) <= LIMITS['KVL_v']))


def _independent_local_root(kernel, command, u, check_budget):
    """Independent finite-domain root; current interpolation uses table corners."""
    if check_budget is not None:
        check_budget()
    valid, gamma = _domain(kernel, [u])
    if not valid or not math.isfinite(command):
        raise ValueError('Independent readout state/command outside finite domain')
    sign = 1 if command >= 0. else -1
    tx, ti = (kernel.base.tx,kernel.base.ti) if sign > 0 else (kernel.negative_tx,kernel.negative_ti)
    if not (tx[0] == 0. and ti[0] == 0. and np.all(np.diff(tx) > 0.)
            and np.all(np.diff(ti) >= 0.) and np.all(np.isfinite(tx)) and np.all(np.isfinite(ti))):
        raise ValueError('Independent transistor table is not finite passive monotone from zero')
    if command == 0.:
        return {'current_a':0.,'local_voltage_v':0.,'transistor_voltage_v':0.,
                'KCL_residual_a':0.,'KVL_residual_v':0.,'local_bracket_width_v':0.,'bisections':0}
    magnitude = abs(command)
    lg, lv = kernel.lg, kernel.lv
    if magnitude < kernel.vmin or math.log(magnitude) > lv[-1]:
        raise ValueError('Independent native command outside original voltage table')
    spectrum = kernel.base.li if sign > 0 else kernel.li_negative
    log_gamma = math.log(float(gamma[0]))
    gi = min(int(np.searchsorted(lg,log_gamma,side='right'))-1,len(lg)-2)
    if gi < 0:
        raise ValueError('Independent Gamma below original table')
    tg = (log_gamma-lg[gi])/(lg[gi+1]-lg[gi])

    def material(local):
        if not math.isfinite(local) or local < kernel.vmin or math.log(local) > lv[-1]:
            raise ValueError('Independent local-voltage trial outside original table')
        log_local = math.log(local)
        if log_local < lv[0]:
            raise ValueError('Independent local voltage below original log-grid resolution')
        vi = min(int(np.searchsorted(lv,log_local,side='right'))-1,len(lv)-2)
        if vi < 0:
            raise ValueError('Independent material trial below table')
        tv = (log_local-lv[vi])/(lv[vi+1]-lv[vi])
        log_current = ((1.-tg)*(1.-tv)*spectrum[gi,vi]
            +(1.-tg)*tv*spectrum[gi,vi+1]
            +tg*(1.-tv)*spectrum[gi+1,vi]+tg*tv*spectrum[gi+1,vi+1])
        return math.exp(float(log_current)+kernel.base.log_scale)

    def residual(local):
        if check_budget is not None:
            check_budget()
        transistor = magnitude-local
        if not 0. <= transistor <= tx[-1]:
            raise ValueError('Independent transistor trial outside actual lookup')
        return material(local)-float(np.interp(transistor,tx,ti))

    # Physical intersection of both tables, not extrapolation or state clipping.
    low, high = max(kernel.vmin,magnitude-float(tx[-1])), magnitude
    if magnitude-low > tx[-1]:
        low = math.nextafter(low,math.inf)
    if not kernel.vmin <= low <= high or magnitude-low > tx[-1]:
        raise ValueError('No representable independent in-table root bracket')
    flo, fhi = residual(low), residual(high)
    if flo > 0. or fhi < 0.:
        raise ValueError('Independent root not bracketed inside both original tables')
    for _ in range(48):
        middle = .5*(low+high)
        if residual(middle) >= 0.:
            high = middle
        else:
            low = middle
    local = .5*(low+high);transistor = magnitude-local
    current = material(local)
    transistor_current = float(np.interp(transistor,tx,ti))
    return {'current_a':sign*current,'local_voltage_v':sign*local,
            'transistor_voltage_v':sign*transistor,
            'KCL_residual_a':sign*(current-transistor_current),
            'KVL_residual_v':command-sign*local-sign*transistor,
            'local_bracket_width_v':high-low,'bisections':48}


def phase_integrity(kernel, result, data, initial_u, b, independent=False, *, check_budget=None):
    """Return an audit dict; add independent arrays/audits to the supplied result.

    Search without rhs_trace may pass native-only integrity, explicitly marked
    incomplete. Final independent=True requires complete saved RHS records.
    Timeout is re-raised after attaching the partial audit/independent arrays.
    Other audit failures return passed=False with their preserved explanation.
    """
    checks=[]
    audit={'status':'running','passed':False,'complete_integrity_passed':False,
           'scope':'pending','independent_requested':bool(independent),'checks':checks,
           'limits':dict(LIMITS),'physical_rejection':False}
    a,s=result['arrays'],result['summary']
    independent_report=None
    def check(name, condition, **detail):
        checks.append({'check':name,'passed':bool(condition),**detail})
    def budget():
        if check_budget is not None:
            check_budget()
    try:
        budget()
        v=np.asarray(data['command_voltage_v'],dtype=float)
        observed=np.asarray(data['observed_current_a'],dtype=float)
        mask=np.asarray(data['scoring_mask'])
        if mask.dtype != np.bool_ or v.ndim != 1 or observed.shape != v.shape or mask.shape != v.shape:
            raise ValueError('Original native data or boolean score mask shape invalid')
        phase=data.get('phase');expected=(503,500) if phase=='set' else (173,170) if phase=='reset' else None
        check('completed_phase',s.get('status')=='completed')
        check('original_native_arrays',np.array_equal(a['command_voltage_v'],v)
              and np.array_equal(a['observed_current_a'],observed)
              and np.array_equal(a['original_scoring_mask'],mask))
        check('original_declared_counts',s['n_native']==len(v) and s['n_scored']==int(mask.sum())
              and (expected is None or (len(v),int(mask.sum()))==expected))
        keys=('current_a','local_voltage_v','transistor_voltage_v','KCL_residual_a','KVL_residual_v','state_u','state_q','path_voltage_s')
        if any(np.asarray(a[k]).shape!=v.shape for k in keys):
            raise ValueError('Native predicted arrays have inconsistent shapes')
        finite=all(np.all(np.isfinite(a[k])) for k in keys)
        check('native_finiteness',finite)
        if not finite:
            raise ValueError('Native predicted arrays contain nonfinite entries')
        domain,gamma=_domain(kernel,a['state_u'])
        check('native_state_Gamma_domain',domain)
        check('q_equals_exp_minus_u',np.allclose(a['state_q'],np.exp(-a['state_u']),rtol=1e-13,atol=0.)
              and np.all((a['state_q']>0.)&(a['state_q']<1.)))
        check('native_signed_passive_KCL_KVL',_ports(v,a['current_a'],a['local_voltage_v'],
              a['transistor_voltage_v'],a['KCL_residual_a'],a['KVL_residual_v']))
        if gamma is not None:
            audit.update(gamma_min_ev=float(min(gamma)),gamma_max_ev=float(max(gamma)))
        clock=np.cumsum(abs(np.diff(np.r_[data['initial_command_voltage_v'],v])))
        check('actual_voltage_distance_clock',np.allclose(a['path_voltage_s'],clock,rtol=0.,atol=1e-12))
        duplicate=np.flatnonzero(np.diff(v)==0.)
        duplicate_error=float(max(abs(a['state_u'][duplicate+1]-a['state_u'][duplicate]),default=0.))
        check('duplicate_endpoint_no_state_reset',duplicate_error<=LIMITS['join_u'],maximum_error=duplicate_error)
        check('explicit_incoming_state_and_no_anchor',s['initial_u']==initial_u
              and s['initial_command_voltage_v']==data['initial_command_voltage_v']
              and s.get('no_conditioned_anchor') is True and s.get('no_hidden_state_reset') is True
              and s.get('no_state_clipping') is True)
        if v[0]==data['initial_command_voltage_v']:
            check('native_incoming_state',abs(float(a['state_u'][0])-initial_u)<=LIMITS['join_u'])
        check('native_terminal_state',abs(float(a['state_u'][-1])-s['terminal_u'])<=LIMITS['join_u']
              and s['terminal_u']==s['final_u'] and s['final_command_voltage_v']==v[-1])
        joined_errors=[];previous_state=float(initial_u);previous_coordinate=0.
        for segment in s['segments']:
            budget();index=segment['index']
            xs=np.asarray(a[f'segment_{index}_coordinate']);us=np.asarray(a[f'segment_{index}_state_u'])
            qs=np.asarray(a[f'segment_{index}_state_q'])
            valid_segment,unused=_domain(kernel,us)
            joined_errors.append(abs(previous_state-segment['initial_u']))
            segment_ok=(segment['status']=='completed' and len(xs)>=2 and xs.shape==us.shape==qs.shape
                and np.all(np.isfinite(xs)) and np.all(np.diff(xs)>0.) and valid_segment
                and np.allclose(qs,np.exp(-us),rtol=1e-13,atol=0.)
                and abs(xs[0]-previous_coordinate)<=1e-12
                and abs(xs[0]-segment['path_start_v'])<=1e-12
                and abs(xs[-1]-segment['path_stop_v'])<=1e-12
                and abs(us[0]-segment['initial_u'])<=LIMITS['join_u']
                and abs(us[-1]-segment['final_u'])<=LIMITS['join_u']
                and abs((xs[-1]-xs[0])-abs(segment['command_stop_v']-segment['command_start_v']))<=1e-12)
            check(f'accepted_segment_{index}',segment_ok)
            previous_state=segment['final_u'];previous_coordinate=segment['path_stop_v']
        check('actual_segment_joins',max(joined_errors,default=0.)<=LIMITS['join_u']
              and abs(previous_state-s['terminal_u'])<=LIMITS['join_u']
              and abs(previous_coordinate-clock[-1])<=1e-12
              and s['max_segment_state_join_error']<=LIMITS['join_u'])
        error=np.log10(abs(a['current_a'][mask])/abs(observed[mask]))
        check('original_mask_residual',np.all(np.isfinite(error)) and np.allclose(result['residual'],error,rtol=1e-12,atol=1e-12)
              and np.allclose(a['log10_current_residual'][mask],error,rtol=1e-12,atol=1e-12))
        check('unchanged_shared_parameters',s['parameters']=={
            'lambda_per_v':kernel.parameters.lambda_per_v,'J_positive_a':kernel.parameters.J_positive_a,
            'kappa_per_v2':kernel.parameters.kappa_per_v2} and s['b']==b)
        omission=s['smallfield_omission'];budget_per_segment=float(data['nearzero_segment_budget'])
        check('explicit_nearzero_budget',0.<budget_per_segment<=1e-10
              and s.get('nearzero_segment_budget')==budget_per_segment)
        physical={}
        for segment in s['segments']:
            va,vb=segment['command_start_v'],segment['command_stop_v']
            if segment['status']=='completed' and min(abs(va),abs(vb))==0. and max(abs(va),abs(vb))>=kernel.vmin:
                physical[segment['index']]=1 if (va if va!=0. else vb)>0. else -1
        omission_ok=True
        for row in omission['evaluations']:
            index=row['segment'];segment=s['segments'][index]
            expected_sign=1 if row['command_v']>0. else -1
            valid_omitted_state,unused=_domain(kernel,[row['state_u']])
            omission_ok=omission_ok and (0.<abs(row['command_v'])<kernel.vmin and expected_sign==row['sign']==segment['sign'] and valid_omitted_state)
            physical[index]=expected_sign
        p=kernel.parameters
        signs={1 if value>0. else -1 for value in np.r_[data['initial_command_voltage_v'],v] if value!=0.}
        bounds={sign:.5*p.kappa_per_v2*kernel.vmin**2*(math.exp((1-sign)*b+p.lambda_per_v*kernel.vmin)
                +math.pi*kernel.gc/(kernel.ron*p.J_positive_a)*math.exp((sign-1)*b)) for sign in signs}
        bound_ok=(set(s['smallfield_bounds_by_sign'])=={str(sign) for sign in signs}
                  and all(math.isfinite(value) and value<budget_per_segment
                          and _close(value,s['smallfield_bounds_by_sign'][str(sign)],atol=1e-30) for sign,value in bounds.items()))
        expected_sum=sum(bounds[sign] for sign in physical.values())
        check('nearzero_physical_segments_and_bounds',omission_ok and bound_ok
              and sorted(physical)==omission['physical_microsegment_indices']
              and _close(expected_sum,omission['local_bound_sum'],atol=1e-30)
              and omission['exact_inactive_claim'] is False,
              physical_microsegments=len(physical),local_integral_bound_sum=expected_sum,
              full_trajectory_certificate=False)
        rhs_present='rhs_trace' in a
        audit['rhs_audit_scope']='complete_saved_RHS_algebra_and_ports' if rhs_present else 'not_collected_native_only_search_audit'
        if rhs_present:
            rows=np.asarray(a['rhs_trace'])
            if rows.ndim!=2 or rows.shape[1]!=7:
                raise ValueError('Saved RHS trace must have seven declared columns')
            check('RHS_trace_count',len(rows)==s['n_rhs_calls'])
            state_ok,unused=_domain(kernel,rows[:,1])
            complete=np.all(np.isfinite(rows[:,[2,3,4,6]]),axis=1)
            missing=rows[~complete]
            missing_ok=(len(missing)==omission['n_RHS_omissions']==len(omission['evaluations'])
                and np.all(np.isnan(missing[:,[2,3,4,6]])) and np.all(np.isfinite(missing[:,[0,1,5]]))
                and np.all((abs(missing[:,0])>0.)&(abs(missing[:,0])<kernel.vmin)) and np.all(missing[:,5]==0.))
            valid=rows[complete]
            port_ok=_ports(valid[:,0],valid[:,2],valid[:,3],valid[:,4],valid[:,6],valid[:,0]-valid[:,3]-valid[:,4])
            q=np.exp(-valid[:,1]);local=abs(valid[:,3]);current=abs(valid[:,2]);sigma=np.where(valid[:,3]>=0.,1.,-1.)
            # Independent transformed q equation; no call to signed_rate().
            kq=p.kappa_per_v2*math.exp(b);j0=p.J_positive_a*math.exp(2*b)
            recovery=kq*local*np.exp(-sigma*b+p.lambda_per_v*local)*q
            capture=kq*local*(current/j0)*np.exp(sigma*b)*(1.-q)/q
            expected_rate=recovery-capture
            tolerance=5e-12*(abs(recovery)+abs(capture))+1e-12
            algebra_ok=np.all(np.isfinite(expected_rate)) and np.all(abs(valid[:,5]-expected_rate)<=tolerance)
            check('RHS_state_ports_omissions_and_algebra',state_ok and missing_ok and port_ok and algebra_ok,
                  n_complete_rows=len(valid),n_omitted_rows=len(missing),
                  max_algebra_absolute_difference=float(max(abs(valid[:,5]-expected_rate),default=0.)))
        else:
            check('final_requires_saved_RHS',not independent)
        check('reported_RHS_conservation',s['max_rhs_KCL_residual_a']<=LIMITS['KCL_a']
              and s['max_rhs_KVL_residual_v']<=LIMITS['KVL_v'])
        audit['native_and_available_RHS_passed']=all(row['passed'] for row in checks)
        if independent and audit['native_and_available_RHS_passed']:
            keys=('current_a','local_voltage_v','transistor_voltage_v','KCL_residual_a','KVL_residual_v','local_bracket_width_v')
            for key in keys:a['independent_'+key]=np.full(len(v),np.nan)
            a['independent_bisections']=np.zeros(len(v),dtype=int)
            a['independent_completed_mask']=np.zeros(len(v),dtype=bool)
            independent_report={'status':'running','passed':False,'root_variable':'local_hBN_voltage_magnitude',
                'bisections_per_nonzero_native':48,'n_native':len(v),'attempted_native_points':0,
                'completed_native_points':0,'failures':[],'includes_original_zero_points':True}
            audit['independent_readout']=independent_report
            for index,(command,state) in enumerate(zip(v,a['state_u'])):
                budget();independent_report['attempted_native_points']+=1
                try:
                    point=_independent_local_root(kernel,float(command),float(state),check_budget)
                    for key in keys:a['independent_'+key][index]=point[key]
                    a['independent_bisections'][index]=point['bisections']
                    a['independent_completed_mask'][index]=True
                    independent_report['completed_native_points']+=1
                except TimeoutError:
                    raise
                except Exception as exc:
                    independent_report['failures'].append({'native_index':index,'command_v':float(command),
                        'state_u':float(state),'error_type':type(exc).__name__,'error':str(exc)})
            if np.all(a['independent_completed_mask']):
                use=v!=0.
                logdiff=float(max(abs(np.log10(abs(a['current_a'][use]/a['independent_current_a'][use]))),default=0.))
                vdiff=float(max(abs(a['local_voltage_v']-a['independent_local_voltage_v']),default=0.))
                independent_ports=_ports(v,a['independent_current_a'],a['independent_local_voltage_v'],
                    a['independent_transistor_voltage_v'],a['independent_KCL_residual_a'],a['independent_KVL_residual_v'])
                independent_report.update(max_log_current_difference_decade=logdiff,max_local_voltage_difference_v=vdiff,
                    max_KCL_residual_a=float(max(abs(a['independent_KCL_residual_a']),default=0.)),
                    max_KVL_residual_v=float(max(abs(a['independent_KVL_residual_v']),default=0.)),
                    passed=bool(independent_ports and logdiff<=LIMITS['independent_log_current_decade']
                                and vdiff<=LIMITS['independent_local_voltage_v']))
            independent_report['status']='completed_independent_native_readout'
            check('independent_all_native_roots',independent_report['passed'])
        elif independent:
            audit['independent_readout']={'status':'not_run_prior_integrity_failed','passed':False}
        audit['passed']=all(row['passed'] for row in checks)
        audit['complete_integrity_passed']=bool(audit['passed'] and rhs_present and (not independent or independent_report is not None and independent_report['passed']))
        audit['scope']='complete_RHS_and_independent_native' if independent else 'complete_RHS_and_native' if rhs_present else 'native_only_search_not_final_validation'
        audit['status']='completed_phase_integrity'
        budget()
    except Exception as exc:
        audit.update(status='caller_budget_exhausted' if isinstance(exc,TimeoutError) else 'failed_integrity_preserved',
                     passed=False,complete_integrity_passed=False,error_type=type(exc).__name__,error=str(exc))
        if independent_report is not None and independent_report['status']=='running':
            independent_report['status']='partial_preserved'
        if isinstance(exc,TimeoutError):
            raise
    finally:
        result['summary']['phase_integrity']=audit
        if independent_report is not None:
            result['summary']['independent_native_readout']=independent_report
    return audit


def compare_phases(nominal, refined):
    """Compare existing phase arrays only; never solve/re-anchor either phase."""
    result={'passed':False,'limits':dict(LIMITS),'scope':'original_score_mask_log_current_and_all_native_states'}
    try:
        a,z=nominal['arrays'],refined['arrays']
        if nominal['summary']['status']!='completed' or refined['summary']['status']!='completed':
            raise ValueError('Both compared phases must be completed')
        for key in ('command_voltage_v','observed_current_a','original_scoring_mask','path_voltage_s'):
            if not np.array_equal(a[key],z[key]):
                raise ValueError('Precision comparison native data/clock differs: '+key)
        mask=a['original_scoring_mask']
        if mask.dtype!=np.bool_ or not np.any(mask):
            raise ValueError('A nonempty original score mask is required')
        if not (np.all(np.isfinite(a['state_u'])) and np.all(np.isfinite(z['state_u']))
                and np.all(np.isfinite(a['current_a'][mask])) and np.all(np.isfinite(z['current_a'][mask]))
                and np.all(abs(a['current_a'][mask])>0.) and np.all(abs(z['current_a'][mask])>0.)):
            raise ValueError('Nonfinite or zero compared outputs')
        logdiff=float(np.max(abs(np.log10(abs(a['current_a'][mask]/z['current_a'][mask])))))
        udiff=float(np.max(abs(a['state_u']-z['state_u'])))
        terminal=abs(nominal['summary']['terminal_u']-refined['summary']['terminal_u'])
        result.update(n_native=len(a['state_u']),n_scored=int(mask.sum()),
            max_log_current_difference_decade=logdiff,max_state_u_difference=udiff,
            terminal_u_difference=terminal,
            passed=bool(logdiff<=LIMITS['precision_log_current_decade']
                        and max(udiff,terminal)<=LIMITS['precision_state_u']))
    except Exception as exc:
        result.update(error_type=type(exc).__name__,error=str(exc))
    return result
