"""Predeclared six-point background inverse/interval diagnostic, not a fit."""
from pathlib import Path
import hashlib
import json
import math
import signal
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from vulcan2d.shared_spectral import SpectralShape


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval_beta(log_lower, log_upper, doses):
    """Exact interval algebra for an unknown initial log-Gamma and common beta.

    Every point obeys L_i <= log_G0 - beta*D_i <= U_i. Initial conditions
    are relaxed to these measured intervals, not estimated for prediction.
    """
    lower, upper = 0.0, math.inf
    active_lower = active_upper = None
    for i in range(len(doses)):
        for j in range(i):
            delta = float(doses[i]-doses[j])
            assert delta > 0
            lo = float((log_lower[j]-log_upper[i])/delta)
            hi = float((log_upper[j]-log_lower[i])/delta)
            if lo > lower:
                lower, active_lower = lo, [j, i]
            if hi < upper:
                upper, active_upper = hi, [j, i]
    return dict(lower_beta_per_v2=lower, upper_beta_per_v2=upper,
                feasible=lower <= upper,
                active_lower_point_pair=active_lower, active_upper_point_pair=active_upper)


def main():
    plan_path = HERE/'soft_return_diagnostic.plan.json'
    plan = json.loads(plan_path.read_text())
    ids, targets, ron = plan['training_ids'], plan['return_voltage_magnitudes_v_in_path_order'], plan['Ron_ohm']
    previous = json.loads((ROOT/'analysis/mechanism_selection_2026-11-03/field_rupture_trial.json').read_text())
    table_path, native_path = Path(plan['table_cache']), Path(plan['native_cache'])
    shape_path = ROOT/plan['shape_source']
    assert sha(table_path) == previous['source_sha256']['analysis/spectral_dynamic_2026-11-03/transport_g010_v0025.npz']
    assert sha(shape_path) == previous['source_sha256'][plan['shape_source']]
    shape = json.loads(shape_path.read_text())['parameters']
    with np.load(table_path, allow_pickle=False) as z:
        lg, lv, li = [z[key].copy() for key in ('log_gamma','log_voltage','log_current')]
        metadata = json.loads(str(z['metadata'].item()))
    original_ron = shape['ron_ohm']
    assert metadata['shape']['ron_ohm'] == original_ron
    observed, signed, native_indices, voltages, keys = [], [], [], [], []
    selected_hash = hashlib.sha256()
    with np.load(native_path, allow_pickle=False) as z:
        for cid in ids:
            prefix = f'1r_cycle_{cid}_reset'
            vv, ii = z[prefix+'_voltage'], z[prefix+'_current']
            for suffix, value in (('voltage',vv),('current',ii)):
                key = prefix+'_'+suffix;keys.append(key)
                selected_hash.update(key.encode());selected_hash.update(value.tobytes())
            ap = int(np.argmax(abs(vv))); eligible = np.arange(len(vv)) > ap
            js = [int(np.argmin(np.where(eligible,abs(abs(vv)-t),np.inf))) for t in targets]
            assert np.all(np.diff(js)>0)
            assert np.allclose(abs(vv[js]),targets,rtol=0,atol=3e-7)
            assert np.all(np.isfinite(ii[js])) and np.all(abs(ii[js])>=1e-12)
            observed.append(abs(ii[js]));signed.append(ii[js]);native_indices.append(js);voltages.append(abs(vv[js]))
    observed, voltages = np.array(observed), np.array(voltages)
    assert np.all(voltages == voltages[0])
    v = voltages[0]
    # Validate the original training RESET arrays against the completed trial
    # without accessing any held-out keys from the native archive.
    with np.load(ROOT/'analysis/mechanism_selection_2026-11-03/field_rupture_trial.npz') as z:
        assert z['record_ids'].tolist() == ids
        for i, js in enumerate(native_indices):
            np.testing.assert_array_equal(observed[i],z['observed_reset_abs_current_a'][i,js])
    constraints = json.loads((ROOT/'analysis/constrained_shared_2026-11-03/feature_constraints.json').read_text())
    gate = constraints['features']['1r.reset.return.I_0p2V']['proposed_empirical_acceptance_interval']
    manifest = dict(plan_sha256=sha(plan_path),script_sha256=sha(__file__),table_sha256=sha(table_path),
                    shape_sha256=sha(shape_path),selected_native_arrays_sha256=selected_hash.hexdigest(),loaded_array_keys=keys)
    (HERE/'soft_return_diagnostic.inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result = dict(status='running',plan=plan,inputs=manifest,table_metadata=metadata,
                  scope=['No fitted kinetics or new independent predictions are produced.',
                         'Pure-soft Gamma_eff is an equality only if the nonnegative parallel path contributes zero; otherwise it bounds Gamma from above.',
                         'No measured background-fraction lower bound, current uncertainty, scan clock or pause duration is supplied.',
                         'All common constants and spectral shape are preselected; no held-out arrays are read.'])
    destination = HERE/'soft_return_diagnostic.json'
    destination.write_text(json.dumps(result,indent=2)+'\n')
    def timeout(*_):
        raise TimeoutError('60-second predeclared computation budget exhausted')
    start=time.monotonic();signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,60.)
    try:
        # Every target uses its native voltage; no resampling of observations.
        columns=[]
        for voltage in v:
            x=math.log(voltage);j=int(np.searchsorted(lv,x)-1)
            assert 0<=j<len(lv)-1
            w=(x-lv[j])/(lv[j+1]-lv[j])
            column=(1-w)*li[1,:,j]+w*li[1,:,j+1]+math.log(original_ron/ron)
            assert np.all(np.diff(column)>0)
            columns.append(column)
        def inverse(current,point):
            x=math.log(float(current));column=columns[point]
            if not column[0] <= x <= column[-1]:
                raise ValueError('Required inverse outside existing table support; no extension permitted')
            return float(np.interp(x,column,lg))
        def forward(log_gamma,point):
            assert lg[0] <= log_gamma <= lg[-1]
            return math.exp(float(np.interp(log_gamma,lg,columns[point])))
        fractions=plan['background_fractions']
        inferred={str(q):np.array([[inverse(q*current,j) for j,current in enumerate(row)] for row in observed]) for q in fractions}
        eff=inferred['1.0'];doses=.5*(v[0]**2-v**2)
        rates=(eff[:,[0]]-eff[:,1:])/doses[None,1:]
        aggregate={};rows=[]
        for i,cid in enumerate(ids):
            static_g=float(np.min(eff[i]));envelope=np.minimum.accumulate(eff[i])
            static_fraction=np.array([forward(static_g,j)/observed[i,j] for j in range(len(v))])
            monotone_fraction=np.array([forward(envelope[j],j)/observed[i,j] for j in range(len(v))])
            row=dict(record=cid,native_indices=native_indices[i],native_voltage_v=v.tolist(),
                     signed_observed_current_a=np.array(signed[i]).tolist(),observed_abs_current_a=observed[i].tolist(),
                     Gamma_eff_ev=np.exp(eff[i]).tolist(),log10_Gamma_eff_ev=(eff[i]/math.log(10)).tolist(),
                     consecutive_return_log10_Gamma_changes=(np.diff(eff[i])/math.log(10)).tolist(),
                     n_point_estimate_increases_along_return=int(np.sum(np.diff(eff[i])>0)),
                     pure_soft_Gamma_2p5_over_0p2=float(math.exp(eff[i,0]-eff[i,-1])),
                     pure_soft_apex_to_point_beta_per_v2=rates[i].tolist(),
                     greatest_static_Gamma_ev=math.exp(static_g),
                     greatest_uniform_static_background_fraction=float(min(static_fraction)),
                     greatest_uniform_nonincreasing_background_fraction=float(min(monotone_fraction)),
                     no_fraction_lower_bound='0 < Gamma_i <= Gamma_eff_i; a sufficiently small static Gamma always satisfies these six positivity bounds.',
                     conditional_Gamma_intervals={},common_field_beta_intervals={})
            for q in fractions:
                row['conditional_Gamma_intervals'][str(q)]=np.stack([np.exp(inferred[str(q)][i]),np.exp(eff[i])],axis=-1).tolist()
                row['common_field_beta_intervals'][str(q)]=interval_beta(inferred[str(q)][i],eff[i],doses)
            rows.append(row)
        for q in fractions:
            intervals=[row['common_field_beta_intervals'][str(q)] for row in rows]
            low=max(x['lower_beta_per_v2'] for x in intervals);high=min(x['upper_beta_per_v2'] for x in intervals)
            aggregate[str(q)]=dict(common_beta_interval=[low,high],common_beta_feasible=low<=high,
                n_individually_feasible_records=sum(x['feasible'] for x in intervals),
                active_lower_record=ids[int(np.argmax([x['lower_beta_per_v2'] for x in intervals]))],
                active_upper_record=ids[int(np.argmin([x['upper_beta_per_v2'] for x in intervals]))],
                n_records_with_any_nonincreasing_Gamma_feasible=sum(row['greatest_uniform_nonincreasing_background_fraction']>=q-1e-12 for row in rows),
                scope='Necessary-condition relaxation with measured apex intervals; not a fit or predictive initial state. A failed interval rejects only this assumed fraction band and fixed spectral shape.')
        common_static_logg=float(np.min(eff))
        common_fractions=np.array([[forward(common_static_logg,j)/observed[i,j] for j in range(len(v))] for i in range(len(ids))])
        gate_logs=[inverse(x,len(v)-1) for x in gate]
        gamma_old=10**previous['models']['prior_SET_conditioned']['full_training_fit']['parameters'][2]
        old_current=forward(math.log(gamma_old),len(v)-1)
        # A batched higher-order direct spectral evaluation measures table
        # interpolation/quadrature mismatch at every reported inverse point.
        direct_shape=SpectralShape(median_energy_ev=shape['median_energy_ev'],sigma_log_energy=shape['sigma_log_energy'],
                                  eta=shape['eta'],ron_ohm=ron,temperature_k=shape['temperature_k'])
        all_logg=np.concatenate([inferred[str(q)].ravel() for q in fractions]+[np.array(gate_logs)])
        all_v=np.concatenate([np.tile(-v,len(ids)) for q in fractions]+[np.array([-v[-1],-v[-1]])])
        intended=np.concatenate([(q*observed).ravel() for q in fractions]+[np.array(gate)])
        direct=abs(direct_shape.current_at_log_gamma(all_v,all_logg,order=64,chunk_size=64))
        direct_error=np.log10(direct/intended)
        result.update(status='completed_fixed_spectrum_return_interval_diagnostic',records=rows,
                      return_path_doses_v2=doses.tolist(),hypothetical_fraction_beta_screens=aggregate,
                      collective_static=dict(greatest_static_Gamma_satisfying_all_84_upper_bounds_ev=math.exp(common_static_logg),
                          greatest_uniform_background_fraction=float(np.min(common_fractions)),
                          pure_soft_Gamma_median_by_voltage_ev=np.median(np.exp(eff),axis=0).tolist(),
                          static_Gamma_from_aggregate_tail_gate_ev=np.exp(gate_logs).tolist(),
                          aggregate_tail_gate_current_a=gate,
                          pure_soft_tail_gate_intersects_all84_positivity_interval=common_static_logg>=gate_logs[0],
                          note='The tail gate is an across-record median acceptance interval, not a per-record confidence interval. Its lower Gamma endpoint applies only if soft current supplies that tail, not merely a>=0.'),
                      previous_frozen_background=dict(Gamma_ev=gamma_old,tail_current_a=old_current,
                          factor_above_original_tail_upper=old_current/gate[1],
                          gamma_reduction_factor_to_tail_upper=gamma_old/math.exp(gate_logs[1]),
                          note='This alone permits lowering a static Gamma; it does not establish the need for a new dynamical state.'),
                      numerical_check=dict(n_table_inversions=len(all_logg),direct_spectral_order=64,
                          max_abs_current_error_decade=float(np.max(abs(direct_error))),
                          median_abs_current_error_decade=float(np.median(abs(direct_error))),
                          per_fraction_max_abs_current_error_decade={str(q):float(np.max(abs(direct_error[k*observed.size:(k+1)*observed.size]))) for k,q in enumerate(fractions)},
                          scope='Direct evaluation at table inverse points checks numerical error, not spectral adequacy or experimental uncertainty. Fractions/interval screens use the fixed table and are conditional, not confidence tests.'),
                      next_kinetic_budget=dict(law='d ln Gamma/ds=-beta*abs(Vh), beta>=0; s in path volts, no seconds inferred',
                          return_solution='ln Gamma(v)=ln Gamma(M)-beta*(M^2-v^2)/2',
                          pure_soft_required_budget='beta*D(v)=ln[Gamma_eff(M)/Gamma_eff(v)] only under zero path contribution and exact fixed spectrum.',
                          interval_budget='L_i <= ln Gamma(M)-beta*D_i <= U_i. For every j<i: (L_j-U_i)/(D_i-D_j) <= beta <= (U_j-L_i)/(D_i-D_j). Intersect all pairs, records and beta>=0.',
                          initial_state='Gamma(M) must ultimately follow the same material dynamics from earlier history. Measured apex intervals are used only to relax the necessary screen; no tail value initializes a prediction.',
                          nonnegative_path_only='With no positive lower bound on background share, the six observations yield no nontrivial lower Gamma or decay-rate bound. Static sufficiently small Gamma remains allowed.',
                          interpretation='Failure of beta intervals can come from fixed spectral shape, assumed background share, state dynamics, or unknown protocol. It does not uniquely select a microscopic mechanism.'))
    except Exception as exc:
        result.update(status='partial_or_failed_diagnostic_preserved',error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.);result['compute_elapsed_s']=time.monotonic()-start
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','compute_elapsed_s','hypothetical_fraction_beta_screens','collective_static','previous_frozen_background','numerical_check') if k in result},indent=2))


if __name__=='__main__':
    main()
