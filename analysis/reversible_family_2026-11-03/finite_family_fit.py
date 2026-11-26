"""One bounded three-parameter conditional SET calibration, all trials retained."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import math
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import minimize
from forward_kernel import ForwardKernel, Parameters, Precision, KernelFailure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(path, value):
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n')


class SearchStop(RuntimeError):
    pass


def main():
    destination = HERE/'finite_family_fit.json'
    planpath = HERE/'finite_family_fit.plan.json'
    trialdir = HERE/'forward_trials'
    if destination.exists() or planpath.exists() or trialdir.exists():
        raise FileExistsError('Preserve every existing attempt; no overwrite or automatic rerun')
    initialization_path = HERE/'integral_initialization.json'
    initialization = json.loads(initialization_path.read_text())
    seeds = initialization['seeds']
    if not 1 <= len(seeds) <= 3:
        raise ValueError('Need one to three declared finite initialization seeds')
    if not all(s['ready_for_forward_initialization'] for s in seeds):
        raise ValueError('Initialization quadrature has not passed its declared check')
    kernel = ForwardKernel.load()
    search_precision = Precision('search', 2e-6, 2e-7, .04)
    precisions = [Precision('nominal', 1e-7, 1e-8, .02),
                  Precision('refined', 1e-8, 1e-9, .01)]
    old = json.loads((ROOT/'analysis/soft_drive_structure_2026-11-03/sinh_field_probe.json').read_text())
    prior = next(r for r in old['runs'] if r['precision']['name']=='refined')['metrics']['all500']['RMSE_decade']
    plan = dict(
        scope='Same three-parameter reversible family, conditional series cycle4 SET; no RESET, bare, holdout or aggregate validation.',
        motivation='Test dynamic calibration instead of assuming two measured return points are equilibria.',
        equation='uprime=kappa*v*(exp(lambda*v)-I*expm1(u)/J); Gamma=gc/expm1(u)',
        seed_source_sha256=sha(initialization_path), maximum_seed_forwards=3,
        initialization_seeds=seeds, frozen_initial_u=kernel.anchor_u,
        parameter_count=3, initial_state_fitted=False, state_reset=False,
        search_precision=asdict(search_precision), verification_precisions=[asdict(p) for p in precisions],
        score='Mean squared log10 current residual over original500 points including conditional anchor; no point removal.',
        optimizer='One Nelder-Mead run from lowest full500 seed MSE; maximum80 function evaluations/iterations, no restarts.',
        optimization_coordinates=['lambda_per_v','ln_kappa_per_v2','ln_J_positive_a'],
        domain='lambda in[0,2]; each log coefficient within ln100 of selected seed. Finite exploration, not physical bounds.',
        initial_simplex='Selected seed plus .1 in lambda (inward at upper boundary), plus .25 in each log coefficient.',
        optimizer_settings=dict(maxfev=80,maxiter=80,xatol=1e-4,fatol=1e-8,adaptive=True),
        invalid_trials='Outside declared parameter domain returns infinity, serialized null with status; no clipping or finite fake residual.',
        failure_policy='All numerical failures retain parameters, last RHS and available arrays; not physical rejections.',
        timeout_policy='Any trajectory timeout stops entire search immediately; no optimizer restart. Existing best may enter reserved final validation.',
        per_search_trajectory_s=12, search_total_s=180, stage_total_s=300,
        per_verification_trajectory_s=45,
        budget_start='After inputs/hashes and plan persistence; includes circuit checks, search and final verification.',
        circuit_check='36 predefined states/commands versus old42-bisection; select fast only if all tolerances pass, otherwise validated strict_bisection42.',
        precision_limits=dict(max_log_current_change_decade=1e-4,max_state_u_change=1e-3),
        legacy_limits=dict(max_log_current_difference_decade=5e-7,max_local_voltage_difference_v=1e-9),
        nearzero='Existing local Lorentz-area omission bound below1e-12; not a global solution error or zero-field retention prediction.',
        physical_clock='Cumulative command voltage, not measured seconds',
        prior_conditional_500point_RMSE_decade=prior,
        limitations='Fit uses same cycle and three actual fitted parameters. Lower training error is not independent validation or microscopic identification.',
        no_expansion_or_new_seed=True)
    dump(planpath, plan)
    sources = dict(kernel.source_sha256)
    sources.update({str(p):sha(p) for p in [initialization_path, planpath, Path(__file__)]})
    report = dict(status='running_bounded_family_calibration', plan=plan, source_sha256=sources,
                  trials=[], final_verifications=[], full_unified_model_achieved=False)
    trialdir.mkdir()
    dump(destination, report)
    started = time.monotonic()
    best = None
    selected_seed = None
    search_bounds = None
    trial_counter = 0

    def persist():
        report['elapsed_s'] = time.monotonic()-started
        dump(destination, report)

    def stage_check():
        if time.monotonic()-started >= plan['stage_total_s']:
            raise TimeoutError('Overall300second experiment budget exhausted')

    def search_check():
        stage_check()
        if time.monotonic()-started >= plan['search_total_s']:
            raise TimeoutError('Search180second budget exhausted; no restart')

    def evaluate(x, label, precision, *, verification=False):
        nonlocal best, trial_counter
        trial_counter += 1
        name=f'{trial_counter:03d}_{label}'
        x=np.asarray(x,float)
        row=dict(trial_id=name, coordinates=x.tolist(), phase=label, precision=asdict(precision))
        report['trials'].append(row)
        if (not np.all(np.isfinite(x)) or (not verification and search_bounds is not None
                and (np.any(x<search_bounds[0]) or np.any(x>search_bounds[1])))):
            row.update(status='outside_declared_parameter_domain', objective_MSE=None,
                       objective_for_optimizer='positive_infinity', new_ODE=False)
            persist()
            return None
        trial_started=time.monotonic()
        limit=plan['per_verification_trajectory_s'] if verification else plan['per_search_trajectory_s']
        def budget():
            stage_check() if verification else search_check()
            if time.monotonic()-trial_started >= limit:
                raise TimeoutError(f'Per-trajectory{limit}second budget exhausted; stop search')
        try:
            budget()
            parameters=Parameters(float(x[0]),float(math.exp(x[2])),float(math.exp(x[1])))
            result=kernel.evaluate(parameters,precision,circuit_mode=report['selected_circuit_mode'],
                                   check_budget=budget,collect_rhs=verification)
            mse=float(np.mean(result['residual']**2))
            row.update(status='completed',objective_MSE=mse,summary=result['summary'])
            np.savez_compressed(trialdir/(name+'.npz'),**result['arrays'])
            if not verification and (best is None or mse < best['mse']):
                best=dict(mse=mse,x=x.copy(),trial_id=name)
                report['best_search_trial']=dict(trial_id=name,coordinates=x.tolist(),MSE=mse,
                                               RMSE_decade=math.sqrt(mse))
            return result
        except KernelFailure as exc:
            row.update(status='numerical_failure_preserved',objective_MSE=None,
                       objective_for_optimizer='positive_infinity',summary=exc.summary)
            np.savez_compressed(trialdir/(name+'.npz'),**exc.arrays)
            if exc.summary['failure_scope']=='caller_budget_exhausted':
                raise SearchStop(str(exc)) from exc
            return None
        except TimeoutError as exc:
            row.update(status='budget_exhausted_before_trajectory',objective_MSE=None,error=str(exc))
            raise SearchStop(str(exc)) from exc
        finally:
            row['elapsed_s']=time.monotonic()-trial_started
            dump(trialdir/(name+'.json'),row)
            persist()

    try:
        report['circuit_validation']=kernel.validate_circuit(check_budget=search_check)
        modes=report['circuit_validation']['modes']
        if modes['fast']['passed']:
            report['selected_circuit_mode']='fast'
        elif modes['strict_bisection42']['passed']:
            report['selected_circuit_mode']='strict_bisection42'
        else:
            raise RuntimeError('Neither declared circuit implementation passed; no ODE executed')
        try:
            for n,seed in enumerate(seeds):
                x=np.array([seed['lambda_per_v'],math.log(seed['kappa_per_v2']),math.log(seed['J_positive_a'])])
                result=evaluate(x,f'seed{n+1}',search_precision)
                print(json.dumps(dict(event='seed_forward',seed=n+1,status='completed' if result else 'numerical_failure',
                    RMSE=None if result is None else result['summary']['metrics']['all500']['RMSE_decade'])),flush=True)
            if best is not None:
                selected_seed=dict(best)
                x0=best['x'].copy()
                lower=np.array([0.,x0[1]-math.log(100),x0[2]-math.log(100)])
                upper=np.array([2.,x0[1]+math.log(100),x0[2]+math.log(100)])
                search_bounds=(lower,upper)
                simplex=np.tile(x0,(4,1))
                simplex[1,0]+=.1 if x0[0]+.1<=2. else -.1
                simplex[2,1]+=.25;simplex[3,2]+=.25
                report['optimizer_initialization']=dict(source_trial=selected_seed['trial_id'],
                    coordinates=x0.tolist(),lower=lower.tolist(),upper=upper.tolist(),initial_simplex=simplex.tolist())
                persist()
                def objective(x):
                    result=evaluate(x,'fit',search_precision)
                    return np.inf if result is None else float(np.mean(result['residual']**2))
                options=dict(plan['optimizer_settings']);options['initial_simplex']=simplex
                fit=minimize(objective,x0,method='Nelder-Mead',options=options)
                report['optimizer']=dict(success=bool(fit.success),message=str(fit.message),
                    nfev=int(fit.nfev),nit=int(fit.nit),coordinates=fit.x.tolist(),objective_MSE=float(fit.fun),
                    scope='Bounded local calibration; no global optimum claim')
        except SearchStop as exc:
            report['search_stopped_by_budget']=str(exc)
        if best is not None:
            verified=[]
            for precision in precisions:
                try:
                    result=evaluate(best['x'],precision.name,precision,verification=True)
                except SearchStop as exc:
                    report['verification_stopped_by_budget']=str(exc)
                    break
                if result is not None:
                    audit,rows=kernel.audit_native_legacy(result,check_budget=stage_check)
                    result['summary']['native_legacy_audit']=audit
                    result['arrays']['native_legacy_audit']=rows
                    np.savez_compressed(HERE/f'final_{precision.name}.npz',**result['arrays'])
                    dump(HERE/f'final_{precision.name}.json',result['summary'])
                    report['final_verifications'].append(result['summary'])
                    verified.append(result)
            if len(verified)==2:
                a,b=[v['arrays'] for v in verified]
                comparison=dict(max_log_current_change_decade=float(max(abs(np.log10(a['current_a'][kernel.mask]/b['current_a'][kernel.mask])))),
                    max_state_u_change=float(max(abs(a['state_u']-b['state_u']))))
                comparison['passed']=all(comparison[k]<=v for k,v in plan['precision_limits'].items())
                report['precision_comparison']=comparison
                audited=all(not v['summary']['native_legacy_audit']['failures'] and all(
                    v['summary']['native_legacy_audit'][k]<=limit for k,limit in plan['legacy_limits'].items()) for v in verified)
                report['native_legacy_audits_passed']=audited
                final_rmse=verified[-1]['summary']['metrics']['all500']['RMSE_decade']
                integrity=all(v['summary']['max_native_KCL_residual_a']<=1e-13
                    and v['summary']['max_rhs_KCL_residual_a']<=1e-13
                    and abs(v['summary']['apex_state_continuity_error'])<=1e-12
                    and abs(v['summary']['anchor_outward_join_error'])<=1e-12 for v in verified)
                report['native_continuity_and_KCL_passed']=integrity
                report['refined_candidate_RMSE_decade']=final_rmse
                numerical_passed=bool(comparison['passed'] and audited and integrity)
                report['numerical_validation_passed']=numerical_passed
                if numerical_passed:
                    report['verified_conditional_RMSE_decade']=final_rmse
                report['lower_than_old_same_conditioned_score']=bool(numerical_passed and final_rmse<prior)
                if search_bounds is not None:
                    lower,upper=search_bounds
                    report['best_parameter_domain_margin']=(np.minimum(best['x']-lower,upper-best['x'])/(upper-lower)).tolist()
        report['status']='completed_bounded_attempt_no_global_or_full_model_claim'
    except Exception as exc:
        report.update(status='partial_or_failed_attempt_preserved',error_type=type(exc).__name__,error=str(exc))
    finally:
        report['n_actual_ODE_trials']=sum('summary' in t for t in report['trials'])
        report['n_completed_trials']=sum(t['status']=='completed' for t in report['trials'])
        report['n_numerical_failures']=sum(t['status']=='numerical_failure_preserved' for t in report['trials'])
        report['source_hashes_still_match']={p:sha(p)==value for p,value in sources.items()}
        if not all(report['source_hashes_still_match'].values()):
            report['status']='invalid_source_changed'
            report['numerical_validation_passed']=False
            report['lower_than_old_same_conditioned_score']=False
            report.pop('verified_conditional_RMSE_decade',None)
        persist()
    print(json.dumps({k:report.get(k) for k in ['status','n_actual_ODE_trials','n_completed_trials',
        'n_numerical_failures','verified_conditional_RMSE_decade','lower_than_old_same_conditioned_score','elapsed_s']},indent=2),flush=True)


if __name__=='__main__':
    main()
