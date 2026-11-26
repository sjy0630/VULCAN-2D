"""One predeclared shared-gap family fitted to both recovered experiments.

No measured current enters simulate_gap. Training uses four median branches;
held-out records are not used to select parameters. Source/clock limitations
are recorded with every output. Run from the repository root.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares, brentq, minimize, OptimizeResult

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vulcan2d.shared_gap import GapLaw, BareGapCircuit, LookupGapCircuit, simulate_gap
from vulcan2d.transistor import measured_transistor_mapping

FOLDER = ROOT / 'analysis/shared_hbn_2026-11-03'
DATA = FOLDER / 'data/native_observations.npz'
NAMES = ['log10_v0', 'log10_i0', 'log10_Icrit_set', 'log10_Icrit_reset',
         'log10_kset', 'log10_kreset', 'log10_escale', 'initial_u_1r', 'initial_u_1t1r']
LOWER = np.array([-2.3, -8., -5., -5., -3., -3., -2.3, .1, .1])
UPPER = np.array([.3, -.5, -2., -.5, 3., 3., .3, 29., 29.])


def load_data():
    with np.load(DATA) as raw:
        bare_ids = sorted({int(k.split('_')[2]) for k in raw.files if k.startswith('1r_')})
        split = {'1r': {'train': bare_ids[:14], 'test': bare_ids[14:]},
                 '1t1r': {'train': list(range(4, 36)), 'test': list(range(36, 54))}}
        data = {}
        for topology, groups in split.items():
            for group, ids in groups.items():
                for phase in ('set', 'reset'):
                    vv = np.array([raw[f'{topology}_cycle_{c}_{phase}_voltage'] for c in ids])
                    ii = np.abs(np.array([raw[f'{topology}_cycle_{c}_{phase}_current'] for c in ids]))
                    if not np.all(vv == vv[0]):
                        raise ValueError('This median benchmark requires identical waveform grids')
                    data[topology, group, phase] = (vv[0], ii)
    # Independent low-bias training observation fixes the common closed-path R.
    v, current = data['1r', 'train', 'reset']
    apex = int(np.argmax(abs(v)))
    keep = (np.arange(len(v)) <= apex) & (abs(v) >= .049) & (abs(v) <= .201)
    slopes = np.sum(abs(v[keep]) * current[:, keep], axis=1) / np.sum(v[keep]**2)
    ron = float(np.median(1. / slopes))
    return data, split, ron


def material(q, ron):
    v0, i0, icset, icreset, ks, kr, escale = 10.**np.asarray(q[:7])
    return GapLaw(ron_ohm=ron, v0_v=v0, i0_a=i0, u_max=30.,
                  kset_per_v=ks, kreset_per_v=kr,
                  eth_set_v=v0*np.arcsinh(icset/i0),
                  eth_reset_v=v0*np.arcsinh(icreset/i0), escale_v=escale,
                  reset_initiation_drive_v=float(q[9]) if len(q)>9 else 0.)


def median_curve(data, topology, phase, stride=1):
    v, ii = data[topology, 'train', phase]
    obs = 10.**np.median(np.log10(np.maximum(ii, 1e-15)), axis=0)
    # The bare SET return has only one nonlimited positive read point. Uniform
    # decimation can drop it and make a falsely soft endpoint look acceptable.
    indices = np.unique(np.r_[np.arange(0, len(v), stride), np.flatnonzero(abs(v) <= .201),
                              np.argmax(abs(v)), len(v)-1])
    return v[indices], obs[indices]


def pair(law, initial, voltages, circuits, step):
    first = simulate_gap(voltages[0], initial, law, circuits[0], max_step_v=step)
    second = simulate_gap(voltages[1], first['final_u'], law, circuits[1], max_step_v=step,
                          initial_command_voltage_v=float(voltages[0][-1]))
    return first, second


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-nfev', type=int, default=120)
    parser.add_argument('--starts', type=int, default=3)
    parser.add_argument('--step', type=float, default=.04)
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--output', default='gap_joint_fit.json')
    parser.add_argument('--reset-initiation', action='store_true')
    parser.add_argument('--reset-only', action='store_true')
    args = parser.parse_args()
    data, split, ron = load_data()
    trans = LookupGapCircuit(measured_transistor_mapping(), 1.1)
    circuits = {'1r': (BareGapCircuit(1e-3), BareGapCircuit()), '1t1r': (trans, trans)}
    samples = {top: [median_curve(data, top, p, args.stride) for p in ('set', 'reset')]
               for top in circuits}
    masks = {}
    for top, curves in samples.items():
        for phase, (v, obs) in zip(('set', 'reset'), curves):
            floor = 1e-11 if phase == 'set' else 1e-12
            use = (abs(v) > 1e-12) & (obs >= floor)
            if top == '1r' and phase == 'set':
                outward = np.arange(len(v)) <= np.argmax(abs(v))
                masks[top, phase] = [use & (obs < .995e-3) & outward,
                                     use & (obs < .995e-3) & ~outward,
                                     use & (obs >= .995e-3)]
            else:
                masks[top, phase] = [use]
    nres = sum(int(sum(np.count_nonzero(m) for m in mm)) for mm in masks.values())
    # A second continuous 1T1R pair constrains cycle-to-cycle state carryover.
    nres += sum(int(sum(np.count_nonzero(m) for m in masks['1t1r', p]))
                for p in ('set', 'reset'))
    calls, failures, started, best_seen = 0, 0, time.monotonic(), float('inf')

    def physical_residual(q):
        pieces = []
        law = material(q, ron)
        for ti, top in enumerate(circuits):
            voltage = [row[0] for row in samples[top]]
            results = pair(law, q[7+ti], voltage, circuits[top], args.step)
            repeats = [results]
            if top == '1t1r':
                repeats.append(pair(law, results[-1]['final_u'], voltage, circuits[top], args.step))
            for results in repeats:
                for phase, result, (_, obs) in zip(('set', 'reset'), results, samples[top]):
                    pred = abs(result['current_a'])
                    if not np.all(np.isfinite(pred)):
                        raise ValueError('Nonfinite model output')
                    delta = np.log10(np.maximum(pred, 1e-16)) - np.log10(obs)
                    for mask in masks[top, phase]:
                        # Each phase gets equal weight. Bare SET pre-switch,
                        # nonlimited return and platform each get one third.
                        scale = np.sqrt(np.count_nonzero(mask)*len(masks[top, phase])*len(repeats))
                        pieces.append(delta[mask]/scale)
        out = np.concatenate(pieces)
        if len(out) != nres or not np.all(np.isfinite(out)):
            raise ValueError('Invalid objective dimensions or values')
        return out

    def residual(q):
        nonlocal calls, failures, best_seen
        calls += 1
        try:
            out = physical_residual(q)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError):
            failures += 1
            out = np.full(nres, 100./np.sqrt(nres))
        score = float(np.linalg.norm(out))
        if score < best_seen:
            best_seen = score
        if calls % 50 == 1:
            print(json.dumps({'evaluations':calls,'elapsed_s':round(time.monotonic()-started,1),
                              'best_balanced_norm':best_seen,'invalid_trials':failures}),flush=True)
        return out

    # Identify two pre-switch transport combinations using training data only;
    # these seed the one shared family, not additional fitted model families.
    vbare, ibare = median_curve(data, '1r', 'set')
    pre = (np.arange(len(vbare)) <= np.argmax(abs(vbare))) & (vbare >= .15) & (vbare <= 2.2)
    prefit = least_squares(lambda ab: np.log10(10.**ab[0]*np.sinh(vbare[pre]/10.**ab[1]))
                          - np.log10(ibare[pre]), [-10., -.5], bounds=([-15., -1.], [-5., 1.]))
    a, b = 10.**prefit.x
    iform = a*np.sinh(2.675/b)
    vt, it = median_curve(data, '1t1r', 'set')
    gsoft = float(it[np.argmin(abs(vt[:np.argmax(vt)]-.1))]/.1)
    seeds = []
    for ic in (2e-4, 7e-4, 8e-5):
        ur = float(np.clip(np.log(ic/iform), 2., 20.))
        v0, i0 = b/ur, a*np.exp(ur)
        ut = brentq(lambda u: ron+v0*u*np.exp(u)/i0-1./gsoft, .000001, 29.)
        seeds.append(np.clip(np.array([np.log10(v0), np.log10(i0), np.log10(ic), np.log10(.005),
                                      1., 1., np.log10(.05), ur, ut]), LOWER+1e-6, UPPER-1e-6))
    lower,upper,names=LOWER,UPPER,NAMES
    if args.reset_initiation:
        baseline=json.loads((FOLDER/'gap_base_fit.json').read_text())
        baseq=np.array(baseline['fits'][baseline['selected_start']]['q'])
        seeds=[np.r_[baseq,value] for value in (.1,.3,.6)]
        lower,upper=np.r_[LOWER,0.],np.r_[UPPER,1.]
        names=NAMES+['reset_initiation_drive_V']
        if args.reset_only:
            base_law=material(baseq,ron)
            seeds=[]
            for threshold in (.18,.22):
                start=np.r_[baseq,.44-threshold]
                start[3]=np.log10(base_law.i0_a*np.sinh(threshold/base_law.v0_v))
                start[5]=0.
                seeds.append(np.clip(start,lower+1e-6,upper-1e-6))
    fits = []
    for index, seed in enumerate(seeds[:args.starts]):
        initial_residual = residual(seed)
        if args.reset_only:
            free=np.array([3,5,9])
            def reset_objective(values):
                full=seed.copy(); full[free]=values
                error=residual(full)
                return float(np.dot(error,error))
            opt=minimize(reset_objective,seed[free],method='Powell',
                         bounds=list(zip(lower[free],upper[free])),
                         options={'maxfev':args.max_nfev,'maxiter':12,'ftol':1e-4,'xtol':1e-4})
            fitted=seed.copy(); fitted[free]=opt.x
            result=OptimizeResult(x=fitted,cost=opt.fun/2.,success=opt.success,
                                  message=opt.message,nfev=opt.nfev,optimality=None)
        else:
            result = least_squares(residual, seed, bounds=(lower, upper), max_nfev=args.max_nfev,
                                   diff_step=1e-3, x_scale='jac', ftol=2e-5, xtol=2e-5, gtol=1e-5)
        validation_error = None
        try:
            final_residual = physical_residual(result.x)
            valid = True
            final_cost = float(np.dot(final_residual, final_residual)/2.)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            valid, final_cost, validation_error = False, None, str(exc)
        fits.append({'start':index,'initial_norm':float(np.linalg.norm(initial_residual)),
                     'q':result.x.tolist(),'cost':float(result.cost),'success':bool(result.success),
                     'valid_trajectory':valid,'verified_cost':final_cost,'validation_error':validation_error,
                     'message':str(result.message),'nfev':int(result.nfev),
                     'optimality':float(result.optimality) if result.optimality is not None else None})
        valid_fits = [r for r in fits if r['valid_trajectory']]
        selected = min(valid_fits, key=lambda r:r['verified_cost']) if valid_fits else None
        report = {'status':'common_gap_candidate_training_only' if selected else 'no_valid_candidate',
                  'parameter_names':names,
                  'shared_ron_ohm':ron, 'split':split, 'settings':vars(args), 'fits':fits,
                  'selected_start':selected['start'] if selected else None,
                  'objective':'equal four phase weight; bare SET nonlimited outward, nonlimited return, and platform equal; two continuous 1T1R pairs',
                  'n_actual_evaluations':calls,'invalid_trials':failures,
                  'elapsed_s':time.monotonic()-started,
                  'input_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
                  'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (Path(__file__), ROOT/'vulcan2d/shared_gap.py')},
                  'limitations':['Voltage-path dynamics, not physical seconds or overshoot.',
                                 'Same-label bare SET/RESET pairing is conditional; cell24 has a mismatch.',
                                 'One initial state per topology, shared material parameters, no per-phase reset.',
                                 'Holdout data excluded from fitting; prior project development used these datasets.',
                                 'No physical validation follows from optimizer convergence.']}
        report['selected_law'] = asdict(material(selected['q'],ron)) if selected else None
        (FOLDER/args.output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'completed_start':index, **fits[-1]}),flush=True)


if __name__ == '__main__':
    main()
