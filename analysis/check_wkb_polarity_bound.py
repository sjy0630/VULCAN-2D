"""Independent WKB class proof checks, then apply archived measured ratios."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.constants import Boltzmann, elementary_charge
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vulcan2d.wkb_polarity import log_current_integral, log_transmission, log_fermi_window, thermal_log_ratio_bound


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    archived_path = ROOT / 'analysis/parallel_rectification_bound.json'
    archived = json.loads(archived_path.read_text())
    changed = [p for p, h in archived['input_sha256'].items() if digest(ROOT / p) != h]
    if changed:
        raise RuntimeError('Measured-ratio audit inputs changed: '+str(changed))
    rng = np.random.default_rng(921); rows = []
    # Samples test implementation and range, not the universal proof itself.
    for j in range(1200):
        v = float(rng.uniform(.02, .5)); temp = float(rng.uniform(250., 400.))
        left, right = np.exp(rng.uniform(np.log(.005), np.log(4.), 2))
        width = float(rng.uniform(.05, 6.)); mass = float(rng.uniform(.1, 1.))
        values = [log_current_integral(v, left, right, width, sign=s, temperature_k=temp, mass_ratio=mass, order=48) for s in (1, -1)]
        refined = [log_current_integral(v, left, right, width, sign=s, temperature_k=temp, mass_ratio=mass, order=96) for s in (1, -1)]
        bound = float(thermal_log_ratio_bound(v, temp)); difference = refined[1]-refined[0]
        energy = np.linspace(-2., 6., 121)
        spectral = float(np.max(log_transmission(energy, v, left, right, width, mass_ratio=mass)-
                                log_transmission(energy+v, v, left, right, width, sign=-1, mass_ratio=mass)))
        window = float(np.max(np.abs(log_fermi_window(energy-v, v, temp)-log_fermi_window(energy, v, temp)))-bound)
        rows.append({'voltage_V':v, 'temperature_K':temp, 'left_height_ev':float(left), 'right_height_ev':float(right),
                     'width_nm':width, 'mass_ratio':mass, 'log_positive_integral':refined[0], 'log_negative_integral':refined[1],
                     'natural_log_ratio':difference, 'absolute_log_ratio_bound':bound,
                     'quadrature_max_log_difference':float(np.max(np.abs(np.array(values)-refined))),
                     'spectral_order_max_residual':spectral, 'thermal_window_max_bound_residual':window})
    # Same v/T within each independent positive mixture is essential.
    mixtures = []
    for j in range(200):
        v = float(rng.uniform(.02, .5)); temp = float(rng.uniform(250., 400.)); positive, negative = [], []
        n = int(rng.integers(2, 13))
        for _ in range(n):
            left, right = np.exp(rng.uniform(np.log(.005), np.log(4.), 2))
            width = float(rng.uniform(.05, 6.)); mass = float(rng.uniform(.1, 1.)); weight = float(rng.normal(0., 4.))
            positive.append(weight+log_current_integral(v, left, right, width, temperature_k=temp, mass_ratio=mass))
            negative.append(weight+log_current_integral(v, left, right, width, sign=-1, temperature_k=temp, mass_ratio=mass))
        ratio = float(logsumexp(negative)-logsumexp(positive)); bound = float(thermal_log_ratio_bound(v, temp))
        mixtures.append({'n_paths':n, 'voltage_V':v, 'temperature_K':temp, 'natural_log_ratio':ratio, 'absolute_log_ratio_bound':bound})
    if max(abs(r['natural_log_ratio'])-r['absolute_log_ratio_bound'] for r in rows+mixtures) > 1e-8:
        raise RuntimeError('Numerical WKB rectification bound failed')
    max_quad = max(r['quadrature_max_log_difference'] for r in rows)
    if max_quad > 2e-6:
        raise RuntimeError('Independent-order quadrature refinement inadequate')
    first3 = []; methods = []
    for processing, summary in archived['processing_sensitivity'].items():
        for row in summary['at_0p1_first3']:
            ratio = row['zero_reference_ratio_upper']; v = row['hbn_voltage_V']
            lower = float(np.exp(-thermal_log_ratio_bound(v, 300.)))
            delta = max(Boltzmann*300/elementary_charge*abs(np.log(ratio))-v, 0.) if ratio and ratio < 1. else 0.
            item = {'processing':processing, 'cycle_id':row['cycle_id'], 'internal_voltage_V':v,
                    'observed_ratio':row['ratio'], 'favorable_zero_reference_ratio_upper':ratio,
                    'frozen_passive_wkb_ratio_lower':lower, 'favorable_upper_still_below_bound':bool(ratio < lower),
                    'conditional_extra_landscape_difference_lower_ev':float(delta),
                    'temperature_sensitivity_lower_ev':{str(t):float(max(Boltzmann*t/elementary_charge*abs(np.log(ratio))-v, 0.))
                                                        for t in (250., 300., 350.)}}
            methods.append(item)
            if processing == 'mean_linear_current':
                first3.append(item)
    robust = [{'cycle_id':cid,
               'minimum_extra_landscape_difference_lower_ev_over_8_processing_methods':min(r['conditional_extra_landscape_difference_lower_ev'] for r in methods if r['cycle_id'] == cid),
               'all_8_favorable_ratio_upper_bounds_violate':all(r['favorable_upper_still_below_bound'] for r in methods if r['cycle_id'] == cid)}
              for cid in (1, 2, 3)]
    sources = [Path(__file__), ROOT / 'vulcan2d/wkb_polarity.py',
               ROOT / 'analysis/WKB_FROZEN_STATE_POLARITY_BOUND_2026-09-21.md', archived_path]
    sources += [ROOT / p for p in archived['input_sha256']]
    result = {'scope':'Conditional frozen passive WKB class bound, independently derived; not a universal tunneling exclusion or novelty claim',
              'bound':'abs(ln(I_minus/I_plus)) <= abs(V_internal)/V_T', 'matched_temperature_K':300.,
              'random_seed':921, 'single_path_checks':rows, 'parallel_mixture_checks':mixtures,
              'maximum_quadrature_refinement_natural_log':max_quad,
              'maximum_fraction_of_thermal_bound':max(abs(r['natural_log_ratio'])/r['absolute_log_ratio_bound'] for r in rows),
              'minimum_numerical_bound_margin':min(r['absolute_log_ratio_bound']-abs(r['natural_log_ratio']) for r in rows+mixtures),
              'all53_measured_coverage_inherited':archived['groups'], 'first3_primary_at_0p1V':first3,
              'first3_all_processing_at_0p1V':methods, 'robust_first3_summary':robust,
              'limitations':['The common frozen state, equal temperature and passive potential-difference condition are essential.',
                             'The proof uses WKB order and common energy-independent supply weights; it does not cover resonant interference or arbitrary energy-dependent contacts.',
                             'Two separate SET-return / RESET-initial measurements need not share a frozen state; the violation tests that assumption.',
                             'Zero-reference perturbations and lookup variations do not exhaust measurement systematic uncertainty.',
                             'Extra-landscape lower bounds are conditional on retaining the other hypotheses, not a measured trap energy, charge or geometry.',
                             'Random numerical cases illustrate the proved class, not additional experiments or a substitute for proof.'],
              'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in dict.fromkeys(sources)}}
    (ROOT / 'analysis/wkb_frozen_polarity_bound.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Numerical cases:',len(rows),len(mixtures),'max quadrature log error:',max_quad, flush=True)
    print('First3 primary:',json.dumps(first3, indent=2), flush=True)
    print('Processing-robust:',json.dumps(robust, indent=2), flush=True)


if __name__ == '__main__':
    run()
