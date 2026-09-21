"""Exact descriptive error floor for I_i,s(v)=A_i*f_s(v) on a common grid.

This is a mathematical constraint on an amplitude-only state model, not a
statistical test, a dimension proof for every latent-state model, or evidence
that cross-scan differences specifically arise from relaxation.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vulcan2d.contact_network import log10_shape


def common_shape_floor(positive_log,negative_log):
    yp,yn=np.asarray(positive_log,dtype=float),np.asarray(negative_log,dtype=float)
    if yp.ndim!=2 or yp.shape!=yn.shape or min(yp.shape)<2 or np.any(~np.isfinite(yp+yn)):
        raise ValueError('Finite paired cycle-by-voltage matrices required')
    mean=(yp+yn)/2;difference=yn-yp
    cycle_amp=mean.mean(axis=1)-mean.mean()
    mean_shape=mean.mean(axis=0);ratio_shape=difference.mean(axis=0)
    predp=cycle_amp[:,None]+mean_shape[None,:]-ratio_shape[None,:]/2
    predn=cycle_amp[:,None]+mean_shape[None,:]+ratio_shape[None,:]/2
    em=mean-cycle_amp[:,None]-mean_shape[None,:];ed=difference-ratio_shape[None,:]
    return {'ratio_only_lower_bound_rmse_decade':float(np.sqrt(np.mean(ed**2))/2),
            'exact_arbitrary_common_shapes_rmse_decade':float(np.sqrt(np.mean(em**2)+np.mean(ed**2)/4)),
            'mean_shape_residual_rmse_decade':float(np.sqrt(np.mean(em**2))),
            'cycle_mean_log_ratio_std_decade':float(np.std(difference.mean(axis=1))),
            'cycle_mean_log_ratio':difference.mean(axis=1).tolist(),
            'best_common_log_ratio_at_each_voltage':ratio_shape.tolist(),
            'cycle_log_amplitudes_centered':cycle_amp.tolist(),
            'positive_prediction':predp.tolist(),'negative_prediction':predn.tolist()}


def relaxed_interval_floor(lower,upper):
    """Conservative ratio-only bound, independently relaxing each grid point."""
    lower,upper=np.asarray(lower),np.asarray(upper)
    if lower.shape!=upper.shape or lower.ndim!=2 or np.any(lower>upper):raise ValueError('Invalid interval matrices')
    losses=[]
    for lo,hi in zip(lower.T,upper.T):
        def mse(center):return np.mean((center-np.clip(center,lo,hi))**2)
        a=float(lo.min());b=float(hi.max())
        losses.append(mse(a) if a==b else minimize_scalar(mse,bounds=(a,b),method='bounded').fun)
    return float(np.sqrt(np.mean(losses))/2)


def run():
    source=ROOT/'analysis/cross_polarity_physics_eta_grid.csv'
    network_source=ROOT/'analysis/contact_network_comparison.json'
    network=json.loads(network_source.read_text())
    with source.open(encoding='utf-8-sig') as f:
        rows=[r for r in csv.DictReader(f) if r['gate_V']=='1.1' and r['lookup_branch']=='mean'
              and r['status']=='valid' and .1-1e-10<=float(r['hbn_voltage_V'])<=.3+1e-10]
    volts=sorted({float(r['hbn_voltage_V']) for r in rows})
    lut={(int(r['cycle_id']),float(r['hbn_voltage_V'])):r for r in rows}
    all_cycles=sorted({int(r['cycle_id']) for r in rows});groups={}
    for tag,cycles in [('all53',all_cycles),('cycles4to53',[c for c in all_cycles if c>=4]),
        ('complete_signal_guard',[c for c in all_cycles if all(max(float(lut[c,v]['set_zero_reference_fraction']),float(lut[c,v]['reset_zero_reference_fraction']))<.1 for v in volts)])]:
        yp=np.array([[np.log10(float(lut[c,v]['set_current_A'])) for v in volts] for c in cycles])
        yn=np.array([[np.log10(float(lut[c,v]['reset_current_A'])) for v in volts] for c in cycles])
        result=common_shape_floor(yp,yn);result.update(n_cycles=len(cycles),cycle_ids=cycles,voltage_grid_V=volts)
        fit_group='all53' if tag=='all53' else 'cycles4to53'
        result['descriptive_models_on_identical_grid']={}
        for name in ('fixed_level','equilibrium_right','coulomb_right','empirical_branches'):
            pars=network['groups'][fit_group]['candidates'][name]['all_data_fit']['parameters']
            sp=log10_shape(np.array(volts),name,pars);sn=log10_shape(-np.array(volts),name,pars)
            amp=((yp-sp).mean(axis=1)+(yn-sn).mean(axis=1))/2
            ep=yp-amp[:,None]-sp;en=yn-amp[:,None]-sn
            result['descriptive_models_on_identical_grid'][name]={
                'source_fit_group':fit_group,'rmse_decade':float(np.sqrt((np.mean(ep**2)+np.mean(en**2))/2))}
        groups[tag]=result
    complete=[c for c in all_cycles if all(lut[c,v]['zero_reference_sensitivity_available_combinations']=='4'
                    and lut[c,v]['zero_reference_sensitivity_eta_min'] for v in volts)]
    low=np.array([[np.log10(float(lut[c,v]['zero_reference_sensitivity_eta_min'])) for v in volts] for c in complete])
    high=np.array([[np.log10(float(lut[c,v]['zero_reference_sensitivity_eta_max'])) for v in volts] for c in complete])
    observed=np.array([[float(lut[c,v]['log10_eta']) for v in volts] for c in complete])
    result={'definition':'y_i,+ = a_i+s_+(v); y_i,- = a_i+s_-(v), arbitrary common voltage functions',
        'derivation':'mean squared branch error = mean residual squared of m=(y+ + y-)/2 plus one-quarter mean residual squared of d=y- - y+. The shared amplitude cancels from d.',
        'groups':groups,'zero_reference_relaxation':{'n_cycles':len(complete),'cycle_ids':complete,
            'excluded_incomplete_interval_cycle_ids':sorted(set(all_cycles)-set(complete)),
            'observed_ratio_only_floor_on_same_cycles_decade':float(np.sqrt(np.mean((observed-observed.mean(axis=0))**2))/2),
            'conservative_ratio_only_floor_decade':relaxed_interval_floor(low,high),
            'interpretation':'Per-point ratio intervals allow four zero-reference perturbations and even independent choices across voltages. This relaxed optimization is not a confidence interval or a noise model.'},
        'limitations':['The bound applies to the same common grid and weights; it must not be subtracted from scores using different windows or native sampling.',
            'Interpolated points are correlated; no independent-sample significance claim is made.',
            'Rules out amplitude-only fixed-shape descriptions at the observed residual level; does not prove that every possible scalar state is impossible.',
            'Differences could reflect cycle-specific shape/rectification, cross-scan changes, or unquantified measurement effects.'],
        'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (Path(__file__),source,network_source,ROOT/'vulcan2d/contact_network.py')}}
    out=ROOT/'analysis/shared_amplitude_bound.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(out)
    for tag,g in groups.items():print(tag,g['n_cycles'],g['ratio_only_lower_bound_rmse_decade'],g['exact_arbitrary_common_shapes_rmse_decade'])
    print('offset relaxed',result['zero_reference_relaxation'])


if __name__=='__main__':run()
