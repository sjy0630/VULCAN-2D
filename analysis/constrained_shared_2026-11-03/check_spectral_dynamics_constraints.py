"""Necessary-constraint diagnostics for a frozen spectral shape; no ODE fit.

Uses training-only summary scales and the existing conditional readout fit.
The narrow-spectrum J_beta RESET formula is an asymptotic screening relation,
not a complete dynamic prediction or a fitted physical activation law.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, roots_hermitenorm

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from vulcan2d.transistor import measured_transistor_mapping
from spectral_readout import spectral_current


def window_fraction(v, p, order):
    """Narrow Lorentzian total spectral fraction at signed local voltage."""
    if p['width_ev']==0:
        energies=np.array([p['epsilon_ev']]);weights=np.ones(1)
    else:
        nodes,weights=roots_hermitenorm(order);weights=weights/weights.sum()
        energies=p['epsilon_ev']+p['width_ev']*nodes
    thermal=8.617333262145e-5*p['temperature_k']
    values=expit(-(energies-(1-p['eta'])*v)/thermal)-expit(-(energies+p['eta']*v)/thermal)
    return abs(float(weights@values))


def main():
    source=OUT/'spectral_readout_screen.json'
    raw=json.loads(source.read_text());family=raw['families']['gaussian_ensemble']
    chosen=family['fits'][family['selected_start']];p=chosen['parameters']
    mapping=measured_transistor_mapping()
    plateau_current=2.67851e-6
    vt=mapping.voltage_for_current(plateau_current,1.1,'reset')
    # Fixed current deembedding here is an oracle screen of an ideal plateau.
    # It is not the actual current trajectory nor a new transistor calibration.
    commands=np.array([1.,1.04,1.2,1.36,1.4])
    locals_=commands-vt
    if np.any(locals_<=0):raise ValueError('Invalid assumed plateau deembedding')
    sf=np.array([window_fraction(-float(v),p,1024) for v in locals_])
    coarse=np.array([window_fraction(-float(v),p,512) for v in locals_])
    sb=window_fraction(-2.5,p,1024)
    bare_median=2.138307664267815e-7
    ratio=plateau_current/bare_median
    q=math.log(ratio)/math.log(sb/sf[2])
    beta=q/(1.+q)
    interior_log_span=q*math.log10(sf[3]/sf[1])
    max_q=.16922/math.log10(sf[3]/sf[1])
    anchors=[]
    d=raw['data']['set'];commands_set=np.array(d['command'])
    for a in (5.,4.,2.,1.,.2):
        index=int(np.argmin(abs(commands_set-a)))
        voltage=float(d['hbn'][index]);current=float(d['current'][index])
        def residual(loggamma):
            predicted=spectral_current(voltage,math.exp(loggamma),p['epsilon_ev'],p['width_ev'],
                p['eta'],p['ron_ohm'],p['temperature_k'],order=1024)
            return math.log(predicted/current)
        loggamma=brentq(residual,-30.,10.,xtol=1e-11)
        anchors.append({'command_v':float(commands_set[index]),'hbn_v':voltage,
                        'current_a':current,'inferred_gamma_ev':math.exp(loggamma),
                        'relative_current_residual':math.expm1(residual(loggamma))})
    report={
        'status':'conditional_narrow_spectrum_threshold_screen_no_fit',
        'spectral_parameters':p,
        'selected_static_start':family['selected_start'],
        'positive_return_gamma_inversion':{
            'status':'Oracle anchor inversion with the same fixed selected spectral shape; not a fitted dynamic path.',
            'anchors_in_return_order':anchors,
            'gamma_2V_over_gamma_5V':anchors[2]['inferred_gamma_ev']/anchors[0]['inferred_gamma_ev'],
            'monotone_set_closure_conflict_at_selected_shape':anchors[2]['inferred_gamma_ev']<anchors[0]['inferred_gamma_ev'],
            'scope':'A decreasing gamma during positive return requires u to increase for every gamma_c>0, contradicting G_SET<=0 at this selected shape. Other accepted shape parameters or mapping hypotheses were not profiled.'},
        'reset_narrow_lock':{
            'assumed_series_plateau_current_a':plateau_current,
            'transistor_drop_v':vt,
            'series_command_v':commands.tolist(),
            'series_internal_magnitude_v':locals_.tolist(),
            'series_spectral_fraction':sf.tolist(),
            'bare_2p5V_spectral_fraction':sb,
            'quadrature_512_to_1024_max_abs_change':float(np.max(abs(sf-coarse))),
            'bare_tail_median_current_a':bare_median,
            'required_midpoint_series_to_bare_ratio':ratio,
            'q_required_by_median_tail_ratio':q,
            'beta_required_by_median_tail_ratio':beta,
            'predicted_log_current_span_1p04_to_1p36_decade_at_required_q':interior_log_span,
            'predeclared_q90_log_span_gate_decade':.16922,
            'q_max_from_smooth_quantile_span_gate':max_q,
            'maximum_midpoint_to_bare_ratio_at_span_gate':float((sb/sf[2])**max_q),
            'median_tail_and_smooth_platform_incompatible_at_selected_shape':interior_log_span>.16922,
        },
        'assumptions':[
            'Same fixed epsilon, eta, width, temperature; both RESET windows have gamma much smaller than thermal/window scales AND u much larger than one, so I=A exp(-u) S(V). Small gamma alone does not establish large u because gamma_c is not calibrated.',
            'Both RESET windows nearly track the same constant J_beta threshold, with 0<beta<1.',
            'Nominal common transistor mapping; ideal plateau current used only to estimate internal voltage.',
            'The 1.04 and 1.36V points represent continuum 10th/90th positions of a smooth monotone 1.0-1.4V model, not quantiles of observed noisy data.',
            'Median ratio and existing Q90 span gate are descriptive training constraints, not a likelihood or a formal population rejection.',
            'Failure excludes this locked asymptotic subcase at the selected readout shape, not every spectral or finite-rate model.'
        ],
        'sources_sha256':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__),source,OUT/'spectral_readout.py',ROOT/'vulcan2d/transistor.py']}
    }
    (OUT/'spectral_dynamics_constraints.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
