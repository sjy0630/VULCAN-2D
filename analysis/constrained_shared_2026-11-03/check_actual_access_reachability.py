"""Algebraic reachability checks; no experimental curve fitting or ODE fitting.

Synthetic points test necessity versus sufficiency. The old shared parameters
are read only as a documented numerical example, not selected using a new fit.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def vcrit(u,ron,jcrit,eth,c):
    return ron*jcrit*np.exp(-u)+eth*u+c*u*np.exp(-3.*u)


def path_maximum(ron,jcrit,eth,c,lo,hi):
    def derivative(u):return -ron*jcrit*math.exp(-u)+eth+c*math.exp(-3.*u)*(1.-3.*u)
    knots=np.linspace(lo,hi,20001)
    values=[derivative(float(u)) for u in knots]
    roots=[]
    for a,b,fa,fb in zip(knots[:-1],knots[1:],values[:-1],values[1:]):
        if fa==0.:roots.append(float(a))
        elif fa*fb<0.:roots.append(brentq(derivative,float(a),float(b),xtol=1e-14))
    points=np.array([lo,hi]+roots)
    vv=vcrit(points,ron,jcrit,eth,c)
    k=int(np.argmax(vv))
    # Independent dense-grid lower estimate guards accidental missed maxima.
    dense=float(np.max(vcrit(knots,ron,jcrit,eth,c)))
    if float(vv[k])+1e-9<dense:raise RuntimeError('Stationary-point maximum below sampled maximum')
    return {'u_of_maximum':float(points[k]),'maximum_voltage_v':float(vv[k]),
            'stationary_points':roots,'dense_grid_maximum_v':dense,
            'path_interval_u':[lo,hi]}


def static_reconstruct(ar,dr,at,dt,kt):
    difference=math.log(at/ar)
    if difference==0. or dr==dt:raise ValueError('Degenerate common-state reconstruction')
    v0=(dr-dt)/difference
    ut=dt/v0;ur=ut+difference
    if v0<=0. or ut<=0. or ur<=0.:raise ValueError('No positive reconstructed states/material scale')
    return {'v0_v':v0,'initial_soft_u':ut,'pre_bare_u':ur,
            'i0_a':at*math.exp(ut),'access_voltage_v':kt*1e-9/ut,
            'bare_cubic_coefficient':kt*ur/ut}


def main():
    umin=(math.sqrt(17.)-1.)/4.
    minimum=math.exp(3.*umin)*(umin+1.)/(2.*umin-1.)
    nearby=[math.exp(3.*u)*(u+1.)/(2.*u-1.) for u in (umin-.001,umin+.001)]
    assert all(value>minimum for value in nearby)
    report={'status':'algebraic_screen_only_no_fit_no_new_trajectory',
            'locked_J_rising_Rh':{'u_of_minimum':umin,'minimum_C_over_eth':minimum},
            'static_peak_cubic_to_barrier_amplification':{
                str(u):math.exp(3.*u)/(3.*math.e*u) for u in (.5,.7807764064,1.,1.05,1.2,2.,3.)}}
    prior_path=ROOT/'analysis/shared_gap_refinement_2026-11-03/accurate_fit.json'
    old=json.loads(prior_path.read_text())
    p=old['selected_law'];q=next(row['q'] for row in old['fits'] if row['start']==old['selected_start'])
    j=p['i0_a']*math.sinh(p['eth_set_v']/p['v0_v'])
    cases=[('old_parameters_B1e4',p['ron_ohm'],j,p['eth_set_v'],1e4,max(0.,math.log(j/.001)),q[7]),
           ('synthetic_reachability_and_Rh_compatible',95.,1e-4,.35,2e4,0.,7.6)]
    report['numerical_examples']={}
    for name,ron,jcrit,eth,b,lo,hi in cases:
        c=b*(jcrit/.001)**3
        row={'ron_ohm':ron,'Jcrit_a':jcrit,'eth_set_v':eth,'B_v':b,'C_v':c,
             'C_over_eth':c/eth,'positive_compliance_a':.001,
             'can_reach_closed_gap_under_compliance':jcrit<=.001,
             **path_maximum(ron,jcrit,eth,c,lo,hi)}
        if lo<=1./3.<=hi:
            row['one_third_point_voltage_v']=float(vcrit(1./3.,ron,jcrit,eth,c))
        report['numerical_examples'][name]=row
    # A synthetic known state reconstruction, no observed targets involved.
    v0,i0,ur,ut,b=.04,3e-7,7.,1.2,20000.
    recovered=static_reconstruct(i0*math.exp(-ur),v0*ur,i0*math.exp(-ut),v0*ut,b*ut/1e-9)
    np.testing.assert_allclose([recovered[k] for k in ('v0_v','i0_a','pre_bare_u','initial_soft_u','access_voltage_v')],
                               [v0,i0,ur,ut,b],rtol=1e-13)
    report['synthetic_static_reconstruction']=recovered
    report['soft_voltage_excess_example']={'soft_hbn_voltage_v':3.3,'bare_first_compliance_v':2.7,
        'conditional_minimum_material_voltage_excess_v':.6,
        'minimum_J_over_Jcrit_from_positive_sum_degree3':(3.3/2.7)**(1./3.),
        'scope':'Only when the soft state lies within the bare traversed interval and both deembedded voltages are valid; finite-rate excess is not a contradiction.'}
    # Independently audit the separate frozen-envelope screen, if available.
    screen_path=OUT/'actual_access_reachability_screen.json'
    if screen_path.exists():
        screen=json.loads(screen_path.read_text())
        fits=screen['fits'];sr=fits['1r'];st=fits['1t1r']
        rec=static_reconstruct(sr['a_A'],sr['d_V'],st['a_A'],st['d_V'],st['k_V_per_A3'])
        derived=screen['derived_shared_parameters_and_necessary_gates']
        np.testing.assert_allclose([rec[k] for k in ('v0_v','initial_soft_u','pre_bare_u','i0_a','access_voltage_v')],
                                   [derived[k] for k in ('v0_V','u_soft','u_bare_pre','i0_A','access_B_V')],rtol=2e-14)
        ref=json.loads((ROOT/'analysis/shared_gap_refinement_2026-11-03/block_fit.json').read_text())
        ron=ref['shared_ron_ohm'];ut=rec['initial_soft_u']
        ipk=derived['minimum_Jcrit_A']*math.exp(-ut)
        # Use the predeclared training low-resistance Q90, not a claimed
        # universal resistance upper bound or an arbitrarily exact u=0.
        vread=.05;r_upper=187.68;iread_min=vread/r_upper
        uread_upper=(vread-ron*iread_min)*ut/(st['k_V_per_A3']*iread_min**3)
        allowed_drop=math.log(.001/ipk)
        report['independent_frozen_envelope_audit']={
            'reconstructed':rec,'soft_peak_current_a':ipk,'compliance_a':.001,
            'read_voltage_v':vread,'predeclared_low_resistance_q90_ohm':r_upper,
            'minimum_current_at_read_a':iread_min,'low_state_u_upper':uread_upper,
            'required_soft_to_low_state_drop_lower':ut-uread_upper,
            'maximum_state_drop_from_frozen_soft_and_compliance':allowed_drop,
            'conditional_state_drop_incompatible':ut-uread_upper>allowed_drop,
            'scope':'Frozen windows, nominal transistor mapping, inherited G and selected static fit; rejects this subcase, not the entire finite-history family.',
            'screen_sha256':hashlib.sha256(screen_path.read_bytes()).hexdigest()}
    report['source_sha256']={'script':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             str(prior_path.relative_to(ROOT)):hashlib.sha256(prior_path.read_bytes()).hexdigest()}
    (OUT/'actual_access_reachability.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
