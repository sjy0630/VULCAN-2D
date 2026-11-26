"""Evaluate a shared field-activated high-conductance formation rule on IV41."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

from analysis import audit_iv41_high_conductance as reads
from analysis import refit_iv41 as iv
from analysis import validate_field_log_soft as field
from analysis import validate_log_soft_cycle as cycle
from analysis import validate_set_outward as set_fit
from vulcan2d import validate_shared_device as shared


GATES = (.9,1.,1.1,1.2,1.3,1.4,1.5)
VOLTAGE_REFERENCE = 1.5


def formation_probability(applied, local, params):
    """First-passage probability along the SET voltage path."""
    log_k, field_slope = params
    hazard = np.exp(log_k + field_slope*(local-VOLTAGE_REFERENCE))
    return -np.expm1(-cumulative_trapezoid(hazard,applied,initial=0.))


def evidence(cohort,circuit):
    rows = []
    for row in cohort:
        terminal = reads.terminal_estimate(row,False,.15)
        if terminal is not None:
            rows.append((row,terminal,reads.local_estimate(row,circuit,False,.15)))
    return rows


def fit_probability(cohorts, cut):
    train = [x for g,x in cohorts.items() if g<=1.3]

    def objective(params):
        errors = 0.
        for applied,local,rows,_ in train:
            p = formation_probability(applied,local,params)[-1]
            p = np.clip(p,1e-14,1.-1e-14)
            n = len(rows)
            k = sum(g>cut for _,g,_ in rows)
            errors -= k*np.log(p)+(n-k)*np.log1p(-p)
        return errors

    fits = [minimize(objective,start,bounds=((-35.,8.),(0.,45.)),method="L-BFGS-B")
            for start in ((-5.,7.),(-12.,20.),(-2.,2.))]
    return min((fit for fit in fits if fit.success),key=lambda f:f.fun)


def low_bias_hard_score(rows,circuit,conductance):
    residual = []
    voltage = []
    data = []
    predicted = []
    for r,term,_ in rows:
        if term<=30.:
            continue
        v = np.abs(r.voltage)
        mask = ~r.masks["branch_outward"] & r.masks["use_unclamped"] & (v>=.05) & (v<=.25)
        if not np.any(mask):
            continue
        i,_ = circuit.operating_point(v[mask],lambda x:conductance*1e-6*x)
        obs = np.abs(r.current[mask])
        residual.extend(np.log10(i/obs))
        voltage.extend(v[mask])
        data.extend(obs*1e6)
        predicted.extend(i*1e6)
    return ({"points":len(residual),"rmse_decade":float(np.sqrt(np.mean(np.square(residual))))
             if residual else None},(voltage,data,predicted))


def render(path, stats, hard):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = list(stats)
    fig,axes = plt.subplots(1,2,figsize=(10,4.3),layout="constrained")
    ax = axes[0]
    measured = [stats[x]["fraction_measured"] for x in g]
    predicted = [stats[x]["probability_predicted"] for x in g]
    ax.axvspan(1.35,1.53,color=".91",label="Held-out gates")
    ax.plot(g,predicted,"-",color="#a1263d",label="Shared formation rule")
    ax.plot(g,measured,"o",color=".15",label="Measured high-state fraction")
    ax.set(ylim=(-.025,1.025),xlabel="Gate voltage (V)",ylabel="High-state fraction",
           title="High state formation after SET")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    ax = axes[1]
    for gate,color in ((1.4,"#486d93"),(1.5,"#4b8266")):
        v,m,p = [np.asarray(x) for x in hard[gate]]
        ax.scatter(v,m,s=7,alpha=.12,color=color,label=f"{gate:g} V measured high group")
        ids = np.argsort(v)
        ax.plot(v[ids],p[ids],color=color,lw=1.6,label=f"{gate:g} V shared hard conductance")
    ax.set(xlabel="SET return terminal voltage (V)",ylabel="Current magnitude (µA)",
           title="High state after formation")
    ax.legend(fontsize=7)
    ax.grid(alpha=.2)
    fig.savefig(path,dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache","revisions","output","figure"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    idx = json.loads((args.cache/"index.json").read_text())
    data = lambda name:json.loads((args.revisions/name).read_text())
    fitted = data("set_outward_refinement.json")
    original = data("field_log_soft_validation.json")["field_model"]
    form = fitted["selected"]
    parameters = fitted["fits"][form]
    initial,ratio = original["initial_log_conductance"],original["readout_positive_negative_ratio"]
    training = cycle.data()[1.1][:32]
    device,prep = shared.fit_soft_set([s for s,_,_,_ in training],training[0][2])

    def current(v,s):
        return set_fit.currents(v,s,initial,ratio,device.soft_exponent,
                                parameters["background_parameters"],form)

    cohorts = {}
    for g in GATES:
        raw,_ = iv.load_cohort(idx,args.cache,round((g-.9)*10)+5,g)
        sc,rc = iv.circuits(g)
        representative = [(raw["SET"][0],raw["RESET"][0],sc,rc)]
        ramp = field.SetRamp(representative,device,prep,initial,ratio,.0125,current_law=current)
        _,_,local = ramp.predict(parameters["growth_parameters"])
        cohorts[g] = (ramp.wave[:ramp.apex+1],local[:ramp.apex+1],evidence(raw["SET"],sc),sc)
    fit = fit_probability(cohorts,30.)
    hard_conductances = [local for g,(_,_,rows,_) in cohorts.items() if g<=1.3
                         for _,terminal,local in rows if terminal>30. and local is not None]
    hard = float(np.median(hard_conductances))
    report, traces = {},{}
    for g,(applied,local,rows,sc) in cohorts.items():
        p = formation_probability(applied,local,fit.x)
        assert np.all(np.diff(p)>=0) and p[0]==0 and p[-1]<=1.
        counts = {str(c):sum(value>c for _,value,_ in rows) for c in (20,30,40)}
        hard_stat,traces[g] = low_bias_hard_score(rows,sc,hard)
        report[g] = {"sweeps":len(rows),"high_counts_by_terminal_uS_cut":counts,
                     "fraction_measured":counts["30"]/len(rows),"probability_predicted":float(p[-1]),
                     "peak_hBN_V_before_formation":float(local.max()),
                     "conditional_hard_set_return":hard_stat}
    v14,h14,_,_ = cohorts[1.4]
    v15,h15,_,_ = cohorts[1.5]
    sensitivity = {str(c):{str(g):float(formation_probability(*cohorts[g][:2],fit_probability(cohorts,c).x)[-1])
                           for g in (1.4,1.5)} for c in (20,40)}
    result = {"protocol":"Fit the first-passage rate to SET-return conductance labels from gate 0.9–1.3 V. Entire 1.4 and 1.5 V excluded from optimization. Their distributions were previously explored to define the task. Hard label uses terminal conductance at 0.15 V >30 uS, a descriptive threshold, never part of the formation rule. Gate only enters the transistor load line. Rates are per sweep voltage, not per unit time. Conditional hard current is checked on SET return only; hard RESET is not simulated.",
              "formation_fit":{"log_rate_at_1p5V":float(fit.x[0]),"log_slope_per_V":float(fit.x[1])},
              "shared_hard_conductance_uS":hard,"hard_conductance_training_count":len(hard_conductances),
              "sensitivity_to_label_cut":sensitivity,
              "field_ordering":{"min_Vh_1p5_minus_1p4_V":float(np.min(h15-np.interp(v15,v14,h14)))},
              "results":report}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.figure.parent.mkdir(parents=True,exist_ok=True)
    render(args.figure,report,traces)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
