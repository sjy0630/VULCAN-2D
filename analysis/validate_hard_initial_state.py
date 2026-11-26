"""Separate initial-conductance load-line effects from direct rate dependence."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import rankdata

from analysis import audit_iv41_high_conductance as reads
from analysis import refit_iv41 as iv
from analysis import validate_field_log_soft as field
from analysis import validate_hard_formation as formation
from analysis import validate_log_soft_cycle as cycle
from analysis import validate_set_outward as set_fit
from vulcan2d import validate_shared_device as shared

VREAD = .5
HIGH_CUT_US = 30.
GATES = formation.GATES


def simulate_conditioned(ramp, device, initial, ratio, background, form, growth, scale, step):
    peak = ramp.wave[ramp.apex]
    grid = np.linspace(0.,peak,int(np.ceil(peak/step))+1)
    state = np.full(len(scale),initial)
    log_k, threshold, upper = growth
    k = np.exp(log_k)

    def local(a,s):
        weight = ratio*1e-6*field.REF*np.maximum(10**s-10**initial,0.)
        def law(v):
            return scale*set_fit.background(v,background,form)+weight*(v/field.REF)**device.soft_exponent
        return ramp.circuit.operating_point(np.full(len(scale),a),law)[1]

    path = np.empty((len(scale),len(grid)))
    path[:,0] = local(grid[0],state)
    for j,delta in enumerate(np.diff(grid)):
        v0 = path[:,j]
        speed = k*np.maximum(v0-threshold,0.)**2
        mid = upper-(upper-state)*np.exp(-speed*delta/2.)
        v_mid = local(.5*(grid[j]+grid[j+1]),mid)
        speed = k*np.maximum(v_mid-threshold,0.)**2
        state = upper-(upper-state)*np.exp(-speed*delta)
        path[:,j+1] = local(grid[j+1],state)
    return grid,path


def probabilities(block, params, pivot, conditioned):
    a,b = params[:2]
    path = block["path"]
    local = block["local"] if conditioned else block["nominal"]
    delta = np.diff(path)
    terms = b*(local-formation.VOLTAGE_REFERENCE)
    log_exposure = logsumexp(np.log(delta/2.)+np.logaddexp(terms[:,:-1],terms[:,1:]),axis=1)
    offset = params[2]*(block["initial"]-pivot) if len(params)==3 else 0.
    p = -np.expm1(-np.exp(np.clip(a+log_exposure+offset,-45.,30.)))
    return np.broadcast_to(p, block["high"].shape)


def calibration(blocks,pivot,conditioned,direct):
    def loss(params):
        total = 0.
        for gate,block in blocks.items():
            if gate>1.3:
                continue
            p = np.clip(probabilities(block,params,pivot,conditioned),1e-12,1.-1e-12)
            y = block["high"]
            total -= (np.log(p[y]).sum()+np.log1p(-p[~y]).sum())
        return total
    starts = [(-5.,7.),(-20.,40.),(-2.,0.)]
    solutions = [minimize(loss,[*p,*([0.] if direct else [])],
                     bounds=[(-50.,10.),(0.,60.),*([(-5.,5.)] if direct else [])],
                     method="L-BFGS-B") for p in starts]
    return min((s for s in solutions if s.success),key=lambda s:s.fun)


def measures(prob,outcome):
    n1,n0 = outcome.sum(),(~outcome).sum()
    auc = (rankdata(prob)[outcome].sum()-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else None
    return {"predicted_fraction":float(prob.mean()),"brier":float(np.mean((prob-outcome)**2)),
            "auc":float(auc) if auc is not None else None}


def draw(path, results, all_prob):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig,axes = plt.subplots(1,2,figsize=(10,4.1),layout="constrained")
    methods = {"same_trajectory":"Same trajectory","initial_through_divider":"Initial state via divider",
               "initial_also_changes_rate":"Initial state via divider + rate"}
    styles = ("--","-", "-.")
    colors = ("#52749c","#a1263d","#4b8266")
    gs = list(results)
    ax = axes[0]
    ax.axvspan(1.35,1.53,color=".92")
    for name,style,color in zip(methods,styles,colors):
        ax.plot(gs,[results[g]["models"][name]["predicted_fraction"] for g in gs],
                style,color=color,label=methods[name])
    ax.plot(gs,[results[g]["measured_fraction"] for g in gs],"o",color=".15",label="Measured")
    ax.set(xlabel="Gate voltage (V)",ylabel="High-state fraction",ylim=(-.02,1.02),
           title="Formation fractions (gray region held out)")
    ax.legend(fontsize=7)
    ax.grid(alpha=.2)
    ax = axes[1]
    for g,marker in ((1.4,"o"),(1.5,"^")):
        b,prob = all_prob[g]
        for flag,color in ((False,"#52749c"),(True,"#a1263d")):
            ids = b["high"]==flag
            ax.scatter(b["initial"][ids],prob[ids],s=13,marker=marker,alpha=.56,color=color,
                       label=f"{g:g} V · {'high' if flag else 'low'} observed")
    ax.set(xlabel="Initial conductance at local 0.5 V (log₁₀ µS)",
           ylabel="Probability using divider only",title="Individual outcomes in held-out gates")
    ax.legend(fontsize=7)
    ax.grid(alpha=.2)
    fig.savefig(path,dpi=175)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache","revisions","output","figure"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    idx = json.loads((args.cache/"index.json").read_text())
    def archive(name):
        return json.loads((args.revisions/name).read_text())
    spec = archive("set_outward_refinement.json")
    form = spec["selected"]
    spec = spec["fits"][form]
    original = archive("field_log_soft_validation.json")["field_model"]
    initial,ratio = original["initial_log_conductance"],original["readout_positive_negative_ratio"]
    nominal = cycle.data()[1.1][:32]
    device,prep = shared.fit_soft_set([s for s,_,_,_ in nominal],nominal[0][2])
    background,growth = spec["background_parameters"],spec["growth_parameters"]
    reference = set_fit.background(VREAD,background,form)
    data = {}
    for g in GATES:
        raw,_ = iv.load_cohort(idx,args.cache,round((g-.9)*10)+5,g)
        sc,rc = iv.circuits(g)
        rows = []
        for r in raw["SET"]:
            before = reads.local_estimate(r,sc,True,VREAD)
            after = reads.terminal_estimate(r,False,.15)
            if before is not None and after is not None:
                rows.append((before,after>HIGH_CUT_US))
        values = np.asarray(rows)
        rep = [(raw["SET"][0],raw["RESET"][0],sc,rc)]
        ramp = field.SetRamp(rep,device,prep,initial,ratio,.0125)
        scales = values[:,0]*1e-6*VREAD/reference
        path,profiles = simulate_conditioned(ramp,device,initial,ratio,background,form,growth,
                                             np.r_[1.,scales],.0125)
        data[g] = {"path":path,"nominal":profiles[:1],"local":profiles[1:],
                   "initial":np.log10(values[:,0]),"high":values[:,1].astype(bool)}
        if g in (1.4,1.5):
            fine_path,fine = simulate_conditioned(ramp,device,initial,ratio,background,form,growth,
                                                  np.r_[1.,scales],.00625)
            data[g]["precision"] = (fine_path,fine)
    pivot = float(np.median(np.concatenate([x["initial"] for g,x in data.items() if g<=1.3])))
    options = {"same_trajectory":(False,False),"initial_through_divider":(True,False),
               "initial_also_changes_rate":(True,True)}
    fits = {k:calibration(data,pivot,*flags) for k,flags in options.items()}
    results,all_prob = {},{}
    for g,b in data.items():
        p = {k:probabilities(b,fits[k].x,pivot,cond) for k,(cond,_) in options.items()}
        y = b["high"]
        slices = [b["local"].max(axis=1)[f] for f in (y,~y)]
        results[g] = {"scans":len(y),"high_scans":int(y.sum()),"measured_fraction":float(y.mean()),
                      "peak_local_V_high_low_medians":[float(np.median(v)) if len(v) else None for v in slices],
                      "models":{name:measures(prob,y) for name,prob in p.items()}}
        if g in (1.4,1.5):
            path,profiles = b["precision"]
            fine = dict(b,path=path,nominal=profiles[:1],local=profiles[1:])
            results[g]["max_refinement_probability_change"] = {
                k:float(np.max(np.abs(p[k]-probabilities(fine,fits[k].x,pivot,cond))))
                for k,(cond,_) in options.items()}
        all_prob[g] = (b,p["initial_through_divider"])
    output = {"protocol":"Use measured early SET reading at intrinsic 0.5 V to scale only the shared positive background-current amplitude. Recompute the subsequent no-hard SET trajectory point by point with the same hBN switching coefficients. Compare field-based formation with or without an additional explicit initial-state term, refitting each on 0.9–1.3 V only. 1.4 and 1.5 V were not fitted. This is conditioned on measured target-prefix current and does not predict the initial state. Label uses terminal SET-return conductance over 30 µS.",
              "fits":{k:{"parameters":x.x.tolist(),"training_log_loss_total":float(x.fun)} for k,x in fits.items()},
              "results":results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.figure.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    draw(args.figure,results,all_prob)
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    main()
