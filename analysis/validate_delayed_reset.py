"""Inferred RESET conductance and a controlled test of delayed erasure."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from analysis import audit_log_soft as rd
from analysis import validate_activated_reset as ar
from analysis import validate_log_soft_cycle as cy


def response(voltage, beta):
    return voltage*np.exp(beta*(np.sqrt(voltage)-np.sqrt(rd.REFERENCE_V)))


def diagnose(rows, beta):
    matrix, starts, ends = [],[],[]
    counts = {"cycles":0,"plateau_points":0,"above_entry":0,
              "above_twice_entry":0,"below_end":0,"plateau_cycles_compared":0,
              "rises_by_twofold_between_plateau_halves":0}
    for _,r,_,rc in rows:
        start,end = cy.observed_state(r,rc),ar.measured_end(r,rc)
        if start is None or end is None or end>start:
            continue
        ids,v,i = rd.intrinsic(r,rc,outward=True)
        app = np.abs(r.voltage)
        curve = np.full(len(app),np.nan)
        curve[ids] = np.log10(i[ids]/response(v[ids],beta)/1e-6)
        plateau = curve[ids[(app[ids]>=.8)&(app[ids]<=1.35)]]
        if not len(plateau):
            continue
        counts["cycles"] += 1
        counts["plateau_points"] += len(plateau)
        counts["above_entry"] += int(np.sum(plateau>start))
        counts["above_twice_entry"] += int(np.sum(plateau>start+np.log10(2)))
        counts["below_end"] += int(np.sum(plateau<end))
        early = plateau[:len(plateau)//2]
        late = plateau[len(plateau)//2:]
        if len(early)>=3 and len(late)>=3:
            counts["plateau_cycles_compared"] += 1
            counts["rises_by_twofold_between_plateau_halves"] += int(
                np.median(late)-np.median(early)>np.log10(2))
        matrix.append(curve)
        starts.append(start)
        ends.append(end)
    return counts,np.asarray(matrix),np.asarray(starts),np.asarray(ends)


def delayed_rate(parameters):
    log_rate,threshold = parameters
    # The rate increases with the local voltage above a common onset.
    return lambda v: np.exp(log_rate)*np.maximum(v-threshold,0.)


def render(path, rows, base, delayed, all_states, starts, ends):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = rows[0][1]
    x = np.abs(first.voltage)
    apex = int(x.argmax())
    fig,axes = plt.subplots(1,3,figsize=(13.8,4.4),layout="constrained")
    observed = np.abs(np.stack([r.current for _,r,_,_ in rows]))
    mask = np.stack([r.masks["use_unclamped"] for _,r,_,_ in rows])
    for ax, ids,title in zip(axes[:2],(np.arange(apex+1),np.arange(apex,len(x))),
                              ("RESET outward","RESET return (right to left)")):
        ids = ids[mask[:,ids].any(axis=0)]
        q = np.nanquantile(np.where(mask[:,ids],observed[:,ids],np.nan),[.1,.5,.9],axis=0)*1e6
        ax.fill_between(x[ids],q[0],q[2],color=".85",label="Measured p10–p90")
        ax.plot(x[ids],q[1],color=".14",label="Measured median")
        ax.plot(x[ids],base[0][ids]*1e6,"--",color="#486d93",label="Previous kinetics")
        ax.plot(x[ids],delayed[0][ids]*1e6,color="#a12f3e",label="Delayed-onset candidate")
        ax.set(title=title,yscale="log",xlabel="Reverse terminal voltage magnitude (V)",
               ylabel="Current magnitude (µA)")
    axes[0].legend(fontsize=7)
    ax = axes[2]
    ids = np.flatnonzero(np.sum(np.isfinite(all_states),axis=0)>=5)
    q = 10**np.nanquantile(all_states[:,ids],[.1,.5,.9],axis=0)
    ax.fill_between(x[ids],q[0],q[2],color=".85")
    ax.plot(x[ids],q[1],color=".15",label="Inferred from I–V")
    ax.plot(x[:apex+1],10**base[1][:apex+1],"--",color="#486d93",label="Previous kinetics")
    ax.plot(x[:apex+1],10**delayed[1][:apex+1],color="#a12f3e",label="Delayed-onset candidate")
    ax.scatter([0.,x[apex]],10**np.array([np.median(starts),np.median(ends)]),
               color=".1",marker="D",label="Low-bias endpoint estimates",zorder=5)
    ax.set(title="Outward conductance state",yscale="log",xlabel="Reverse terminal voltage magnitude (V)",
           ylabel="Conductance at local 0.15 V (µS)")
    ax.legend(fontsize=7)
    for panel in axes:
        panel.grid(alpha=.2)
    fig.suptitle("1.3 V gate held out | RESET prediction and state diagnostic")
    fig.savefig(path,dpi=185)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--readout",type=Path,required=True)
    ap.add_argument("--set-results",type=Path,required=True)
    ap.add_argument("--endpoint-weight",type=float,default=0.)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--figure",type=Path,required=True)
    args = ap.parse_args()
    if not np.isfinite(args.endpoint_weight) or args.endpoint_weight < 0.:
        ap.error("--endpoint-weight must be finite and nonnegative")
    prior = json.loads(args.readout.read_text())
    write = json.loads(args.set_results.read_text())["results"]
    beta = prior["transport_beta_per_sqrt_V"]
    previous = [np.log(prior["eraser_rate_at_0p5V_per_V"]),prior["eraser_field_coefficient_per_V"]]
    pairs = cy.data()
    train = {.9:pairs[.9][:70],1.1:pairs[1.1][:32]}
    test = {.9:pairs[.9][70:],1.1:pairs[1.1][32:],1.3:pairs[1.3]}
    ends = [s for rows in train.values() for _,r,_,rc in rows if (s:=ar.measured_end(r,rc)) is not None]
    floor = float(np.quantile(ends,.1))

    def transport(v,s,_,__):
        return 1e-6*10**s*response(v,beta)

    fit_groups = []
    for rows in train.values():
        known = [(r,s) for _,r,_,rc in rows if (s:=cy.observed_state(r,rc)) is not None]
        mask = np.stack([r.masks["use_unclamped"] for r,_ in known])
        current = np.log10(np.abs(np.stack([r.current for r,_ in known])[mask]))
        endpoints = np.array([np.nan if (end:=ar.measured_end(r,rows[0][3])) is None else end
                              for r,_ in known])
        fit_groups.append((ar.ResetRamp(rows,None,None,.02,transport),
                           [s for _,s in known],mask,current,endpoints))

    def residual(p):
        chunks = []
        for ramp,states,mask,observed,endpoints in fit_groups:
            trace,end = ramp.predict(states,floor,p,rate_law=delayed_rate(p))
            chunks.append((np.log10(trace[mask])-observed)/np.sqrt(len(observed)))
            keep = np.isfinite(endpoints)
            if args.endpoint_weight and np.any(keep):
                chunks.append(np.sqrt(args.endpoint_weight/np.sum(keep))*(end[keep]-endpoints[keep]))
        return np.concatenate(chunks)

    fits = [least_squares(residual,p,bounds=([-7.,0.],[9.,1.2]),diff_step=1e-4,
                         ftol=1e-5,max_nfev=100) for p in ([2.,.15],[4.,.65])]
    best = min((fit for fit in fits if fit.success),key=lambda fit:fit.cost)
    result = {}
    for g,rows in test.items():
        start = np.log10(write[str(g)]["entry_predicted_uS"])
        ramp = ar.ResetRamp(rows,None,None,.01,transport)
        first = tuple(a[0] for a in ramp.predict([start],floor,previous,return_trajectory=True))
        second = tuple(a[0] for a in ramp.predict(
            [start],floor,best.x,rate_law=delayed_rate(best.x),return_trajectory=True))
        fine,_ = ar.ResetRamp(rows,None,None,.005,transport).predict(
            [start],floor,best.x,rate_law=delayed_rate(best.x))
        actual = np.abs(np.stack([r.current for _,r,_,_ in rows]))
        eligible = np.stack([r.masks["use_unclamped"] for _,r,_,_ in rows])
        outward = np.arange(len(ramp.wave))<=ramp.apex
        scores = {}
        for key,part in (("full",np.ones(len(outward),bool)),("outward",outward),("return",~outward),
                         ("platform",outward&(ramp.wave>=.8)&(ramp.wave<=1.35))):
            mask = eligible & part
            scores[key] = {name:rd.rmse(np.log10(np.broadcast_to(item[0],actual.shape)[mask]/actual[mask]))
                           for name,item in (("previous",first),("delayed",second))}
        diagnostic,states,a,b = diagnose(rows,beta)
        keep = eligible.any(axis=0)
        assert np.all(np.diff(second[1][:ramp.apex+1]) <= 1e-10)
        result[str(g)] = {"scores_decade":scores,"state_diagnostic":diagnostic,
                          "end_prediction_uS":{"previous":float(10**first[1][-1]),
                                               "delayed":float(10**second[1][-1])},
                          "matched_end_observed_uS":float(10**np.median(b)),
                          "max_refinement_delta_decade":float(np.max(np.abs(
                              np.log10(second[0][keep]/fine[0,keep]))))}
        if g == 1.3:
            args.figure.parent.mkdir(parents=True,exist_ok=True)
            render(args.figure,rows,first,second,states,a,b)
    report = {"protocol":"Frozen SET inputs and field-assisted readout. Fit delayed loss on identical training cases as the previous activated law, with cycles 1–70 at 0.9 V and 4–35 at 1.1 V; 1.3 V entirely excluded. All test forecasts use the predicted SET starting state. Inferred conductance uses target current and deembedded voltage and is therefore a diagnostic. Rate is per voltage path, not per second.",
              "local_onset_V":float(best.x[1]),"rate_coefficient_per_V2":float(np.exp(best.x[0])),
              "training_cost":float(best.cost),"endpoint_weight":args.endpoint_weight,"results":result}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
