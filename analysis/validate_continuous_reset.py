"""Compare one RESET state equation across both branches against an apex freeze."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from analysis import audit_log_soft as readout
from analysis import validate_activated_reset as activated
from analysis import validate_delayed_reset as diagnostics
from analysis import validate_log_soft_cycle as cycle


def predict(ramp, start, floor, params):
    current, state = ramp.predict([start], floor, params, return_trajectory=True)
    return current[0], state[0]


def compare(rows, ramp, curves):
    observed = np.abs(np.stack([r.current for _,r,_,_ in rows]))
    mask = np.stack([r.masks["use_unclamped"] for _,r,_,_ in rows])
    outward = np.arange(len(ramp.wave)) <= ramp.apex
    windows = {
        "full": np.ones(len(outward), dtype=bool),
        "outward": outward, "return": ~outward,
        "return_above_0p3V": ~outward & (ramp.wave >= .3),
    }
    return {key: {name:readout.rmse(np.log10(
            np.broadcast_to(curve[0],observed.shape)[mask&part]/observed[mask&part]))
            for name,curve in curves.items()} for key,part in windows.items()}


def render(path, rows, ramp, old, new, beta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1,3,figsize=(13.8,4.5),layout="constrained")
    volts = ramp.wave
    samples = np.arange(len(volts))
    actual = np.abs(np.stack([r.current for _,r,_,_ in rows]))
    mask = np.stack([r.masks["use_unclamped"] for _,r,_,_ in rows])
    for ax, part, title in zip(
            axes[:2],(samples<=ramp.apex,samples>=ramp.apex),
            ("RESET outward","RESET return (right to left)")):
        ids = np.flatnonzero(part & mask.any(axis=0))
        lo,mid,hi = np.nanquantile(np.where(mask[:,ids],actual[:,ids],np.nan),[.1,.5,.9],axis=0)*1e6
        ax.fill_between(volts[ids],lo,hi,color=".85",label="Measured p10–p90")
        ax.plot(volts[ids],mid,color=".15",label="Measured median")
        ax.plot(volts[ids],old[0][ids]*1e6,"--",color="#476e96",label="Frozen on return")
        ax.plot(volts[ids],new[0][ids]*1e6,color="#9b2e3e",label="Continuous RESET")
        ax.set(yscale="log",title=title,xlabel="Reverse terminal voltage magnitude (V)",
               ylabel="Current magnitude (µA)")
    axes[0].legend(fontsize=7)

    path_length = np.r_[0.,np.cumsum(np.abs(np.diff(volts)))]
    inferred,entry,ends = [],[],[]
    for _,r,_,rc in rows:
        s0,s1 = cycle.observed_state(r,rc),activated.measured_end(r,rc)
        if s0 is None or s1 is None:
            continue
        curve = np.full(len(volts),np.nan)
        for outward in (True,False):
            ids,local,current = readout.intrinsic(r,rc,outward)
            curve[ids] = np.log10(current[ids]/diagnostics.response(local[ids],beta)/1e-6)
        inferred.append(curve)
        entry.append(s0)
        ends.append(s1)
    inferred = np.asarray(inferred)
    ids = np.flatnonzero(np.sum(np.isfinite(inferred),axis=0)>=5)
    lo,mid,hi = 10**np.nanquantile(inferred[:,ids],[.1,.5,.9],axis=0)
    ax = axes[2]
    ax.fill_between(path_length[ids],lo,hi,color=".87")
    ax.plot(path_length[ids],mid,color=".15",label="Inferred from I–V")
    ax.plot(path_length,10**old[1],"--",color="#476e96",label="Frozen on return")
    ax.plot(path_length,10**new[1],color="#9b2e3e",label="Continuous RESET")
    ax.scatter(path_length[[0,-1]],10**np.median([entry,ends],axis=1),marker="D",
               color=".18",label="Low-bias endpoint estimates",zorder=5)
    ax.axvline(path_length[ramp.apex],color=".4",lw=.8,ls=":",label="Sweep reversal")
    ax.set(yscale="log",title="Conductance across both branches",xlabel="Cumulative sweep path (V)",
           ylabel="Conductance at local 0.15 V (µS)")
    ax.legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle("1.3 V gate held out | Shared current law and RESET kinetics")
    fig.savefig(path,dpi=185)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("readout","set-results","output","figure"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    prior = json.loads(args.readout.read_text())
    writes = json.loads(args.set_results.read_text())["results"]
    beta = prior["transport_beta_per_sqrt_V"]
    previous = [np.log(prior["eraser_rate_at_0p5V_per_V"]),prior["eraser_field_coefficient_per_V"]]
    pairs = cycle.data()
    train = {.9:pairs[.9][:70],1.1:pairs[1.1][:32]}
    test = {.9:pairs[.9][70:],1.1:pairs[1.1][32:],1.3:pairs[1.3]}
    endpoints = [s for rows in train.values() for _,r,_,rc in rows
                 if (s:=activated.measured_end(r,rc)) is not None]
    floor = np.quantile(endpoints,.1)

    def current(v,state,_,__):
        return 1e-6*10**state*diagnostics.response(v,beta)

    learning = {}
    for g,rows in train.items():
        data = [(r,s,activated.measured_end(r,rc)) for _,r,_,rc in rows
                if (s:=cycle.observed_state(r,rc)) is not None]
        mask = np.stack([r.masks["use_unclamped"] for r,_,_ in data])
        obs = np.log10(np.abs(np.stack([r.current for r,_,_ in data])[mask]))
        end = np.array([z if z is not None else np.nan for _,_,z in data])
        learning[g] = ([s for _,s,_ in data],mask,obs,end)

    fits = {}
    for evolve in (False,True):
        ramps = {g:activated.ResetRamp(rows,None,None,.02,current,evolve_return=evolve)
                 for g,rows in train.items()}

        def residual(params):
            parts = []
            for g,ramp in ramps.items():
                starts,mask,obs,ends = learning[g]
                prediction,ending = ramp.predict(starts,floor,params)
                parts.append((np.log10(prediction[mask])-obs)/np.sqrt(len(obs)))
                valid = np.isfinite(ends)
                parts.append((ending[valid]-ends[valid])/np.sqrt(np.sum(valid)))
            return np.concatenate(parts)

        candidates = [least_squares(residual,p,bounds=([-8.,0.],[9.,30.]),
                                    diff_step=1e-4,ftol=1e-5,max_nfev=120)
                      for p in ([0.,1.],[2.,7.])]
        fits[evolve] = min((p for p in candidates if p.success),key=lambda p:p.cost)

    scores = {}
    for g,rows in test.items():
        start = np.log10(writes[str(g)]["entry_predicted_uS"])
        frozen = activated.ResetRamp(rows,None,None,.01,current)
        whole = activated.ResetRamp(rows,None,None,.01,current,evolve_return=True)
        curves = {"previous":predict(frozen,start,floor,previous),
                  "continuous_same_parameters":predict(whole,start,floor,previous),
                  "frozen_refit":predict(frozen,start,floor,fits[False].x),
                  "continuous_refit":predict(whole,start,floor,fits[True].x)}
        assert np.allclose(curves["previous"][1][:frozen.apex+1],
                           curves["continuous_same_parameters"][1][:whole.apex+1],atol=1e-10)
        assert np.all(np.diff(curves["continuous_refit"][1])<=1e-10)
        tight = activated.ResetRamp(rows,None,None,.005,current,evolve_return=True)
        precision = predict(tight,start,floor,fits[True].x)
        valid = np.any([r.masks["use_unclamped"] for _,r,_,_ in rows],axis=0)
        observed = [s for _,r,_,rc in rows if (s:=activated.measured_end(r,rc)) is not None]
        scores[str(g)] = {"cycles":len(rows),"scores_decade":compare(rows,whole,curves),
                          "final_conductance_uS":{k:float(10**v[1][-1]) for k,v in curves.items()},
                          "apex_conductance_uS":{k:float(10**v[1][whole.apex]) for k,v in curves.items()},
                          "observed_end_median_uS":float(10**np.median(observed)),
                          "max_refinement_delta_decade":float(np.max(np.abs(np.log10(
                              precision[0][valid]/curves["continuous_refit"][0][valid]))))}
        if g == 1.3:
            args.figure.parent.mkdir(parents=True,exist_ok=True)
            render(args.figure,rows,whole,curves["frozen_refit"],curves["continuous_refit"],beta)
    result = {"protocol":"Keep the same SET forecasts and current law. Continuous state propagation follows cumulative |dVapp| during both RESET limbs. Compare a change with fixed coefficients and refit both variants to the same 0.9 and 1.1 V training current and low-bias ending conductance. 1.3 V remains entirely held out. Endpoint estimates used only for training and scoring. No measured time.",
              "background_uS":10**floor,
              "fit_parameters":{("continuous" if k else "frozen"):{"rate_at_local_0p5V_per_V":float(np.exp(v.x[0])),
                               "field_slope_per_V":float(v.x[1]),"cost":float(v.cost)} for k,v in fits.items()},
              "results":scores}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
