"""Calibrate gate-only current corrections and validate the entire IV41 cycle."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from analysis import refit_iv41 as refit
from analysis import validate_programming_bridge as bridge
from analysis.fit_reset_population import load_data
from vulcan2d import gate_only as model

PHASES = model.PHASES
METHODS = (("local_delta",0.),("local_delta",.5),("local_bounded_delta",0.))


def typical(y, eligible, voltage):
    """Estimate the log-current mean with equal weight per measured sweep."""
    weight = eligible/np.maximum(eligible.sum(axis=1),1)[:,None]
    w = weight.sum(axis=0)
    mean = np.divide(np.sum(weight*y,axis=0),w,out=np.zeros(len(voltage)),where=w>0)
    a = int(np.argmax(np.abs(voltage)))
    for limb in (np.arange(a+1),np.arange(a+1,len(voltage))):
        known=limb[w[limb]>0]
        if not len(known):
            raise ValueError("No valid observations on a sweep limb")
        mean[limb]=gaussian_filter1d(np.interp(limb,known,mean[known]),1.5)
    return mean


def summarize_anchors(data, waveforms, physics):
    profiles, original = {}, {}
    for gate, phases in data.items():
        circuits = refit.circuits(gate)
        original[gate] = model.original(physics,gate,waveforms,circuits)
        if gate not in bridge.PROGRAM_GATES:
            continue
        profiles[gate]={}
        for i,phase in enumerate(PHASES):
            filtered, raw = phases[phase]
            current = np.abs(np.stack([r.current for r in filtered]))
            mask = np.stack([r.masks["use_unclamped"] for r in raw])
            v=waveforms[phase]
            cv=circuits[i].transistor
            terminal = typical(model.log_abs(current),mask,v)
            local,ok = model.apparent_log_conductance(v,current,cv)
            mean_conductance = typical(local,mask&ok,v)
            base_log_i=original[gate][phase]
            base_conductance=model.apparent_log_conductance(v,10**base_log_i,cv)[0]
            profiles[gate][phase] = {
                "local_mean":mean_conductance.tolist(), "terminal_mean":terminal.tolist(),
                "local_delta":(mean_conductance-base_conductance).tolist(),
                "terminal_delta":(terminal-base_log_i).tolist()}
    return profiles, original


def select(profiles, original, data, waves):
    result, diagnostics={},{}
    for index, phase in enumerate(PHASES):
        scores={}
        for method,edge in METHODS:
            errors=[]
            for gate in profiles:
                learn={g:r[phase] for g,r in profiles.items() if g!=gate}
                estimate=model.current_for_method(waves[phase],refit.circuits(gate)[index],
                    original[gate][phase],learn,gate,method,edge)
                errors.append(refit.error_scores(data[gate][phase][1],phase,estimate)["valid"])
            scores[method,edge] = float(np.sqrt(np.mean(np.square(errors))))
        best=min(scores.values())
        chosen=next(rule for rule,value in scores.items() if value <= best*1.002)
        result[phase] = chosen
        diagnostics[phase] = {f"{name}_{edge:g}":v for (name,edge),v in scores.items()}
    return result,diagnostics


def figure(path, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes=plt.subplots(4,2,figsize=(10.,12.),squeeze=False)
    colors={"measured median":"#25496b","gate-only forecast":"#087d64","previous physical":"#777"}
    for row,(gate,phases) in enumerate(curves.items()):
        for col,(phase,case) in enumerate(phases.items()):
            ax=axes[row,col]
            wave,obs,new,previous=case
            a=int(np.argmax(abs(wave)))
            ids=np.flatnonzero(abs(wave)>1e-12)
            for branch, style in ((ids[ids<=a],"-"),(ids[ids>=a],"--")):
                for (label,color),y in zip(colors.items(),(obs,new,previous)):
                    ax.plot(abs(wave[branch]),y[branch]+6,style,color=color,lw=1.55)
            ax.set_title(f"{gate:g} V gate  |  {phase}",loc="left",weight="bold")
            ax.set_xlabel("Terminal voltage magnitude (V)")
            ax.set_ylabel(r"$\log_{10} (|I| / \mu\mathrm{A})$")
            ax.grid(alpha=.15)
    handles=[Line2D([],[],color=color,label=label) for label,color in colors.items()]
    handles.append(Line2D([],[],color="#555",linestyle="--",label="Return sweep"))
    fig.legend(handles=handles,loc="lower center",ncols=4,frameon=False)
    fig.tight_layout(rect=(0,.035,1,1))
    fig.savefig(path,dpi=160)
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook",type=Path)
    parser.add_argument("prior",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--cache",type=Path,required=True)
    args=parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    data=load_data(refit.origin_cache(args.workbook,args.cache),args.cache)
    waves={p:data[1.1][p][1][0].voltage for p in PHASES}
    if any(not np.array_equal(waves[p],r.voltage)
           for phases in data.values() for p in PHASES for r in phases[p][1]):
        raise ValueError("Using one voltage protocol requires aligned voltage grids")
    prior={"physical_prior":json.loads(args.prior.read_text())}
    anchors,base=summarize_anchors(data,waves,prior)
    chosen,dev=select(anchors,base,data,waves)
    cfg={"schema_version":1, **prior,
         "validated_gate_range_v":[min(data),max(data)],
         "selected":chosen, "waveforms":{p:v.tolist() for p,v in waves.items()},
         "anchors":anchors}
    metrics, curves = {}, {}
    for gate in bridge.ASSESS_GATES:
        phase_metrics, phase_curves={},{}
        predictions=model.predict(cfg,gate)
        for index,p in enumerate(PHASES):
            raw=data[gate][p][1]
            v,current=predictions[p]
            old=10**base[gate][p]
            empirical=model.current_for_method(v,refit.circuits(gate)[index],base[gate][p],
                         {g:r[p] for g,r in anchors.items()},gate,"terminal_mean",0.)
            cases={k:refit.error_scores(raw,p,y) for k,y in
                   (("prior",old),("new",current),("terminal_linear",empirical))}
            cases["oracle_floor"]=refit.deterministic_floor(raw,p)
            phase_metrics[p]=cases
            usable = np.stack([np.where(r.masks["use_unclamped"],model.log_abs(r.current),np.nan) for r in raw])
            ok=np.any(np.isfinite(usable),axis=0)
            summary=np.full(len(v),np.nan)
            summary[ok]=np.nanmedian(usable[:,ok],axis=0)
            phase_curves[p]=(v,summary,model.log_abs(current),base[gate][p])
        metrics[gate],curves[gate]=phase_metrics,phase_curves
        print(gate,{p:{k:round(s.get("valid",s.get("valid_rmse_decade")),4)
                      for k,s in m.items()} for p,m in phase_metrics.items()},flush=True)
    means={}
    for key in ("prior","new","terminal_linear","oracle_floor"):
        phases={}
        for phase in PHASES:
            scores=[r[phase][key] for r in metrics.values()]
            values=[r.get("valid",r.get("valid_rmse_decade")) for r in scores]
            phases[phase]=float(np.sqrt(np.average(np.square(values),weights=[r["n"] for r in scores])))
        phases["equal_phase_cycle"]=float(np.sqrt(np.mean(np.square(list(phases.values())))))
        means[key]=phases
    model_error={k:{phase:float(np.sqrt(max(0., means[k][phase]**2-means["oracle_floor"][phase]**2)))
                    for phase in (*PHASES,"equal_phase_cycle")}
                 for k in ("prior","new","terminal_linear")}
    result={"workbook_sha256":refit.sha_file(args.workbook),"prior_sha256":refit.sha_file(args.prior),
            "anchor_gates":list(anchors),"holdout_gates":list(metrics),"selected":chosen,
            "anchor_residual_validation":dev,"assessment":metrics,"pooled":means,
            "systematic_center_error":model_error}
    (args.output/"gate_only_calibration.json").write_text(json.dumps(cfg,allow_nan=False)+"\n")
    (args.output/"gate_only_results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    figure(args.output/"Vulcan_gate_only_SET_RESET.png",curves)
    print("Selected",chosen,means,flush=True)


if __name__ == "__main__":
    main()
