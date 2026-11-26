"""Validate fixed-state SET transport with disjoint per-sweep voltage windows."""
import argparse
import json
from pathlib import Path

import numpy as np

from analysis import audit_iv41_high_conductance as reads
from analysis import refit_iv41 as iv
from analysis import validate_set_current_bottleneck as bottleneck


REF = .5
GATES = (.9,1.,1.1,1.2,1.3,1.4,1.5)
R_GRID = (0.,250.,500.,1000.,2000.,3000.,5000.,8000.,23000.)
ALPHA_GRID = (0.,.005,.015,.05)
SLOPES = np.linspace(.15,18.,150)
WINDOWS = {".58–.80":(.58,.80),".80–1.00":(.80,1.00),"1.00–1.20":(1.,1.20)}


def read_curve(row, circuit, outward):
    g = reads.local_estimate(row,circuit,outward,REF)
    label = reads.terminal_estimate(row,False,.15)
    if g is None or label is None:
        return None
    ids,v,i = reads.reader.intrinsic(row,circuit,outward)
    ids = ids[(v[ids]>=.30)&(v[ids]<=1.20)]
    vol,current = v[ids],i[ids]
    prefix = (vol>=.30)&(vol<=.48)
    future = vol>=.58
    if prefix.sum()<6 or not np.any(future):
        return None
    return {"voltage":vol,"current":current,"prefix":prefix,"future":future,
            "anchor":g*1e-6*REF,"high":label>30.}


def series(v,slope,anchor,R):
    """Invert V = asinh(I/A)/beta + RI for constant state."""
    eps = REF-R*anchor
    if eps<=0:
        return None
    a = anchor/np.sinh(slope*eps)
    upper = a*np.sinh(slope*v)
    if R:
        upper = np.minimum(upper,v/R)
    lo = np.zeros_like(upper)
    for _ in range(26):
        mid = (lo+upper)/2
        is_low = np.arcsinh(mid/a)/slope+R*mid<v
        lo,upper = np.where(is_low,mid,lo),np.where(is_low,upper,mid)
    return (lo+upper)/2


def predict(record,kind,param,common_slope=None):
    v = record["voltage"]
    anchor = record["anchor"]
    pre = record["prefix"]
    observed = record["current"]
    slopes = SLOPES[:,None] if common_slope is None else np.array([[common_slope]])
    if kind=="series":
        predicted = series(v[None,:],slopes,anchor,param)
    else:
        predicted = anchor*bottleneck.limited_ratio(v[None,:],slopes,param)
    if predicted is None:
        return None
    err = np.log10(predicted[:,pre]/observed[None,pre])
    loss = np.mean(np.where(abs(err)<.15,.5*err**2,.15*(abs(err)-.075)),axis=1)
    return predicted[int(np.argmin(loss))]


def evaluate(records,spec):
    kind,param,common = spec
    fail = 0
    bins = {k:[] for k in ("all",*WINDOWS)}
    for record in records:
        pred = predict(record,kind,param,common)
        if pred is None:
            fail += 1
            continue
        voltage,obs = record["voltage"],record["current"]
        err = np.log10(pred/obs)
        bins["all"].extend(err[record["future"]])
        for key,(lo,hi) in WINDOWS.items():
            bins[key].extend(err[(voltage>=lo)&(voltage<=hi)])
    return {"scans":len(records),"unrepresentable_anchors":fail,
            "windows":{k:{"samples":len(v),"rmse_decade":float(np.sqrt(np.mean(np.square(v)))) if v else None,
                         "median_log_error":float(np.median(v)) if v else None} for k,v in bins.items()}}


def select(train):
    # Candidate shape is fitted from the low voltage prefix; select common
    # physical parameters by predicting higher voltage SET-return points.
    rows = [r for _,rec in train.items() for r in rec[False]]
    Rmax = min(REF/r["anchor"] for r in rows)
    valid_R = [R for R in R_GRID if R<Rmax]
    series_scores = {R:evaluate(rows,("series",R,None))["windows"]["all"]["rmse_decade"] for R in valid_R}
    cap_scores = {a:evaluate(rows,("finite",a,None))["windows"]["all"]["rmse_decade"] for a in ALPHA_GRID}
    return {"one_shape":("series",0.,7.696509584641415),
            "prefix_shape":("series",0.,None),
            "shared_series_resistance":("series",min(series_scores,key=series_scores.get),None),
            "finite_transport_rate":("finite",min(cap_scores,key=cap_scores.get),None)}, {
                "train_return_R_upper_ohm":Rmax,"series_training_rmse":series_scores,
                "finite_training_rmse":cap_scores}


def plot(path,test,spec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.linspace(.30,1.20,91)
    fig,axes = plt.subplots(2,2,figsize=(9.5,6.7),layout="constrained")
    items = ((.9,True,False),(1.5,True,False),(1.4,False,True),(1.5,False,True))
    for ax,(gate,outward,high) in zip(axes.flat,items):
        rows = [r for r in test[gate][outward] if r["high"]==high]
        actual,series_models,limited = [],[],[]
        for row in rows:
            order = np.argsort(row["voltage"])
            vv = row["voltage"][order]
            def grid_curve(x):
                return np.interp(grid,vv,x[order]/row["anchor"],left=np.nan,right=np.nan)
            actual.append(grid_curve(row["current"]))
            for name,target in (("shared_series_resistance",series_models),("finite_transport_rate",limited)):
                candidate = predict(row,*spec[name])
                if candidate is not None:
                    target.append(grid_curve(candidate))
        for data,color,style,label in ((actual,".12","-","Measured median"),
                                        (series_models,"#486d93","--","Gap + common series resistance"),
                                        (limited,"#a1263d","-","Finite transport rate")):
            values = np.stack(data)
            ok = np.isfinite(values).sum(axis=0)>=max(3,len(values)//5)
            ax.plot(grid[ok],np.nanmedian(values[:,ok],axis=0),style,color=color,label=label)
        ax.axvspan(.30,.50,color=".90",label="Input and anchor")
        ax.axvline(.58,color=".5",ls=":",label="Scoring begins")
        ax.set(yscale="log",xlabel="hBN voltage after removing transistor drop (V)",
               ylabel="Current / current at 0.5 V",
               title=f"{gate:g} V gate · {'outward' if outward else 'return'} · {'higher' if high else 'lower'} written state")
        ax.grid(alpha=.19)
    axes[0,0].legend(fontsize=7)
    fig.suptitle("SET readout check | low-bias input, disjoint voltage prediction")
    fig.savefig(path,dpi=175)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("cache","output","figure"):
        p.add_argument("--"+k,type=Path,required=True)
    args=p.parse_args()
    idx=json.loads((args.cache/"index.json").read_text())
    train,test={},{}
    for g in GATES:
        cohort,_=iv.load_cohort(idx,args.cache,round((g-.9)*10)+5,g)
        circuit,_=iv.circuits(g)
        rows=cohort["SET"]
        cut=int(.7*len(rows)) if g<=1.3 else 0
        train[g]={o:[r for row in rows[:cut] if (r:=read_curve(row,circuit,o)) is not None]
                  for o in (True,False)}
        test[g]={o:[r for row in rows[cut:] if (r:=read_curve(row,circuit,o)) is not None]
                  for o in (True,False)}
    spec,fit=select(train)
    result={}
    for g,d in test.items():
        result[str(g)]={}
        for outward,rows in d.items():
            key="outward" if outward else "return"
            result[str(g)][key]={}
            for label,part in (("all",rows),("lower",[r for r in rows if not r["high"]]),
                               ("higher",[r for r in rows if r["high"]])):
                if not part:
                    continue
                result[str(g)][key][label]={name:evaluate(part,model) for name,model in spec.items()}
            result[str(g)][key]["R_ceiling_ohm_for_all_anchors"]=min(REF/r["anchor"] for r in rows)
    output={"protocol":"Deembedded hBN positive SET voltage. Per-trace shape uses only 0.30–0.48 V; current normalization uses hBN 0.5 V. Score separate points at 0.58–1.20 V. Pick common transport parameters on lower-gate training SET return only; split lower-gate scans by workbook row order and hold out 1.4/1.5 V entirely from common fitting. SET return is a presumed fixed-state comparison, not independently verified. SET outward can already include state evolution, so its results are a consistency check rather than fixed-state ground truth. Finite-rate expression is physically motivated, not a microscopic identification.",
            "selected":spec,"fit":fit,"results":result}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.figure.parent.mkdir(parents=True,exist_ok=True)
    plot(args.figure,test,spec)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps(output,indent=2))


if __name__=="__main__":
    main()
