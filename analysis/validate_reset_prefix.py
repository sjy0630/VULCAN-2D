"""Compare fixed and rolling held-out RESET forecasts at explicit read cutoffs."""
import argparse
import json
from pathlib import Path

import numpy as np

from analysis import refit_iv41 as refit
from analysis import validate_programming_bridge as bridge
from analysis.fit_reset_population import load_data
from vulcan2d import reset_population
from vulcan2d import reset_prefix as prefix

FIRST = 40
CUTS = (.23, .45, .85, 1.25, 1.50)
OPTIONS = (("neighbors", 6.), ("neighbors", 10.), ("linear", .025),
           ("linear", .15), ("quadratic", .15))
CROSS_OPTIONS = (("neighbors", 3.), ("neighbors", 6.), ("neighbors", 10.),
                 ("linear", .025), ("linear", .15))


def cohort(data, gate, cutoff):
    raw = data[gate]["RESET"][1]
    wave = raw[0].voltage
    v = np.abs(wave)
    a = int(np.argmax(v))
    outward = np.arange(len(v)) <= a
    known = outward & (v <= cutoff+1e-9)
    future = ~known
    transistor = refit.circuits(gate)[1].transistor
    x = np.stack([prefix.features(r.voltage[known],r.current[known],transistor,cutoff) for r in raw])
    y = refit.log_i(np.stack([r.current for r in raw]))
    fit_y = refit.log_i(np.stack([r.current for r in data[gate]["RESET"][0]]))
    mask = np.stack([r.masks["use_unclamped"] for r in raw]) & future
    return dict(x=x, y=y, fit_y=fit_y, mask=mask, wave=wave, outward=outward,
                cap=np.interp(v,*transistor), known=known)


def observations(g):
    return (g["x"],g["fit_y"],g["mask"])


def forecast(fitted, x, cap):
    result = prefix.predict(fitted,x)
    return np.minimum(result,refit.log_i(cap))


def region_errors(predicted, observed, mask):
    n=mask.sum(axis=1)
    mse=np.divide((((predicted-observed)**2)*mask).sum(axis=1),n,
                  out=np.full(len(n),np.nan),where=n>0)
    return mse


def scores(predicted, observed, mask, outward):
    regions = {"future": mask, "outward": mask & outward, "return": mask & ~outward}
    result = {}
    for name, keep in regions.items():
        e=region_errors(predicted, observed,keep)
        valid=np.isfinite(e)
        result[name] = {"rmse_decade": float(np.sqrt(np.mean(e[valid]))) if valid.any() else None,
                        "scans": int(valid.sum()), "points": int(np.sum(keep))}
    return result


def validated_choice(g):
    x,y,m = observations(g)
    ranking={}
    for option in OPTIONS:
        best=[]
        for a,b in ((20,30),(30,FIRST)):
            model=prefix.fit(x[:a],y[:a],m[:a],*option)
            pred=forecast(model,x[a:b],g["cap"])
            best.extend(region_errors(pred,g["y"][a:b],m[a:b]))
        ranking[option]=float(np.nanmean(best))
    return min(ranking,key=ranking.get)


def same_gate(g, conditioned, option, policy):
    x,y,m=observations(g)
    blocks = ((FIRST, len(x)),) if policy=="fixed" else ((FIRST,60),(60,80),(80,len(x)))
    estimates, simple = [], []
    for first,last in blocks:
        fit=prefix.fit(x[:first],y[:first],m[:first],*option)
        estimates.append(forecast(fit,x[first:last],g["cap"]))
        simple.append(np.broadcast_to(fit["mean"],y[first:last].shape))
    new, base = np.concatenate(estimates),np.concatenate(simple)
    unknown=g["y"][FIRST:]
    valid=m[FIRST:]
    return {
        k:scores(q,unknown,valid,g["outward"]) for k,q in
        (("conditioned",new),("same_gate_mean",base),
         ("physical_unconditional",np.broadcast_to(conditioned,unknown.shape)))}


def fit_transfer(cohorts, gates, option):
    train=[cohorts[a] for a in gates]
    x,y,m=[np.concatenate(parts) for parts in zip(*(observations(a) for a in train))]
    return prefix.fit(x,y,m,*option)


def select_transfer(cohorts):
    anchors=bridge.PROGRAM_GATES
    ranking={}
    for option in CROSS_OPTIONS:
        dev=[]
        for g in anchors:
            h=cohorts[g]
            fitted=fit_transfer(cohorts,tuple(a for a in anchors if a != g),option)
            estimated=forecast(fitted,h["x"],h["cap"])
            dev.append(scores(estimated,h["y"],h["mask"],h["outward"]))
        ranking[option]=aggregate(dev)
    option=min(ranking,key=ranking.get)
    return fit_transfer(cohorts,anchors,option), option, {str(k):v for k,v in ranking.items()}


def cross_gate(cohorts, g, physical, fitted, start=FIRST):
    h = cohorts[g]
    if any(not np.array_equal(cohorts[a]["wave"],h["wave"]) for a in bridge.PROGRAM_GATES):
        raise ValueError("Cross-gate comparison requires a common measured voltage grid")
    target=h["x"][start:]
    pred=forecast(fitted,target,h["cap"])
    return {k:scores(v,h["y"][start:],h["mask"][start:],h["outward"])
            for k,v in (("conditioned",pred),
                        ("physical_unconditional",np.broadcast_to(physical,pred.shape)))}


def aggregate(per_gate):
    future=[r["future"] for r in per_gate]
    return float(np.sqrt(np.average([r["rmse_decade"]**2 for r in future],
                                    weights=[r["scans"] for r in future])))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook",type=Path)
    parser.add_argument("physical",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--cache",type=Path,required=True)
    args=parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    index=refit.origin_cache(args.workbook,args.cache)
    raw=load_data(index,args.cache)
    physical=json.loads(args.physical.read_text())
    baseline={}
    for g in bridge.ASSESS_GATES:
        v,i,_=reset_population.predict(physical,g)
        target=raw[g]["RESET"][1][0].voltage
        baseline[g]=refit.align(target,v,refit.log_i(i))
    all_results={}
    for cutoff in CUTS:
        data={g:cohort(raw,g,cutoff) for g in (*bridge.PROGRAM_GATES,*bridge.ASSESS_GATES)}
        transferred, transfer_option, selection=select_transfer(data)
        if cutoff in (.23, 1.5):
            np.savez_compressed(args.output/f"cross_gate_prefix_{cutoff:.2f}V.npz", **transferred,
                                wave=data[1.1]["wave"],cutoff=cutoff,
                                minimum_gate=min(data),maximum_gate=max(data))
        outcome={}
        for g in bridge.ASSESS_GATES:
            h=data[g]
            option=validated_choice(h)
            fixed=same_gate(h,baseline[g],option,"fixed")
            rolling=same_gate(h,baseline[g],option,"rolling")
            plain=same_gate(h,baseline[g],("neighbors",6.),"rolling")
            strict=cross_gate(data,g,baseline[g],transferred)
            full=cross_gate(data,g,baseline[g],transferred,start=0)
            outcome[g]=dict(method=option,fixed=fixed,rolling=rolling,
                            rolling_fixed_rule=plain,cross_gate=strict,cross_gate_full=full)
            print(cutoff,g,option,*[round(x["conditioned"]["future"]["rmse_decade"],4)
                                    for x in (fixed,rolling,plain,strict,full)],flush=True)
        policies=("fixed","rolling","rolling_fixed_rule","cross_gate","cross_gate_full")
        first=next(iter(outcome.values()))
        all_results[str(cutoff)]={"gates":outcome,"cross_gate_selected":transfer_option,
            "anchor_selection_rmse_decade":selection,
            "pooled_rmse_decade": {
                policy:{name:aggregate([r[policy][name] for r in outcome.values()]) for name in first[policy]}
                for policy in policies
            }}
    report={"source_sha256":refit.sha_file(args.workbook),"prior_sha256":refit.sha_file(args.physical),
            "first_same_gate_training_scans":FIRST,"results":all_results}
    (args.output/"reset_prefix_validation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
