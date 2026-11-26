"""Distinguish high-conductance cohorts from a load-line deembedding artifact."""
import argparse
import json
from pathlib import Path

import numpy as np

from analysis import audit_log_soft as reader
from analysis import refit_iv41 as iv


GATES = (1.3,1.4,1.5)
VREF = .15


def local_estimate(row,circuit,outward,reference):
    ids,v,i = reader.intrinsic(row,circuit,outward=outward)
    for first,second in zip(ids[:-1],ids[1:]):
        if second!=first+1:
            continue
        a,b = sorted((first,second),key=lambda k:v[k])
        if v[a]<=reference<=v[b] and v[b]>v[a]:
            return float(np.interp(reference,v[[a,b]],i[[a,b]])/reference*1e6)
    return None


def terminal_estimate(row,outward,reference):
    v = np.abs(row.voltage)
    mask = (row.masks["branch_outward"]==outward) & row.masks["eligible"]
    ids = np.flatnonzero(mask)
    for first,second in zip(ids[:-1],ids[1:]):
        if second!=first+1:
            continue
        a,b = sorted((first,second),key=lambda k:v[k])
        if v[a]<=reference<=v[b] and v[b]>v[a]:
            return float(np.interp(reference,v[[a,b]],np.abs(row.current[[a,b]]))/reference*1e6)
    return None


def distribution(values):
    use = [v for v in values if v is not None]
    return {"n":len(use),"p10_median_p90":np.quantile(use,[.1,.5,.9]).tolist() if use else None,
            "count_above_30uS":sum(v>30. for v in use),"count_above_100uS":sum(v>100. for v in use)}


def diagnostics(rows,circuit,outward):
    values = {}
    records = []
    for row in rows:
        x = {f"{scope}_{v:.2f}V": estimate(row,circuit,outward,v) if scope=="local"
             else estimate(row,outward,v) for scope,estimate in (
                ("local",local_estimate),("terminal",terminal_estimate))
             for v in (.10,.15,.20)}
        records.append(x)
    for key in records[0]:
        values[key] = [row[key] for row in records]
    shape = {"high":[],"other":[]}
    for row in records:
        g,a,b = (row[f"local_{v:.2f}V"] for v in (.15,.10,.20))
        if g is not None and a is not None and b is not None:
            shape["high" if g>30. else "other"].append(b/a)
    return values,{"sweeps":len(rows),"conductance_uS":{k:distribution(v) for k,v in values.items()},
                   "G_at_0p20V_over_G_at_0p10V":{
                       k:{"n":len(v),"p10_median_p90":np.quantile(v,[.1,.5,.9]).tolist() if v else None}
                       for k,v in shape.items()}}


def draw(path, all_values):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(9,6.7),layout="constrained")
    for j,phase in enumerate(("SET return","RESET entry")):
        for k,scope in enumerate(("terminal","local")):
            ax = axes[j,k]
            for g,color in zip(GATES,("#486d93","#a1263d","#4b8266")):
                x = np.sort([v for v in all_values[g][phase][f"{scope}_0.15V"] if v is not None])
                ax.step(x,np.arange(1,len(x)+1)/len(x),where="post",label=f"{g:g} V gate (n={len(x)})",color=color)
            ax.set(xscale="log",ylim=(0,1.03),title=f"{phase} | {scope} 0.15 V",
                   xlabel="Effective conductance (µS)",ylabel="Fraction of recorded sweeps")
            ax.grid(alpha=.2)
    axes[0,0].legend(loc="best",fontsize=8)
    fig.suptitle("IV41 | Low-voltage conductance before and after branch reversal")
    fig.savefig(path,dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache","output","figure"):
        parser.add_argument("--"+name,required=True,type=Path)
    args = parser.parse_args()
    index = json.loads((args.cache/"index.json").read_text())
    result,raw = {},{}
    for gate in GATES:
        cohort,rejected = iv.load_cohort(index,args.cache,round((gate-.9)*10)+5,gate)
        sc,rc = iv.circuits(gate)
        raw[gate],result[str(gate)] = {},{"curve_screen":rejected}
        for phase,key,circuit,outward in (("SET","SET return",sc,False),("RESET","RESET entry",rc,True)):
            raw[gate][key],result[str(gate)][key] = diagnostics(cohort[phase],circuit,outward)
    output = {"protocol":"Separate cohorts, without assuming one-to-one SET–RESET pairing. Conductance is magnitude I/V at a common voltage. Terminal checks use eligible raw readings without load-line correction; local uses the measured/interpolated transistor characteristic and excludes points near the estimated transistor cap. The 30 and 100 uS levels are descriptive cutoffs and do not identify a mechanism.",
              "results":result}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.figure.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    draw(args.figure,raw)
    for gate,parts in result.items():
        for key in ("SET return","RESET entry"):
            item = parts[key]
            for cut in ("terminal_0.15V","local_0.15V"):
                print(gate,key,cut,item["conductance_uS"][cut])
            print("shape",item["G_at_0p20V_over_G_at_0p10V"])


if __name__ == "__main__":
    main()
