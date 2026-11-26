"""Score the shared hBN model on the four additional IV41 gate conditions."""
import argparse
import json
from pathlib import Path

import numpy as np

from analysis import refit_iv41 as iv
from analysis import validate_delayed_reset as reset_readout
from analysis import validate_field_log_soft as field
from analysis import validate_log_soft_cycle as cycle
from analysis import validate_set_outward as new_set
from vulcan2d import validate_shared_device as shared
from analysis import validate_activated_reset as reset


GATES = (1.,1.2,1.4,1.5)


def errors(rows, curve):
    residual = np.concatenate([cycle.log_errors(r,curve,r.masks["use_unclamped"]) for r in rows])
    return {"points":len(residual),"rmse_decade":float(np.sqrt(np.mean(residual**2)))}


def draw(path, panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig,axes = plt.subplots(4,2,figsize=(9.5,11),layout="constrained")
    for j,(gate,cohort,set_curve,reset_curve) in enumerate(panels):
        for col,(phase,pred) in enumerate((("SET",set_curve),("RESET",reset_curve))):
            ax = axes[j,col]
            rows = cohort[phase]
            v = np.abs(rows[0].voltage)
            masks = np.stack([r.masks["use_unclamped"] for r in rows])
            measured = np.abs(np.stack([r.current for r in rows]))
            outward = rows[0].masks["branch_outward"]
            for branch,style in ((outward,"-"),(~outward,"--")):
                ids = np.flatnonzero(branch & masks.any(axis=0))
                lo,mid,hi = np.nanquantile(np.where(masks[:,ids],measured[:,ids],np.nan),
                                          [.1,.5,.9],axis=0)*1e6
                ax.fill_between(v[ids],lo,hi,color=".87",alpha=.75)
                ax.plot(v[ids],mid,style,color=".15",lw=1.4)
                ax.plot(v[ids],pred[ids]*1e6,style,color="#9b2e3e",lw=1.45)
            ax.set(yscale="log",title=f"{gate:g} V gate · {phase} · {len(rows)} sweeps",
                   xlabel="Applied voltage magnitude (V)",ylabel="Current magnitude (µA)")
            ax.grid(alpha=.2)
    axes[0,0].legend(handles=[Line2D([],[],color=".15",label="Measured median"),
                      Patch(color=".87",label="Measured p10–p90"),
                      Line2D([],[],color="#9b2e3e",label="Frozen prediction"),
                      Line2D([],[],color=".4",ls="--",label="Return sweep")],
                      fontsize=7,ncols=2)
    fig.suptitle("IV41 | Four additional gate voltages, without refitting")
    fig.savefig(path,dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workbook","cache","revisions","output","figure"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    index = iv.origin_cache(args.workbook,args.cache)
    def result(name):
        return json.loads((args.revisions/name).read_text())
    set_fit = result("set_outward_refinement.json")
    field_fit = result("field_log_soft_validation.json")["field_model"]
    reset_fit = result("continuous_reset.json")
    beta = result("reset_platform_readout.json")["transport_beta_per_sqrt_V"]
    nominal = cycle.data()[1.1][:32]
    device,prep = shared.fit_soft_set([s for s,_,_,_ in nominal],nominal[0][2])
    form = set_fit["selected"]
    chosen = set_fit["fits"][form]
    initial,ratio = field_fit["initial_log_conductance"],field_fit["readout_positive_negative_ratio"]

    def forward(v,s):
        return new_set.currents(v,s,initial,ratio,device.soft_exponent,
                                chosen["background_parameters"],form)

    def reverse(v,s,_a,_b):
        return 1e-6*10**s*reset_readout.response(v,beta)

    reset_par = reset_fit["fit_parameters"]["continuous"]
    params = [np.log(reset_par["rate_at_local_0p5V_per_V"]),reset_par["field_slope_per_V"]]
    floor = np.log10(reset_fit["background_uS"])
    report,panels = {},[]
    for g in GATES:
        cohort,rejected = iv.load_cohort(index,args.cache,round((g-.9)*10)+5,g)
        sc,rc = iv.circuits(g)
        # Each phase uses its recorded common waveform; the single predicted
        # SET end is passed to RESET. The origin labels are not paired by index.
        synthetic = [(cohort["SET"][0],cohort["RESET"][0],sc,rc)]
        state,i_set,_ = field.SetRamp(synthetic,device,prep,initial,ratio,.0125,
                                     current_law=forward).predict(chosen["growth_parameters"])
        i_reset,end = reset.ResetRamp(synthetic,None,None,.01,reverse,evolve_return=True).predict(
            [state],floor,params)
        i_reset = i_reset[0]
        starts = [s for r in cohort["RESET"] if (s:=cycle.observed_state(r,rc)) is not None]
        report[str(g)] = {"sheet":index["names"][round((g-.9)*10)+4],
            "sweeps":{p:len(r) for p,r in cohort.items()},"excluded_traces":rejected,
            "SET":errors(cohort["SET"],i_set),"RESET":errors(cohort["RESET"],i_reset),
            "entry_predicted_uS":10**state,"entry_observed_median_uS":float(10**np.median(starts)),
            "final_predicted_uS":float(10**end[0])}
        panels.append((g,cohort,i_set,i_reset))
    output = {"workbook_sha256":index and (args.cache/"origin_sha256.txt").read_text().strip(),
              "protocol":"No refitting on new workbook. Rebuild the original nominal SET fit on existing 1.1 V cycles 4–35 and use saved shared SET and continuous RESET parameters. Score raw retained points; phase grids follow IV41 modal-waveform selection. SET and RESET traces are scored separately without assuming a cross-phase curve-ID mapping. All gates use their measured/interpolated transistor lookup.",
              "results":report}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    draw(args.figure,panels)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
