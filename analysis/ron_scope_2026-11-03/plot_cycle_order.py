"""Observed training low reads and conditional scalar-family error floors."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    report = json.loads((HERE/"SCALAR_CYCLE_ORDER_AUDIT.json").read_text())
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42})
    fig, (ax, bx) = plt.subplots(1,2,figsize=(10.4,3.8),gridspec_kw={"width_ratios":[1.55,1]})
    for phase,color,marker,label in (("set","#176B9B","o","SET return, +50 mV"),
                                     ("reset","#C45A27","s","RESET outward, -50 mV")):
        for j,row in enumerate(r for r in report["low_bias_checks"] if r["phase"] == phase):
            ax.plot(row["record_ids"],np.asarray(row["currents_a"])*1e6,
                    color=color,marker=marker,ms=4,lw=1.3,label=label if j==0 else None)
    ax.axvspan(3.5,4.5,color="#D5D8DC",alpha=.65,lw=0)
    ax.text(4,65,"Cycle 4 missing",ha="center",va="bottom",rotation=90,fontsize=8,color="#555555")
    ax.set(xlabel="Measured cycle label",ylabel="Current magnitude (µA)",ylim=(0,925),
           title="Paired reads agree; cycle values rise and fall")
    ax.set_xticks([1,3,5,7,9,11,13,15])
    ax.legend(frameon=False,loc="upper left",fontsize=9)
    ax.grid(axis="y",alpha=.15)
    labels=["1R SET","1R RESET","1T1R SET","1T1R RESET"]
    keys=["1r_set","1r_reset","1t1r_set","1t1r_reset"]
    values=[report["raw_phase_lower_bounds"][k]["raw_RMSE_lower_bound_decade"] for k in keys]
    bars=bx.bar(np.arange(4),values,color=["#176B9B","#176B9B","#6B929F","#6B929F"],width=.65)
    for bar,value in zip(bars,values):
        bx.text(bar.get_x()+bar.get_width()/2,value+.015,f"{value:.3f}",ha="center",fontsize=9)
    bx.set_xticks(range(4),labels,rotation=25,ha="right")
    bx.set(ylabel="Optimistic raw RMSE floor (decade)",ylim=(0,.69),
           title="Conditional scalar-model accuracy limits")
    bx.grid(axis="y",alpha=.15)
    fig.text(.5,.015,"Assumptions: identical complete protocol, fixed parameters, deterministic scalar state, monotone readout.\nFloors relax state dynamics; they are not fitted model predictions or measured noise limits.",
             ha="center",va="bottom",fontsize=8,color="#555555")
    fig.tight_layout(rect=(0,.12,1,1))
    for suffix in ("png","pdf"):
        fig.savefig(HERE/f"scalar_cycle_order.{suffix}",dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
