"""Test sparse positive-gate interpolation before attributing return curvature.

Four processing alternatives preserve the measured gate curves. Held-gate
tests use only the other measured gates. The resulting spread is not a
confidence interval or evidence for the exact 1.1-V transfer characteristic.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.interpolate import PchipInterpolator

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from diagnose_cross_polarity_physics import read_pairs
from vulcan2d.transistor import load_default_lookup
from vulcan2d.transport_candidates import Curve, fit_shared_shape

METHODS=("linear_current","sqrt_current","log_current","pchip_current")


def gate_interpolate(gates, matrix, target, method):
    """Matrix is gate by voltage. No extrapolation in the gate coordinate."""
    if not gates[0] <= target <= gates[-1]:
        raise ValueError("Gate extrapolation forbidden")
    exact=np.flatnonzero(np.isclose(gates,target,rtol=0,atol=1e-12))
    if len(exact):return matrix[exact[0]].copy()
    if method=="pchip_current":
        out=PchipInterpolator(gates,matrix,axis=0,extrapolate=False)(target)
    else:
        hi=int(np.searchsorted(gates,target));lo=hi-1
        w=(target-gates[lo])/(gates[hi]-gates[lo])
        a,b=matrix[lo],matrix[hi]
        if method=="linear_current":out=(1-w)*a+w*b
        elif method=="sqrt_current":out=((1-w)*np.sqrt(a)+w*np.sqrt(b))**2
        elif method=="log_current":
            out=np.zeros_like(a);positive=(a>0)&(b>0)
            out[positive]=np.exp((1-w)*np.log(a[positive])+w*np.log(b[positive]))
        else:raise ValueError(method)
    if np.any(out < -1e-20) or np.any(np.diff(out)<-1e-15):
        raise ValueError("Gate interpolation generated non-passive/non-monotone voltage law")
    return np.maximum(out,0)


def invert(vds, currents, target):
    if not 0 <= target <= currents[-1]:raise ValueError("Outside drain-current domain")
    hi=int(np.searchsorted(currents,target,side="left"))
    if hi==0:return float(vds[0])
    lo=hi-1
    if currents[hi]<=currents[lo]:raise ValueError("Non-unique inverse")
    return float(vds[lo]+(target-currents[lo])/(currents[hi]-currents[lo])*(vds[hi]-vds[lo]))


def run():
    source=load_default_lookup()
    gates=source.gate_voltages
    grid=np.unique(np.concatenate([c["drain_voltage_v"] for c in source.curves()]))
    matrix=np.array([[source.current(v,g) for v in grid] for g in gates])
    report={"scope":"Exploratory interpolation sensitivity; positive-gate lookup only",
            "measured_gates_V":gates.tolist(),"target_gate_V":1.1,
            "held_gate_tests":[],"return_shape_sensitivity":{},"gate_interpolation_assumptions":{
                "linear_current":"Existing bilinear-current rule",
                "sqrt_current":"Sensitivity motivated by strong-inversion saturation square law, not valid as a universal MOS law",
                "log_current":"Sensitivity motivated by subthreshold exponential gate response, not valid as a universal MOS law",
                "pchip_current":"Shape-preserving cubic interpolation in gate current, not a calibrated transistor device model"}}
    for method in METHODS:
        for k in range(1,len(gates)-1):
            mask=np.arange(len(gates))!=k
            predicted=gate_interpolate(gates[mask],matrix[mask],gates[k],method)
            target=matrix[k]
            for lo,hi in ((.05,.30),(.30,1.0),(1.0,5.0),(.05,5.0)):
                use=(grid>=lo)&(grid<=hi)&(target>0)&(predicted>0)
                error=np.log10(predicted[use]/target[use])
                report["held_gate_tests"].append({"method":method,"held_gate_V":float(gates[k]),
                    "voltage_window_V":[lo,hi],"n_points":int(sum(use)),
                    "rmse_decade":float(np.sqrt(np.mean(error**2))),"bias_decade":float(np.mean(error))})
    pairs=read_pairs("1.1")
    for method in METHODS:
        id_curve=gate_interpolate(gates,matrix,1.1,method)
        series=[];excluded={}
        for pair in pairs:
            raw=pair["set"];v=np.abs(raw["voltage"]);i=np.abs(raw["current"])
            start=len(v)-1-int(np.argmax(v[::-1]));vv=[];ii=[];dropped=0
            zero=float(np.max(i[v<=1e-12]))
            for n in range(start,len(v)):
                if v[n]<=0 or i[n]<=10*zero:continue
                try:
                    vd=invert(grid,id_curve,i[n]);vh=v[n]-vd
                    cap=float(np.interp(v[n],grid,id_curve))
                    if vh<=0 or vd/v[n]>.5 or i[n]/cap>.7:
                        dropped+=1;continue
                    vv.append(vh);ii.append(i[n])
                except ValueError:dropped+=1
            series.append((pair["cycle_id"],np.array(vv),np.log10(ii)))
            excluded[pair["cycle_id"]]=dropped
        windows=[]
        for lo,hi in ((.15,.4),(.4,.8),(.8,1.2),(1.2,2),(2,3),(3,4.2)):
            curves=[];exponents=[]
            for cid,v,y in series:
                use=(v>=lo)&(v<=hi)
                if sum(use)<5:continue
                c=Curve(cid,v[use],y[use]);curves.append(c)
                x=np.log10(c.voltage_v);z=c.log10_current_a
                exponents.append(float(np.sum((x-x.mean())*(z-z.mean()))/np.sum((x-x.mean())**2)))
            if len(curves)<2:raise RuntimeError("Insufficient cycles for a common shape sensitivity window")
            fit=fit_shared_shape(curves,"power")
            windows.append({"window_V":[lo,hi],"n_cycles":len(curves),"cycle_ids":[c.cycle_id for c in curves],
                            "shared_exponent":fit["parameter"],
                            "n_cycles_local_exponent_below_one":sum(x<1 for x in exponents),
                            "local_exponent_q10_median_q90":np.quantile(exponents,[.1,.5,.9]).tolist()})
        report["return_shape_sensitivity"][method]={"window_results":windows,"excluded_counts":excluded,
             "predicted_current_at_Vds_0p1_1_5_A":np.interp([.1,1,5],grid,id_curve).tolist()}
    report["limitations"]=[
        "Held-gate spacing is 1 V versus 0.5 V for the target bracket; its error is not a direct uncertainty bound at gate 1.1 V.",
        "All variants use the same standalone transistor samples and share possible device mismatch, bias-network, and measurement system errors.",
        "A robust local slope is not proof that a return branch is in one frozen state.",
        "No gate correction is optimized against the target memristor current; no production interpolation is changed.",
        "Alternative numerical choices are diagnostics, not independently verified physical transistor laws.",
    ]
    files=[HERE/"audit_transistor_gate_interpolation.py",HERE.parent/"vulcan2d/data/fig2b_transistor_lookup.json",HERE.parent/"1T1M写入.xlsx"]
    report["input_sha256"]={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    out=HERE/"transistor_gate_interpolation_audit.json";out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(out)
    for method,g in report["return_shape_sensitivity"].items():
        print(method,[(r['window_V'],round(r['shared_exponent'],3),r['n_cycles_local_exponent_below_one']) for r in g['window_results']])


if __name__=="__main__":run()
