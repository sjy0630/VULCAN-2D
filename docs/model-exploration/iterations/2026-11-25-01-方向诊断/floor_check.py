"""Quick diagnostic: deterministic waveform-only floor, bimodality and row-order memory in IV41."""
import json, numpy as np
from collections import defaultdict
C = '/Users/sujiangyu/Desktop/SIM2RRAM/outputs/vulcan-handoff/iv41_cache/'
idx = json.load(open(C + 'index.json'))
hdr = idx['curves'][0]
print('curve header:', hdr)
sheets = {5: .9, 6: 1.0, 7: 1.1, 8: 1.2, 9: 1.3, 10: 1.4, 11: 1.5}
def li(i): return np.log10(np.maximum(np.abs(i), 1e-13))
for s, g in sheets.items():
    name = idx['names'][s - 1]
    meta = {str(r[4]): r for r in idx['curves'][1:] if r[13] == name}
    d = np.load(C + f's{s}.npz')
    grids = defaultdict(list)
    for k in sorted(d.files, key=int):
        o, v, i = d[k].T
        if not (np.isfinite(v).all() and np.isfinite(i).all()): continue
        ph = meta[k][6]
        grids[(ph, v.tobytes())].append((int(k), v, i))
    out = {}
    for ph in ('正', '负'):
        cands = [x for key, x in grids.items() if key[0] == ph]
        rows = max(cands, key=len)
        v = rows[0][1]; Y = np.stack([li(r[2]) for r in rows])
        apex = np.argmax(np.abs(v)); dv = np.median(np.abs(np.diff(v)))
        # mask out compliance-like plateau? keep simple: all points with |V|>0.05
        m = np.abs(v) > 0.05
        mean = Y.mean(0); med = np.median(Y, 0)
        floor_mean = np.sqrt(np.mean((Y[:, m] - mean[m]) ** 2))
        floor_med = np.sqrt(np.mean((Y[:, m] - med[m]) ** 2))
        # outward-only
        mo = m & (np.arange(len(v)) <= apex)
        fo = np.sqrt(np.mean((Y[:, mo] - mean[mo]) ** 2))
        # low-bias read at |V|~0.15 on return limb (index after apex)
        ret = np.arange(len(v)) > apex
        j = np.where(ret)[0][np.argmin(np.abs(np.abs(v[ret]) - .15))]
        jo = np.where(~ret)[0][np.argmin(np.abs(np.abs(v[~ret]) - .15))]
        Gr = np.abs(np.stack([r[2][j] for r in rows])) / .15 * 1e6
        Go = np.abs(np.stack([r[2][jo] for r in rows])) / .15 * 1e6
        lg = np.log10(np.maximum(Gr, 1e-6))
        ac = np.corrcoef(lg[:-1], lg[1:])[0, 1]
        # conditional floor: regress each point's logI on log read-at-0.15-outward (one measured scalar)
        x = np.log10(np.maximum(Go, 1e-7))
        X = np.c_[np.ones_like(x), x]
        beta, *_ = np.linalg.lstsq(X, Y[:, mo], rcond=None)
        fc = np.sqrt(np.mean((Y[:, mo] - X @ beta) ** 2))
        out[ph] = dict(n=len(rows), Vmax=float(np.abs(v).max()), dV=float(dV) if (dV := dv) else 0, npts=len(v),
                       floor_all_mean=round(floor_mean, 3), floor_all_median=round(floor_med, 3),
                       floor_outward=round(fo, 3), floor_outward_given_initial_read=round(fc, 3),
                       return_G015_uS_q10_50_90=np.round(np.quantile(Gr, [.1, .5, .9]), 2).tolist(),
                       frac_return_G_gt30uS=round(float((Gr > 30).mean()), 2),
                       lag1_autocorr_logG_return=round(float(ac), 2))
    print(g, json.dumps(out, ensure_ascii=False))
