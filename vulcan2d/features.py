"""Feature extraction that MIRRORS analysis/extract_features.py exactly, so the
model and the measured data are scored through identical estimators."""
import numpy as np


def _smooth(y, wdw=5):
    if len(y) < wdw:
        return y
    return np.convolve(y, np.ones(wdw) / wdw, mode="same")


def _at(V, I, vt, tol=0.06):
    j = np.argmin(np.abs(V - vt))
    if abs(V[j] - vt) > tol:
        return np.nan
    return abs(I[j])


def set_feat(Vs, Is):
    apex = int(np.argmax(np.abs(Vs)))
    Vf, If = Vs[:apex + 1], np.abs(Is[:apex + 1])
    logI = np.log10(np.clip(If, 1e-13, None))
    dlog = np.gradient(_smooth(logI), Vf)
    win = (Vf > 0.3) & (Vf < 3.0)
    Vset = Vf[np.argmax(np.where(win, dlog, -np.inf))]
    Icc = abs(Is[apex])
    I_hrs = _at(Vf, If, 0.2)
    Vr, Ir = Vs[apex:], np.abs(Is[apex:])
    I_lrs = _at(Vr, Ir, 0.2)
    return dict(Vset=Vset, Icc=Icc,
                R_HRS=0.2 / I_hrs if I_hrs else np.nan, I_HRS=I_hrs,
                R_LRS=0.2 / I_lrs if I_lrs else np.nan, I_LRS=I_lrs)


def reset_feat(Vr, Ir):
    apex = int(np.argmax(np.abs(Vr)))
    Vf, If = Vr[:apex + 1], np.abs(Ir[:apex + 1])
    logI = np.log10(np.clip(If, 1e-13, None))
    dlog = np.gradient(_smooth(logI), Vf)
    win = (Vf < -0.5) & (Vf > -1.7)
    Vreset = Vf[np.argmax(np.where(win, dlog, -np.inf))]
    return dict(Vreset=Vreset)


def vset_phi(Vs, pbs, lvl=0.5):
    """Model switching voltage = applied V where phi_bar first crosses lvl on the
    SET upsweep. Physical and numerically stable (the I-V dlogI peak is fine for
    the data but jitters on the model's very sharp transition)."""
    apex = int(np.argmax(np.abs(Vs)))
    pb, V = pbs[:apex + 1], Vs[:apex + 1]
    idx = np.where(pb >= lvl)[0]
    return V[idx[0]] if len(idx) else np.nan


def vreset_phi(Vr, pbr, lvl=0.5):
    apex = int(np.argmax(np.abs(Vr)))
    pb, V = pbr[:apex + 1], Vr[:apex + 1]
    idx = np.where(pb <= lvl)[0]
    return V[idx[0]] if len(idx) else np.nan


def extract_all(cycles):
    """V_set/V_reset use the SAME dlogI-peak estimator as the measured data
    (analysis/extract_features.py) so model and data are compared fairly.
    The phi-crossing values are kept as *_phi diagnostics only."""
    import pandas as pd
    rows = []
    for c in cycles:
        f = set_feat(c["Vs"], c["Is"])          # f["Vset"] = dlogI peak (data-style)
        f.update(reset_feat(c["Vr"], c["Ir"]))  # f["Vreset"] = dlogI peak (data-style)
        f["Vset_phi"] = vset_phi(c["Vs"], c["pbs"])    # model-native diagnostics
        f["Vreset_phi"] = vreset_phi(c["Vr"], c["pbr"])
        rows.append(f)
    return pd.DataFrame(rows)


def summary(df, col):
    x = np.asarray(df[col], float)
    x = x[np.isfinite(x)]
    cv = np.std(x) / abs(np.mean(x)) * 100
    ax = np.abs(x)
    sigln = np.std(np.log(ax[ax > 0])) if (ax > 0).any() else np.nan
    return dict(mean=np.mean(x), cv=cv, sigma_ln=sigln, n=len(x))
