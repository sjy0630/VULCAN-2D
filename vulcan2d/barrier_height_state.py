"""A barrier-height state with frozen geometry and finite-temperature supply."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .barrier_state import supply_conductance, LN10


class HeightStateTable:
    def __init__(self, width_nm, voltage_grid, height_grid, log_conductance):
        self.width_nm = float(width_nm)
        self.voltage = np.asarray(voltage_grid, dtype=float)
        self.height = np.asarray(height_grid, dtype=float)
        self.log_conductance = np.asarray(log_conductance, dtype=float)
        if self.log_conductance.shape != (len(self.voltage), len(self.height)):
            raise ValueError('Voltage-by-height conductance table required')

    def interpolate(self, voltage, height_ev):
        v, h = np.broadcast_arrays(voltage, height_ev)
        if np.any(~np.isfinite(v)) or np.any(~np.isfinite(h)) or np.any(v < 0) or np.any(v > self.voltage[-1]+1e-12):
            raise ValueError('Finite voltage inside table domain required')
        if np.any(h < self.height[0]-1e-12) or np.any(h > self.height[-1]+1e-12):
            raise ValueError('Height inside table domain required')
        dv, dh = self.voltage[1]-self.voltage[0], self.height[1]-self.height[0]
        iv = np.clip(((v-self.voltage[0])/dv).astype(int), 0, len(self.voltage)-2)
        ih = np.clip(((h-self.height[0])/dh).astype(int), 0, len(self.height)-2)
        fv, fh = (v-self.voltage[iv])/dv, (h-self.height[ih])/dh
        a, b = self.log_conductance[iv, ih], self.log_conductance[iv+1, ih]
        c, d = self.log_conductance[iv, ih+1], self.log_conductance[iv+1, ih+1]
        value = (1-fv)*((1-fh)*a+fh*c)+fv*((1-fh)*b+fh*d)
        deriv_v = ((1-fh)*(b-a)+fh*(d-c))/dv
        deriv_h = ((1-fv)*(c-a)+fv*(d-b))/dh
        return value, deriv_v, deriv_h

    def internal(self, voltage, log10_prefactor_S, height_ev):
        v = np.asarray(voltage, dtype=float)
        logc, dlogc_dv, dlogc_dh = self.interpolate(v, height_ev)
        conductance = 10**(log10_prefactor_S+logc)
        current = v*conductance; derivative = conductance*(1+LN10*v*dlogc_dv)
        if np.any(derivative <= 0):
            raise RuntimeError('Nonmonotone internal transport interpolation')
        return current, derivative, np.ones_like(current), dlogc_dh


def build_height_tables(widths_nm, *, mass_ratio=.5, temperature_k=300., max_voltage=.32,
                        min_height_ev=.02, max_height_ev=4., voltage_step=.0025,
                        height_step_ev=.01, order=24, cache=True):
    widths = np.array(widths_nm, dtype=float)
    if np.any(widths < 0) or len(set(widths.tolist())) != len(widths):
        raise ValueError('Distinct nonnegative fixed widths required')
    voltage = np.linspace(0., max_voltage, int(np.ceil(max_voltage/voltage_step))+1)
    height = np.linspace(min_height_ev, max_height_ev, int(np.ceil((max_height_ev-min_height_ev)/height_step_ev))+1)
    if min_height_ev <= 0:
        raise ValueError('Positive barrier heights required')
    source = Path(__file__); root = source.parents[1]
    settings = {'widths_nm':widths.tolist(), 'voltage':voltage.tolist(), 'height':height.tolist(),
                'mass_ratio':mass_ratio, 'temperature_k':temperature_k, 'order':order,
                'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'supply_kernel_sha256':hashlib.sha256(source.with_name('barrier_state.py').read_bytes()).hexdigest()}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    folder = root / '.venv/.cache/barrier_height'; path = folder / (key+'.npz')
    if cache and path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['cache_key']) != key:
                raise RuntimeError('Wrong table cache identity')
            values = saved['log_conductance']
    else:
        values = np.empty((len(widths), len(voltage), len(height)))
        for j, phi in enumerate(height):
            for k, v in enumerate(voltage):
                values[:, k, j] = np.log10(supply_conductance(v, widths, phi, mass_ratio=mass_ratio,
                                                            temperature_k=temperature_k, order=order))
        if cache:
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, cache_key=key, log_conductance=values)
    if values.shape != (len(widths), len(voltage), len(height)) or np.any(~np.isfinite(values)):
        raise RuntimeError('Invalid table cache dimensions or values')
    return {float(g):HeightStateTable(g, voltage, height, values[i]) for i, g in enumerate(widths)}
