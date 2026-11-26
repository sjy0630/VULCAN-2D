"""Shared reversible q/a material, actual external circuits, continuous history.

Prepared implementation only: import performs no load, experiment or fit.
The soft recovery order is the previously studied fixed q**2 choice. Rates
are per command-voltage path, not identified physical-time kinetics.

V2 differs only in the closed voltage endpoint representation: the frozen
metadata endpoint is used after exact log-endpoint equality is verified.
This corrects exp(log(5)) rounding below native 5 V; it adds no physical
parameter, extends no table domain and changes no state/current clipping.
Frozen parent SHA256: 1288ae379fbefff679bebd3ad110a10cee344f970d8feb6a8dff7838b58a26d7.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
import hashlib
import json
import math
import sys

sys.dont_write_bytecode = True
import numpy as np
from scipy.integrate import DOP853, Radau
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT/'analysis/zero_crossing_compatibility_2026-11-04',
                  ROOT/'analysis/signed_soft_history_2026-11-04'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from minimal_static_loader import load_minimal
from signed_forward_kernel import _path_clock


class DomainFailure(ValueError):
    """Unresolved physical/table domain, never silently clipped."""


class PhaseFailure(RuntimeError):
    def __init__(self, message, summary, arrays):
        super().__init__(message)
        self.summary, self.arrays = summary, arrays


class HistoryFailure(RuntimeError):
    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _immutable_array(value):
    array = np.asarray(value, dtype=float)
    # Immutable bytes own the data; setting writeable=True cannot unfreeze it.
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


@dataclass(frozen=True)
class Parameters:
    lambda_per_v: float
    J_positive_a: float
    kappa_positive_per_v2: float
    b: float
    a_max_s: float
    kappa_a_per_v2: float
    rho: float
    enable_a: bool

    def __post_init__(self):
        for key in ('lambda_per_v', 'J_positive_a', 'kappa_positive_per_v2',
                    'b', 'a_max_s', 'kappa_a_per_v2', 'rho'):
            object.__setattr__(self, key, _finite(getattr(self, key), key))
        if type(self.enable_a) is not bool:
            raise TypeError('enable_a must be an explicit bool')
        if (self.lambda_per_v < 0 or self.J_positive_a <= 0
                or self.kappa_positive_per_v2 < 0 or self.a_max_s < 0
                or self.kappa_a_per_v2 < 0 or self.rho <= 0):
            raise ValueError('Require lambda,kappa_positive,A,kappa_a>=0 and J,rho>0')
        if self.enable_a and (self.a_max_s <= 0 or self.kappa_a_per_v2 <= 0):
            raise ValueError('Enabled a requires A>0 and kappa_a>0')


@dataclass(frozen=True)
class Precision:
    name: str
    rtol: float
    atol_u: float
    atol_x: float
    max_step: float

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('Explicit precision name required')
        for key in ('rtol', 'atol_u', 'atol_x', 'max_step'):
            value = _finite(getattr(self, key), key)
            if value <= 0:
                raise ValueError('Precision values must be positive')
            object.__setattr__(self, key, value)


@dataclass(frozen=True)
class FrozenReadout:
    lg: np.ndarray
    lv: np.ndarray
    li_positive: np.ndarray
    li_negative: np.ndarray
    tx_positive: np.ndarray
    ti_positive: np.ndarray
    tx_negative: np.ndarray
    ti_negative: np.ndarray
    ron_ohm: float
    gamma_c_ev: float
    log_scale: float
    vmin_v: float
    vmax_v: float
    reference_kappa_positive_per_v2: float
    shape_metadata: object


@dataclass(frozen=True)
class CoupledMaterial:
    readout: FrozenReadout
    parameters: Parameters
    nearzero_anchor_v: float
    candidate_source_id: str
    candidate_source_sha256: str
    source_sha256: object

    def validate_state(self, state, *, table=True):
        if len(state) != 2:
            raise DomainFailure('State must contain exactly q and a [S]')
        q, a = map(float, state)
        if not all(map(math.isfinite, (q, a))) or not 0 <= q <= 1:
            raise DomainFailure(f'q outside [0,1]: {q!r}')
        if not 0 <= a <= self.parameters.a_max_s:
            raise DomainFailure(f'a outside [0,A]: {a!r}')
        if not self.parameters.enable_a and a != 0:
            raise DomainFailure('Disabled-a control requires a=0 throughout')
        if table:
            if not 0 < q < 1:
                raise DomainFailure('Finite spectral table requires 0<q<1')
            u = -math.log(q)
            if not 0 < u <= 80:
                raise DomainFailure('Finite inherited spectral domain requires 0<u<=80')
            lg = math.log(self.readout.gamma_c_ev/math.expm1(u))
            if not self.readout.lg[0] <= lg <= self.readout.lg[-1]:
                raise DomainFailure('State Gamma outside frozen spectral table')
        return q, a

    def _table_abs(self, magnitude, q, sign, *, reference=False):
        r = self.readout
        u = -math.log(q)
        lg = math.log(r.gamma_c_ev/math.expm1(u))
        lv = math.log(magnitude)
        if not r.lv[0] <= lv <= r.lv[-1]:
            raise DomainFailure('Requested spectral voltage outside frozen table')
        spectrum = r.li_positive if sign > 0 else r.li_negative
        gi = min(int(np.searchsorted(r.lg, lg, side='left'))-1, len(r.lg)-2)
        vi = min(int(np.searchsorted(r.lv, lv, side='left'))-1, len(r.lv)-2)
        # Exact lower table endpoints use the first cell, never extrapolation.
        if gi == -1 and lg == r.lg[0]:
            gi = 0
        if vi == -1 and lv == r.lv[0]:
            vi = 0
        if gi < 0 or vi < 0:
            raise DomainFailure('Interpolation index below frozen domain')
        tg = (lg-r.lg[gi])/(r.lg[gi+1]-r.lg[gi])
        tv = (lv-r.lv[vi])/(r.lv[vi+1]-r.lv[vi])
        if reference:
            value = ((1-tg)*(1-tv)*spectrum[gi,vi]
                     +(1-tg)*tv*spectrum[gi,vi+1]
                     +tg*(1-tv)*spectrum[gi+1,vi]
                     +tg*tv*spectrum[gi+1,vi+1])
        else:
            # The frozen signed kernel's log-current bilinear interpolation.
            value = ((1-tg)*((1-tv)*spectrum[gi,vi]+tv*spectrum[gi,vi+1])
                     +tg*((1-tv)*spectrum[gi+1,vi]+tv*spectrum[gi+1,vi+1]))
        return math.exp(float(value)+r.log_scale)

    def components(self, voltage_hbn_v, state, *, reference=False):
        q, a = self.validate_state(state)
        voltage = _finite(voltage_hbn_v, 'actual Vh')
        if voltage == 0:
            return {'soft_current_a': 0., 'hard_current_a': 0., 'current_a': 0.,
                    'nearzero_extension_used': False}
        sign, v = (1 if voltage > 0 else -1), abs(voltage)
        if v > self.readout.vmax_v:
            raise DomainFailure('Actual material voltage exceeds frozen table')
        extension = v < self.nearzero_anchor_v
        if extension:
            soft = sign*self._table_abs(self.nearzero_anchor_v, q, sign,
                                       reference=reference)*(v/self.nearzero_anchor_v)
        else:
            soft = sign*self._table_abs(v, q, sign, reference=reference)
        hard = a*voltage
        total = soft+hard
        if not all(map(math.isfinite, (soft, hard, total))) or soft*voltage < 0 or hard*voltage < 0:
            raise DomainFailure('Nonfinite or nonpassive material branches')
        return {'soft_current_a': soft, 'hard_current_a': hard,
                'current_a': total, 'nearzero_extension_used': extension}

    def current(self, voltage_hbn_v, state):
        return self.components(voltage_hbn_v, state)['current_a']

    def rate_components(self, state, point):
        q, a = self.validate_state(state, table=False)
        if point.get('_material_id') != id(self) or tuple(point.get('_state', ())) != (q,a):
            raise ValueError('Rates require this material/state and its actual solved point')
        vh = _finite(point['local_voltage_v'], 'Vh')
        current = _finite(point['current_a'], 'total material current')
        soft, hard = point['soft_current_a'], point['hard_current_a']
        if (vh*current < 0 or vh*soft < 0 or vh*hard < 0
                or not math.isclose(current, soft+hard, rel_tol=1e-12, abs_tol=1e-18)):
            raise DomainFailure('Inconsistent solved branch currents')
        if vh == 0:
            if current != 0:
                raise DomainFailure('Nonzero current at zero actual voltage')
            return dict(capture_q_per_v=0., recovery_q_per_v=0.,
                        formation_a_s_per_v=0., loss_a_s_per_v=0.)
        p = self.parameters
        sign, v, i = (1 if vh > 0 else -1), abs(vh), abs(current)
        chi = (1-sign)*p.b
        capture = p.kappa_positive_per_v2*v*(i/p.J_positive_a)*math.exp(-chi)*(1-q)
        recovery = p.kappa_positive_per_v2*v*math.exp(chi+p.lambda_per_v*v)*q*q
        formation = loss = 0.
        if p.enable_a:
            formation = p.kappa_a_per_v2*v*math.exp(sign*p.b)*q*(p.a_max_s-a)
            loss = p.kappa_a_per_v2*v*p.rho*math.exp(-sign*p.b)*(1-q)*a
        values = dict(capture_q_per_v=capture, recovery_q_per_v=recovery,
                      formation_a_s_per_v=formation, loss_a_s_per_v=loss)
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise DomainFailure('Positive rate component overflow/domain failure')
        return values

    def rates(self, state, point):
        terms = self.rate_components(state, point)
        return (terms['capture_q_per_v']-terms['recovery_q_per_v'],
                terms['formation_a_s_per_v']-terms['loss_a_s_per_v'])

    def metadata(self):
        return dict(parameters=asdict(self.parameters), fixed_recovery_order=2,
                    state_definition='q=exp(-u)=Gamma/(Gamma+Gamma_c); a in siemens',
                    total_current_drives_capture=True, shared_b_for_both_states=True,
                    readout=dict(Ron_ohm=self.readout.ron_ohm,
                                 gamma_c_ev=self.readout.gamma_c_ev),
                    nearzero_policy='linear_endpoint_full_integration',
                    nearzero_anchor_v=self.nearzero_anchor_v,
                    nearzero_extension_is_approximation=True,
                    candidate_source_id=self.candidate_source_id,
                    candidate_source_sha256=self.candidate_source_sha256,
                    source_sha256=dict(self.source_sha256))


def load_material(parameters, *, candidate_source_id, candidate_source_sha256,
                  nearzero_policy='linear_endpoint', nearzero_anchor_v=None):
    """Load only frozen readout/lookup data; no observations, roots or ODEs."""
    if not isinstance(parameters, Parameters):
        raise TypeError('Explicit frozen Parameters required')
    if not isinstance(candidate_source_id, str) or not candidate_source_id.strip():
        raise ValueError('Explicit candidate source ID required')
    if (not isinstance(candidate_source_sha256, str) or len(candidate_source_sha256) != 64
            or any(c not in '0123456789abcdefABCDEF' for c in candidate_source_sha256)):
        raise ValueError('Explicit 64-hex candidate source SHA256 required')
    if nearzero_policy != 'linear_endpoint':
        raise ValueError('Only the declared continuous linear-endpoint extension is implemented')
    kernel = load_minimal()
    for path, digest in kernel.source_sha256.items():
        if sha(path) != digest:
            raise ValueError(f'Frozen source changed: {path}')
    with np.load(kernel.cache/'transport_g010_v0025.npz') as archive:
        shape = json.loads(str(archive['metadata']))['shape']
    endpoint_v = _finite(shape['voltage_limit_v'], 'frozen metadata voltage endpoint')
    if endpoint_v <= 0 or math.log(endpoint_v) != float(kernel.lv[-1]):
        raise ValueError('Frozen metadata voltage endpoint differs from the log-table endpoint')
    r = FrozenReadout(*[_immutable_array(value) for value in (
        kernel.lg, kernel.lv, kernel.base.li, kernel.li_negative,
        kernel.base.tx, kernel.base.ti, kernel.negative_tx, kernel.negative_ti)],
        ron_ohm=kernel.ron, gamma_c_ev=kernel.gc, log_scale=kernel.base.log_scale,
        vmin_v=kernel.vmin, vmax_v=endpoint_v,
        reference_kappa_positive_per_v2=kernel.parameters.kappa_per_v2,
        shape_metadata=MappingProxyType(dict(shape)))
    anchor = r.vmin_v if nearzero_anchor_v is None else _finite(nearzero_anchor_v, 'nearzero anchor')
    if not r.vmin_v <= anchor <= r.vmax_v:
        raise ValueError('Near-zero anchor must be inside the original voltage table')
    for tx, ti in ((r.tx_positive,r.ti_positive),(r.tx_negative,r.ti_negative)):
        if (tx[0] != 0 or ti[0] != 0 or not np.all(np.diff(tx)>0)
                or not np.all(np.diff(ti)>=0) or not np.all(np.isfinite(tx))
                or not np.all(np.isfinite(ti))):
            raise ValueError('Frozen transistor branch must be finite passive monotone from zero')
    sources = dict(kernel.source_sha256)
    sources[str(Path(__file__).resolve())] = sha(Path(__file__))
    return CoupledMaterial(r, parameters, anchor, candidate_source_id,
                           candidate_source_sha256.lower(), MappingProxyType(sources))


def _root(function, low, high, *, reference=False):
    flo, fhi = function(low), function(high)
    if not all(map(math.isfinite, (flo,fhi))) or flo > 0 or fhi < 0:
        raise DomainFailure(f'Actual total-current root is not bracketed: {flo}, {fhi}')
    if flo == 0:
        return low
    if fhi == 0:
        return high
    if reference:
        for _ in range(64):
            mid = .5*(low+high)
            if function(mid) >= 0:
                high = mid
            else:
                low = mid
        return .5*(low+high)
    return brentq(function, low, high, xtol=5e-324, rtol=4*np.finfo(float).eps, maxiter=160)


def _material_voltage(material, state, sign, current, upper, reference):
    return _root(lambda v: abs(material.components(sign*v, state,
                       reference=reference)['current_a'])-current,
                 0., upper, reference=reference)


def _transistor_voltage_interval(target, tx, ti):
    """All voltages carrying target, including measured flat-current plateaux."""
    if not 0 <= target <= ti[-1]:
        raise DomainFailure('Current lies outside the measured transistor branch')
    equal = np.flatnonzero(ti == target)
    if len(equal):
        return float(tx[equal[0]]), float(tx[equal[-1]])
    j = int(np.searchsorted(ti, target, side='right'))
    if not 0 < j < len(ti) or ti[j] <= ti[j-1]:
        raise DomainFailure('No increasing measured transistor segment brackets current')
    voltage = float(tx[j-1]+(target-ti[j-1])*(tx[j]-tx[j-1])/(ti[j]-ti[j-1]))
    return voltage, voltage


def _point(command, material, state, local, transistor, compliance, limited,
           transistor_current=None, reference=False):
    components = material.components(local, state, reference=reference)
    current = components['current_a']
    kcl = 0. if transistor_current is None else current-transistor_current
    return dict(command_voltage_v=command, supply_voltage_v=local+transistor,
                local_voltage_v=local, transistor_voltage_v=transistor,
                compliance_voltage_v=compliance, compliance_limited=bool(limited),
                KCL_residual_a=kcl, KVL_residual_v=command-local-transistor-compliance,
                _material_id=id(material), _state=tuple(map(float,state)), **components)


def _validate_compliance(value):
    if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
        raise ValueError('Compliance must be positive finite amperes or None')


@dataclass(frozen=True)
class BareCircuit:
    compliance_a: float | None = None

    def __post_init__(self):
        _validate_compliance(self.compliance_a)

    def solve(self, command, material, state, *, reference=False):
        command = _finite(command, 'command')
        material.validate_state(state)
        if command == 0:
            return _point(0., material, state, 0., 0., 0., False, reference=reference)
        sign, magnitude = (1 if command > 0 else -1), abs(command)
        upper = min(magnitude, material.readout.vmax_v)
        top_i = abs(material.components(sign*upper,state,reference=reference)['current_a'])
        if self.compliance_a is not None and (top_i > self.compliance_a
                or (top_i == self.compliance_a and magnitude > upper)):
            local = sign*_material_voltage(material,state,sign,self.compliance_a,upper,reference)
            return _point(command,material,state,local,0.,command-local,True,reference=reference)
        if magnitude > material.readout.vmax_v:
            raise DomainFailure('Unregulated bare voltage exceeds frozen material domain')
        return _point(command,material,state,command,0.,0.,False,reference=reference)


@dataclass(frozen=True)
class SeriesCircuit:
    compliance_a: float | None = None

    def __post_init__(self):
        _validate_compliance(self.compliance_a)

    def solve(self, command, material, state, *, reference=False):
        command = _finite(command, 'command')
        material.validate_state(state)
        if command == 0:
            return _point(0.,material,state,0.,0.,0.,False,0.,reference=reference)
        sign, magnitude = (1 if command > 0 else -1), abs(command)
        r = material.readout
        tx, ti = (r.tx_positive,r.ti_positive) if sign > 0 else (r.tx_negative,r.ti_negative)
        def intrinsic(v):
            return abs(material.components(sign*v,state,reference=reference)['current_a'])
        # A current cap regulates actual supplied voltage, including on a
        # transistor plateau. It can act before an unregulated root leaves data.
        if self.compliance_a is not None and self.compliance_a <= ti[-1]:
            cap = float(self.compliance_a)
            upper = min(magnitude,r.vmax_v)
            if intrinsic(upper) >= cap:
                vh_cap = _material_voltage(material,state,sign,cap,upper,reference)
                vt_low, vt_high = _transistor_voltage_interval(cap,tx,ti)
                if magnitude >= vh_cap+vt_low:
                    limited = magnitude > vh_cap+vt_high
                    vt = vt_high if limited else magnitude-vh_cap
                    local, transistor = sign*vh_cap, sign*vt
                    supplied = local+transistor
                    return _point(command,material,state,local,transistor,
                        command-supplied if limited else 0.,limited,
                        sign*float(np.interp(vt,tx,ti)),reference=reference)
        if reference:
            low, high = max(0.,magnitude-float(tx[-1])), min(magnitude,r.vmax_v)
            if magnitude-low > tx[-1]:
                low = math.nextafter(low,math.inf)
            if low > high:
                raise DomainFailure('No common material/transistor voltage domain')
            def residual(vh):
                vt = magnitude-vh
                if not 0 <= vt <= tx[-1]:
                    raise DomainFailure('Reference transistor trial outside measured domain')
                return intrinsic(vh)-float(np.interp(vt,tx,ti))
            vh = _root(residual,low,high,reference=True)
            vt = magnitude-vh
        else:
            low, high = max(0.,magnitude-r.vmax_v), min(magnitude,float(tx[-1]))
            if magnitude-low > r.vmax_v:
                low = math.nextafter(low,math.inf)
            if low > high:
                raise DomainFailure('No common material/transistor voltage domain')
            def residual(vt):
                return float(np.interp(vt,tx,ti))-intrinsic(magnitude-vt)
            vt = _root(residual,low,high)
            vh = magnitude-vt
        return _point(command,material,state,sign*vh,sign*vt,0.,False,
                      sign*float(np.interp(vt,tx,ti)),reference=reference)


POINT_FIELDS = ('supply_voltage_v','current_a','soft_current_a','hard_current_a',
                'local_voltage_v','transistor_voltage_v','compliance_voltage_v',
                'KCL_residual_a','KVL_residual_v','compliance_limited','nearzero_extension_used')
RHS_COLUMNS = ('path_voltage_s','command_voltage_v','state_u','state_q','state_a_s',
               'dq_per_v','da_s_per_v','capture_q_per_v','recovery_q_per_v',
               'formation_a_s_per_v','loss_a_s_per_v')+POINT_FIELDS


def _internal(material, state):
    q, a = material.validate_state(state)
    return np.array([-math.log(q),a/material.parameters.a_max_s if material.parameters.enable_a else 0.])


def _physical(material, values):
    u, x = map(float,values)
    if not math.isfinite(u) or not 0 < u <= 80 or not math.isfinite(x) or not 0 <= x <= 1:
        raise DomainFailure(f'Numerical u/x trial outside physical domain: {(u,x)!r}')
    if not material.parameters.enable_a and x != 0:
        raise DomainFailure('Disabled-a numerical state changed')
    return material.validate_state((math.exp(-u),x*material.parameters.a_max_s))


def _expanded_solver_state(material, values):
    """Inactive a is absent from the solver, not a clipped second coordinate."""
    values = np.asarray(values,dtype=float)
    expected = (2,) if material.parameters.enable_a else (1,)
    if values.shape != expected:
        raise ValueError(f'Unexpected active solver coordinate shape: {values.shape}')
    return values.copy() if material.parameters.enable_a else np.array([values[0],0.])


def simulate_phase(material, circuit, data, initial_state, precision, method,
                   check_budget, collect_rhs=True):
    """Jointly integrate both states, retaining failed and partial traces."""
    if not isinstance(material,CoupledMaterial) or not isinstance(circuit,(BareCircuit,SeriesCircuit)):
        raise TypeError('One frozen CoupledMaterial and an explicit external circuit required')
    if not isinstance(precision,Precision) or method not in ('DOP853','Radau'):
        raise ValueError('Explicit Precision and DOP853/Radau required')
    if check_budget is not None and not callable(check_budget):
        raise TypeError('check_budget must be callable or None')
    y0 = _internal(material,initial_state)
    initial_state = material.validate_state(initial_state)
    command = np.asarray(data['command_voltage_v'],dtype=float)
    initial_command = _finite(data['initial_command_voltage_v'],'incoming command')
    if command.ndim != 1 or not len(command) or not np.all(np.isfinite(command)):
        raise ValueError('Finite native one-dimensional waveform required')
    observed, mask = data.get('observed_current_a'), data.get('scoring_mask')
    if (observed is None) != (mask is None):
        raise ValueError('Observations and original boolean scoring mask must be supplied together')
    if observed is not None:
        observed,mask = np.asarray(observed,dtype=float),np.asarray(mask)
        if observed.shape != command.shape or mask.shape != command.shape or mask.dtype != np.bool_:
            raise ValueError('Original observation/mask shapes or mask dtype invalid')
        if np.any(mask & (~np.isfinite(observed) | (abs(observed)<=0) | (command==0))):
            raise ValueError('Original scoring mask selects invalid observation/zero voltage')
    unique,inverse,segments = _path_clock(command,initial_command)
    states = np.full((len(unique),2),np.nan)
    states[0] = y0
    summary = dict(status='running',**material.metadata(),circuit=type(circuit).__name__,
                   compliance_a=circuit.compliance_a,precision=asdict(precision),method=method,
                   internal_coordinates='u=-ln(q), x=a/A; saved disabled-a x=0',
                   active_solver_coordinates=['u','x'] if material.parameters.enable_a else ['u'],
                   inactive_atol_x_not_used=not material.parameters.enable_a,
                   initial_state_q=initial_state[0],initial_state_a_s=initial_state[1],
                   initial_command_voltage_v=initial_command,n_native=len(command),
                   record_id=data.get('record_id'),phase=data.get('phase'),segments=[],
                   no_state_clipping=True,no_state_reanchoring=True,no_failed_phase_retry=True,
                   no_nearzero_freeze=True,all_native_points_retained=True,
                   clock='cumulative actual command-voltage distance; repeated endpoint has no inferred dwell',
                   rhs_columns=list(RHS_COLUMNS),rhs_trace_requested=bool(collect_rhs),
                   rhs_calls=0,last_rhs={},accepted_steps=0)
    arrays = dict(command_voltage_v=command.copy(),path_voltage_s=unique[inverse][1:].copy())
    if observed is not None:
        arrays.update(observed_current_a=observed.copy(),original_scoring_mask=mask.copy())
    rhs_rows,accepted_rows = [],[]
    solver = None

    def budget():
        if check_budget is not None:
            check_budget()

    def snapshot():
        arrays['unique_path_voltage_s'] = unique.copy()
        arrays['unique_internal_state_ux'] = states.copy()
        arrays['rhs_trace'] = np.asarray(rhs_rows,dtype=float).reshape((-1,len(RHS_COLUMNS)))
        arrays['accepted_steps_ux'] = np.asarray(accepted_rows,dtype=float).reshape((-1,3))

    try:
        budget()
        y = y0.copy()
        for index,(start,stop,va,vb) in enumerate(segments):
            budget()
            row = dict(index=index,path_start_v=start,path_stop_v=stop,
                       command_start_v=va,command_stop_v=vb,initial_internal_state_ux=y.tolist())
            summary['segments'].append(row)
            direction = (vb-va)/(stop-start)
            def rhs(s,trial):
                budget()
                full_trial = _expanded_solver_state(material,trial)
                applied = va+direction*(float(s)-start)
                if s == start:
                    applied = va
                elif s == stop:
                    applied = vb
                summary['last_rhs'] = dict(segment=index,path_voltage_s=float(s),
                    command_voltage_v=float(applied),trial_internal_state_ux=full_trial.tolist())
                state = _physical(material,full_trial)
                point = circuit.solve(applied,material,state)
                terms = material.rate_components(state,point)
                dq = terms['capture_q_per_v']-terms['recovery_q_per_v']
                da = terms['formation_a_s_per_v']-terms['loss_a_s_per_v']
                du = -dq/state[0]
                dx = da/material.parameters.a_max_s if material.parameters.enable_a else 0.
                if not math.isfinite(du) or not math.isfinite(dx):
                    raise DomainFailure('Nonfinite joint state derivative')
                summary['rhs_calls'] += 1
                if collect_rhs:
                    trace = dict(path_voltage_s=s,command_voltage_v=applied,state_u=float(full_trial[0]),
                                 state_q=state[0],state_a_s=state[1],dq_per_v=dq,da_s_per_v=da,
                                 **terms,**{key:point[key] for key in POINT_FIELDS})
                    rhs_rows.append([float(trace[key]) for key in RHS_COLUMNS])
                return np.array([du,dx]) if material.parameters.enable_a else np.array([du])
            p = material.parameters
            stiffness_scale = max(1.,p.kappa_positive_per_v2/material.readout.reference_kappa_positive_per_v2,
                                  p.kappa_a_per_v2 if p.enable_a else 0.)
            first = min(stop-start,precision.max_step,1e-4/stiffness_scale)
            constructor = DOP853 if method == 'DOP853' else Radau
            solver_initial = y if p.enable_a else y[:1]
            solver_atol = ([precision.atol_u,precision.atol_x] if p.enable_a else [precision.atol_u])
            solver = constructor(rhs,start,solver_initial,stop,rtol=precision.rtol,
                atol=np.array(solver_atol),max_step=precision.max_step,first_step=first)
            row['first_step_v'] = first
            while solver.status == 'running':
                budget()
                previous = float(solver.t)
                message = solver.step()
                if solver.status == 'failed':
                    raise RuntimeError(f'{method} failed: {message}')
                accepted_state = _expanded_solver_state(material,solver.y)
                _physical(material,accepted_state)
                summary['accepted_steps'] += 1
                accepted_rows.append([float(solver.t),float(accepted_state[0]),float(accepted_state[1])])
                selected = np.flatnonzero((unique>previous)&(unique<=solver.t))
                if len(selected):
                    values = solver.dense_output()(unique[selected]).T
                    for native,value in zip(selected,values):
                        expanded = _expanded_solver_state(material,value)
                        _physical(material,expanded)
                        states[native] = expanded
            y = _expanded_solver_state(material,solver.y)
            row.update(final_internal_state_ux=y.tolist(),solver_nfev=solver.nfev,status='complete')
        if not np.all(np.isfinite(states)):
            raise RuntimeError('Native state interpolation has unresolved values')
        native = states[inverse][1:]
        physical = np.array([_physical(material,value) for value in native])
        arrays.update(state_u=native[:,0].copy(),state_q=physical[:,0],state_a_s=physical[:,1])
        points=[]
        for applied,state in zip(command,physical):
            budget()
            points.append(circuit.solve(float(applied),material,state))
        for key in POINT_FIELDS:
            arrays[key] = np.array([point[key] for point in points],dtype=bool if key in (
                'compliance_limited','nearzero_extension_used') else float)
        residual = np.empty(0)
        if observed is not None:
            if np.any(abs(arrays['current_a'][mask])<=0):
                raise DomainFailure('Nonpositive predicted current at an original scoring point')
            residual = np.log10(abs(arrays['current_a'][mask]))-np.log10(abs(observed[mask]))
            if not np.all(np.isfinite(residual)):
                raise DomainFailure('Nonfinite original-mask residual')
            summary['metrics'] = dict(n_residuals=len(residual),
                log10_rmse_decade=float(np.sqrt(np.mean(residual**2))) if len(residual) else None)
        final = _physical(material,y)
        summary.update(status='complete',final_state_q=final[0],final_state_a_s=final[1],
                       final_state_u=float(y[0]),final_command_voltage_v=float(command[-1]),
                       nearzero_omitted_state_change=0.)
        snapshot()
        return dict(summary=summary,arrays=arrays,residual=residual)
    except Exception as exc:
        summary.update(status='numerical_or_domain_failure_retained',error_type=type(exc).__name__,
                       error=str(exc),interpretation='Unresolved numerical/domain result, not a physical rejection')
        if solver is not None:
            summary['last_solver_state'] = dict(path_voltage_s=float(solver.t),
                internal_state_ux=_expanded_solver_state(material,solver.y).tolist())
        snapshot()
        raise PhaseFailure(str(exc),summary,arrays) from exc


def simulate_history(material, phases, initial_state, precision, method,
                     check_budget, collect_rhs=True):
    """One declared continuous segment; no phase initial-state overrides.

    phases contains {'data': phase_data, 'circuit': external_circuit} items.
    Missing measured circles require a separate segment/call with a declared
    unknown entry state. Numeric record IDs, when present, are checked here.
    """
    if not phases:
        raise ValueError('A nonempty declared continuous segment is required')
    state = material.validate_state(initial_state)
    # Validate all structural joins before evaluating any forward phase.
    previous_data = None
    for item in phases:
        if set(item) != {'data','circuit'}:
            raise ValueError('History items accept only data and circuit; no state overrides')
        if not isinstance(item['circuit'],(BareCircuit,SeriesCircuit)):
            raise TypeError('Every history phase requires a supported external circuit')
        data = item['data']
        waveform = np.asarray(data['command_voltage_v'],dtype=float)
        if waveform.ndim != 1 or not len(waveform) or not np.all(np.isfinite(waveform)):
            raise ValueError('Every history phase needs a finite native waveform')
        _finite(data['initial_command_voltage_v'],'phase incoming command')
        if previous_data is not None:
            last_command = float(previous_data['command_voltage_v'][-1])
            if float(data['initial_command_voltage_v']) != last_command:
                raise ValueError('History command handoff is discontinuous or omitted')
            prev_id,cur_id = previous_data.get('record_id'),data.get('record_id')
            prev_phase,cur_phase = previous_data.get('phase'),data.get('phase')
            if prev_id is not None and cur_id is not None:
                valid = ((prev_phase=='set' and cur_phase=='reset' and cur_id==prev_id)
                         or (prev_phase=='reset' and cur_phase=='set' and cur_id==prev_id+1))
                if not valid:
                    raise ValueError('Observed history must be SET/RESET/consecutive SET; no bridging missing circles')
        previous_data = data
    results,joins,residuals = [],[],[]
    for index,item in enumerate(phases):
        data = item['data']
        incoming = state
        try:
            result = simulate_phase(material,item['circuit'],data,incoming,precision,
                                    method,check_budget,collect_rhs=collect_rhs)
        except Exception as exc:
            if isinstance(exc,PhaseFailure):
                failed = dict(summary=exc.summary,arrays=exc.arrays,residual=np.empty(0))
            else:
                failed = dict(summary=dict(status='phase_input_or_setup_failure_retained',
                    error_type=type(exc).__name__,error=str(exc)),arrays={},residual=np.empty(0))
            raise HistoryFailure(str(exc),dict(summary=dict(status='failed_history_retained',
                failed_phase_index=index,joins=joins,**material.metadata()),phases=results+[failed],
                residual=np.concatenate(residuals) if residuals else np.empty(0))) from exc
        results.append(result)
        residuals.append(result['residual'])
        state = (result['summary']['final_state_q'],result['summary']['final_state_a_s'])
        joins.append(dict(phase_index=index,initial_state_q=incoming[0],initial_state_a_s=incoming[1],
                          final_state_q=state[0],final_state_a_s=state[1]))
    residual = np.concatenate(residuals)
    return dict(summary=dict(status='complete',n_phases=len(results),joins=joins,
                initial_state_q=float(initial_state[0]),initial_state_a_s=float(initial_state[1]),
                final_state_q=state[0],final_state_a_s=state[1],n_residuals=len(residual),
                log10_rmse_decade=float(np.sqrt(np.mean(residual**2))) if len(residual) else None,
                one_shared_material_object=True,no_within_segment_state_resets=True,
                **material.metadata()),phases=results,residual=residual)
