"""Training-only terminal-current refinement of the RESET transition law.

The physical model and parameter count are unchanged.  The existing
``e50_mvc`` and ``width_mvc`` are initialized from de-embedded observable
states, then refitted against complete training RESET terminal currents.  This
removes a calibration-objective mismatch without adding a second state or a
new transport term.  Held-out target currents must never enter this function.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from . import model_unified as U
from . import model_v11 as v11
from . import model_v12 as v12


@dataclass(frozen=True)
class TerminalResetTransitionRefinement:
    e50_mvc: float
    width_mvc: float
    initial_rmse_decade: float
    refined_rmse_decade: float
    n_training_cycles: int
    n_training_points: int
    n_function_evaluations: int
    optimizer_status: int
    optimizer_message: str
    current_floor_a: float = 1e-12


def _same_reset_grid(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("At least one training row is required")
    reference = np.asarray(rows[0]["reset_voltage"], dtype=float)
    if any(not np.array_equal(
            reference, np.asarray(row["reset_voltage"], dtype=float))
           for row in rows[1:]):
        raise ValueError(
            "Terminal RESET refinement currently requires a common voltage grid")
    return reference


def fit_terminal_reset_transition(
        bundle, training_rows, base_params, reset_mapping, spatial=None,
        current_floor_a=1e-12, max_nfev=60):
    """Refit only RESET ``E50`` and width using training terminal currents."""
    training_rows = list(training_rows)
    if len(training_rows) < 4:
        raise ValueError("At least four training cycles are required")
    spatial = U.SpatialConfiguration.mean_field() if spatial is None else spatial
    reset_voltage = _same_reset_grid(training_rows)
    set_voltage = np.asarray(training_rows[0]["set_voltage"], dtype=float)
    if any(not np.array_equal(
            set_voltage, np.asarray(row["set_voltage"], dtype=float))
           for row in training_rows[1:]):
        raise ValueError(
            "Terminal RESET refinement currently requires a common SET grid")

    set_result = U.run_unified_set(
        set_voltage, training_rows[0].get("gate_V", 1.1),
        bundle["set_transport"], bundle["set_transition"],
        spatial=spatial)
    reset_params = v11.params_with_transport_law(
        base_params, bundle["reset_transport"])
    gate_voltage = float(bundle["reset_transition"].gate_voltage_v)
    g_lrs = bundle["link"].predict_reset_lrs_amplitude(
        set_result["a_connected"],
        bundle["set_transport"].connected_exponent,
        reset_params.reset_alpha_lrs_identified)
    a_hrs = v12._cap_reset_endpoint(
        bundle["endpoint"].predict(), g_lrs, reset_params)
    initial_state = set_result["z_local"][-1]

    observed = np.asarray(
        [np.abs(np.asarray(row["reset_current"], dtype=float))
         for row in training_rows], dtype=float)
    voltage_mask = np.abs(reset_voltage) > 1e-12
    valid = (
        np.isfinite(observed)
        & (observed >= float(current_floor_a))
        & voltage_mask[None, :]
    )
    if np.count_nonzero(valid) < 100:
        raise ValueError("Too few valid terminal-current training points")

    def predicted(values):
        e50 = float(values[0])
        width = float(np.exp(values[1]))
        result = U.run_unified_reset(
            reset_voltage, gate_voltage, initial_state,
            g_lrs, a_hrs, reset_params, spatial, reset_mapping,
            e50, width)
        return np.abs(result["Is"])

    log_observed = np.log10(observed[valid])

    def residual(values):
        current = np.broadcast_to(predicted(values), observed.shape)
        return np.log10(current[valid]) - log_observed

    transition = bundle["reset_transition"]
    initial = np.array([
        transition.e50_mvc,
        np.log(transition.width_mvc),
    ], dtype=float)
    initial_residual = residual(initial)
    fit = least_squares(
        residual,
        initial,
        bounds=([0.30, np.log(0.01)], [1.50, np.log(0.80)]),
        x_scale=[0.20, 0.40],
        max_nfev=int(max_nfev),
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    if not fit.success or np.any(~np.isfinite(fit.x)):
        raise RuntimeError(
            f"Terminal RESET refinement failed: {fit.message}")
    refined_residual = residual(fit.x)
    return TerminalResetTransitionRefinement(
        e50_mvc=float(fit.x[0]),
        width_mvc=float(np.exp(fit.x[1])),
        initial_rmse_decade=float(np.sqrt(np.mean(initial_residual ** 2))),
        refined_rmse_decade=float(np.sqrt(np.mean(refined_residual ** 2))),
        n_training_cycles=len(training_rows),
        n_training_points=int(np.count_nonzero(valid)),
        n_function_evaluations=int(fit.nfev),
        optimizer_status=int(fit.status),
        optimizer_message=str(fit.message),
        current_floor_a=float(current_floor_a),
    )


def simulate_with_terminal_refinement(
        row, bundle, refinement, base_params, reset_mapping, spatial=None):
    """Predict one target cycle without reading either target current trace."""
    return U.run_unified_full_cycle(
        row["set_voltage"], row["reset_voltage"],
        float(bundle["reset_transition"].gate_voltage_v),
        bundle["set_transport"], bundle["set_transition"],
        bundle["link"], bundle["endpoint"], bundle["reset_transport"],
        reset_base_params=base_params,
        reset_mapping=reset_mapping,
        reset_transition_law=bundle["reset_transition"],
        reset_e50_mvc=refinement.e50_mvc,
        reset_width_mvc=refinement.width_mvc,
        spatial=spatial,
    )
