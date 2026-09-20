"""Validate the effective field-assisted HRS refinement without target leakage.

For each paired target ``S_i -> R_i``:

* the SET-to-initial-RESET law excludes ``R_i``;
* the HRS transport curvature excludes the complete target ``R_i`` curve;
* the protocol HRS endpoint excludes the target endpoint amplitude;
* the target RESET current is used only for scoring.

The RESET transition-field law remains the cross-gate 0.9/1.3-V result.  This
isolates the effect of correcting the HRS transport curvature at the fixed
1.1-V protocol instead of refitting the complete loop to its own target.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from . import cycle_link as CL
from . import hrs_transport as HT
from . import model_v11 as M11
from . import validate_full_cycle as VF
from . import validate_v5 as V5
from .validate_v7 import _deterministic_thresholds
from .reset_identification import _branch_mask, _log_sinh, reconstruct_reset


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "analysis" / "full_cycle_pf_refinement_summary.json"
FIGURE_PATH = ROOT / "analysis" / "figures" / "19_vulcan_pf_hrs_refinement.png"
PROTOCOL = "paired 0 to -1.7 to 0 V DC RESET"


def _log_error(observed, predicted):
    return VF._log_error(observed, predicted)


def _sinh_shape_mad(cycles, alpha):
    residuals = []
    for cycle in cycles:
        mask = _branch_mask(cycle, "hrs")
        if np.count_nonzero(mask) < 5:
            continue
        log_amplitude = (
            np.log10(cycle.current_a[mask])
            - _log_sinh(alpha, cycle.voltage_hbn_v[mask])
        )
        residuals.extend(log_amplitude - np.median(log_amplitude))
    return float(np.median(np.abs(residuals)))


def _branch_diagnostics(data, model, vmax=1.7):
    grid = np.arange(0.1, vmax + 1e-9, 0.02)
    bins = ((0.1, 0.4), (0.4, 0.8), (0.8, 1.2), (1.2, 1.7))
    result = {}
    for forward, label in ((True, "forward"), (False, "return")):
        measured, q10, q90 = V5._median_branch(data, grid, forward)
        predicted, _, _ = V5._median_branch(model, grid, forward)
        residual = np.log10(predicted) - np.log10(measured)
        branch = {
            "rmse_decade": float(np.sqrt(np.mean(residual ** 2))),
            "median_signed_decade": float(np.median(residual)),
            "fraction_inside_measured_10_90_band": float(np.mean(
                (predicted >= q10) & (predicted <= q90))),
            "voltage_bins": {},
        }
        for low, high in bins:
            select = (grid >= low) & (grid <= high + 1e-12)
            local = residual[select]
            branch["voltage_bins"][f"{low:.1f}-{high:.1f} V"] = {
                "rmse_decade": float(np.sqrt(np.mean(local ** 2))),
                "median_signed_decade": float(np.median(local)),
            }
        result[label] = branch
    return result


def _split_half_floor(actual, repetitions=400):
    rng = np.random.default_rng(20260916)
    forward = []
    returning = []
    count = len(actual)
    for _ in range(int(repetitions)):
        order = rng.permutation(count)
        left = [actual[index] for index in order[:count // 2]]
        right = [actual[index] for index in order[count // 2:]]
        score = V5._loop_rmse(left, right, vmax=1.7)
        forward.append(score["forward_decade"])
        returning.append(score["return_decade"])
    return {
        "forward_q10_median_q90_decade": [
            float(value) for value in np.quantile(forward, (0.1, 0.5, 0.9))
        ],
        "return_q10_median_q90_decade": [
            float(value) for value in np.quantile(returning, (0.1, 0.5, 0.9))
        ],
    }


def _run_pf_cycle(row, transfer_law, endpoint_law, transport_law,
                  threshold, base_p, mapping):
    return M11.run_refined_set_conditioned_cycle(
        row["set_voltage"],
        row["set_current"],
        row["reset_voltage"],
        VF.GATE_VOLTAGE,
        transfer_law,
        endpoint_law,
        transport_law,
        base_params=base_p,
        mapping=mapping,
        e50_mvc=threshold,
        width_mvc=base_p.reset_width_mvc,
    )


def _chronological_check(records, reconstructed, base_p, mapping,
                         training_count=35):
    train_records = records[:training_count]
    test_records = records[training_count:]
    train_reconstructed = reconstructed[:training_count]
    exposure = np.asarray([row["set_exposure"] for row in records])
    read_g = np.asarray([row["read_g_lrs"] for row in records])
    set_law = CL.fit_set_reset_transfer(
        exposure[:training_count], read_g[:training_count])
    pf_law = HT.fit_effective_pf_law(
        train_reconstructed, VF.GATE_VOLTAGE, PROTOCOL)
    train_amplitudes = [
        HT.estimate_hrs_amplitude(cycle, pf_law)
        for cycle in train_reconstructed
    ]
    endpoint_law = CL.fit_reset_endpoint_attractor(
        train_amplitudes, VF.GATE_VOLTAGE, PROTOCOL)
    thresholds = _deterministic_thresholds(
        len(test_records),
        base_p.reset_e50_mvc,
        base_p.reset_e50_sigma_mvc,
    )

    actual = []
    refined = []
    observed_amplitude = []
    predicted_amplitude = []
    for row, cycle, threshold in zip(
            test_records, reconstructed[training_count:], thresholds):
        actual.append((row["reset_voltage"], row["reset_current"]))
        refined.append(_run_pf_cycle(
            row, set_law, endpoint_law, pf_law, threshold, base_p, mapping))
        observed_amplitude.append(HT.estimate_hrs_amplitude(cycle, pf_law))
        predicted_amplitude.append(endpoint_law.predict())

    # Comparator: same chronological information, original sinh HRS envelope.
    old_endpoint = CL.fit_reset_endpoint_attractor(
        [row["identified_state"].g_hrs_a for row in train_records],
        VF.GATE_VOLTAGE,
        PROTOCOL,
    )
    old = []
    for row, threshold in zip(test_records, thresholds):
        old.append(CL.run_set_conditioned_cycle(
            row["set_voltage"], row["set_current"], row["reset_voltage"],
            VF.GATE_VOLTAGE, set_law, p=base_p, mapping=mapping,
            e50_mvc=threshold, width_mvc=base_p.reset_width_mvc,
            endpoint_law=old_endpoint,
        ))

    return {
        "training_cycles": int(training_count),
        "later_test_cycles": int(len(test_records)),
        "transport_law": {
            "beta_per_sqrt_V": pf_law.beta_per_sqrt_v,
            "training_shape_MAD_decade": pf_law.residual_mad_decade,
        },
        "terminal_RESET_error": {
            "original_sinh_HRS": V5._loop_rmse(
                actual, V5._as_curves(old), vmax=1.7),
            "effective_PF_HRS": V5._loop_rmse(
                actual, V5._as_curves(refined), vmax=1.7),
        },
        "PF_amplitude_endpoint_error": _log_error(
            observed_amplitude, predicted_amplitude),
    }


def main():
    base_p, mapping = VF._fit_reset_parameters()
    records, rejected = VF._load_paired_cycles(base_p, mapping)
    reconstructed = [
        reconstruct_reset(
            row["reset_voltage"], row["reset_current"],
            VF.GATE_VOLTAGE, mapping)
        for row in records
    ]
    exposure = np.asarray([row["set_exposure"] for row in records])
    read_g = np.asarray([row["read_g_lrs"] for row in records])
    _, set_loo_laws = CL.leave_one_out_transfer_predictions(exposure, read_g)
    pf_loo_laws = HT.leave_one_out_pf_laws(
        reconstructed, VF.GATE_VOLTAGE, PROTOCOL)
    full_pf_law = HT.fit_effective_pf_law(
        reconstructed, VF.GATE_VOLTAGE, PROTOCOL)

    thresholds = _deterministic_thresholds(
        len(records), base_p.reset_e50_mvc, base_p.reset_e50_sigma_mvc)
    actual = []
    original = []
    refined = []
    observed_amplitude = []
    predicted_amplitude = []
    beta_values = []

    original_endpoint_predictions, original_endpoint_laws = (
        CL.leave_one_out_endpoint_predictions(
            [row["identified_state"].g_hrs_a for row in records],
            VF.GATE_VOLTAGE,
            PROTOCOL,
        )
    )

    for target, (row, threshold, pf_law) in enumerate(zip(
            records, thresholds, pf_loo_laws)):
        training_amplitudes = [
            HT.estimate_hrs_amplitude(cycle, pf_law)
            for index, cycle in enumerate(reconstructed)
            if index != target
        ]
        endpoint_law = CL.fit_reset_endpoint_attractor(
            training_amplitudes, VF.GATE_VOLTAGE, PROTOCOL)
        actual.append((row["reset_voltage"], row["reset_current"]))
        original.append(CL.run_set_conditioned_cycle(
            row["set_voltage"], row["set_current"], row["reset_voltage"],
            VF.GATE_VOLTAGE, set_loo_laws[target], p=base_p,
            mapping=mapping, e50_mvc=threshold,
            width_mvc=base_p.reset_width_mvc,
            endpoint_law=original_endpoint_laws[target],
        ))
        refined.append(_run_pf_cycle(
            row, set_loo_laws[target], endpoint_law, pf_law,
            threshold, base_p, mapping))
        observed_amplitude.append(HT.estimate_hrs_amplitude(
            reconstructed[target], pf_law))
        predicted_amplitude.append(endpoint_law.predict())
        beta_values.append(pf_law.beta_per_sqrt_v)

    original_curves = V5._as_curves(original)
    refined_curves = V5._as_curves(refined)
    original_error = V5._loop_rmse(actual, original_curves, vmax=1.7)
    refined_error = V5._loop_rmse(actual, refined_curves, vmax=1.7)
    chronological = _chronological_check(
        records, reconstructed, base_p, mapping)

    report = {
        "model": (
            "VULCAN-2D measured-SET-conditioned full cycle with effective "
            "Poole-Frenkel-like HRS transport"
        ),
        "data": {
            "set_workbook": VF.SET_FILE,
            "reset_workbook": VF.RESET_FILE,
            "gate_voltage_V": VF.GATE_VOLTAGE,
            "protocol": PROTOCOL,
            "paired_cycles_total": 53,
            "paired_cycles_accepted": len(records),
            "rejected_cycles": rejected,
        },
        "prediction_protocol": {
            "validation": (
                "leave one complete S_i-to-R_i pair out of every fitted "
                "SET transfer, HRS curvature and HRS endpoint law"
            ),
            "target_inputs": (
                "target measured SET trace and target RESET voltage protocol"
            ),
            "target_RESET_current_used_for_prediction": False,
            "transition_field_law": (
                "unchanged 0.9/1.3-V cross-gate identification"
            ),
        },
        "effective_PF_HRS_transport": {
            "equation": "I_HRS=A_HRS*V_hBN*exp(beta_PF*sqrt(V_hBN))",
            "full_group_beta_per_sqrt_V": full_pf_law.beta_per_sqrt_v,
            "full_group_shape_MAD_decade": (
                full_pf_law.residual_mad_decade),
            "leave_one_out_beta_per_sqrt_V": {
                "minimum": float(np.min(beta_values)),
                "median": float(np.median(beta_values)),
                "maximum": float(np.max(beta_values)),
            },
            "original_sinh_shape_MAD_decade": _sinh_shape_mad(
                reconstructed, base_p.reset_alpha_hrs_identified),
            "scope": HT.claim_scope(),
        },
        "leave_one_cycle_out_validation": {
            "terminal_RESET_error": {
                "original_sinh_HRS": original_error,
                "effective_PF_HRS": refined_error,
                "relative_change_percent": {
                    "forward": float(100.0 * (
                        refined_error["forward_decade"]
                        / original_error["forward_decade"] - 1.0)),
                    "return": float(100.0 * (
                        refined_error["return_decade"]
                        / original_error["return_decade"] - 1.0)),
                },
            },
            "PF_amplitude_endpoint_error": _log_error(
                observed_amplitude, predicted_amplitude),
            "original_sinh_endpoint_error": _log_error(
                [row["identified_state"].g_hrs_a for row in records],
                original_endpoint_predictions,
            ),
            "original_branch_diagnostics": _branch_diagnostics(
                actual, original_curves),
            "effective_PF_branch_diagnostics": _branch_diagnostics(
                actual, refined_curves),
        },
        "chronological_first35_to_later16": chronological,
        "experimental_split_half_median_floor": _split_half_floor(actual),
        "claim_scope": {
            **CL.claim_scope(),
            **M11.claim_scope(M11.params_with_transport_law(
                base_p, full_pf_law)),
            "transport_model_selection": (
                "the effective PF-like curve is selected for predictive "
                "curvature; the available single-temperature data do not "
                "prove a unique microscopic conduction mechanism"
            ),
        },
        "verdict": {
            "target_RESET_current_leakage_closed": True,
            "return_error_reduced": bool(
                refined_error["return_decade"]
                < original_error["return_decade"]),
            "chronological_return_error_reduced": bool(
                chronological["terminal_RESET_error"]
                ["effective_PF_HRS"]["return_decade"]
                < chronological["terminal_RESET_error"]
                ["original_sinh_HRS"]["return_decade"]),
            "additional_experiment_required_for_this_upgrade": False,
            "remaining_limits": (
                "measured SET is still an input; no physical-time constant "
                "or unique ionic/vacancy species is identified"
            ),
        },
    }

    SUMMARY_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))
        grid = np.arange(0.1, 1.7 + 1e-9, 0.02)

        ax = axes[0, 0]
        V5._median_plot(ax, actual, grid, "measured", True,
                        x_sign=-1, color="black", lw=1.8)
        V5._median_plot(ax, actual, grid, None, False,
                        x_sign=-1, color="black", lw=1.2, ls="--")
        V5._median_plot(ax, original_curves, grid, "original sinh HRS", True,
                        x_sign=-1, color="#999999", lw=1.4)
        V5._median_plot(ax, original_curves, grid, None, False,
                        x_sign=-1, color="#999999", lw=1.0, ls="--")
        V5._median_plot(ax, refined_curves, grid, "effective PF HRS", True,
                        x_sign=-1, color="#d95f02", lw=1.8)
        V5._median_plot(ax, refined_curves, grid, None, False,
                        x_sign=-1, color="#d95f02", lw=1.2, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(1e-12, 2e-3)
        ax.set_xlabel("Applied voltage (V)")
        ax.set_ylabel("|I| (A)")
        ax.set_title("(a) Target-excluding median RESET prediction")
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        for curves, label, color in (
                (original_curves, "original sinh HRS", "#999999"),
                (refined_curves, "effective PF HRS", "#d95f02")):
            measured, _, _ = V5._median_branch(actual, grid, False)
            predicted, _, _ = V5._median_branch(curves, grid, False)
            ax.plot(grid, np.log10(predicted) - np.log10(measured),
                    lw=1.8, label=label, color=color)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xlabel("|Applied voltage| on RESET return (V)")
        ax.set_ylabel("log10(I_model/I_data)")
        ax.set_title("(b) Return-branch systematic residual")
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        labels = ["Original\nsinh HRS", "Effective\nPF HRS"]
        x = np.arange(2)
        ax.bar(x - 0.18,
               [original_error["forward_decade"],
                refined_error["forward_decade"]],
               0.36, label="forward")
        ax.bar(x + 0.18,
               [original_error["return_decade"],
                refined_error["return_decade"]],
               0.36, label="return")
        ax.set_xticks(x, labels)
        ax.set_ylabel("Median-loop RMSE (decade)")
        ax.set_title("(c) Leave-one-cycle-out validation")
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        chronological_errors = chronological["terminal_RESET_error"]
        ax.bar(x - 0.18,
               [chronological_errors["original_sinh_HRS"]
                ["forward_decade"],
                chronological_errors["effective_PF_HRS"]
                ["forward_decade"]],
               0.36, label="forward")
        ax.bar(x + 0.18,
               [chronological_errors["original_sinh_HRS"]
                ["return_decade"],
                chronological_errors["effective_PF_HRS"]
                ["return_decade"]],
               0.36, label="return")
        ax.set_xticks(x, labels)
        ax.set_ylabel("Later-16 median-loop RMSE (decade)")
        ax.set_title("(d) Train first 35, predict later 16")
        ax.legend(fontsize=8)

        fig.suptitle("VULCAN-2D: field-assisted HRS transport refinement",
                     fontsize=15, fontweight="bold")
        fig.tight_layout()
        FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("saved", SUMMARY_PATH)
    if plt is not None:
        print("saved", FIGURE_PATH)


if __name__ == "__main__":
    main()
