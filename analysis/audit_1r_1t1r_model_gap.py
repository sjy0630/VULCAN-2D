"""Audit the existing 1R/1T1R mathematical-model gap without fitting data.

Uses deployed calibration and archived protocol statements, not fresh raw
workbook reads. The numerical counterexamples are model extrapolation tests;
they do not establish matched physical devices or a microscopic mechanism.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vulcan2d import model_v4 as M4
from vulcan2d import model_v7 as M7
from vulcan2d import model_v12 as M12


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bracketed_zero(function, lower, upper, iterations=60):
    """Independent scalar bisection, with an explicit sign-changing bracket."""
    left = float(function(lower))
    right = float(function(upper))
    if not np.isfinite(left + right) or left * right >= 0:
        raise ValueError("A finite sign-changing bracket is required")
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        value = float(function(midpoint))
        if left * value > 0:
            lower, left = midpoint, value
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def protocol_evidence():
    source = ROOT / "analysis/V5_DATA_CONSTRAINED_PHYSICS_2026-07-26.md"
    text = source.read_text()
    quotes = [
        "将裸 1R SET 的限流从错误的 10 mA 改为老师确认、原始曲线也直接显示的\n   1 mA",
        "22 圈裸 1R SET 的最大电流中位数为 `0.999990 mA`，确认其平台是 1 mA。",
    ]
    for quote in quotes:
        if quote not in text:
            raise AssertionError("Archived protocol quotation changed")
    second = ROOT / "analysis/V4_PHYSICS_UPDATE_2026-07-19.md"
    second_text = second.read_text()
    reset_quote = "RESET 未设置人为限流。"
    if reset_quote not in second_text:
        raise AssertionError("Archived RESET protocol quotation changed")
    return [
        {"path": str(source.relative_to(ROOT)), "quote": quote,
         "evidence_type": "archived project report; not independently reverified raw measurement"}
        for quote in quotes
    ] + [{"path": str(second.relative_to(ROOT)), "quote": reset_quote,
          "evidence_type": "archived project report; not instrument metadata read this run"}]


def raw_file_availability():
    result = []
    for name in ("1R数据.xlsx", "裸RRESET.xlsx"):
        path = ROOT / name
        metadata = path.stat()
        flags = getattr(metadata, "st_flags", 0)
        # macOS SF_DATALESS is 0x40000000; stat exports it on supported builds.
        dataless_bit = getattr(stat, "SF_DATALESS", 0x40000000)
        result.append({
            "path": name, "size_bytes": metadata.st_size,
            "file_flags": flags, "dataless": bool(flags & dataless_bit),
            "contents_read": False,
            "note": "Only filesystem metadata inspected; no claim of a fresh workbook fit or provenance check",
        })
    return result


def run():
    calibration = M12.default_calibration()
    law = calibration.set_transport
    bare = M4.Params()
    reset = M7.Params()
    pre_amplitude, connected_amplitude = law.pre_amplitude(), law.connected_amplitude()

    def pre(voltage):
        return M12.set_hbn_current(voltage, 0.0, pre_amplitude, connected_amplitude, law)

    def connected(voltage):
        return M12.set_hbn_current(voltage, 1.0, pre_amplitude, connected_amplitude, law)

    crossover = bracketed_zero(lambda v: connected(v) - pre(v), 1.0, 5.0)
    residual = connected(crossover) - pre(crossover)
    if abs(residual) > 1e-15:
        raise AssertionError("Envelope crossover did not close numerically")
    if not pre(5.0) > connected(5.0):
        raise AssertionError("The documented high-voltage state-order counterexample no longer holds")

    g_reset = calibration.set_reset_link.predict_reset_lrs_amplitude(
        connected_amplitude, law.connected_exponent, reset.reset_alpha_lrs_identified)
    small_signal_conductance = g_reset * reset.reset_alpha_lrs_identified
    small_signal_resistance = 1.0 / small_signal_conductance
    probe_v = 1e-6
    probe_current = M7.reset_hbn_current(probe_v, 1.0, g_reset, 1e-15, reset)
    derivative_error = abs(probe_current / probe_v / small_signal_conductance - 1.0)
    if derivative_error > 1e-9:
        raise AssertionError("RESET analytic small-signal slope differs from direct law")

    command_v = 5.0
    hard_resistance = bare.Rhard_median
    compliance = bare.Icomp_1r
    frozen_raw_current = command_v / hard_resistance
    clipped_current = min(frozen_raw_current, compliance)
    compliant_actual_voltage = hard_resistance * clipped_current
    report = {
        "date": "2026-11-03",
        "scope": "Structural audit of existing separately implemented 1R and 1T1R h-BN laws; no refit",
        "input_interpretation": "Existing default calibration and archived reports; not fresh raw measurement extraction",
        "set_envelope_extrapolation": {
            "law": "I_h(V,x)=(1-x) A_pre V exp(beta sqrt(V)) + x A_connected V**n",
            "parameters": {
                "A_pre": pre_amplitude, "beta_per_sqrt_V": law.beta_pre_per_sqrt_v,
                "A_connected": connected_amplitude, "n": law.connected_exponent,
            },
            "nonzero_crossover_voltage_hbn_V": crossover,
            "crossover_current_residual_A": residual,
            "samples": [{"voltage_hbn_V": v, "pre_current_A": pre(v),
                         "connected_current_A": connected(v),
                         "dI_dx_A": connected(v) - pre(v),
                         "connected_equivalent_resistance_ohm": v / connected(v)}
                        for v in (0.05, 0.1, 0.2, 1.0, 2.7, 5.0)],
            "meaning": "The local soft-state envelopes reverse their state-current ordering on bare-voltage extrapolation; merely removing the transistor is not a validated 1R model",
        },
        "reset_conductance_gap": {
            "LRS_amplitude_A": g_reset,
            "alpha_per_V": reset.reset_alpha_lrs_identified,
            "small_signal_conductance_S": small_signal_conductance,
            "small_signal_resistance_ohm": small_signal_resistance,
            "finite_probe_voltage_V": probe_v,
            "finite_probe_relative_slope_error": derivative_error,
            "archived_1R_hard_resistance_parameter_ohm": hard_resistance,
            "resistance_ratio_1T1R_to_1R": small_signal_resistance / hard_resistance,
            "meaning": "The calibrated soft-state endpoint and archived hard-state scale differ; this does not prove different intrinsic physics or matched device geometry",
        },
        "old_1r_clipping_counterexample": {
            "assumption": "Frozen fully formed h=1, Rhard at old default median; ideal static current compliance",
            "command_voltage_V": command_v,
            "hard_resistance_ohm": hard_resistance,
            "raw_current_at_command_A": frozen_raw_current,
            "compliance_A": compliance,
            "old_clipped_current_A": clipped_current,
            "actual_voltage_required_by_ohmic_law_V": compliant_actual_voltage,
            "constitutive_current_residual_if_command_is_treated_as_actual_A": frozen_raw_current - clipped_current,
            "meaning": "Clipping I while retaining commanded V as h-BN stress does not close the component law; a compliant source must solve actual V_h",
            "limitation": "The source workbook voltage column's command/readback semantics and instrument transient response are not established here",
        },
        "protocol_evidence": protocol_evidence(),
        "raw_file_availability": raw_file_availability(),
        "limitations": [
            "No experimental currents were fitted or scored in this audit.",
            "No fresh XLSX data read was attempted; normal-operation first-three-cycle exclusion does not authorize deleting historical files.",
            "No evidence here establishes matched 1R/1T1R cells, common geometry, paired bare SET/RESET cycles, or an identical parameter set.",
            "Counterexamples assess the old models' mathematical scope, not the viability of all common h-BN mechanisms.",
            "Different forming/history states may be legitimate, but independently resetting every branch is not state continuity.",
        ],
    }
    sources = [Path(__file__), ROOT / "vulcan2d/model_v4.py",
               ROOT / "vulcan2d/model_v7.py", ROOT / "vulcan2d/model_v12.py",
               ROOT / "vulcan2d/model_v13.py", ROOT / "vulcan2d/set_identification.py",
               ROOT / "analysis/V5_DATA_CONSTRAINED_PHYSICS_2026-07-26.md",
               ROOT / "analysis/V4_PHYSICS_UPDATE_2026-07-19.md"]
    report["source_sha256"] = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "analysis/shared_hbn_2026-11-03/structural_gap.json")
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(args.output)
    print(json.dumps({
        "crossover_V": report["set_envelope_extrapolation"]["nonzero_crossover_voltage_hbn_V"],
        "reset_R_ohm": report["reset_conductance_gap"]["small_signal_resistance_ohm"],
        "reset_R_ratio": report["reset_conductance_gap"]["resistance_ratio_1T1R_to_1R"],
        "ideal_compliant_actual_voltage_V": report["old_1r_clipping_counterexample"]["actual_voltage_required_by_ohmic_law_V"],
    }, indent=2))


if __name__ == "__main__":
    main()
