"""Conditional classical PF consistency check; not a permittivity measurement.

Run from any directory with Python 3. Uses the archived full-cycle calibration,
not a new fit. Temperature is assumed because no temperature series is used.
"""
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "full_cycle_autonomous_summary.json"
Q = 1.602176634e-19  # C, exact
KB = 1.380649e-23  # J/K, exact
EPS0 = 8.8541878188e-12  # F/m, CODATA 2022
D_M = 6e-9


def find_value(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_value(value, key)
            if found is not None:
                return found
    return None


def beta_pf(epsilon_r, temperature_k, thickness_m=D_M):
    return math.sqrt(Q ** 3 / (math.pi * EPS0 * epsilon_r * thickness_m)) / (KB * temperature_k)


def epsilon_apparent(beta, temperature_k, thickness_m=D_M):
    return Q ** 3 / (math.pi * EPS0 * thickness_m * (KB * temperature_k * beta) ** 2)


def main():
    source = json.loads(SOURCE.read_text())
    coefficients = {
        label: float(find_value(source, key))
        for label, key in {
            "SET_pre": "SET_pre_beta_per_sqrt_V",
            "RESET_HRS": "RESET_PF_beta_per_sqrt_V",
        }.items()
    }
    rows = []
    for label, beta in coefficients.items():
        for temperature in (250.0, 300.0, 350.0):
            eps = epsilon_apparent(beta, temperature)
            # Independent forward reconstruction prevents logarithm/unit mistakes.
            assert math.isclose(beta_pf(eps, temperature), beta, rel_tol=1e-12)
            rows.append({"branch": label, "beta_per_sqrt_V": beta,
                         "assumed_temperature_K": temperature,
                         "apparent_relative_permittivity": eps})
    report = {
        "calibration_source": SOURCE.name,
        "interpretation": "Conditional classical uniform-field PF check, not measured permittivity or proof of a transport mechanism.",
        "assumed_thickness_m": D_M,
        "beta_definition": "Natural exponential in I=A*v*exp(beta*sqrt(v)); log10(I/v) slope is beta/ln(10).",
        "equations": {
            "epsilon_r": "q^3 / [pi*epsilon_0*d*(k_B*T*beta)^2]",
            "local_field": "If E_loc=f*v/d_eff, epsilon_app=epsilon_true*d_eff/(f*d_nominal).",
        },
        "si_constants": {"q_C": Q, "k_B_J_per_K": KB, "epsilon_0_F_per_m": EPS0},
        "results": rows,
        "external_comparator": {
            "doi": "10.1088/2053-1591/ac4fe1",
            "dynamic_epsilon_r": 3.1,
            "note": "Pierret et al., different 10–100 nm Au/hBN/Au capacitors; comparison is not a measured parameter of this target device.",
            "predicted_beta_at_300K_6nm": beta_pf(3.1, 300.0),
        },
        "limitations": [
            "No target temperature series; 250/300/350 K are assumptions, not observations.",
            "Degraded CVD hBN/interfaces need not have the same dielectric response as reference capacitors.",
            "Shorter effective gap or field enhancement alone increases beta and does not explain a smaller coefficient in this classical model.",
            "Schottky requires its own prefactor and refit; the present PF-like beta cannot be reused unchanged.",
            "This is an internal consistency check using an already selected compact law, not a new validation dataset.",
        ],
    }
    output = ROOT / "pf_parameter_consistency_2026-09-20.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "results": rows,
                      "reference_beta": report["external_comparator"]["predicted_beta_at_300K_6nm"]}, indent=2))


if __name__ == "__main__":
    main()
