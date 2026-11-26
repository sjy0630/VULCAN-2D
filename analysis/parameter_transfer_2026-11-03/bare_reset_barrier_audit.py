"""Supplement old bare-entry candidates with a cached-node RESET barrier.

No new spectrum, interpolation, ODE, parameter search, or optimization.  The
minimum over finitely many path states is an UPPER bound on the true path
minimum, appropriate for a necessary threshold ceiling (not a sufficient
reachability condition).
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main():
    prior_path = OUT / "bare_entry_budget_audit.json"
    table_path = ROOT / "analysis/spectral_dynamic_2026-11-03/transport_g010_v0025.npz"
    data_path = ROOT / "analysis/shared_hbn_2026-11-03/data/native_observations.npz"
    prior = json.loads(prior_path.read_text())
    assert prior["fixed_beta"] == 1.0 and prior["fixed_A_RESET"] == 0.0
    constraints = prior["aggregate_constraints"]
    with np.load(table_path) as z:
        gamma = np.exp(z["log_gamma"])
        voltage = np.exp(z["log_voltage"])
        current = np.exp(z["log_current"])
        metadata = json.loads(str(z["metadata"]))
    assert np.all(np.diff(gamma) > 0) and np.all(np.diff(voltage) > 0)
    assert np.min(np.diff(current[1], axis=1)) >= -1e-15
    assert np.min(np.diff(current[1], axis=0)) >= -1e-15
    with np.load(data_path) as z:
        wave = z[f'1r_cycle_{constraints["training_ids"][0]}_reset_voltage']
        for cid in constraints["training_ids"]:
            assert np.array_equal(wave, z[f"1r_cycle_{cid}_reset_voltage"])
        vmax = float(np.max(np.abs(wave)))
        apex = int(np.argmax(np.abs(wave)))
        tail_index = apex + 1 + int(np.argmin(np.abs(np.abs(wave[apex+1:]) - 0.2)))
        vtail = float(abs(wave[tail_index]))
    jmax = int(np.searchsorted(voltage, vmax, side="left"))
    jtail = int(np.searchsorted(voltage, vtail, side="right") - 1)
    assert 0 <= jtail < jmax < len(voltage)
    gread = constraints["SET_read_gamma_lower_ev"]
    gtail = constraints["gamma_RESET_tail_upper_ev"]
    # Strict interior states remove any ambiguity about merely reaching an
    # endpoint at which the derivative vanishes. Every admissible continuous
    # monotone RESET trajectory must CROSS each of these states.
    ids = np.flatnonzero((gamma > gtail) & (gamma < gread))
    assert len(ids) > 0
    rows = []
    for old_row in prior["cases"]:
        if "candidate" not in old_row:
            continue
        candidate = old_row["candidate"]
        gc, jr = candidate["gamma_c_ev"], candidate["JR_a"]
        u = np.log1p(gc / gamma[ids])
        # Vupper >= Vmax, so this is an upper bound for the current at a
        # concrete necessary path state, over the entire RESET waveform.
        ceiling_at_nodes = current[1, ids, jmax] * (1.0 + gc / gamma[ids])
        p = int(np.argmin(ceiling_at_nodes))
        tight = int(ids[p])
        blocked = ceiling_at_nodes <= jr
        row = {
            "gamma_c_ev": gc,
            "JR_a": jr,
            "u_read_max": math.log1p(gc / gread),
            "u_tail_min": math.log1p(gc / gtail),
            "necessary_JR_strict_upper_a": float(ceiling_at_nodes[p]),
            "JR_over_necessary_upper": float(jr / ceiling_at_nodes[p]),
            "most_restrictive_tested_state": {
                "gamma_ev": float(gamma[tight]),
                "u": float(u[p]),
                "current_upper_a": float(current[1, tight, jmax]),
                "reset_drive_upper": float(math.log(ceiling_at_nodes[p] / jr)),
            },
            "certified_barrier_node_count": int(sum(blocked)),
            "rejected_by_barrier": bool(np.any(blocked)),
            "status": "rejected" if np.any(blocked) else "not_rejected_not_validated",
        }
        if np.any(blocked):
            # Descending gamma is increasing RESET state. The largest blocked
            # gamma gives the first certified barrier among the saved nodes.
            first = int(ids[np.flatnonzero(blocked)[-1]])
            ilow = float(current[1, first, jtail])
            row["first_certified_barrier_from_LRS"] = {
                "gamma_ev": float(gamma[first]),
                "u": math.log1p(gc / gamma[first]),
                "current_upper_at_Vmax_a": float(current[1, first, jmax]),
                "return_0p2V_current_lower_a": ilow,
                "return_lower_over_empirical_upper": ilow / constraints["RESET_return_0p2V_current_Q90_a"],
            }
        rows.append(row)
    report = {
        "status": "cached_node_RESET_reachability_supplement_no_ODE_no_fit",
        "fixed_beta": 1.0,
        "fixed_A_RESET": 0.0,
        "n_old_candidates": len(rows),
        "n_rejected": sum(r["rejected_by_barrier"] for r in rows),
        "n_not_rejected": sum(not r["rejected_by_barrier"] for r in rows),
        "constraints": {
            "RESET_Vmax_actual_v": vmax,
            "RESET_Vmax_upper_cached_node_v": float(voltage[jmax]),
            "RESET_return_read_actual_v": vtail,
            "RESET_return_read_lower_cached_node_v": float(voltage[jtail]),
            "SET_read_gamma_lower_ev": gread,
            "RESET_tail_gamma_upper_ev": gtail,
            "RESET_tail_current_Q90_a": constraints["RESET_return_0p2V_current_Q90_a"],
            "n_strict_interior_gamma_nodes": len(ids),
            "gamma_node_range_ev": [float(gamma[ids[0]]), float(gamma[ids[-1]])],
        },
        "necessary_condition": "JR < inf_{uRead<u<uH} |F(-Vmax,u)| exp(u). The finite-node/Vupper result is a deliberately relaxed upper bound on this infimum.",
        "rows": rows,
        "scope": [
            "Tests exactly the five previously selected gamma_c/JR pairs; does not scan or exclude other JR values or the continuous parameter family.",
            "Every RESET trajectory satisfying the aggregate common low-read and tail state budgets must cross the sampled interior states. No physical pairing across measured records is asserted.",
            "At a barrier, positive-part drive is zero for all permitted negative voltages. The locally unique scalar flow cannot cross it regardless of positive finite kR or extra path length.",
            "A non-rejected candidate may still fail finite-time arrival, peak timing/current, complete curve gates, or shared 1R/1T1R transfer.",
            "The state law is beta1 A0. For fixed gamma_c/JR, A>0 only reduces RESET drive, but allowing JR and other parameters to change is not excluded; no whole-A>0-family conclusion follows.",
            "Fixed numerical spectrum cache and monotonic interpolation contract are assumed; this is not new spectral quadrature error certification or interval arithmetic.",
        ],
        "loaded_spectrum_metadata": metadata,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [prior_path, table_path, data_path]
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "bare_reset_barrier_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("n_old_candidates", "n_rejected", "n_not_rejected", "rows")}, indent=2))


if __name__ == "__main__":
    main()
