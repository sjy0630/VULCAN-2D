"""Read-only local derivative audit at the frozen accurate baseline fit."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from refine_shared_gap import JointProblem, LOWER, UPPER, NAMES

OUT = Path(__file__).resolve().parent
SOURCES = [ROOT / "analysis/refine_shared_gap.py", ROOT / "vulcan2d/shared_gap.py",
           ROOT / "vulcan2d/shared_gap_ivp.py", OUT / "accurate_fit.json"]


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}


def main():
    started = time.monotonic()
    frozen_hashes = hashes()
    report = json.loads((OUT / "accurate_fit.json").read_text())
    selected = report["fits"][report["selected_start"]]
    q = np.array(selected["q"], float)
    options = dict(state_rtol=2e-8, state_atol=2e-10, current_rtol=2e-5)
    problem = JointProblem(**options)
    base = problem.residual(q)
    base_cost = float(np.dot(base, base)/2.)
    result = {"status": "running", "purpose": "Local derivative audit, no optimization or parameter replacement",
              "initial_q": q.tolist(), "parameter_names": NAMES,
              "source_sha256": frozen_hashes, "strict_integrator_options": options,
              "reported_optimizer_cost": selected["verified_bias_cost"],
              "reported_optimizer_optimality": selected["optimality"],
              "strict_base_cost": base_cost, "trials": [], "jacobians": {}}
    path = OUT / "finite_difference_audit.json"

    def save():
        if hashes() != frozen_hashes:
            raise RuntimeError("Audited source or baseline changed while audit was running")
        result["elapsed_seconds"] = time.monotonic()-started
        path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")

    js = {}
    for h in (1e-3, 1e-4):
        columns = []
        for j, name in enumerate(NAMES):
            trial = q.copy()
            trial[j] += h
            if np.any(trial >= UPPER) or np.any(trial <= LOWER):
                raise RuntimeError("Unintended parameter boundary crossing")
            r = problem.residual(trial)
            cost = float(np.dot(r, r)/2.)
            columns.append((r-base)/h)
            row = {"parameter": name, "index": j, "absolute_step": h,
                   "cost": cost, "cost_change_from_strict_base": cost-base_cost,
                   "forward_cost_slope": (cost-base_cost)/h,
                   "residual_change_norm": float(np.linalg.norm(r-base))}
            result["trials"].append(row)
            print(json.dumps(row), flush=True)
            save()
        jacobian = np.array(columns).T
        gradient = jacobian.T @ base
        distance = np.where(gradient < 0., UPPER-q, q-LOWER)
        result["jacobians"][str(h)] = {
            "gradient": gradient.tolist(),
            "gradient_norm_inf": float(np.max(abs(gradient))),
            "trf_bound_scaled_gradient_norm_inf": float(np.max(abs(gradient*distance))),
            "singular_values": np.linalg.svd(jacobian, compute_uv=False).tolist(),
            "column_norms": np.linalg.norm(jacobian, axis=0).tolist(),
        }
        js[h] = jacobian
        save()
    change = js[1e-4]-js[1e-3]
    result["difference_between_step_sizes"] = {
        "relative_frobenius_norm": float(np.linalg.norm(change)/np.linalg.norm(js[1e-4])),
        "relative_column_norms": (np.linalg.norm(change, axis=0)
                                  / np.maximum(np.linalg.norm(js[1e-4], axis=0), 1e-30)).tolist(),
        "gradient_sign_agreement": (np.sign(result["jacobians"]["0.001"]["gradient"])
                                    == np.sign(result["jacobians"]["0.0001"]["gradient"])).tolist(),
    }
    lower_cost = [r for r in result["trials"] if r["cost_change_from_strict_base"] < -1e-7]
    result["observed_lower_cost_neighbor_count"] = len(lower_cost)
    result["best_sampled_neighbor"] = min(result["trials"], key=lambda x: x["cost"])
    result["status"] = "completed_no_optimization"
    save()
    print(json.dumps({k: result[k] for k in ("strict_base_cost", "jacobians", "difference_between_step_sizes",
                                           "observed_lower_cost_neighbor_count", "best_sampled_neighbor")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
