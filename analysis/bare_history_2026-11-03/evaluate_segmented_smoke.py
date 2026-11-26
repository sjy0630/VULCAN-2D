"""One real 14-pair adapter check at the original first diagnostic candidate.

u5=u1 is an explicit test input, not an inferred state after missing cycle4.
This verifies implementation and reports a conditional training score; no fit.
"""
from pathlib import Path
import hashlib
import json
import signal
import time

import numpy as np

from segmented_problem import make_problem, HERE, PREVIOUS, TRAINING_SEGMENTS
from evaluate_bare_entry import OPTIONS


def main():
    started = time.monotonic()
    first_path = PREVIOUS / "bare_entry_forward.json"
    old = json.loads(first_path.read_text())
    q = np.r_[old["q"], old["q"][5]]
    output = HERE / "segmented_smoke.json"
    report = dict(status="running_single_frozen_candidate_segmented_check",
                  q=q.tolist(), options=OPTIONS, maximum_pair_calls=14,
                  initial_state_scenario="u5 equals u1 solely for a reproducible adapter check; not measured or fitted",
                  evidence_scope="Full14 training integration of the known supplied segments. No missing cycle4 trajectory, no optimization, no holdout, no improvement claim.")
    output.write_text(json.dumps(report, indent=2)+"\n")

    def timeout(*_):
        raise TimeoutError("90-second segmented check budget exhausted")

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 90.)
    try:
        problem = make_problem(integrator_options=OPTIONS)
        result = problem.evaluate(q)
        arrays = {"q":q, "residual":result["residual"]}
        comparisons = []
        with np.load(HERE / "repeat_protocol_diagnostic.npz") as previous:
            for ids in TRAINING_SEGMENTS:
                for position, cid in enumerate(ids):
                    # Existing cycles2 and3 are bitwise identical and have
                    # exactly the same terminal state. Compare later cycles
                    # to that stored fixed trajectory, without reusing it.
                    reference_cycle = min(position+1, 3)
                    for phase in ("set", "reset"):
                        trace = result["predictions"]["1r",cid,phase]
                        old_prefix = f"candidate_0_simulation_cycle_{reference_cycle}_{phase}_"
                        prefix = f"1r_cycle_{cid}_{phase}_"
                        arrays[prefix+"command_v"] = problem.waveforms["1r",phase]
                        for key in ("current_a","state_u","hbn_voltage_v","transistor_voltage_v"):
                            arrays[prefix+key] = trace[key]
                        arrays[prefix+"final_u"] = np.asarray(trace["final_u"])
                        comparisons.append(dict(record_id=cid, phase=phase,
                            reference_simulation_cycle=reference_cycle,
                            current_and_state_bit_equal=bool(
                                np.array_equal(trace["current_a"],previous[old_prefix+"current_a"])
                                and np.array_equal(trace["state_u"],previous[old_prefix+"state_u"]))))
        report.update(status="completed_full14_conditional_segmented_check_no_refit",
                      details=result["details"], checkpoint_metadata=problem.checkpoint_metadata(),
                      reference_comparisons=comparisons,
                      all28_phase_trajectories_reproduce_reference=all(x["current_and_state_bit_equal"] for x in comparisons))
        if not report["all28_phase_trajectories_reproduce_reference"]:
            raise AssertionError("Segmented trajectories differ from the frozen protocol reference")
        np.savez_compressed(output.with_suffix(".npz"), **arrays)
    except Exception as exc:
        report.update(status="failed_segmented_check_preserved",error_type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        report["elapsed_s"] = time.monotonic()-started
        report["source_sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__).resolve(),HERE/"segmented_problem.py",first_path,HERE/"repeat_protocol_diagnostic.npz")}
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report.get(k) for k in
          ("status","elapsed_s","all28_phase_trajectories_reproduce_reference","error")},indent=2))
    if report["status"] == "failed_segmented_check_preserved":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
