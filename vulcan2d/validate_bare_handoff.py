"""Compare frozen transfer and bare calibration with a paired SET–RESET handoff."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from . import model_shared_device as model
from . import validate_shared_device as shared


def _by_cycle(rows):
    indexed = {int(row.label.lstrip("SR")): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate cycle ID")
    return indexed


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported report value: {type(value).__name__}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    groups, circuits = shared.load_groups()
    source_set, source_reset = ("1T1M", 1.1, "SET"), ("1T1M", 1.1, "RESET")
    bare_set, bare_reset = ("1M", None, "SET"), ("1M", None, "RESET")
    source_sets, source_resets = (_by_cycle(groups[key]) for key in (source_set, source_reset))
    source_train = list(range(4, 36))
    device, preparation = shared.fit_soft_set(
        [source_sets[cycle] for cycle in source_train], circuits[source_set])
    initial = shared.simulation(
        source_sets[4], "SET", device, preparation, circuits[source_set])["final"]
    device, preparation = shared.fit_soft_reset(
        [source_resets[cycle] for cycle in source_train], circuits[source_reset],
        device, preparation, initial)

    sets, resets = (_by_cycle(groups[key]) for key in (bare_set, bare_reset))
    if sets.keys() != resets.keys() or len(sets) != 22:
        raise ValueError("Expected 22 same-numbered bare SET and RESET pairs")
    train, test = sorted(sets)[:14], sorted(sets)[14:]
    set_circuit, reset_circuit = circuits[bare_set], circuits[bare_reset]
    hard_initial = model.State(hard=1.0)
    bare_device, bare_preparation = shared.fit_bare_set(
        [sets[cycle] for cycle in train], set_circuit, device, preparation)
    bare_device, bare_preparation = shared.fit_bare_reset(
        [resets[cycle] for cycle in train], reset_circuit, bare_device, bare_preparation,
        hard_initial)

    report = {
        "source_training_cycles": source_train,
        "bare_calibration_cycles": train,
        "bare_test_cycles": test,
        "protocol": "Each observed pair starts SET in State(). Paired RESET uses the simulated SET end. There is no cross-cycle state propagation.",
        "cases": {},
    }
    set_rows, reset_rows = [sets[cycle] for cycle in test], [resets[cycle] for cycle in test]
    amplitude_only = replace(preparation, off_set_a=bare_preparation.off_set_a,
                             off_reset_a=bare_preparation.off_reset_a)
    for label, law, prep in (
        ("frozen_1t1m", device, preparation),
        ("bare_off_amplitudes_only", device, amplitude_only),
        ("bare_off_amplitudes_and_soft_availability", device,
         replace(amplitude_only, soft_available=bare_preparation.soft_available)),
        ("bare_calibrated", bare_device, bare_preparation),
    ):
        set_predictions = [shared.simulation(row, "SET", law, prep, set_circuit)
                           for row in set_rows]
        paired = [shared.simulation(row, "RESET", law, prep, reset_circuit, written["final"])
                  for row, written in zip(reset_rows, set_predictions)]
        forced = [shared.simulation(row, "RESET", law, prep, reset_circuit, hard_initial)
                  for row in reset_rows]
        report["cases"][label] = {
            "device": asdict(law),
            "preparation": asdict(prep),
            "SET": shared.scores(set_rows, set_predictions, "SET", set_circuit),
            "RESET_paired": shared.scores(reset_rows, paired, "RESET", reset_circuit),
            "RESET_assumed_hard": shared.scores(reset_rows, forced, "RESET", reset_circuit),
            "SET_end_states": {str(cycle): asdict(written["final"])
                               for cycle, written in zip(test, set_predictions)},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False, default=_json_scalar) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
