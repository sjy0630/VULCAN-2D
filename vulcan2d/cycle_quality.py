"""Screen entire sweeps for a persistent change at comparable read voltages.

The screen uses the measured branches, never a fit residual. Reports keep
the raw sheet identifiers and all decision features. It identifies failed
or nonstandard transitions within a sweep, not prior forming provenance.
"""
from dataclasses import asdict, dataclass, replace
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import validate_transition_regions as audit


READ_BANDS_V = ((.10, .30), (.30, .50), (.55, .80))
READ_RATIO_NO_CHANGE = 2.0
READ_RATIO_WEAK = 3.0
ON_FRACTION_WEAK = .10
MIN_POINTS_PER_BRANCH = 3
TRAINING_BOUNDARY = {None: 14, .9: 66, 1.1: 35, 1.3: 67}


@dataclass(frozen=True)
class CycleQuality:
    topology: str
    gate_v: float
    phase: str
    sheet: str
    status: str
    excluded: bool
    outward_conductance_s: tuple
    return_conductance_s: tuple
    expected_state_contrast: tuple
    on_conductance_fraction: tuple
    compliance_onset_v: float = None


def _read_conductances(sweep, phase):
    v = np.abs(sweep.voltage)
    current = np.abs(sweep.current)
    valid = (np.isfinite(current) & ~sweep.masks["transistor_dominated"]
             & ~sweep.masks["lookup_mismatch"] & ~sweep.masks["smu_compliance"])
    outward = sweep.masks["branch_outward"]
    result = np.full((len(READ_BANDS_V), 2), np.nan)
    for index, (start, end) in enumerate(READ_BANDS_V):
        band = valid & (v >= start-1e-8) & (v <= end+1e-8)
        for side, branch in enumerate((outward, ~outward)):
            use = band & branch
            if np.sum(use) >= MIN_POINTS_PER_BRANCH:
                # Treat observations below the analysis floor at that floor
                # rather than letting one near-zero sample imply huge gain.
                result[index, side] = np.median(
                    np.maximum(current[use], audit.CURRENT_FLOOR_A[phase]) / v[use])
    return result


def _record(key, sweep, reads, on_reference):
    topology, gate, phase = key
    out, back = reads.T
    on, off = (back, out) if phase == "SET" else (out, back)
    contrast = on / off
    fractions = on / on_reference
    clamp = sweep.masks["smu_compliance"]
    peak_hits = np.flatnonzero(clamp & sweep.masks["branch_outward"])
    first_clamp = float(abs(sweep.voltage[peak_hits[0]])) if len(peak_hits) else None

    # Earlier paired 1.1-V calibration identifies R1–R3 as missing normal
    # RESET initialization (their positive SET traces still have contrast).
    if gate == 1.1 and sweep.label in {"S1", "S2", "S3", "R1", "R2", "R3"}:
        status = "exclude_prior_abnormal_pair"
    elif topology == "1M" and phase == "SET":
        has_return_clamp = np.any(clamp & ~sweep.masks["branch_outward"])
        status = ("retain_persistent_compliance" if len(peak_hits) and has_return_clamp
                  else "review_no_persistent_compliance")
    elif np.all(np.isfinite(contrast)) and np.all(contrast <= READ_RATIO_NO_CHANGE):
        status = "exclude_no_persistent_state_change"
    elif (np.all(np.isfinite(contrast)) and np.all(contrast <= READ_RATIO_WEAK)
          and np.all(fractions < ON_FRACTION_WEAK)):
        status = "exclude_weak_on_and_weak_contrast"
    elif not np.all(np.isfinite(contrast)):
        status = "review_read_window_incomplete"
    else:
        status = "retain"

    def finite_tuple(values):
        return tuple(float(x) if np.isfinite(x) else None for x in values)
    return CycleQuality(
        topology, gate, phase, sweep.label, status, status.startswith("exclude_"),
        finite_tuple(out), finite_tuple(back), finite_tuple(contrast),
        finite_tuple(fractions), first_clamp)


def screen_groups(groups):
    """Classify each sweep using gate- and phase-specific read levels."""
    result = {}
    for key, rows in groups.items():
        values = np.stack([_read_conductances(row, key[2]) for row in rows])
        # Reference only the calibration portion of this gate and polarity.
        # Limited 1M SET rows can have no uncensored read on their return.
        training_on = values[:TRAINING_BOUNDARY[key[1]], :, 1 if key[2] == "SET" else 0]
        on_reference = np.asarray([
            np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else np.nan
            for column in training_on.T
        ])
        result[key] = [_record(key, row, measured, on_reference)
                       for row, measured in zip(rows, values)]
    paired_keys = (("1T1M", 1.1, "SET"), ("1T1M", 1.1, "RESET"))
    bad_pair_numbers = {row.sheet[1:] for key in paired_keys
                        for row in result.get(key, []) if row.excluded}
    for key in paired_keys:
        if key in result:
            result[key] = [
                replace(row, status="exclude_corresponding_pair", excluded=True)
                if row.sheet[1:] in bad_pair_numbers and not row.excluded else row
                for row in result[key]
            ]
    return result


def split_with_screen(rows, flags, start=None, stop=None):
    """Apply filtering inside the original split, leaving its boundary intact."""
    if len(rows) != len(flags) or any(row.label != flag.sheet for row, flag in zip(rows, flags)):
        raise ValueError("Quality records do not align to the original sweep order")
    return [row for row, flag in zip(rows[start:stop], flags[start:stop])
            if not flag.excluded]


def main(argv=None):
    from .validate_shared_device import load_groups

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parsed = parser.parse_args(argv)
    parsed.output_dir.mkdir(parents=True, exist_ok=True)
    groups, _ = load_groups(include_abnormal_pairs=True)
    records = [row for group in screen_groups(groups).values() for row in group]
    with (parsed.output_dir / "cycle_quality.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        for row in records:
            writer.writerow(asdict(row))
    info = {
        "provenance": "The user confirmed that the three gate voltages come from the same 1T1M device. Sweep labels alone do not establish cross-gate chronology or SET/RESET pairing outside the established 1.1 V records.",
        "read_bands_abs_v": READ_BANDS_V,
        "contrast_no_change_upper": READ_RATIO_NO_CHANGE,
        "contrast_weak_upper": READ_RATIO_WEAK,
        "on_fraction_weak_upper": ON_FRACTION_WEAK,
        "reference_counts_before_screening": {
            str(gate): n for gate, n in TRAINING_BOUNDARY.items()},
        "status": "Excluded traces leave normal-cycle calibration and evaluation. Review traces stay eligible. Prior forming or initialization cannot be dated from these traces alone.",
        "records": [asdict(row) for row in records],
    }
    (parsed.output_dir / "cycle_quality.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in records:
        if row.excluded:
            print(row.topology, row.gate_v, row.phase, row.sheet, row.status,
                  row.expected_state_contrast)


if __name__ == "__main__":
    main()
