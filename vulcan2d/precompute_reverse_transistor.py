"""Convert the locally measured negative-``V_DS`` transistor sweep to JSON.

The source workbook ``Bookoutput2.xls`` contains one long table with columns
``Vd``, ``Vg``, ``Ig`` and an unlabeled fourth column.  The fourth column is
treated as ``Id`` because its magnitude and gate/drain dependence match the
reported output characteristic; this interpretation remains an explicit
metadata limitation until the experimenter confirms the header.

Usage::

    python -m vulcan2d.precompute_reverse_transistor /path/to/Bookoutput2.xls
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_payload(source):
    import pandas as pd

    source = Path(source)
    frame = pd.read_excel(source, header=None)
    if frame.shape[1] < 4:
        raise ValueError("Expected at least four columns: Vd, Vg, Ig, and Id")
    header_candidates = frame.index[
        (frame.iloc[:, 0].astype(str).str.strip() == "Vd") &
        (frame.iloc[:, 1].astype(str).str.strip() == "Vg")
    ].tolist()
    if len(header_candidates) != 1:
        raise ValueError("Could not uniquely locate the Vd/Vg header row")
    header = int(header_candidates[0])
    data = frame.iloc[header + 1:, :4].copy()
    data.columns = ["Vd", "Vg", "Ig", "Id"]
    for name in data.columns:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    data = data.dropna(subset=["Vd", "Vg", "Id"])

    curves = []
    counts = set()
    for gate, group in data.groupby("Vg", sort=True):
        vds = group["Vd"].to_numpy(float)
        current = group["Id"].to_numpy(float)
        if len(vds) < 3 or abs(vds[0]) > 1e-12 or abs(vds[-1]) > 1e-12:
            raise ValueError(f"V_G={gate:g} V is not a complete 0->negative->0 sweep")
        if float(vds.min()) >= 0.0:
            raise ValueError(f"V_G={gate:g} V does not contain negative V_DS")
        counts.add(len(vds))
        curves.append({
            "gate_voltage_v": float(gate),
            "drain_voltage_sign": "negative",
            "drain_voltage_v": abs(vds).tolist(),
            "drain_current_a": abs(current).tolist(),
        })
    if len(curves) < 2 or len(counts) != 1:
        raise ValueError("Expected equal-length sweeps at multiple gate voltages")

    return {
        "schema_version": 1,
        "source_file": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "measurement_date_in_file": "2026-09-08",
        "device_label_in_file": "CHIP-CE4-10",
        "interpretation": (
            "Independent transistor output sweeps at negative physical Vd. "
            "The JSON stores |Vd| and |Id| for the magnitude-domain 1T1M solver."
        ),
        "limitations": [
            "The fourth source column is unlabeled and is provisionally interpreted as Id.",
            "Drain/source/body-well wiring and identity relative to the 1T1M cell require experimenter confirmation.",
            "Large-|Vd| gate-independent current is retained and is compatible with a parasitic/body-diode contribution, not proof of one.",
        ],
        "curves": curves,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=(Path(__file__).with_name("data") /
                 "bookoutput2_reverse_transistor_lookup.json"),
    )
    args = parser.parse_args(argv)
    payload = build_payload(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({len(payload['curves'])} gate curves)")


if __name__ == "__main__":
    main()
