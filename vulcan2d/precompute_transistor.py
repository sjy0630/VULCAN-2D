"""Convert the public Nature Figure 2b workbook to a transistor lookup JSON.

The canonical source is DOI ``10.5281/zenodo.7607096`` (CC BY 4.0). Usage:

    python -m vulcan2d.precompute_transistor /path/to/Fig.2b.xlsx
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_payload(source):
    import pandas as pd

    source = Path(source)
    curves = []
    workbook = pd.ExcelFile(source)
    for sheet_name in workbook.sheet_names:
        frame = workbook.parse(sheet_name)
        if list(frame.columns[:2]) != ["DrainV", "DrainI"]:
            raise ValueError(f"Unexpected columns in {sheet_name}: {list(frame.columns)}")
        vds = frame["DrainV"].astype(float).tolist()
        current = frame["DrainI"].astype(float).tolist()
        curves.append({
            "gate_voltage_v": float(sheet_name.removesuffix("V")),
            "sheet": sheet_name,
            "drain_voltage_v": vds,
            "drain_current_a": current,
        })
    return {
        "schema_version": 1,
        "source_file": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_url": "https://zenodo.org/records/7607096",
        "source_doi": "10.5281/zenodo.7607096",
        "license": "CC-BY-4.0",
        "paper": "Zhu et al., Nature 618, 57-62 (2023)",
        "interpretation": (
            "Official Figure 2b standalone 1T output characteristics. Sweep "
            "order is preserved; each curve is 0 to 5 to 0 V."),
        "limitations": [
            "Only non-negative V_DS is provided.",
            "Device geometry, terminal convention, temperature, and mapping to the target 1T1M cell are not yet confirmed.",
        ],
        "curves": curves,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("data") / "fig2b_transistor_lookup.json")
    args = parser.parse_args(argv)
    payload = build_payload(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(payload['curves'])} gate-voltage curves)")


if __name__ == "__main__":
    main()
