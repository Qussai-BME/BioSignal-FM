"""Inspect a MIT-BIH WFDB record and emit only sanitized structural metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import wfdb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record_path", type=Path, help="Path without WFDB suffix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record_path = args.record_path.expanduser().resolve()
    if not record_path.with_suffix(".hea").is_file():
        raise FileNotFoundError(record_path.with_suffix(".hea"))
    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), "atr")
    report = {
        "record": record_path.name,
        "sampling_rate_hz": float(record.fs),
        "n_samples": int(record.sig_len),
        "n_channels": int(record.n_sig),
        "channel_names": list(record.sig_name),
        "units": list(record.units),
        "annotation_count": int(len(annotation.sample)),
        "annotation_symbol_counts": dict(sorted(Counter(annotation.symbol).items())),
        "annotation_sample_minimum": int(min(annotation.sample))
        if len(annotation.sample)
        else None,
        "annotation_sample_maximum": int(max(annotation.sample))
        if len(annotation.sample)
        else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
