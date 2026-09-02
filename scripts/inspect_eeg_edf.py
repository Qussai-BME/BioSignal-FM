"""Inspect an EEGMMID EDF file and emit only sanitized structural metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edf_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.edf_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = mne.io.read_raw_edf(source, preload=False, verbose="ERROR")
    events, event_ids = mne.events_from_annotations(raw, verbose="ERROR")
    report = {
        "source_file": source.name,
        "n_channels": len(raw.ch_names),
        "channel_names": list(raw.ch_names),
        "sampling_rate_hz": float(raw.info["sfreq"]),
        "n_samples": int(raw.n_times),
        "duration_seconds": float(raw.times[-1]) if raw.n_times else 0.0,
        "annotations": [
            {
                "onset_seconds": float(annotation["onset"]),
                "duration_seconds": float(annotation["duration"]),
                "description": str(annotation["description"]),
            }
            for annotation in raw.annotations
        ],
        "event_ids": {str(key): int(value) for key, value in event_ids.items()},
        "event_count": int(len(events)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
