"""Inspect a NinaPro MATLAB file and emit only sanitized structural metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, whosmat


def _scalar(value: Any) -> int | float | str | None:
    array = np.asarray(value).squeeze()
    if array.size != 1:
        return None
    item = array.item()
    return item if isinstance(item, (int, float, str)) else str(item)


def _vector_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value).squeeze()
    summary: dict[str, Any] = {"shape": list(np.asarray(value).shape), "dtype": str(array.dtype)}
    if np.issubdtype(array.dtype, np.number) and array.size:
        summary["minimum"] = float(np.nanmin(array))
        summary["maximum"] = float(np.nanmax(array))
        summary["unique_count"] = int(np.unique(array).size)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.mat_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    structures = [
        {"name": name, "shape": list(shape), "matlab_type": matlab_type}
        for name, shape, matlab_type in whosmat(source)
    ]
    fields = [
        item["name"]
        for item in structures
        if item["name"]
        in {
            "emg",
            "frequency",
            "stimulus",
            "restimulus",
            "repetition",
            "rerepetition",
            "exercise",
            "subject",
        }
    ]
    values = loadmat(source, variable_names=fields, squeeze_me=False)

    report: dict[str, Any] = {
        "source_file": source.name,
        "variables": structures,
        "emg": _vector_summary(values["emg"]) if "emg" in values else None,
        "frequency_hz": _scalar(values["frequency"]) if "frequency" in values else None,
        "exercise": _scalar(values["exercise"]) if "exercise" in values else None,
        "subject": _scalar(values["subject"]) if "subject" in values else None,
        "labels": {
            key: _vector_summary(values[key])
            for key in ("stimulus", "restimulus", "repetition", "rerepetition")
            if key in values
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
