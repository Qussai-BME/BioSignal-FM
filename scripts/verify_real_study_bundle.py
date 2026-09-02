"""Verify a sanitized real-data study artifact bundle before release review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(mapping: dict[str, Any], name: str) -> Any:
    if name not in mapping or mapping[name] in (None, "", [], {}):
        raise ValueError(f"Missing required evidence field: {name}")
    return mapping[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle_dir.expanduser().resolve()
    manifest_path = bundle / "manifest.json"
    metrics_path = bundle / "metrics.json"
    summary_path = bundle / "prediction_summary.json"
    for required_path in (manifest_path, metrics_path, summary_path):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _require(manifest, "experiment_id")
    if not str(manifest["experiment_id"]).startswith("exp_"):
        raise ValueError("experiment_id must begin with exp_")
    if manifest.get("research_readiness_issues"):
        raise ValueError("Manifest declares unresolved research readiness issues")
    provenance = _require(manifest, "dataset_provenance")
    protocol = _require(manifest, "protocol")
    for field_name in ("dataset_id", "dataset_version", "license_id", "origin"):
        _require(provenance, field_name)
    for field_name in ("protocol_id", "split", "metrics", "unit_of_analysis"):
        _require(protocol, field_name)
    if provenance["origin"] != "real":
        raise ValueError("This verifier only accepts real-data evidence bundles")
    if summary.get("raw_predictions_exported") is not False:
        raise ValueError(
            "Prediction summary must explicitly confirm raw predictions were not exported"
        )
    if "claim_boundary" not in metrics:
        raise ValueError("Metrics artifact must declare a claim boundary")
    output_hashes = _require(manifest, "output_hashes")
    expected_hashes = {
        "metrics.json": _sha256(metrics_path),
        "prediction_summary.json": _sha256(summary_path),
    }
    for artifact_name, expected_hash in expected_hashes.items():
        actual_hash = output_hashes.get(artifact_name)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Hash mismatch for {artifact_name}: {actual_hash!r} != {expected_hash!r}"
            )

    verification = {
        "bundle_dir": str(bundle),
        "manifest_run_id": manifest["run_id"],
        "experiment_id": manifest["experiment_id"],
        "dataset_id": provenance["dataset_id"],
        "protocol_id": protocol["protocol_id"],
        "claim_boundary": metrics["claim_boundary"],
        "artifact_hashes_verified": sorted(expected_hashes),
        "privacy_safe_summary_confirmed": True,
        "status": "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verification, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verification, sort_keys=True))


if __name__ == "__main__":
    main()
