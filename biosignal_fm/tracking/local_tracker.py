"""Local JSON file-based tracker.

Fallback tracker when MLflow is not available. Uses the
:class:`NumpyAwareJSONEncoder` from :mod:`biosignal_fm.reproducibility`
to handle numpy and torch types properly — this is the fix for the
MyoControl v2.0 audit defect where ``LocalTracker`` used
``default=str`` which silently stringified unknown types.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..reproducibility import NumpyAwareJSONEncoder, compute_sha256

__all__ = ["LocalTracker"]


class LocalTracker:
    """JSON file-based tracker.

    Writes three files in ``output_dir``:

    - ``params.json`` — last params logged (overwritten)
    - ``metrics.jsonl`` — append-only newline-delimited JSON, one entry per step
    - ``artifacts.jsonl`` — append-only, records artifact paths + SHA-256

    Examples
    --------
    >>> from biosignal_fm.tracking import LocalTracker
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     tracker = LocalTracker(output_dir=tmp)
    ...     tracker.log_params({"lr": 1e-4})
    ...     tracker.log_metrics({"loss": 0.5}, step=1)
    ...     tracker.finish()
    """

    def __init__(self, output_dir: Path | str, run_name: str | None = None) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or f"run_{int(time.time())}"
        self._started_at = time.time()

        self._params_path = self.output_dir / "params.json"
        self._metrics_path = self.output_dir / "metrics.jsonl"
        self._artifacts_path = self.output_dir / "artifacts.jsonl"

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters (overwrites existing params)."""
        with self._params_path.open("w", encoding="utf-8") as fh:
            json.dump(params, fh, cls=NumpyAwareJSONEncoder, indent=2)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics (append to .jsonl file)."""
        entry = {"step": int(step), "timestamp": time.time(), **metrics}
        with self._metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, cls=NumpyAwareJSONEncoder) + "\n")

    def log_artifact(self, path: Path | str) -> None:
        """Record an artifact path + SHA-256 (append to .jsonl file)."""
        path = Path(path)
        sha = compute_sha256(path) if path.exists() else "missing"
        entry = {
            "path": str(path.resolve()),
            "sha256": sha,
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "logged_at": time.time(),
        }
        with self._artifacts_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, cls=NumpyAwareJSONEncoder) + "\n")

    def finish(self) -> None:
        """Mark the run as finished (writes a summary)."""
        summary = {
            "run_name": self.run_name,
            "started_at": self._started_at,
            "finished_at": time.time(),
            "duration_seconds": time.time() - self._started_at,
        }
        summary_path = self.output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, cls=NumpyAwareJSONEncoder, indent=2)
