"""Base tracker protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = ["BaseTracker"]


@runtime_checkable
class BaseTracker(Protocol):
    """Tracker protocol implemented by LocalTracker and MLflowTracker."""

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...

    def log_artifact(self, path: Path | str) -> None: ...

    def finish(self) -> None: ...
