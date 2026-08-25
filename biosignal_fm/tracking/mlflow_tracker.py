"""MLflow tracker (optional dependency).

Falls back to :class:`LocalTracker` if MLflow is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .local_tracker import LocalTracker

__all__ = ["MLflowTracker"]


class MLflowTracker:
    """MLflow-backed tracker with LocalTracker fallback.

    Parameters
    ----------
    tracking_uri : str, optional
        MLflow tracking URI. Default ``file:./mlruns``.
    experiment_name : str, optional
    run_name : str, optional
    output_dir : Path or str, optional
        If MLflow is unavailable, a LocalTracker is created here.

    Examples
    --------
    >>> from biosignal_fm.tracking import MLflowTracker
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     tracker = MLflowTracker(experiment_name="exp1", output_dir=tmp)
    ...     tracker.log_params({"lr": 1e-4})
    ...     tracker.log_metrics({"loss": 0.5}, step=1)
    ...     tracker.finish()
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "biosignal_fm",
        run_name: str | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("./mlruns_local")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fallback: LocalTracker | None = None
        self._mlflow = None
        self._run = None

        try:
            import mlflow

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
            self._run = mlflow.start_run(run_name=run_name)
        except ImportError:
            # Fall back to local JSON
            self._fallback = LocalTracker(
                output_dir=self.output_dir, run_name=run_name or experiment_name
            )

    def log_params(self, params: dict[str, Any]) -> None:
        # __init__ guarantees exactly one of _mlflow/_fallback is non-None
        # (whichever branch of the try/except ran); asserting here narrows
        # that invariant for mypy at each use site.
        if self._mlflow is not None:
            self._mlflow.log_params(params)
        else:
            assert self._fallback is not None
            self._fallback.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self._mlflow is not None:
            # Sanitize: MLflow requires float values
            sanitized = {k: float(v) for k, v in metrics.items() if v is not None}
            self._mlflow.log_metrics(sanitized, step=step)
        else:
            assert self._fallback is not None
            self._fallback.log_metrics(metrics, step)

    def log_artifact(self, path: Path | str) -> None:
        if self._mlflow is not None:
            self._mlflow.log_artifact(str(path))
        else:
            assert self._fallback is not None
            self._fallback.log_artifact(path)

    def finish(self) -> None:
        if self._mlflow is not None:
            self._mlflow.end_run()
        else:
            assert self._fallback is not None
            self._fallback.finish()
