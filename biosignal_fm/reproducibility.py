"""Reproducibility utilities for BioSignal-FM.

This module provides:

1. :func:`set_global_seed` — Sets deterministic seeds across Python's
   ``random``, NumPy, and PyTorch (CPU and CUDA).
2. :class:`RunManifest` — Records the full reproducibility metadata of a
   training/evaluation run: git HEAD, environment fingerprint, config hash,
   and SHA-256 hashes of all output artifacts.
3. :func:`compute_sha256` — SHA-256 of a file (used by RunManifest).
4. :func:`env_fingerprint` — Captures ``pip freeze`` and platform info.

These primitives are used to satisfy the NeurIPS / ICML reproducibility
checklist requirement that every figure in a paper can be regenerated
bit-identically from the manifest.

Example
-------
>>> from biosignal_fm.reproducibility import set_global_seed, RunManifest
>>> set_global_seed(42)
>>> manifest = RunManifest.create(name="test_run", seed=42)
>>> _ = manifest.save(Path("/tmp/manifest.json"))
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "set_global_seed",
    "compute_sha256",
    "env_fingerprint",
    "RunManifest",
]


def set_global_seed(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int, optional
        The seed value to use. Default is 42.

    Notes
    -----
    This function sets seeds for:

    - Python's :mod:`random` module
    - NumPy (global and local RNG)
    - PyTorch CPU and CUDA (if installed)
    - Environment variables ``PYTHONHASHSEED`` and ``CUBLAS_WORKSPACE_CONFIG``

    For full determinism on CUDA, call ``torch.use_deterministic_algorithms(True)``
    after this function (may raise errors for some non-deterministic ops).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_sha256(path: Path | str, chunk_size: int = 8192) -> str:
    """Compute the SHA-256 hash of a file.

    Parameters
    ----------
    path : Path or str
        Path to the file.
    chunk_size : int, optional
        Read chunk size in bytes. Default 8 KiB.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def env_fingerprint() -> dict[str, Any]:
    """Capture a fingerprint of the current Python environment.

    Returns
    -------
    dict
        Dictionary containing:

        - ``python_version`` : str
        - ``platform`` : str
        - ``platform_release`` : str
        - ``platform_version`` : str
        - ``architecture`` : str
        - ``processor`` : str
        - ``packages`` : dict[str, str] — name -> version of all installed packages
        - ``biosignal_fm_version`` : str
    """
    fp: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "packages": {},
    }

    # Capture installed packages via pip
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        for line in result.stdout.strip().splitlines():
            if "==" in line:
                name, _, version = line.partition("==")
                fp["packages"][name.lower()] = version
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        fp["packages"] = {"_error": "pip freeze failed"}

    # BioSignal-FM version
    try:
        from . import __version__

        fp["biosignal_fm_version"] = __version__
    except ImportError:
        fp["biosignal_fm_version"] = "unknown"

    return fp


def _get_git_head() -> str:
    """Get the current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "unknown"


def _get_git_dirty() -> bool:
    """Check if the git working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


@dataclass
class RunManifest:
    """Reproducibility manifest for a single training/evaluation run.

    A RunManifest records everything needed to reproduce a run:

    - ``run_id`` : unique UUID4
    - ``name`` : human-readable name
    - ``timestamp`` : ISO-8601 UTC timestamp
    - ``git_head`` : git commit hash (or 'unknown')
    - ``git_dirty`` : whether the working tree had uncommitted changes
    - ``env_fingerprint`` : full Python environment snapshot
    - ``config`` : the experiment configuration (as dict)
    - ``config_hash`` : SHA-256 of the serialized config (for quick diffing)
    - ``output_hashes`` : SHA-256 of every output artifact
    - ``metrics`` : final metrics of the run
    - ``seed`` : the random seed used
    - ``dataset_provenance`` : evidence class, source, version, and licensing
    - ``protocol`` : explicit split and metric protocol used by the run
    - ``runtime_context`` : local execution context captured at run creation
    """

    run_id: str
    name: str
    timestamp: str
    git_head: str
    git_dirty: bool
    env_fingerprint: dict[str, Any]
    config: dict[str, Any]
    config_hash: str
    output_hashes: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    seed: int = 42
    notes: str = ""
    dataset_provenance: dict[str, Any] = field(default_factory=dict)
    protocol: dict[str, Any] = field(default_factory=dict)
    runtime_context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        config: Any | dict[str, Any] | None = None,
        seed: int = 42,
        notes: str = "",
        dataset_provenance: dict[str, Any] | None = None,
        protocol: dict[str, Any] | None = None,
    ) -> RunManifest:
        """Create a new RunManifest with auto-populated metadata.

        Parameters
        ----------
        name : str
            Human-readable run name.
        config : ExperimentConfig or dict, optional
            The experiment configuration. If a dataclass, it will be
            serialized via :mod:`dataclasses.asdict`. If None, an empty
            dict is used.
        seed : int, optional
            Random seed. Default 42.
        notes : str, optional
            Free-form notes. Default "".

        Returns
        -------
        RunManifest
            A new manifest with auto-populated run_id, timestamp, git_head,
            env_fingerprint, and config_hash.
        """
        # Serialize config
        if config is None:
            config_dict: dict[str, Any] = {}
        elif isinstance(config, dict):
            config_dict = config
        else:
            try:
                # Try the ExperimentConfig.to_dict() first (handles Path/Enum)
                config_dict = config.to_dict() if hasattr(config, "to_dict") else asdict(config)
            except TypeError as e:
                # Raise instead of silently stringifying the config. The old
                # behavior ({"_raw": str(config)}) produced a meaningless SHA-256
                # hash and hid the real problem from the user.
                raise TypeError(
                    f"Cannot serialize config of type {type(config).__name__} to a "
                    f"dict for RunManifest hashing: {e}. Convert the config to a "
                    f"plain dict (or implement to_dict()) before passing it."
                ) from e

        # Compute config hash using a numpy-aware JSON encoder (NOT
        # default=str, which would silently stringify unexpected types and
        # produce a misleading hash). If the config contains objects the
        # encoder cannot handle, we raise — the user must convert explicitly.
        try:
            from .tracking.local_tracker import NumpyAwareJSONEncoder

            config_json = json.dumps(config_dict, sort_keys=True, cls=NumpyAwareJSONEncoder)
        except TypeError as e:
            raise TypeError(
                f"Cannot JSON-serialize config for hashing: {e}. "
                "Convert non-serializable fields (Path, numpy types, custom "
                "classes) to plain Python before passing the config."
            ) from e
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()

        return cls(
            run_id=str(uuid.uuid4()),
            name=name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            git_head=_get_git_head(),
            git_dirty=_get_git_dirty(),
            env_fingerprint=env_fingerprint(),
            config=config_dict,
            config_hash=config_hash,
            seed=seed,
            notes=notes,
            dataset_provenance=dict(dataset_provenance or {}),
            protocol=dict(protocol or {}),
            runtime_context={
                "cwd": str(Path.cwd()),
                "cpu_count": os.cpu_count(),
                "python_executable": sys.executable,
            },
        )

    def record_dataset_provenance(self, provenance: dict[str, Any]) -> None:
        """Record dataset provenance without silently merging contradictory values."""
        if not isinstance(provenance, dict):
            raise TypeError("provenance must be a plain dictionary")
        self.dataset_provenance = dict(provenance)

    def record_protocol(self, protocol: dict[str, Any]) -> None:
        """Record an explicit research protocol for later reproduction."""
        if not isinstance(protocol, dict):
            raise TypeError("protocol must be a plain dictionary")
        self.protocol = dict(protocol)

    def add_output(self, path: Path | str, alias: str | None = None) -> str:
        """Add an output file to the manifest and compute its SHA-256.

        Parameters
        ----------
        path : Path or str
            Path to the output file.
        alias : str, optional
            Custom key for the output_hashes dict. If None, the file name
            is used.

        Returns
        -------
        str
            The SHA-256 hash of the file.
        """
        path = Path(path)
        key = alias or path.name
        self.output_hashes[key] = compute_sha256(path)
        return self.output_hashes[key]

    def add_metric(self, name: str, value: float) -> None:
        """Record a metric in the manifest."""
        self.metrics[name] = float(value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest to a plain dict."""
        return {
            "run_id": self.run_id,
            "name": self.name,
            "timestamp": self.timestamp,
            "git_head": self.git_head,
            "git_dirty": self.git_dirty,
            "env_fingerprint": self.env_fingerprint,
            "config": self.config,
            "config_hash": self.config_hash,
            "output_hashes": self.output_hashes,
            "metrics": self.metrics,
            "seed": self.seed,
            "notes": self.notes,
            "dataset_provenance": self.dataset_provenance,
            "protocol": self.protocol,
            "runtime_context": self.runtime_context,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the manifest to a JSON string.

        Uses :class:`NumpyAwareJSONEncoder` to handle any numpy types that
        may have leaked into the metrics dict.
        """
        return json.dumps(self.to_dict(), cls=NumpyAwareJSONEncoder, indent=indent)

    def save(self, path: Path | str) -> Path:
        """Save the manifest to a JSON file.

        Parameters
        ----------
        path : Path or str
            Destination file path. Parent directories are created.

        Returns
        -------
        Path
            The absolute path of the written file.
        """
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write(self.to_json())
        return path

    @classmethod
    def load(cls, path: Path | str) -> RunManifest:
        """Load a RunManifest from a JSON file.

        Parameters
        ----------
        path : Path or str
            Path to the manifest JSON file.

        Returns
        -------
        RunManifest
            The loaded manifest.
        """
        path = Path(path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**data)


class NumpyAwareJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy and torch types gracefully.

    This replaces the dangerous ``default=str`` pattern that silently
    stringifies anything unknown — including custom objects that should
    raise. Instead, we explicitly handle common scientific types and
    raise ``TypeError`` for genuinely unknown types.
    """

    def default(self, o: Any) -> Any:
        # NumPy types
        try:
            import numpy as np

            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.generic):
                return o.item()
        except ImportError:
            pass

        # PyTorch types
        try:
            import torch

            if isinstance(o, torch.Tensor):
                return o.detach().cpu().tolist()
        except ImportError:
            pass

        # Path
        if isinstance(o, Path):
            return str(o)

        # Enum
        from enum import Enum

        if isinstance(o, Enum):
            return o.value

        # Datetime
        if isinstance(o, datetime):
            return o.isoformat()

        # Fallback — raise explicitly instead of stringify
        raise TypeError(
            f"Object of type {type(o).__name__} is not JSON serializable. "
            f"Convert it explicitly before logging."
        )
