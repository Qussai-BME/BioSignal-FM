"""Explicit modality registry for BioSignal-FM V4.

The registry is intentionally small and in-process. It is not a plugin
marketplace; it provides one auditable location for modality capabilities,
edge adapters, preprocessing factories, and encoder factories.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from ..core import Signal

__all__ = [
    "ModalityStatus",
    "SignalAdapter",
    "ModalityPlugin",
    "ModalityRegistry",
    "default_registry",
]


class ModalityStatus(str, Enum):
    """Maturity classification communicated to users and reports."""

    CORE = "core"
    EXPERIMENTAL = "experimental"
    LEGACY_OPTIONAL = "legacy_optional"


class SignalAdapter(Protocol):
    """Boundary adapter that converts a source object to a canonical Signal."""

    def to_signal(self, source: Any) -> Signal:
        """Convert a source-specific object into the canonical signal contract."""


@dataclass(frozen=True)
class ModalityPlugin:
    """Declarative capability entry for one biosignal modality."""

    identifier: str
    display_name: str
    status: ModalityStatus
    adapter_factory: Callable[[], SignalAdapter] | None = None
    preprocessing_factory: Callable[..., Any] | None = None
    encoder_factory: Callable[..., Any] | None = None
    supported_tasks: tuple[str, ...] = ()
    visualization_capabilities: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    optional_dependencies: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        normalized = self.identifier.strip().lower()
        if not normalized:
            raise ValueError("ModalityPlugin.identifier must not be empty")
        object.__setattr__(self, "identifier", normalized)


@dataclass
class ModalityRegistry:
    """Small explicit registry that owns modality capability declarations."""

    _plugins: dict[str, ModalityPlugin] = field(default_factory=dict)

    def register(self, plugin: ModalityPlugin, *, replace: bool = False) -> None:
        """Register a plugin, rejecting accidental replacement by default."""
        if plugin.identifier in self._plugins and not replace:
            raise ValueError(f"Modality already registered: {plugin.identifier}")
        self._plugins[plugin.identifier] = plugin

    def get(self, identifier: str) -> ModalityPlugin:
        """Return a modality plugin by its canonical identifier."""
        normalized = identifier.strip().lower()
        try:
            return self._plugins[normalized]
        except KeyError as error:
            available = ", ".join(self.identifiers())
            raise KeyError(f"Unknown modality {identifier!r}. Available: {available}") from error

    def identifiers(self, *, status: ModalityStatus | None = None) -> tuple[str, ...]:
        """Return registered identifiers in stable registration order."""
        return tuple(
            identifier
            for identifier, plugin in self._plugins.items()
            if status is None or plugin.status is status
        )

    def plugins(self, *, status: ModalityStatus | None = None) -> tuple[ModalityPlugin, ...]:
        """Return plugins in stable registration order."""
        return tuple(
            plugin for plugin in self._plugins.values() if status is None or plugin.status is status
        )

    def supports(self, identifier: str, task: str) -> bool:
        """Return whether a modality advertises support for a task."""
        return task.strip().lower() in self.get(identifier).supported_tasks

    def validate_signals(self, signals: Iterable[Signal]) -> None:
        """Reject signals whose declared modality is not registered."""
        for signal in signals:
            self.get(signal.metadata.modality)

    def as_dict(self) -> Mapping[str, Mapping[str, Any]]:
        """Return a JSON-friendly registry description for reports and UIs."""
        return {
            plugin.identifier: {
                "display_name": plugin.display_name,
                "status": plugin.status.value,
                "supported_tasks": plugin.supported_tasks,
                "visualization_capabilities": plugin.visualization_capabilities,
                "datasets": plugin.datasets,
                "optional_dependencies": plugin.optional_dependencies,
                "notes": plugin.notes,
            }
            for plugin in self._plugins.values()
        }


def _lazy_preprocessing_factory(*args: Any, **kwargs: Any) -> Any:
    """Load scientific preprocessing only when a caller actually requests it."""
    from .preprocessing import preprocessing_factory

    return preprocessing_factory(*args, **kwargs)


def default_registry() -> ModalityRegistry:
    """Build the canonical V4 registry without importing optional libraries."""
    from .adapters import ECGAdapter, ECoGAdapter, EEGAdapter, EMGAdapter, FNIRSAdapter

    registry = ModalityRegistry()
    registry.register(
        ModalityPlugin(
            identifier="emg",
            display_name="Surface electromyography (EMG)",
            status=ModalityStatus.CORE,
            adapter_factory=EMGAdapter,
            preprocessing_factory=_lazy_preprocessing_factory,
            supported_tasks=("classification", "regression", "representation_learning"),
            visualization_capabilities=("timeseries", "spectrum", "channels"),
            datasets=("NinaPro DB5",),
            notes="Most mature legacy pathway; existing protocols remain configurable.",
        )
    )
    registry.register(
        ModalityPlugin(
            identifier="eeg",
            display_name="Electroencephalography (EEG)",
            status=ModalityStatus.CORE,
            adapter_factory=EEGAdapter,
            preprocessing_factory=_lazy_preprocessing_factory,
            supported_tasks=("classification", "representation_learning"),
            visualization_capabilities=("timeseries", "topography", "spectrum", "epochs"),
            datasets=("EEG Motor Movement/Imagery", "BCI Competition IV-2a", "WAY-EEG-GAL"),
            optional_dependencies=("mne",),
            notes="MNE and BIDS are adapter concerns, not core dependencies.",
        )
    )
    registry.register(
        ModalityPlugin(
            identifier="ecg",
            display_name="Electrocardiography (ECG)",
            status=ModalityStatus.CORE,
            adapter_factory=ECGAdapter,
            preprocessing_factory=_lazy_preprocessing_factory,
            supported_tasks=("classification", "rhythm_analysis", "representation_learning"),
            visualization_capabilities=("timeseries", "beats", "spectrum"),
            datasets=("MIT-BIH Arrhythmia",),
            optional_dependencies=("wfdb",),
            notes="ECG preprocessing remains distinct from EMG preprocessing.",
        )
    )
    registry.register(
        ModalityPlugin(
            identifier="ecog",
            display_name="Electrocorticography / intracranial EEG (ECoG/iEEG)",
            status=ModalityStatus.EXPERIMENTAL,
            adapter_factory=ECoGAdapter,
            preprocessing_factory=_lazy_preprocessing_factory,
            supported_tasks=("representation_learning",),
            visualization_capabilities=("timeseries", "electrode_locations", "epochs"),
            optional_dependencies=("mne",),
            notes="Architectural adapter only; no V4 benchmark claim is made.",
        )
    )
    registry.register(
        ModalityPlugin(
            identifier="fnirs",
            display_name="Functional near-infrared spectroscopy (fNIRS)",
            status=ModalityStatus.LEGACY_OPTIONAL,
            adapter_factory=FNIRSAdapter,
            preprocessing_factory=_lazy_preprocessing_factory,
            supported_tasks=("classification", "representation_learning"),
            visualization_capabilities=("timeseries", "channels"),
            optional_dependencies=("h5py",),
            notes="Preserved as a legacy-compatible optional extension; not a V4 core modality.",
        )
    )
    return registry
