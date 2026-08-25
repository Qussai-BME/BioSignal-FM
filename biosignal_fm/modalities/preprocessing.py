"""Registry-facing preprocessing factories.

Scientific transforms remain in ``biosignal_fm.preprocessing``. This module
only chooses an existing modality-aware pipeline at the application edge.
"""

from __future__ import annotations

from ..config import Modality, PreprocessingConfig
from ..preprocessing import PreprocessingPipeline

__all__ = ["preprocessing_factory"]


def preprocessing_factory(
    modality: str,
    config: PreprocessingConfig | None = None,
) -> PreprocessingPipeline:
    """Create the existing preprocessing pipeline for a registered modality."""
    return PreprocessingPipeline(
        config=config or PreprocessingConfig(),
        modality=Modality.from_str(modality),
    )
