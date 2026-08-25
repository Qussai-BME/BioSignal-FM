"""Classical biosignal baselines for comparison with BioSignal-FM.

This module implements the canonical baselines used in the biosignal ML
literature, so that BioSignal-FM can be compared on a level playing field.

Baselines implemented:

1. **LDA + TD features** — Hudgins 1993 (classical.py)
2. **SVM + TD features** — classical ML (classical.py)
3. **Random Forest + TD features** — ensemble (classical.py)
4. **CNN1D** — Atzori 2016 (classical.py)
5. **EEGNet** — Lawhern 2018 (deep.py)
6. **ResNet1D** — He 2016 adapted (deep.py)

All baselines conform to a common ``Baseline`` protocol.
"""

from __future__ import annotations

from .classical import (
    Baseline,
    CNN1DBaseline,
    LDATDBaseline,
    RandomForestTDBaseline,
    SVMTDBaseline,
    extract_td_features,
    run_baseline_loso,
)
from .deep import EEGNetBaseline, ResNet1DBaseline

__all__ = [
    "Baseline",
    "extract_td_features",
    "LDATDBaseline",
    "SVMTDBaseline",
    "RandomForestTDBaseline",
    "CNN1DBaseline",
    "EEGNetBaseline",
    "ResNet1DBaseline",
    "run_baseline_loso",
]
