"""Classical biosignal baselines: LDA+TD, SVM+TD, RF+TD, CNN1D.

This module implements the canonical baselines used in the biosignal ML
literature, so that BioSignal-FM can be compared on a level playing field.

References:

- Hudgins, B., Parker, P., & Scott, R. N. (1993). "A new strategy for
  multifunction myoelectric control." IEEE TBME, 40(1), 82-94.
- Atzori, M., et al. (2016). "Electromyography data for non-invasive
  naturally-controlled robotic hand prostheses." Scientific Data, 1, 140053.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Only for type annotations below -- the real imports stay lazy (inside
    # fit()) so `import biosignal_fm` doesn't eagerly pull in sklearn/torch.
    import torch.nn as nn
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

__all__ = [
    "Baseline",
    "extract_td_features",
    "LDATDBaseline",
    "SVMTDBaseline",
    "RandomForestTDBaseline",
    "CNN1DBaseline",
    "run_baseline_loso",
]


# --------------------------------------------------------------------------- #
# Time-domain feature extraction (Hudgins et al. 1993)
# --------------------------------------------------------------------------- #


def extract_td_features(
    signals: np.ndarray,
    threshold: float = 0.01,
) -> np.ndarray:
    """Extract Hudgins time-domain features from biosignal windows.

    For each window of shape (n_channels, n_samples), extract 4 features
    per channel:

    1. **MAV** (Mean Absolute Value): ``mean(|x|)``
    2. **ZC** (Zero Crossings): count of sign changes (with threshold)
    3. **SSC** (Slope Sign Changes): count of sign changes in the derivative
    4. **WL** (Waveform Length): ``sum(|x[i+1] - x[i]|)``

    Parameters
    ----------
    signals : np.ndarray
        Input of shape (n_windows, n_channels, n_samples) or
        (n_channels, n_samples).
    threshold : float
        Amplitude threshold for ZC and SSC to suppress noise.

    Returns
    -------
    np.ndarray
        Features of shape (n_windows, n_channels * 4) or (n_channels * 4,)
        if input was 2D.

    References
    ----------
    Hudgins, B., Parker, P., & Scott, R. N. (1993). "A new strategy for
    multifunction myoelectric control." IEEE TBME 40(1): 82-94.
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 2:
        signals = signals[np.newaxis, ...]  # (1, C, T)
        squeeze = True
    elif signals.ndim == 3:
        squeeze = False
    else:
        raise ValueError(f"signals must be 2D or 3D, got {signals.shape}")

    n_windows, n_channels, n_samples = signals.shape
    features = np.zeros((n_windows, n_channels * 4), dtype=np.float64)

    for w in range(n_windows):
        for c in range(n_channels):
            x = signals[w, c, :]
            # Feature index for this (window, channel): 4 features per channel
            base_idx = c * 4

            # 1. MAV — Mean Absolute Value
            features[w, base_idx] = np.mean(np.abs(x))

            # 2. ZC — Zero Crossings (with threshold to avoid noise)
            sign_change = np.diff(np.sign(x))
            above_threshold = np.abs(np.diff(x)) > threshold
            features[w, base_idx + 1] = np.sum((sign_change != 0) & above_threshold)

            # 3. SSC — Slope Sign Changes
            dx = np.diff(x)
            sign_change_dx = np.diff(np.sign(dx))
            above_threshold_dx = np.abs(dx[1:] - dx[:-1]) > threshold
            features[w, base_idx + 2] = np.sum((sign_change_dx != 0) & above_threshold_dx)

            # 4. WL — Waveform Length
            features[w, base_idx + 3] = np.sum(np.abs(np.diff(x)))

    if squeeze:
        squeezed: np.ndarray = features[0]
        return squeezed
    return features


# --------------------------------------------------------------------------- #
# Baseline abstract base
# --------------------------------------------------------------------------- #


class Baseline(ABC):
    """Abstract base class for all baselines.

    Every baseline must implement ``fit``, ``predict``, and ``score``.
    This uniform interface lets the experiment runner swap baselines
    without changing the evaluation code.
    """

    name: str = "Baseline"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the baseline on features X and labels y."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for features X."""

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy on (X, y)."""
        preds = self.predict(X)
        return float(np.mean(preds == y))


# --------------------------------------------------------------------------- #
# LDA + TD features
# --------------------------------------------------------------------------- #


class LDATDBaseline(Baseline):
    """Linear Discriminant Analysis on Hudgins time-domain features.

    This is THE canonical sEMG gesture-recognition baseline (Hudgins 1993).
    LDA assumes Gaussian class-conditional distributions with shared
    covariance — a reasonable assumption for TD features after
    standardization.

    Parameters
    ----------
    standardize : bool
        If True (default), standardize features (z-score) before LDA.
    """

    name = "LDA+TD"

    def __init__(self, standardize: bool = True) -> None:
        self.standardize = standardize
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._clf: LinearDiscriminantAnalysis | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if self.standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            self._mean = scaler.mean_
            self._std = scaler.scale_
        self._clf = LinearDiscriminantAnalysis()
        self._clf.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Baseline not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        if self.standardize and self._mean is not None and self._std is not None:
            X = (X - self._mean) / self._std
        return np.asarray(self._clf.predict(X))


# --------------------------------------------------------------------------- #
# SVM + TD features
# --------------------------------------------------------------------------- #


class SVMTDBaseline(Baseline):
    """Support Vector Machine (RBF kernel) on TD features."""

    name = "SVM+TD"

    def __init__(self, C: float = 1.0, gamma: str = "scale") -> None:
        self.C = C
        self.gamma = gamma
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._clf: SVC | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        self._mean = scaler.mean_
        self._std = scaler.scale_
        self._clf = SVC(C=self.C, kernel="rbf", gamma=self.gamma, decision_function_shape="ovr")
        self._clf.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Baseline not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        if self._mean is not None and self._std is not None:
            X = (X - self._mean) / self._std
        return np.asarray(self._clf.predict(X))


# --------------------------------------------------------------------------- #
# Random Forest + TD features
# --------------------------------------------------------------------------- #


class RandomForestTDBaseline(Baseline):
    """Random Forest on TD features."""

    name = "RF+TD"

    def __init__(self, n_estimators: int = 100, max_depth: int | None = None) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._clf: RandomForestClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.ensemble import RandomForestClassifier

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self._clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
        )
        self._clf.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Baseline not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return np.asarray(self._clf.predict(X))


# --------------------------------------------------------------------------- #
# CNN1D baseline (Atzori et al. 2016 style)
# --------------------------------------------------------------------------- #


class CNN1DBaseline(Baseline):
    """Simple 1D CNN baseline (Atzori et al. 2016 style).

    Architecture:
        Conv1d(C, 32, 5) -> ReLU -> MaxPool(2)
        Conv1d(32, 64, 5) -> ReLU -> MaxPool(2)
        Conv1d(64, 128, 3) -> ReLU -> MaxPool(2)
        Flatten -> Linear -> ReLU -> Dropout(0.5) -> Linear(n_classes)

    Parameters
    ----------
    n_channels : int
    n_classes : int
    signal_length : int
    epochs : int
    batch_size : int
    lr : float
    device : str
    """

    name = "CNN1D"

    def __init__(
        self,
        n_channels: int = 16,
        n_classes: int = 8,
        signal_length: int = 400,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.signal_length = signal_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self._model: nn.Module | None = None
        self._classes: np.ndarray | None = None

    def _build_model(self) -> None:
        import torch.nn as nn

        class _CNN1D(nn.Module):
            def __init__(self, n_channels: int, n_classes: int, signal_length: int) -> None:
                super().__init__()
                self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=5, padding=2)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
                self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                self.pool = nn.MaxPool1d(2)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.5)
                L = max(1, signal_length // 8)
                self.fc1 = nn.Linear(128 * L, 64)
                self.fc2 = nn.Linear(64, n_classes)

            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.pool(self.relu(self.conv3(x)))
                x = x.flatten(1)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x

        self._model = _CNN1D(self.n_channels, self.n_classes, self.signal_length)
        self._model.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 3:
            raise ValueError(f"CNN1D expects 3D input (N, C, T), got {X.shape}")

        self._classes = np.unique(y)
        label_map = {c: i for i, c in enumerate(self._classes)}
        y_mapped = np.array([label_map[c] for c in y], dtype=np.int64)

        self._build_model()
        # _build_model() always assigns self._model as a side effect;
        # asserting narrows the type for mypy since it can't see that across
        # the method-call boundary.
        assert self._model is not None
        self._model.train()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y_mapped)
        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        for _epoch in range(self.epochs):
            for batch_x, batch_y in dl:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self._model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("Baseline not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"CNN1D expects 3D input (N, C, T), got {X.shape}")

        self._model.eval()
        X_t = torch.from_numpy(X).to(self.device)
        with torch.no_grad():
            logits = self._model(X_t)
            preds: np.ndarray = logits.argmax(dim=-1).cpu().numpy()

        if self._classes is not None:
            return np.asarray(self._classes[preds])
        return preds


# --------------------------------------------------------------------------- #
# LOSO evaluation helper
# --------------------------------------------------------------------------- #


def run_baseline_loso(
    baseline: Baseline,
    signals: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    use_td_features: bool = True,
) -> dict:
    """Run a baseline with Leave-One-Subject-Out cross-validation.

    Parameters
    ----------
    baseline : Baseline
        The baseline to evaluate. Will be re-fitted (deep-copied) for each fold.
    signals : np.ndarray
        Raw signals of shape (n_samples, n_channels, n_samples_per_window).
    labels : np.ndarray
        Class labels of shape (n_samples,).
    subject_ids : np.ndarray
        Subject IDs of shape (n_samples,).
    use_td_features : bool
        If True, extract TD features before fitting (for LDA/SVM/RF).
        If False, pass raw signals (for CNN1D).

    Returns
    -------
    dict
        {"baseline_name", "fold_accuracies", "mean_accuracy",
         "std_accuracy", "per_fold_predictions"}
    """
    signals = np.asarray(signals)
    labels = np.asarray(labels)
    subject_ids = np.asarray(subject_ids)
    unique_subjects = np.unique(subject_ids)

    fold_accuracies: list[float] = []
    per_fold_predictions: list[tuple[np.ndarray, np.ndarray]] = []

    for test_subj in unique_subjects:
        train_mask = subject_ids != test_subj
        test_mask = subject_ids == test_subj

        X_train = signals[train_mask]
        X_test = signals[test_mask]
        y_train = labels[train_mask]
        y_test = labels[test_mask]

        if use_td_features:
            X_train_feat = extract_td_features(X_train)
            X_test_feat = extract_td_features(X_test)
        else:
            X_train_feat = X_train
            X_test_feat = X_test

        fold_baseline = copy.deepcopy(baseline)
        fold_baseline.fit(X_train_feat, y_train)
        preds = fold_baseline.predict(X_test_feat)

        acc = float(np.mean(preds == y_test))
        fold_accuracies.append(acc)
        per_fold_predictions.append((y_test, preds))

    return {
        "baseline_name": baseline.name,
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "per_fold_predictions": per_fold_predictions,
    }
