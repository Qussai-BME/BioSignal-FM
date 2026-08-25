"""Additional deep-learning baselines: EEGNet and ResNet1D.

These are added to the baseline suite for a more comprehensive comparison.

References:

- Lawhern, V. J., et al. (2018). "EEGNet: a compact convolutional neural
  network for EEG-based brain-computer interfaces." JNE 15(5): 056013.
- He, K., et al. (2016). "Deep Residual Learning for Image Recognition."
  CVPR 2016.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .classical import Baseline

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = ["EEGNetBaseline", "ResNet1DBaseline"]


class EEGNetBaseline(Baseline):
    """EEGNet: a compact CNN for EEG (Lawhern et al. 2018).

    Architecture (simplified):
        Conv2d(1, F1, (1, kernel_length))  # temporal filter
        DepthwiseConv2d(F1, F1, (n_channels, 1))  # spatial filter
        SeparableConv2d(F1, F2, (1, 16))
        Linear(F2 * T_reduced, n_classes)

    Parameters
    ----------
    n_channels : int
    n_classes : int
    signal_length : int
    F1 : int
        Temporal filter count. Default 8.
    F2 : int
        Output filter count. Default 16.
    kernel_length : int
        Temporal kernel length. Default 64.
    epochs : int
    batch_size : int
    lr : float
    device : str
    """

    name = "EEGNet"

    def __init__(
        self,
        n_channels: int = 8,
        n_classes: int = 4,
        signal_length: int = 400,
        F1: int = 8,
        F2: int = 16,
        kernel_length: int = 64,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.signal_length = signal_length
        self.F1 = F1
        self.F2 = F2
        self.kernel_length = kernel_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self._model: nn.Module | None = None
        self._classes: np.ndarray | None = None

    def _build_model(self) -> None:
        import torch.nn as nn
        import torch.nn.functional as F

        class _EEGNet(nn.Module):
            def __init__(
                self,
                n_channels: int,
                n_classes: int,
                signal_length: int,
                F1: int,
                F2: int,
                kernel_length: int,
            ) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2))
                self.bn1 = nn.BatchNorm2d(F1)
                # Depthwise conv (spatial)
                self.depthwise = nn.Conv2d(F1, F1, (n_channels, 1), groups=F1)
                self.bn2 = nn.BatchNorm2d(F1)
                self.pool1 = nn.AvgPool2d((1, 4))
                self.drop1 = nn.Dropout(0.5)
                # Separable conv
                self.sep_conv1 = nn.Conv2d(F1, F1, (1, 16), padding=(0, 8))
                self.sep_conv2 = nn.Conv2d(F1, F2, (1, 1))
                self.bn3 = nn.BatchNorm2d(F2)
                self.pool2 = nn.AvgPool2d((1, 8))
                self.drop2 = nn.Dropout(0.5)
                # Compute flattened size
                L = max(1, signal_length // 32)
                self.fc = nn.Linear(F2 * L, n_classes)

            def forward(self, x):
                # x: (B, C, T) -> (B, 1, C, T)
                x = x.unsqueeze(1)
                x = F.elu(self.bn1(self.conv1(x)))
                x = F.elu(self.bn2(self.depthwise(x)))
                x = self.pool1(x)
                x = self.drop1(x)
                x = F.elu(self.bn3(self.sep_conv2(self.sep_conv1(x))))
                x = self.pool2(x)
                x = self.drop2(x)
                x = x.flatten(1)
                x = self.fc(x)
                return x

        self._model = _EEGNet(
            self.n_channels,
            self.n_classes,
            self.signal_length,
            self.F1,
            self.F2,
            self.kernel_length,
        )
        self._model.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 3:
            raise ValueError(f"EEGNet expects 3D input (N, C, T), got {X.shape}")

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
        self._model.eval()
        X_t = torch.from_numpy(X).to(self.device)
        with torch.no_grad():
            logits = self._model(X_t)
            preds: np.ndarray = logits.argmax(dim=-1).cpu().numpy()
        if self._classes is not None:
            return np.asarray(self._classes[preds])
        return preds


class ResNet1DBaseline(Baseline):
    """ResNet adapted to 1D biosignals (He et al. 2016, adapted).

    A 1D ResNet with 3 residual blocks. Each block has 2 conv layers with
    a skip connection. This is a standard deep-learning baseline for
    biosignal classification.

    Parameters
    ----------
    n_channels : int
    n_classes : int
    signal_length : int
    base_filters : int
        Number of filters in the first conv layer. Default 32.
    epochs : int
    batch_size : int
    lr : float
    device : str
    """

    name = "ResNet1D"

    def __init__(
        self,
        n_channels: int = 8,
        n_classes: int = 4,
        signal_length: int = 400,
        base_filters: int = 32,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.signal_length = signal_length
        self.base_filters = base_filters
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self._model: nn.Module | None = None
        self._classes: np.ndarray | None = None

    def _build_model(self) -> None:
        import torch.nn as nn

        class _ResBlock1D(nn.Module):
            def __init__(self, channels: int) -> None:
                super().__init__()
                self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
                self.bn1 = nn.BatchNorm1d(channels)
                self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
                self.bn2 = nn.BatchNorm1d(channels)
                self.relu = nn.ReLU()

            def forward(self, x):
                identity = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out = self.relu(out + identity)
                return out

        class _ResNet1D(nn.Module):
            def __init__(
                self, n_channels: int, n_classes: int, signal_length: int, base_filters: int
            ) -> None:
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv1d(n_channels, base_filters, 7, padding=3),
                    nn.BatchNorm1d(base_filters),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                )
                self.block1 = _ResBlock1D(base_filters)
                self.block2 = _ResBlock1D(base_filters)
                self.block3 = _ResBlock1D(base_filters)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(base_filters, n_classes)

            def forward(self, x):
                x = self.stem(x)
                x = self.block1(x)
                x = self.block2(x)
                x = self.block3(x)
                x = self.pool(x).squeeze(-1)
                x = self.fc(x)
                return x

        self._model = _ResNet1D(
            self.n_channels,
            self.n_classes,
            self.signal_length,
            self.base_filters,
        )
        self._model.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 3:
            raise ValueError(f"ResNet1D expects 3D input (N, C, T), got {X.shape}")

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
        self._model.eval()
        X_t = torch.from_numpy(X).to(self.device)
        with torch.no_grad():
            logits = self._model(X_t)
            preds: np.ndarray = logits.argmax(dim=-1).cpu().numpy()
        if self._classes is not None:
            return np.asarray(self._classes[preds])
        return preds
