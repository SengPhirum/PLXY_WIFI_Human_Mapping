"""Optional deep models (PyTorch): 1D CNN and CNN-GRU over CSI windows.

These are the plan's deep-learning candidates (§11.2). PyTorch is an
*optional* dependency — the default demo runs entirely on sklearn baselines.
Install with:  pip install -r requirements-torch.txt

The models consume flattened window feature vectors by default (same input
as the sklearn baselines, ensuring a fair comparison) reshaped internally to
(links*4 feature maps, subcarriers). They expose the sklearn fit/predict API
via :class:`TorchWindowModel`.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without torch
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class _Cnn1d(nn.Module):
        """1D CNN across the subcarrier axis; channels = per-link features."""

        def __init__(self, in_channels: int, n_out: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, 64, 5, padding=2), nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.AdaptiveAvgPool1d(8),
                nn.Flatten(),
                nn.Linear(128 * 8, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, n_out),
            )

        def forward(self, x):  # x: (batch, channels, subcarriers)
            return self.net(x)

    class _CnnGru(nn.Module):
        """CNN feature extractor + GRU over a short sequence of windows.

        For single-window input it degrades to CNN + 1-step GRU, kept for
        API compatibility; feed sequences for genuine temporal modelling.
        """

        def __init__(self, in_channels: int, n_out: int):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels, 64, 5, padding=2), nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.AdaptiveAvgPool1d(8), nn.Flatten(),
            )
            self.gru = nn.GRU(64 * 8, 128, batch_first=True)
            self.head = nn.Linear(128, n_out)

        def forward(self, x):  # x: (batch, seq, channels, subcarriers)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            b, s = x.shape[:2]
            f = self.cnn(x.flatten(0, 1)).view(b, s, -1)
            out, _ = self.gru(f)
            return self.head(out[:, -1])


class TorchWindowModel:
    """sklearn-compatible wrapper around the torch architectures."""

    def __init__(self, arch: str = "cnn", task: str = "zone", seed: int = 0,
                 epochs: int = 60, batch_size: int = 64, lr: float = 1e-3):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. The 'cnn' and 'cnn_gru' models are "
                "optional — install with: pip install -r requirements-torch.txt, "
                "or use one of the sklearn models (knn/svm/rf/mlp)."
            )
        self.arch = arch
        self.task = task
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model: nn.Module | None = None
        self.classes_: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        """Flattened feature vectors → (batch, channels, subcarriers)."""
        n_ch, n_sc = self._shape
        return X.reshape(len(X), n_ch, n_sc).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, n_subcarriers: int | None = None):
        torch.manual_seed(self.seed)
        n_sc = n_subcarriers or 52
        if X.shape[1] % n_sc != 0:
            n_sc = 1  # PCA-projected features: treat as 1D signal
        self._shape = (X.shape[1] // n_sc, n_sc)

        if self.task == "zone":
            self.classes_ = np.unique(y)
            y_idx = np.searchsorted(self.classes_, y)
            yt = torch.as_tensor(y_idx, dtype=torch.long)
            n_out = len(self.classes_)
            loss_fn = nn.CrossEntropyLoss()
        else:
            yt = torch.as_tensor(y, dtype=torch.float32)
            n_out = y.shape[1]
            loss_fn = nn.MSELoss()

        cls = _Cnn1d if self.arch == "cnn" else _CnnGru
        self.model = cls(self._shape[0], n_out)
        Xt = torch.as_tensor(self._reshape(X))
        loader = DataLoader(TensorDataset(Xt, yt),
                            batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
        return self

    @torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        out = self.model(torch.as_tensor(self._reshape(X))).numpy()
        if self.task == "zone":
            return self.classes_[out.argmax(axis=1)]
        return out
