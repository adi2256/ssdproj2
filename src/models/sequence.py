"""
Temporal sequence model.

A small GRU over a drive's SMART trajectory, replacing the base paper's
Random Forest back end.

THIS IS NOT A NOVELTY CLAIM. Koh et al. (IEEE Access 2024) and MVTRF
(FAST 2023) already established that modelling SMART data as a time
series beats treating each day as an independent snapshot. This module
exists because the project needs a reasonable modern back end, and it
gets one ablation in the paper. It stays out of the abstract.

INTERFACE

Matches RandomForestBaseline, except the input is 3-D:

    fit(X, y)               X is (n_samples, window_len, n_features)
    predict_proba(X) -> p   p is (n_samples,), P(fails within horizon)

CLASS IMBALANCE
    `pos_weight` on the loss, defaulting to the empirical
    negative-to-positive ratio. As with the forest, this changes the
    scale of the output probabilities. Conformal coverage is unaffected
    -- the guarantee holds for any score function, calibrated or not.

SIZE
    Deliberately small: one GRU layer, 32 hidden units, ~5k parameters.
    The project must run on a laptop or a free Colab tier, and with a
    few thousand failed drives a larger model would overfit long before
    it helped. Early stopping on a validation slice carved from the
    training windows -- never from calibration or test.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:                                   # pragma: no cover
    torch = None
    nn = object
    HAVE_TORCH = False

from ..config import CFG


class _GRUNet(nn.Module if HAVE_TORCH else object):
    """GRU over the trajectory, linear head on the final hidden state."""

    def __init__(self, n_features, hidden=32, layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden,
            num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)   # logits


class SequenceModel:
    """
        m = SequenceModel(seed=0).fit(X_train, y_train)
        p = m.predict_proba(X_test)

    X is (n, window_len, n_features), as produced by make_windows.
    """

    name = "gru"

    def __init__(self, hidden: int = 32, layers: int = 1, dropout: float = 0.0,
                 lr: float = 1e-2, epochs: int = 60, batch_size: int = 256,
                 val_frac: float = 0.15, patience: int = 8,
                 pos_weight: float | None = None, seed: int = 0,
                 device: str = "cpu", verbose: bool = False):
        if not HAVE_TORCH:
            raise ImportError(
                "SequenceModel requires torch. Install it, or use "
                "RandomForestBaseline on flattened windows."
            )
        self.hidden, self.layers, self.dropout = hidden, layers, dropout
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.val_frac, self.patience = val_frac, patience
        self.pos_weight = pos_weight
        self.seed, self.device, self.verbose = seed, device, verbose

        self.net = None
        self.n_features_ = None
        self.window_len_ = None
        self.history_: list[dict] = []

    # -----------------------------------------------------------
    def _check(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                f"X must be (n, window_len, n_features), got {X.shape}"
            )
        return np.nan_to_num(X)

    def fit(self, X, y) -> "SequenceModel":
        X = self._check(X)
        y = np.asarray(y).ravel().astype(np.float32)
        if len(X) != len(y):
            raise ValueError("X and y length mismatch")
        if len(X) == 0:
            raise ValueError("empty training set")

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.window_len_, self.n_features_ = X.shape[1], X.shape[2]

        # Validation slice comes out of TRAINING windows only. Taking it
        # from calibration would contaminate the conformal scores.
        idx = rng.permutation(len(X))
        n_val = max(1, int(round(self.val_frac * len(X))))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        if len(tr_idx) == 0:
            tr_idx, val_idx = idx, idx

        dev = torch.device(self.device)
        Xtr = torch.tensor(X[tr_idx], device=dev)
        ytr = torch.tensor(y[tr_idx], device=dev)
        Xva = torch.tensor(X[val_idx], device=dev)
        yva = torch.tensor(y[val_idx], device=dev)

        pw = self.pos_weight
        if pw is None:
            n_pos = float(ytr.sum().item())
            pw = ((len(ytr) - n_pos) / n_pos) if n_pos > 0 else 1.0
        pw_t = torch.tensor([max(pw, 1.0)], device=dev)

        self.net = _GRUNet(self.n_features_, self.hidden, self.layers,
                           self.dropout).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw_t)

        best, best_state, since = float("inf"), None, 0
        n = len(Xtr)

        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, device=dev)
            total = 0.0
            for s in range(0, n, self.batch_size):
                b = perm[s:s + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(b)

            self.net.eval()
            with torch.no_grad():
                vloss = float(loss_fn(self.net(Xva), yva).item())
            self.history_.append({"epoch": epoch,
                                  "train_loss": total / max(n, 1),
                                  "val_loss": vloss})
            if self.verbose:
                print(f"epoch {epoch:3d}  train {total / max(n, 1):.4f}  "
                      f"val {vloss:.4f}")

            if vloss < best - 1e-5:
                best, since = vloss, 0
                best_state = {k: v.detach().clone()
                              for k, v in self.net.state_dict().items()}
            else:
                since += 1
                if since >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("call fit() before predict_proba()")
        X = self._check(X)
        if X.shape[2] != self.n_features_:
            raise ValueError(
                f"expected {self.n_features_} features, got {X.shape[2]}"
            )
        self.net.eval()
        out = []
        dev = torch.device(self.device)
        with torch.no_grad():
            for s in range(0, len(X), 4096):
                b = torch.tensor(X[s:s + 4096], device=dev)
                out.append(torch.sigmoid(self.net(b)).cpu().numpy())
        return (np.concatenate(out) if out
                else np.zeros(0, dtype=np.float64)).astype(np.float64)

    @property
    def n_parameters(self) -> int:
        if self.net is None:
            raise RuntimeError("call fit() first")
        return sum(p.numel() for p in self.net.parameters())


def make_sequence_model(seed: int = None, **kw) -> "SequenceModel":
    seed = seed if seed is not None else CFG.seed
    return SequenceModel(seed=seed, **kw)
