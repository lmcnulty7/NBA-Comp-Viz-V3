"""analytics/ghosting/deep.py: the ghosting GRU, the plan's deep candidate behind ``--deep torch``.

    python -m analytics.ghosting.train --fixtures | --data DATA_ROOT --report reports/ghosting.md --deep torch [...]

The GRU reads exactly what the MLP stand-in reads: the 5-step window of features.window_columns() (lag 1 block
first, then lag 2 ... lag 5, len(features.BASE_FEATURES) features per step), reshaped to (n, N_LAGS, n_features)
and reversed on the time axis so the GRU runs oldest (t minus 5) to newest (t minus 1). Its last hidden state
goes through one linear layer to the displacement from t minus 1 to t (both axes, the same target). It returns a
``predict(window_matrix) -> (n, 2)`` callable, so the trainer's roll-forward feeds it unchanged.

Architecture and training (fixed in advance, never tuned on a held-out or scored row):
    inputs standardised by the training rows' column means and deviations; targets standardised the same way
    GRU(n_features -> GRU_HIDDEN, one layer, batch_first) -> Linear(GRU_HIDDEN -> 2)
    Adam (LR), batch BATCH, at most MAX_EPOCHS epochs, early stopping after PATIENCE epochs without a lower
    validation MSE on an inner split of the training games (analytics.deep_torch.inner_split, by sportvu_game)
    torch.manual_seed(seed), CPU, torch.set_num_threads(threads) for the fit and the predictions

torch is imported inside the fit only; importing this module never needs it. ``fit_gru`` refuses a held-out row
(ValueError) before torch is touched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics import deep_torch as DT
from analytics import schema
from analytics.ghosting import features as F

WHO = "ghosting.deep"
GRU_HIDDEN = 32
MAX_EPOCHS = 100
PATIENCE = 8
BATCH = 256
LR = 2e-3
LABEL = "3 torch GRU (hidden %d) over the 5-step window" % GRU_HIDDEN


def _net(torch, n_steps: int, n_features: int, hidden: int, n_out: int):
    """The GRU module: a flat (n, n_steps * n_features) window in lag-1-first order -> (n, n_out)."""
    nn = torch.nn

    class WindowGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_features, hidden, num_layers=1, batch_first=True)
            self.head = nn.Linear(hidden, n_out)

        def forward(self, x):
            seq = x.reshape(x.shape[0], n_steps, n_features).flip(1)      # oldest step first
            _, h = self.gru(seq)
            return self.head(h[-1])

    return WindowGRU()


class GRUWindowRegressor:
    """sklearn-like fit(X, y, groups) / predict(X) over the flat window matrix."""

    def __init__(self, n_steps: int = F.N_LAGS, n_features: int | None = None, hidden: int = GRU_HIDDEN, seed: int = 0,
                 threads: int = 1, max_epochs: int | None = None, patience: int | None = None, batch_size: int = BATCH,
                 lr: float = LR):
        self.n_steps, self.n_features, self.hidden, self.seed, self.threads = n_steps, n_features, hidden, seed, threads
        self.max_epochs = MAX_EPOCHS if max_epochs is None else max_epochs
        self.patience = PATIENCE if patience is None else patience
        self.batch_size, self.lr = batch_size, lr

    def fit(self, X: np.ndarray, y: np.ndarray, groups=None) -> "GRUWindowRegressor":
        torch = DT.require_torch(WHO)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(len(X), -1)
        nf = self.n_features or X.shape[1] // self.n_steps
        if nf * self.n_steps != X.shape[1]:
            raise ValueError("%s: %d columns is not %d steps of %d features" % (WHO, X.shape[1], self.n_steps, nf))
        self.n_features = nf
        self.xs_, self.ys_ = DT.Standardiser().fit(X), DT.Standardiser().fit(y)
        torch.manual_seed(self.seed)
        self.net_ = _net(torch, self.n_steps, nf, self.hidden, y.shape[1])
        self.history_ = DT.train_net(torch, self.net_, self.xs_.transform(X), self.ys_.transform(y), groups=groups,
                                     seed=self.seed, threads=self.threads, max_epochs=self.max_epochs,
                                     patience=self.patience, batch_size=self.batch_size, lr=self.lr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        torch = DT.require_torch(WHO)
        out = DT.predict_net(torch, self.net_, self.xs_.transform(X), threads=self.threads)
        return self.ys_.inverse(out)


def fit_gru(frame: pd.DataFrame, seed: int = 0, threads: int = 1, columns: list[str] | None = None, **kw):
    """The GRU on the window columns of ``frame`` (training rows of one fold, or every training row for the held-out
    refit) -> predict(window_matrix) -> displacement (n, 2). Raises ValueError on a held-out row."""
    DT.refuse_heldout_rows(frame, WHO)
    cols = list(columns) if columns is not None else F.window_columns()
    X = frame[cols]
    schema.assert_time_forward(X, "states_1hz")
    y = frame[F.TARGETS].to_numpy(dtype=float)
    groups = frame["sportvu_game"].astype(str).to_numpy() if "sportvu_game" in frame.columns else None
    m = GRUWindowRegressor(n_steps=F.N_LAGS, n_features=len(F.BASE_FEATURES), seed=seed, threads=threads, **kw)
    m.fit(X.to_numpy(dtype=float), y, groups=groups)
    return m.predict
