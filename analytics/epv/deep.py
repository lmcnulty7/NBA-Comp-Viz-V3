"""analytics/epv/deep.py: the EPV sequence model, the plan's deep candidate behind ``--deep torch``.

    python -m analytics.epv.train --fixtures | --data DATA_ROOT --report reports/epv.md --deep torch [...]

The model reads exactly what the MLP stand-in reads: features.lagged_columns(N_LAGS), the state's features at t
and at the four earlier states of the same possession (t-1 ... t-4, padded with the row's own values at the start
of the possession; features.lagged reads earlier states only). The flat row is reshaped to (n, N_LAGS,
n_features) and reversed on the time axis, so the GRU runs from t-4 to t and never sees a state after t. Its last
hidden state goes through one linear layer to the expected points (points_to_end), the same regression target;
the trainer's Calibrated wrapper then fits the same isotonic map to P(points > 0) on inner out-of-fold
predictions, exactly as for the stand-in.

Architecture and training (fixed in advance, never tuned on a held-out or scored row):
    median imputation and standardisation fit on the training rows (as the stand-in's pipeline); target standardised
    GRU(n_features -> GRU_HIDDEN, one layer, batch_first) -> Linear(GRU_HIDDEN -> 1)
    Adam (LR, weight decay WEIGHT_DECAY), batch BATCH, at most MAX_EPOCHS epochs, early stopping after PATIENCE
    epochs without a lower validation MSE on an inner split of the training games (analytics.deep_torch.inner_split,
    by sportvu_game), then the net reset to its initial weights and refit for the best epoch count on every row it
    was given; torch.manual_seed(seed), CPU, torch.set_num_threads(THREADS)

torch is imported inside the fit only. ``fit_sequence`` refuses a held-out row (ValueError) before torch is touched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics import deep_torch as DT
from analytics import schema

WHO = "epv.deep"
GRU_HIDDEN = 16
MAX_EPOCHS = 100
PATIENCE = 8
BATCH = 256
LR = 2e-3
WEIGHT_DECAY = 1e-4
THREADS = 1
LABEL = "torch GRU (hidden %d) over the last 5 states" % GRU_HIDDEN


def _net(torch, n_steps: int, n_features: int, hidden: int):
    """The GRU module: a flat (n, n_steps * n_features) row in lag-0-first order -> (n, 1)."""
    nn = torch.nn

    class StateGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_features, hidden, num_layers=1, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            seq = x.reshape(x.shape[0], n_steps, n_features).flip(1)      # t-4 first, t last
            _, h = self.gru(seq)
            return self.head(h[-1])

    return StateGRU()


class GRUSequenceRegressor:
    """sklearn-like fit(X, y, groups) / predict(X) over the flat lagged matrix (NaN allowed: median-imputed)."""

    def __init__(self, n_steps: int = 5, hidden: int = GRU_HIDDEN, seed: int = 0, threads: int | None = None,
                 max_epochs: int | None = None, patience: int | None = None):
        self.n_steps, self.hidden, self.seed = n_steps, hidden, seed
        self.threads = THREADS if threads is None else threads
        self.max_epochs = MAX_EPOCHS if max_epochs is None else max_epochs
        self.patience = PATIENCE if patience is None else patience

    def fit(self, X: np.ndarray, y: np.ndarray, groups=None) -> "GRUSequenceRegressor":
        from sklearn.impute import SimpleImputer
        torch = DT.require_torch(WHO)
        X = np.asarray(X, dtype=float)
        if X.shape[1] % self.n_steps:
            raise ValueError("%s: %d columns is not a multiple of %d steps" % (WHO, X.shape[1], self.n_steps))
        self.n_features = X.shape[1] // self.n_steps
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        self.imp_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        Xi = self.imp_.transform(X)
        self.xs_, self.ys_ = DT.Standardiser().fit(Xi), DT.Standardiser().fit(y)
        torch.manual_seed(self.seed)
        self.net_ = _net(torch, self.n_steps, self.n_features, self.hidden)
        self.history_ = DT.train_net(torch, self.net_, self.xs_.transform(Xi), self.ys_.transform(y), groups=groups,
                                     seed=self.seed, threads=self.threads, max_epochs=self.max_epochs,
                                     patience=self.patience, batch_size=BATCH, lr=LR, weight_decay=WEIGHT_DECAY, who=WHO)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        torch = DT.require_torch(WHO)
        Xi = self.imp_.transform(np.asarray(X, dtype=float))
        out = DT.predict_net(torch, self.net_, self.xs_.transform(Xi), threads=self.threads)
        return self.ys_.inverse(out)[:, 0]


def fit_sequence(X: pd.DataFrame, y, columns: list, seed: int = 0, n_steps: int = 5, **kw) -> GRUSequenceRegressor:
    """The sequence model on ``X[columns]`` (features.lagged_columns order) of the training rows -> a fitted model
    whose predict(ndarray) gives expected points. Raises ValueError on a held-out row (a heldout flag or a held-out
    game id in X.sportvu_game)."""
    DT.refuse_heldout_rows(X, WHO)
    Xc = X[list(columns)]
    schema.assert_time_forward(Xc, "states_1hz")
    groups = X["sportvu_game"].astype(str).to_numpy() if "sportvu_game" in X.columns else None
    m = GRUSequenceRegressor(n_steps=n_steps, seed=seed, **kw)
    return m.fit(Xc.to_numpy(dtype=float), np.asarray(y, dtype=float), groups=groups)
