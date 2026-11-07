"""analytics/embeddings/deep.py: the embeddings autoencoder in torch, the plan's deep candidate behind ``--deep torch``.

    python -m analytics.embeddings.train --fixtures | --data DATA_ROOT --report reports/embeddings.md --deep torch [...]

The autoencoder reads exactly what the sklearn stand-in reads: Z, the sufficient player-seasons' features after
median imputation and StandardScaler (train.standardise), and reconstructs Z. Its layers mirror the stand-in's
AE_LAYERS (32, 8, 32) with tanh, so the embedding is the 8-unit bottleneck activation in [-1, 1], the same range
the contract gives for player_embeddings:

    Linear(d -> 32) tanh, Linear(32 -> 8) tanh   (encoder; the bottleneck is the embedding)
    Linear(8 -> 32) tanh, Linear(32 -> d)        (decoder, linear output on the standardised scale)

Training (fixed in advance): Adam (LR 0.01, as the stand-in's learning_rate_init), batch BATCH, at most MAX_EPOCHS
epochs, early stopping after PATIENCE epochs without a lower validation reconstruction MSE on an inner split of the
player-seasons (by player_id: analytics.deep_torch.inner_split), then the net reset to its initial weights and
refit for the best epoch count on all of Z; torch.manual_seed(seed), CPU, torch.set_num_threads(THREADS). It
returns the same dict as train.fit_autoencoder (name "autoencoder", embed, reconstruction_mse on all of Z,
n_iter = the epochs of the returned net, the best epoch), so the PCA-or-autoencoder rule (train.decide) and the
tables are unchanged.

torch is imported inside the fit only. ``fit_autoencoder`` refuses a held-out row (ValueError: a true heldout flag
in ``rows``, or a held-out id among the training ``games``) before torch is touched.
"""
from __future__ import annotations

import numpy as np

from analytics import deep_torch as DT

WHO = "embeddings.deep"
LAYERS = (32, 8, 32)
MAX_EPOCHS = 400
PATIENCE = 25
BATCH = 32
LR = 0.01
THREADS = 1
LABEL = "torch autoencoder (32, 8, 32), tanh"


def _net(torch, d: int, layers=LAYERS):
    """Encoder (d -> layers[0] -> layers[1], tanh) and decoder (layers[1] -> layers[2] tanh -> d, linear)."""
    nn = torch.nn
    h1, k, h2 = layers

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(d, h1), nn.Tanh(), nn.Linear(h1, k), nn.Tanh())
            self.decoder = nn.Sequential(nn.Linear(k, h2), nn.Tanh(), nn.Linear(h2, d))

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return AE()


def fit_autoencoder(Z: np.ndarray, seed: int = 0, rows=None, games=None, groups=None, max_epochs: int | None = None,
                    patience: int | None = None, threads: int | None = None) -> dict:
    """The torch autoencoder on Z -> dict(name="autoencoder", embed(A) -> (n, 8), reconstruction_mse, n_iter, history).
    ``rows`` (the fitted player-season rows, aligned to Z) gives the player_id groups of the inner split when
    ``groups`` is not passed; ``rows`` and ``games`` (the training game ids) are checked for held-out material first."""
    DT.refuse_heldout_rows(rows, WHO, games=games)
    torch = DT.require_torch(WHO)
    Z = np.asarray(Z, dtype=float)
    if groups is None and rows is not None and "player_id" in getattr(rows, "columns", []):
        groups = rows["player_id"].to_numpy()
    th = THREADS if threads is None else threads
    torch.manual_seed(seed)
    net = _net(torch, Z.shape[1])
    hist = DT.train_net(torch, net, Z, Z, groups=groups, seed=seed, threads=th,
                        max_epochs=MAX_EPOCHS if max_epochs is None else max_epochs,
                        patience=PATIENCE if patience is None else patience, batch_size=BATCH, lr=LR, who=WHO)
    encoder = net.encoder

    def embed(A, enc=encoder):
        """The bottleneck activation (the encoder's tanh output) of A."""
        return DT.predict_net(torch, enc, np.asarray(A, dtype=float), threads=th)

    rec = DT.predict_net(torch, net, Z, threads=th)
    return {"name": "autoencoder", "embed": embed, "reconstruction_mse": float(np.mean((rec - Z) ** 2)),
            "n_iter": int(hist["best_epoch"]), "history": hist}
