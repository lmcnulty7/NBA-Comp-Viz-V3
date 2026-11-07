"""analytics/deep_torch.py: the shared torch plumbing behind the three stage 6 deep models.

The plan names three deep models, each standing in as an sklearn MLP until its baseline gate opens:
the ghosting GRU (analytics/ghosting/deep.py), the EPV sequence model (analytics/epv/deep.py) and the
embeddings autoencoder (analytics/embeddings/deep.py). Each lane's trainer takes ``--deep {mlp,torch}``
(default ``mlp``, the stand-in, so which model ships does not change unless asked). With ``--deep torch``
the torch model fills the lane's single deep slot and is kept or dropped by the lane's unchanged keep rule.

This module never imports torch at import time: ``import_torch`` imports it lazily and returns None when it
is absent, and ``resolve_deep`` turns a ``--deep torch`` request without torch into the MLP stand-in with a
one-line notice (FALLBACK_NOTICE), which the lane's report repeats.

Training (``train_net``): CPU only, ``torch.manual_seed(seed)`` before the net is built, the intra-op pool
set to ``threads`` (1 by default) for the fit and restored after, Adam on the mean squared error, minibatches
shuffled by a seeded generator, early stopping on an inner validation split of the rows passed in (the
training rows of the current fold only: never a held-out row, never the fold being scored). The split is by
group (whole games, or whole players for the embeddings) when there are at least MIN_GROUPS_FOR_GROUP_SPLIT
groups, and a seeded row split otherwise. The epoch with the lowest finite validation loss is best_epoch; the
net is then reset to its initial weights and refit for best_epoch epochs on every row passed in (the same seed),
so it learns from the same rows as the sklearn stand-in. A validation loss that is never finite raises ValueError.
torch.use_deterministic_algorithms(True) is on for the fit and the predictions (restored after), so a fit with
more than one thread reproduces too.

Held-out guard (``refuse_heldout_rows``): every lane's torch fit calls it first, so a held-out row raises
ValueError before torch is even imported (a true ``heldout`` flag, or a game id that
analytics.eval.is_heldout_game flags).
"""
from __future__ import annotations

import contextlib
import logging

import numpy as np

log = logging.getLogger("analytics.deep_torch")

DEEP_CHOICES = ("mlp", "torch")
DEFAULT_DEEP = "mlp"
FALLBACK_NOTICE = "torch not available; MLP stand-in used"
MIN_GROUPS_FOR_GROUP_SPLIT = 5
VAL_FRAC = 0.2


# ----------------------------------------------------------------------------- availability and the --deep option

def import_torch():
    """The torch module, imported lazily; None when it is not importable: absent, blocked in sys.modules, or broken
    (any exception on import is caught and logged in one line, and the caller falls back)."""
    try:
        import torch
    except ImportError:
        return None
    except Exception as e:  # noqa: BLE001  (a broken install raises anything: OSError, RuntimeError, ...)
        log.warning("torch failed to import (%s: %s); treated as not available", type(e).__name__, str(e).splitlines()[0] if str(e) else "")
        return None
    return torch


def torch_available() -> bool:
    """True when torch imports."""
    return import_torch() is not None


def resolve_deep(requested: str, who: str) -> dict:
    """The deep model a run uses -> dict(requested, used, notice). ``torch`` without torch falls back to the MLP
    stand-in with FALLBACK_NOTICE logged once (and returned for the report)."""
    if requested not in DEEP_CHOICES:
        raise ValueError("%s: --deep must be one of %s, not %r" % (who, ", ".join(DEEP_CHOICES), requested))
    if requested == "torch" and not torch_available():
        log.warning("%s: --deep torch asked but %s", who, FALLBACK_NOTICE)
        return {"requested": "torch", "used": "mlp", "notice": FALLBACK_NOTICE}
    return {"requested": requested, "used": requested, "notice": ""}


def add_deep_arg(ap) -> None:
    """The ``--deep {mlp,torch}`` flag every lane trainer with a deep candidate shares."""
    ap.add_argument("--deep", choices=DEEP_CHOICES, default=DEFAULT_DEEP,
                    help="the deep candidate: mlp (the sklearn stand-in, default) or torch (the plan's model, kept only "
                         "through the lane's unchanged keep rule; falls back to mlp when torch is not importable)")


def require_torch(who: str):
    """torch, or RuntimeError naming the caller (a torch fit reached without resolve_deep)."""
    torch = import_torch()
    if torch is None:
        raise RuntimeError("%s: %s (call resolve_deep before a torch fit)" % (who, FALLBACK_NOTICE))
    return torch


# ----------------------------------------------------------------------------- guards

def refuse_heldout_rows(frame, who: str, game_col: str = "sportvu_game", games=None) -> None:
    """ValueError when a held-out row would reach a torch fit: a true ``heldout`` column in ``frame``, or a game id
    (``frame[game_col]`` and/or ``games``) that analytics.eval.is_heldout_game flags."""
    from analytics.eval import is_heldout_game
    if frame is not None and hasattr(frame, "columns"):
        if "heldout" in frame.columns and bool(np.asarray(frame["heldout"].fillna(False), dtype=bool).any()):
            raise ValueError("%s: a held-out row reached the torch fit" % who)
        if game_col in frame.columns:
            bad = [g for g in frame[game_col].astype(str).unique() if is_heldout_game(g)]
            if bad:
                raise ValueError("%s: held-out game %s reached the torch fit" % (who, bad[0]))
    if games is not None:
        bad = [g for g in map(str, games) if is_heldout_game(g)]
        if bad:
            raise ValueError("%s: held-out game %s reached the torch fit" % (who, bad[0]))


# ----------------------------------------------------------------------------- training

@contextlib.contextmanager
def deterministic(torch):
    """torch.use_deterministic_algorithms(True) inside the block (so a multi-threaded fit reproduces too), the
    previous setting restored after."""
    prev, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=warn)


@contextlib.contextmanager
def torch_threads(torch, n: int = 1):
    """torch's intra-op pool pinned to ``n`` threads inside the block, restored after."""
    prev = torch.get_num_threads()
    torch.set_num_threads(max(int(n), 1))
    try:
        yield
    finally:
        torch.set_num_threads(prev)


def inner_split(n: int, groups=None, seed: int = 0, frac: float = VAL_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """(inner-train index, validation index) over the n rows passed in. By whole groups when there are at least
    MIN_GROUPS_FOR_GROUP_SPLIT of them (round(frac * groups), at least one), else a seeded row split; under 5 rows
    the validation set is empty and early stopping watches the training loss."""
    idx = np.arange(n)
    if n < 5:
        return idx, idx[:0]
    rng = np.random.default_rng(seed + 7919)
    if groups is not None:
        g = np.asarray(groups).astype(str)
        uniq = np.unique(g)
        if len(uniq) >= MIN_GROUPS_FOR_GROUP_SPLIT:
            k = max(1, int(round(frac * len(uniq))))
            val_groups = rng.choice(uniq, size=k, replace=False)
            val = np.isin(g, val_groups)
            return idx[~val], idx[val]
    perm = rng.permutation(n)
    k = max(1, int(round(frac * n)))
    return np.sort(perm[k:]), np.sort(perm[:k])


def _epochs(torch, net, Xt, Yt, seed: int, lr: float, weight_decay: float, batch_size: int, max_epochs: int,
            val=None, patience: int = 0) -> dict:
    """Minibatch Adam on the MSE for up to max_epochs epochs, minibatches shuffled by a generator seeded with
    ``seed``. With ``val`` (X, Y), early stopping: stop after ``patience`` epochs without a finite validation MSE
    lower than the best, and return the best epoch and its weights."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    n = len(Xt)
    bs = max(1, min(int(batch_size), n))
    best, best_epoch, bad, epochs, best_state = float("inf"), 0, 0, 0, None
    for epoch in range(int(max_epochs)):
        net.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(net(Xt[b]), Yt[b])
            loss.backward()
            opt.step()
        epochs = epoch + 1
        if val is None:
            continue
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(val[0]), val[1]))
        if np.isfinite(v) and v < best - 1e-7:
            best, best_epoch, bad = v, epochs, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    return {"epochs": epochs, "best_epoch": best_epoch, "best": best, "best_state": best_state}


def train_net(torch, net, X: np.ndarray, Y: np.ndarray, groups=None, seed: int = 0, threads: int = 1,
              max_epochs: int = 100, patience: int = 8, batch_size: int = 256, lr: float = 2e-3,
              weight_decay: float = 0.0, who: str = "deep_torch") -> dict:
    """Fit ``net`` (already built under torch.manual_seed) to Y on X by minibatch Adam on the mean squared error.

    Two passes, both on the rows passed in (the training rows of the current fold only):
      1. early stopping on inner_split: train on the inner-train rows, watch the validation MSE, pick best_epoch;
      2. refit: the net's initial weights restored and the shuffling generator reseeded, then best_epoch epochs on
         every row passed in, so the torch candidate learns from the same rows as the sklearn stand-in.
    Under 5 rows there is no validation split: early stopping watches the training loss and its best weights are kept.
    Runs on ``threads`` intra-op threads with torch.use_deterministic_algorithms(True), both restored after.
    ValueError when the validation loss is never finite (NaN or inf in the inputs or the targets).
    -> dict(epochs, best_epoch, best_val_loss, n_train, n_val, refit, n_refit)."""
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    tr, va = inner_split(len(X), groups, seed)
    init_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
    kw = dict(seed=seed, lr=lr, weight_decay=weight_decay, batch_size=batch_size)
    with torch_threads(torch, threads), deterministic(torch):
        Xa, Ya = torch.from_numpy(X), torch.from_numpy(Y)
        Xt, Yt = Xa[tr], Ya[tr]
        val = (Xa[va], Ya[va]) if len(va) else (Xt, Yt)
        r = _epochs(torch, net, Xt, Yt, max_epochs=max_epochs, val=val, patience=patience, **kw)
        if r["best_epoch"] == 0:
            raise ValueError("%s: the validation loss was never finite in %d epoch(s); check the inputs and targets for "
                             "NaN or inf" % (who, r["epochs"]))
        refit = len(va) > 0
        if refit:
            net.load_state_dict(init_state)
            _epochs(torch, net, Xa, Ya, max_epochs=r["best_epoch"], **kw)
        else:
            net.load_state_dict(r["best_state"])
        net.eval()
    return {"epochs": r["epochs"], "best_epoch": r["best_epoch"], "best_val_loss": r["best"], "n_train": int(len(tr)),
            "n_val": int(len(va)), "refit": refit, "n_refit": int(len(X)) if refit else 0}


def predict_net(torch, net, X: np.ndarray, threads: int = 1, batch_size: int = 65536) -> np.ndarray:
    """net(X) as float64 numpy, in eval mode without gradients, on ``threads`` threads."""
    X = np.asarray(X, dtype=np.float32)
    out = []
    with torch_threads(torch, threads), deterministic(torch), torch.no_grad():
        net.eval()
        for i in range(0, len(X), batch_size):
            out.append(net(torch.from_numpy(X[i:i + batch_size])).numpy().astype(float))
        if not out:
            out.append(net(torch.zeros((1,) + X.shape[1:])).numpy().astype(float)[:0])
    return np.concatenate(out, axis=0)


class Standardiser:
    """Column means and standard deviations of the training rows (a zero deviation scales by 1)."""

    def fit(self, A: np.ndarray) -> "Standardiser":
        A = np.asarray(A, dtype=float)
        self.mean_ = A.mean(axis=0)
        sd = A.std(axis=0)
        self.scale_ = np.where(sd > 1e-12, sd, 1.0)
        return self

    def transform(self, A: np.ndarray) -> np.ndarray:
        return (np.asarray(A, dtype=float) - self.mean_) / self.scale_

    def inverse(self, A: np.ndarray) -> np.ndarray:
        return np.asarray(A, dtype=float) * self.scale_ + self.mean_
