"""tools/research_ghosting.py: the ghosting research experiment ghosting-E1 (PLAN.md D111, D113).

    python3 -m tools.research_ghosting e1-pass [--artifacts DIR] [--max-minutes 25]
    python3 -m tools.research_ghosting e1-fold --fold K [--artifacts DIR]
    python3 -m tools.research_ghosting e1-dev  [--artifacts DIR]

Pre-registration: docs/research/ghosting.md, "ghosting-E1", with its amendment (D113), which fixes every definition
used here before any number existed. The question: does training the window network on its own 5-step rollout lower
the 1.0 s ghost error against the frozen one-step stand-in?

e1-pass   One pass over the development games' frames, one game at a time, resumable: the frozen check_frames,
          check_possessions, build_panel_game, feature_frame and fit_rows (unchanged) give each game's 1 Hz fit rows;
          for every fit row the pass also keeps what a 5-step rollout needs (the context of the next 5 panel rows,
          the defender's actual positions there, the two positions before the origin, and how many steps stay inside
          the sequence). Development games only; held-out games are never read.
e1-fold   One out-of-fold fold, checkpointed: the frozen stand-in refit on the training folds; the one-step and the
          rollout torch networks trained on the same rows (MPS, capped); every model rolled forward with the frozen
          roll_forward from every 5 Hz origin of the fold's games, rebuilt from frames one game at a time; the
          stand-in's ghosts compared with the stored ghost_frames (the gate). Writes the fold's summaries.
e1-dev    After all 5 folds: the gate over every game, the paired bootstraps, the sign tests and the decision.
          Writes reports/research/ghosting-E1-dev.json.

e1-final  Set A once through tools.research_testset.score_once (D113, only if development adopts and only on the
          lead's go): the stand-in and the rollout network refit on every development fit row first. Nothing here writes
under DATA_ROOT or reports/ except the research results; the lane's command line is never run.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

from export.heavy import heavy_job, peak_rss_gb
from tools.frozen_common import REPO_ROOT, _plain, _sha256, artifacts_dir, assert_safe_output, training_games_only
from tools.research_testset import dev_games, refuse_test

WHO = "tools.research_ghosting"
EXPERIMENT = "ghosting-E1"
RESULTS = REPO_ROOT / "reports" / "research"
LANE_REPORT = REPO_ROOT / "reports" / "ghosting.md"
SEASON = "2015-16"
SEED, BOOT, FOLDS = 0, 2000, 5
LR, BATCH, EPOCHS = 1e-3, 1024, 10
HORIZON_BY_EPOCH = (1, 1, 2, 2, 3, 3, 4, 4, 5, 5)
CHUNK_GAMES = 40
GHOST_TOL = 1e-3                 # ft, the D49 ghosting cap
DIFF_1S, GUARD_02S, SIGN_P = -0.10, 0.02, 0.05
WORK = "ghosting_e1"
CANDS = ("persistence", "constant_velocity", "mlp", "onestep", "rollout")
STEPS = 5


def work_dir(artifacts=None) -> Path:
    return assert_safe_output(artifacts_dir(artifacts) / "research" / WORK)


def _frozen():
    from analytics.ghosting import features as F
    from analytics.ghosting import train as T
    from analytics.ghosting import validate as V
    if (T.HIDDEN, T.MAX_ITER, F.N_LAGS, F.FIT_EVERY, F.DT) != ((64, 64), 200, 5, 5, 0.2):
        raise RuntimeError(f"{WHO}: the frozen ghosting settings changed")
    return F, T, V


def layout():
    """Column layout of a cached row: the frozen window (155) and targets (2), then the rollout block: the context
    of panel rows p..p+4 (5 x 24), the defender's actual positions there (5 x 2), the positions at p-2 and p-1 (4),
    and the number of valid steps (1)."""
    F, _, _ = _frozen()
    nw, nc = len(F.window_columns()), len(F.CTX_COLUMNS)
    o = {"win": (0, nw), "tgt": (nw, nw + 2)}
    s = nw + 2
    o["ctx"] = (s, s + STEPS * nc); s += STEPS * nc
    o["act"] = (s, s + STEPS * 2); s += STEPS * 2
    o["hist"] = (s, s + 4); s += 4
    o["nvalid"] = (s, s + 1); s += 1
    o["width"] = s
    return o


# ----------------------------------------------------------------------------- development inputs

def dev_game_list(root: Path) -> list:
    from analytics.eval import is_heldout_game
    frames = {p.stem for p in (root / "sportvu_contract").glob("*.parquet") if not p.name.endswith("_pbp.parquet")}
    poss = {p.stem for p in (root / "analytics" / "possessions" / SEASON).glob("*.parquet")}
    games = dev_games(sorted(g for g in frames & poss if not is_heldout_game(g)), WHO)
    training_games_only(games, WHO)
    return games


def game_tables(root: Path, g: str):
    """One development game's cleaned frames and possessions, panel and feature frame, by the frozen functions."""
    import numpy as np
    import pandas as pd
    F, _, V = _frozen()
    frames, _ = V.check_frames(pd.read_parquet(root / "sportvu_contract" / f"{g}.parquet"))
    poss, _ = V.check_possessions(pd.read_parquet(root / "analytics" / "possessions" / SEASON / f"{g}.parquet"), frames)
    panel, _ = F.build_panel_game(frames, poss, g)
    del frames
    panel["heldout"] = False
    panel["panel_row"] = np.arange(len(panel))
    return poss, panel, F.feature_frame(panel)


def cache_rows(panel, frame):
    """The cached rows of one game: the frozen fit rows plus each row's rollout block (layout())."""
    import numpy as np
    F, T, _ = _frozen()
    keep = (frame["step"].to_numpy() % F.FIT_EVERY) == 0
    fit = F.fit_rows(frame)[T.fit_columns()].to_numpy(dtype=np.float32)
    p = frame.loc[keep, "panel_row"].to_numpy().astype(int)
    ctx = panel[F.CTX_COLUMNS].to_numpy(dtype=float)
    dxy = panel[["def_x", "def_y"]].to_numpy(dtype=float)
    end = T._sequence_end(panel)
    n_panel = len(panel)
    q = np.minimum(p[:, None] + np.arange(STEPS)[None, :], n_panel - 1)
    nvalid = np.minimum(end[p] - p + 1, STEPS)
    block = np.concatenate([ctx[q].reshape(len(p), -1), dxy[q].reshape(len(p), -1), dxy[p - 2], dxy[p - 1],
                            nvalid[:, None]], axis=1).astype(np.float32)
    out = np.concatenate([fit, block], axis=1)
    assert out.shape[1] == layout()["width"]
    return out


def frame_pass(artifacts=None, max_minutes: float = 25.0, log=print) -> dict:
    import os as _os
    import numpy as np
    root = Path(_os.environ["DATA_ROOT"])
    games = dev_game_list(root)
    out = work_dir(artifacts) / "rows"
    out.mkdir(parents=True, exist_ok=True)
    started, done = time.time(), 0
    for g in games:
        if (out / f"{g}.npy").exists():
            continue
        if time.time() - started > max_minutes * 60:
            break
        _, panel, frame = game_tables(root, g)
        rows = cache_rows(panel, frame) if len(frame) else np.zeros((0, layout()["width"]), dtype=np.float32)
        (out / f"{g}.json").write_text(json.dumps({"feature_rows": int(len(frame)), "fit_rows": int(len(rows))}) + "\n")
        np.save(out / f"{g}.part.npy", rows)
        (out / f"{g}.part.npy").rename(out / f"{g}.npy")
        done += 1
        if log and done % 100 == 0:
            log(f"[research] {done} games this run, {time.time() - started:.0f} s")
    have = sum((out / f"{g}.npy").exists() for g in games)
    return {"games": len(games), "written_this_run": done, "present": have, "complete": have == len(games),
            "seconds": round(time.time() - started, 1), "peak_gb": round(peak_rss_gb(), 2)}


def folds(games: list) -> list:
    """The lane's 5 grouped folds by game (seed 0) over the games with fit rows -> [(train ids, test ids)]."""
    from analytics.eval import grouped_folds
    return [(sorted(tr), sorted(te)) for tr, te in grouped_folds(sorted(games), n_folds=FOLDS, seed=SEED)]


# ----------------------------------------------------------------------------- the networks

def torch_device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_net(mean, scale):
    """The stand-in's shape in torch: standardised 155-column window, 64 and 64 ReLU units, a 2-d displacement."""
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
            self.body = nn.Sequential(nn.Linear(len(mean), 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

        def forward(self, W):
            return self.body((W - self.mean) / self.scale)

    return Net()


def base_features_torch(ctx, def_xy, def_back):
    """analytics.ghosting.features.base_features in torch (back_s fixed at 0.4 s, as the roll-forward uses it):
    differentiable in the defender's positions."""
    import torch
    F, _, _ = _frozen()
    C = {c: i for i, c in enumerate(F.CTX_COLUMNS)}
    pt = lambda name: ctx[:, [C[name + "_x"], C[name + "_y"]]]
    cols = [ctx[:, [C["asg_x"], C["asg_y"]]] - def_xy]
    cols += [pt(f"off{i}") - def_xy for i in range(1, 6)]
    ball = ctx[:, [C["ball_x"], C["ball_y"]]]
    cols.append(ball - def_xy)
    cols += [pt(f"odef{j}") - def_xy for j in range(1, 5)]
    right = (ball[:, 0] > F.MID_X).to(ctx.dtype)
    baskets = torch.tensor(F.BASKETS, dtype=ctx.dtype, device=ctx.device)
    basket = baskets[right.long()]
    cols.append(torch.linalg.norm(ball - basket, dim=1, keepdim=True))
    cols.append(torch.linalg.norm(def_xy - basket, dim=1, keepdim=True))
    cols.append((def_xy - def_back) / (F.VEL_STEPS * F.DT))
    cols.append(def_xy)
    cols.append(ctx[:, [C["shot_clock_s"], C["elapsed_s"]]])
    cols.append(right[:, None])
    return torch.cat(cols, dim=1)


def rollout_path(net, B, h: int) -> list:
    """The ghost positions after steps 1..h of the rollout from each row of the batch B (layout() columns): the
    network's own position fed back, its own features recomputed (base_features_torch), everyone else actual, as
    analytics.ghosting.train.roll_forward does at scoring."""
    import torch
    F, _, _ = _frozen()
    L = layout()
    nb, nc = len(F.BASE_FEATURES), len(F.CTX_COLUMNS)
    W0 = B[:, L["win"][0]:L["win"][1]]
    ctx = B[:, L["ctx"][0]:L["ctx"][1]].reshape(-1, STEPS, nc)
    hist = B[:, L["hist"][0]:L["hist"][1]]
    win = [W0[:, k * nb:(k + 1) * nb] for k in range(F.N_LAGS)]
    h0, h1 = hist[:, 0:2], hist[:, 2:4]              # positions at p-2 and p-1, as roll_forward's state["hist"]
    ghost, path = h1, []
    for j in range(h):
        ghost = ghost + net(torch.cat(win, dim=1))
        path.append(ghost)
        if j + 1 < h:
            Fnew = base_features_torch(ctx[:, j], ghost, h0)
            h0, h1 = h1, ghost
            win = [Fnew] + win[:-1]
    return path


def rollout_loss(net, B, h: int):
    """Mean squared position error over the valid steps 1..h of the rollout (a step past the sequence's end is left
    out)."""
    import torch
    L = layout()
    act = B[:, L["act"][0]:L["act"][1]].reshape(-1, STEPS, 2)
    nvalid = B[:, L["nvalid"][0]]
    total = torch.zeros((), device=B.device)
    count = torch.zeros((), device=B.device)
    for j, ghost in enumerate(rollout_path(net, B, h)):
        mask = (nvalid > j).to(B.dtype)
        total = total + (((ghost - act[:, j]) ** 2).sum(dim=1) * mask).sum()
        count = count + mask.sum()
    return total / count.clamp(min=1)


def train_net(rows_of, train_games, mean, scale, objective: str, folder: Path, log=print) -> dict:
    """Train one network (objective "onestep" or "rollout") with the fixed budget, checkpointed by epoch in folder."""
    import numpy as np
    import torch
    torch.manual_seed(SEED)
    dev = torch_device()
    net = make_net(mean, scale).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    L = layout()
    folder.mkdir(parents=True, exist_ok=True)
    start = 0
    for e in range(EPOCHS, 0, -1):
        ck = folder / f"{objective}_epoch{e}.pt"
        if ck.exists():
            state = torch.load(ck, map_location=dev)
            net.load_state_dict(state["net"]); opt.load_state_dict(state["opt"])
            start = e
            break
    rng = np.random.default_rng([SEED, 7])
    for _ in range(start):                        # the shuffles a resumed run has already used
        rng.permutation(len(train_games))
    timing = []
    for epoch in range(start, EPOCHS):
        t0 = time.time()
        h = 1 if objective == "onestep" else HORIZON_BY_EPOCH[epoch]
        order = rng.permutation(len(train_games))
        er = np.random.default_rng([SEED, 8, epoch])
        loss_sum, n_batches = 0.0, 0
        for c0 in range(0, len(order), CHUNK_GAMES):
            chunk = np.concatenate([rows_of(train_games[i]) for i in order[c0:c0 + CHUNK_GAMES]])
            chunk = chunk[er.permutation(len(chunk))]
            for b0 in range(0, len(chunk), BATCH):
                B = torch.from_numpy(chunk[b0:b0 + BATCH]).to(dev)
                if objective == "onestep":
                    W = B[:, L["win"][0]:L["win"][1]]
                    y = B[:, L["tgt"][0]:L["tgt"][1]]
                    loss = ((net(W) - y) ** 2).sum(dim=1).mean()
                else:
                    loss = rollout_loss(net, B, h)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_sum += float(loss.detach()); n_batches += 1
            del chunk
        torch.save({"net": net.state_dict(), "opt": opt.state_dict()}, folder / f"{objective}_epoch{epoch + 1}.pt")
        timing.append({"epoch": epoch + 1, "horizon": h, "mean_loss": loss_sum / max(n_batches, 1),
                       "seconds": round(time.time() - t0, 1)})
        if log:
            log(f"[research] {objective} epoch {epoch + 1}/{EPOCHS} h={h} loss {loss_sum / max(n_batches, 1):.4f} "
                f"{time.time() - t0:.0f} s")
    final = folder / f"{objective}_epoch{EPOCHS}.pt"
    net.eval().cpu()
    return {"net": net, "weights": final.name, "weights_sha256": _sha256(final), "epochs": timing}


# ----------------------------------------------------------------------------- one fold

def predictors(stand_in, onestep, rollout) -> dict:
    """{candidate: predict(window matrix) -> displacement} for the frozen roll_forward."""
    import numpy as np
    import torch
    F, _, _ = _frozen()
    ivx = F.BASE_FEATURES.index("def_vx")

    def torch_fn(net):
        def f(W):
            with torch.no_grad():
                return net(torch.from_numpy(np.asarray(W, dtype=np.float32))).numpy().astype(float)
        return f

    return {"persistence": lambda W: np.zeros((len(W), 2)),
            "constant_velocity": lambda W: np.asarray(W, dtype=float)[:, [ivx, ivx + 1]] * F.DT,
            "mlp": stand_in, "onestep": torch_fn(onestep), "rollout": torch_fn(rollout)}


def evaluate_game(root: Path, g: str, preds: dict, fold: int) -> dict:
    """Every candidate rolled forward by the frozen roll_forward from every 5 Hz origin of one game -> the game's
    error summaries, the ghost possessions of the stand-in and the rollout network, and the gate comparison."""
    import numpy as np
    import pandas as pd
    _, T, _ = _frozen()
    poss, panel, frame = game_tables(root, g)
    if not len(frame):
        return None
    errs = []
    for c, f in preds.items():
        key = "persistence" if c == "persistence" else "mlp"
        e = T.roll_forward({key: f}, frame, panel, fold=fold)
        e["candidate"] = c
        errs.append(e)
    err = pd.concat(errs, ignore_index=True)
    keys = ["sportvu_game", "possession", "frame", "def_player_id"]
    summ = []
    for h in (0.2, 1.0):
        w = err[err["horizon_s"] == h].pivot_table(index=keys, columns="candidate", values="error_ft",
                                                   aggfunc="first")
        row = {"sportvu_game": g, "horizon_s": h, "rows": int(len(w))}
        for c in CANDS:
            row["sum_" + c], row["n_" + c] = float(w[c].sum()), int(w[c].count())
        summ.append(row)
    gp = {}
    for c in ("mlp", "rollout"):
        e = err[err["candidate"] == c].copy()
        e["candidate"] = "mlp"
        gp[c] = T.ghost_tables(e, "mlp", poss)[1]
    mine = err[err["candidate"] == "mlp"][keys + ["horizon_s", "ghost_x", "ghost_y"]]
    stored = pd.read_parquet(Path(os.environ["DATA_ROOT"]) / "analytics" / "ghosting" / SEASON /
                             f"ghost_frames_{g}.parquet", columns=keys + ["horizon_s", "ghost_x", "ghost_y", "fold"])
    mine = mine.assign(horizon_s=mine["horizon_s"].astype(float).round(1))
    stored = stored.assign(horizon_s=stored["horizon_s"].astype(float).round(1))   # stored as float32: 0.2 != 0.2f
    m = mine.merge(stored, on=keys + ["horizon_s"], how="outer", suffixes=("", "_stored"), indicator=True)
    both = m["_merge"] == "both"
    diff = np.hypot(m.loc[both, "ghost_x"] - m.loc[both, "ghost_x_stored"], m.loc[both, "ghost_y"] - m.loc[both, "ghost_y_stored"])
    gate = {"sportvu_game": g, "rows": int(len(mine)), "rows_stored": int(len(stored)), "unmatched": int((~both).sum()),
            "max_abs_diff_ft": float(diff.max()) if both.any() else float("nan"),
            "stored_fold_equal": bool((m.loc[both, "fold"] == fold).all())}
    return {"summ": summ, "gp": gp, "gate": gate}


def run_fold(k: int, artifacts=None, log=print) -> dict:
    import os as _os
    import gc
    import numpy as np
    import pandas as pd
    import torch
    from analytics.eval import fit_threads
    _, T, _ = _frozen()
    root = Path(_os.environ["DATA_ROOT"])
    wd = work_dir(artifacts)
    games = dev_game_list(root)
    have = [g for g in games if (wd / "rows" / f"{g}.npy").exists()]
    if len(have) != len(games):
        raise SystemExit(f"{WHO}: run e1-pass first ({len(games) - len(have)} games missing)")
    rows_of = lambda g: np.load(wd / "rows" / f"{g}.npy", mmap_mode="r")
    with_rows = [g for g in games                    # the frozen run kept the games with feature rows
                 if json.loads((wd / "rows" / f"{g}.json").read_text())["feature_rows"] > 0]
    tr, te = folds(with_rows)[k]
    refuse_test(tr + te, WHO)
    training_games_only(tr + te, WHO)
    folder = wd / f"fold{k}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "summaries.parquet").exists():
        return {"fold": k, "status": "already done"}
    started = time.time()
    L = layout()
    # the frozen stand-in, on the training folds' fit rows in the frozen order (sorted game ids)
    n = sum(rows_of(g).shape[0] for g in tr)
    mm_path = folder / "fit_rows.tmp.npy"
    X = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.float32, shape=(n, L["tgt"][1]))
    i = 0
    for g in tr:
        r = rows_of(g)
        X[i:i + len(r)] = r[:, :L["tgt"][1]]
        i += len(r)
    frame = pd.DataFrame(X, columns=T.fit_columns(), copy=False)
    frame.insert(0, "sportvu_game", np.repeat(tr, [rows_of(g).shape[0] for g in tr]))
    frame["heldout"] = False
    t0 = time.time()
    with fit_threads(1):
        stand_in = T.fit_mlp(frame, seed=SEED)
    t_mlp = time.time() - t0
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    for j in range(0, n, T.SCALER_CHUNK):
        sc.partial_fit(X[j:j + T.SCALER_CHUNK, :L["win"][1]])
    mean, scale = sc.mean_.astype(np.float32), sc.scale_.astype(np.float32)
    del frame, X
    gc.collect()
    mm_path.unlink()
    peak_after_mlp = peak_rss_gb()
    if log:
        log(f"[research] fold {k}: stand-in fitted on {n} rows in {t_mlp:.0f} s, peak {peak_after_mlp:.1f} GB")
    nets = {o: train_net(rows_of, tr, mean, scale, o, folder, log) for o in ("onestep", "rollout")}
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    preds = predictors(stand_in, nets["onestep"]["net"], nets["rollout"]["net"])
    t0 = time.time()
    summ, gp, gates = [], {"mlp": [], "rollout": []}, []
    for c, g in enumerate(te):
        r = evaluate_game(root, g, preds, fold=k)
        if r is None:
            continue
        summ += r["summ"]
        gates.append(r["gate"])
        for c2 in gp:
            gp[c2].append(r["gp"][c2])
        if log and (c + 1) % 25 == 0:
            log(f"[research] fold {k}: evaluated {c + 1} of {len(te)} games, {time.time() - t0:.0f} s")
    for c2 in gp:
        pd.concat(gp[c2], ignore_index=True).to_parquet(folder / f"gp_{c2}.parquet", index=False)
    pd.DataFrame(gates).to_parquet(folder / "gate.parquet", index=False)
    info = {"fold": k, "train_games": len(tr), "test_games": len(te), "fit_rows": int(n),
            "seconds": {"stand_in_fit": round(t_mlp, 1), "evaluation": round(time.time() - t0, 1),
                        "total": round(time.time() - started, 1)},
            "peak_rss_gb": round(peak_rss_gb(), 2), "peak_rss_after_stand_in_gb": round(peak_after_mlp, 2),
            "nets": {o: {k2: v for k2, v in nets[o].items() if k2 != "net"} for o in nets},
            "mps_watermarks": {"high": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
                               "low": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO")},
            "device": str(torch_device())}
    (folder / "fold.json").write_text(json.dumps(_plain(info), indent=2, allow_nan=False) + "\n")
    pd.DataFrame(summ).to_parquet(folder / "summaries.parquet", index=False)      # written last: the fold is done
    return info


# ----------------------------------------------------------------------------- development statistics

def _mean_of(c: str):
    return lambda d: d["sum_" + c].sum() / d["n_" + c].sum() if d["n_" + c].sum() else float("nan")


def printed_numbers(text: str) -> dict:
    import re
    m02 = re.search(r"\| 0\.2 s error, MLP stand-in \| (\d+\.\d+) ", text)
    m10 = re.search(r"\| 1\.0 s error, MLP stand-in \| (\d+\.\d+) ", text)
    ms = re.search(r"\| graded-deviation sign test \(positive / negative games\) \| (\d+) / (\d+), p = (\d+\.\d+)", text)
    if not (m02 and m10 and ms):
        raise SystemExit(f"{WHO}: the frozen report does not print the stand-in's numbers")
    return {"0.2": m02.group(1), "1.0": m10.group(1), "n_pos": int(ms.group(1)), "n_neg": int(ms.group(2)),
            "p": ms.group(3)}


def dev(artifacts=None) -> dict:
    import numpy as np
    import pandas as pd
    from analytics.eval import bootstrap_by_game, paired_bootstrap_by_game
    _, T, _ = _frozen()
    wd = work_dir(artifacts)
    missing = [k for k in range(FOLDS) if not (wd / f"fold{k}" / "summaries.parquet").exists()]
    if missing:
        raise SystemExit(f"{WHO}: folds {missing} are not done")
    S = pd.concat([pd.read_parquet(wd / f"fold{k}" / "summaries.parquet") for k in range(FOLDS)], ignore_index=True)
    gate = pd.concat([pd.read_parquet(wd / f"fold{k}" / "gate.parquet") for k in range(FOLDS)], ignore_index=True)
    gp = {c: pd.concat([pd.read_parquet(wd / f"fold{k}" / f"gp_{c}.parquet") for k in range(FOLDS)], ignore_index=True)
          for c in ("mlp", "rollout")}
    folds_info = [json.loads((wd / f"fold{k}" / "fold.json").read_text()) for k in range(FOLDS)]
    games = sorted(S["sportvu_game"].unique())
    training_games_only(games, WHO)
    refuse_test(games, WHO)
    out = {"experiment": EXPERIMENT, "stage": "development games, the lane's 5 grouped folds by game, seed 0",
           "games": len(games), "folds": folds_info}
    means, pairs = {}, {}
    for h in (0.2, 1.0):
        d = S[S["horizon_s"] == h].reset_index(drop=True)
        means[str(h)] = {c: bootstrap_by_game(d, "sportvu_game", _mean_of(c), n=BOOT, seed=SEED) for c in CANDS}
        means[str(h)]["rows"] = int(d["rows"].sum())
        pairs[str(h)] = {f"{a}_minus_{b}": paired_bootstrap_by_game(d, "sportvu_game", _mean_of(a), _mean_of(b),
                                                                     n=BOOT, seed=SEED)
                         for a, b in (("rollout", "mlp"), ("onestep", "mlp"), ("rollout", "onestep"),
                                      ("mlp", "persistence"), ("mlp", "constant_velocity"))}
    signs = {c: {k: v for k, v in T.sign_test(gp[c]).items() if k != "per_game"} for c in gp}
    printed = printed_numbers(LANE_REPORT.read_text())
    half = lambda s: 0.5 * 10 ** -len(s.split(".")[1])
    gate_rec = {"games_compared": int(len(gate)), "rows": int(gate["rows"].sum()), "rows_stored": int(gate["rows_stored"].sum()),
                "unmatched": int(gate["unmatched"].sum()), "max_abs_diff_ft": float(gate["max_abs_diff_ft"].max()),
                "stored_fold_equal": bool(gate["stored_fold_equal"].all()), "tolerance_ft": GHOST_TOL,
                "printed": printed,
                "recomputed": {"0.2": means["0.2"]["mlp"]["point"], "1.0": means["1.0"]["mlp"]["point"],
                               "n_pos": signs["mlp"]["n_pos"], "n_neg": signs["mlp"]["n_neg"],
                               "p": signs["mlp"]["p_value"]}}
    gate_rec["passed"] = bool(gate_rec["unmatched"] == 0 and gate_rec["rows"] == gate_rec["rows_stored"]
                              and gate_rec["max_abs_diff_ft"] <= GHOST_TOL and gate_rec["stored_fold_equal"]
                              and abs(means["0.2"]["mlp"]["point"] - float(printed["0.2"])) <= half(printed["0.2"])
                              and abs(means["1.0"]["mlp"]["point"] - float(printed["1.0"])) <= half(printed["1.0"])
                              and signs["mlp"]["n_pos"] == printed["n_pos"] and signs["mlp"]["n_neg"] == printed["n_neg"])
    out["gate"] = gate_rec
    out["mean_error_ft"], out["paired_ft"], out["sign_test"] = means, pairs, signs
    p1, p02, s = pairs["1.0"]["rollout_minus_mlp"], pairs["0.2"]["rollout_minus_mlp"], signs["rollout"]
    decision = {"one_s": {"diff": p1["diff"], "hi": p1["hi"], "rule": f"diff <= {DIFF_1S} and hi < 0",
                          "ok": bool(p1["diff"] <= DIFF_1S and p1["hi"] < 0)},
                "guard_02s": {"hi": p02["hi"], "rule": f"hi < +{GUARD_02S}", "ok": bool(p02["hi"] < GUARD_02S)},
                "sign_test": {"n_pos": s["n_pos"], "n_neg": s["n_neg"], "p": s["p_value"],
                              "rule": f"n_pos > n_neg and p < {SIGN_P}",
                              "ok": bool(s["n_pos"] > s["n_neg"] and s["p_value"] < SIGN_P)}}
    decision["adopted"] = bool(gate_rec["passed"] and all(v["ok"] for v in decision.values() if isinstance(v, dict)))
    out["decision"] = decision
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{EXPERIMENT}-dev.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    (RESULTS / f"{EXPERIMENT}-gate.json").write_text(json.dumps(_plain(gate_rec), indent=2, allow_nan=False) + "\n")
    return {"gate_passed": gate_rec["passed"], "decision": decision}


def fold_record(wd: Path, k: int) -> dict:
    """A fold's record. Fold 0 was rescored with its saved weights (the gate fix), so its training epochs live in the
    first run's record, moved aside to fold0/superseded_gate_bug/."""
    rec = json.loads((wd / f"fold{k}" / "fold.json").read_text())
    if not rec["nets"]["rollout"]["epochs"]:
        rec = {**rec, "epochs_from": f"fold{k}/superseded_gate_bug/fold.json",
               "nets": json.loads((wd / f"fold{k}" / "superseded_gate_bug" / "fold.json").read_text())["nets"]}
    return rec


def fold_description(artifacts=None) -> dict:
    """Pre-score description (PLAN.md D115), read by no rule: per fold, the 1.0 s pooled mean error of each model,
    the rollout-minus-stand-in difference with the SD of its per-game values, the differences against constant
    velocity and the one-step net, and the final-epoch training losses; the expected Set A half-width from those
    SDs (1.96 x SD / sqrt(10))."""
    import numpy as np
    import pandas as pd
    wd = work_dir(artifacts)
    folds_out = []
    for k in range(FOLDS):
        S = pd.read_parquet(wd / f"fold{k}" / "summaries.parquet")
        d = S[S["horizon_s"] == 1.0]
        mean = {c: float(d["sum_" + c].sum() / d["n_" + c].sum()) for c in CANDS}
        per_game = (d["sum_rollout"] / d["n_rollout"] - d["sum_mlp"] / d["n_mlp"]).to_numpy()
        rec = fold_record(wd, k)
        folds_out.append({"fold": k, "games": int(len(d)), "mean_error_1s": mean,
                          "rollout_minus_mlp": mean["rollout"] - mean["mlp"],
                          "rollout_minus_constant_velocity": mean["rollout"] - mean["constant_velocity"],
                          "rollout_minus_onestep": mean["rollout"] - mean["onestep"],
                          "per_game_sd_rollout_minus_mlp": float(np.std(per_game, ddof=1)),
                          "final_epoch_loss": {o: rec["nets"][o]["epochs"][-1]["mean_loss"] for o in ("onestep", "rollout")},
                          "epochs_from": rec.get("epochs_from", f"fold{k}/fold.json")})
    sds = [f["per_game_sd_rollout_minus_mlp"] for f in folds_out]
    losses = [f["final_epoch_loss"]["rollout"] for f in folds_out]
    out = {"experiment": EXPERIMENT, "stage": "pre-score description (PLAN.md D115), development folds only; read by no rule",
           "folds": folds_out,
           "expected_set_a_half_width_1s": {"from_sd_min": 1.96 * min(sds) / math.sqrt(10),
                                            "from_sd_max": 1.96 * max(sds) / math.sqrt(10),
                                            "formula": "1.96 x per-game SD / sqrt(10), normal approximation"},
           "rollout_final_epoch_loss_range": [min(losses), max(losses)]}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{EXPERIMENT}-folds.json").write_text(json.dumps(_plain(out), indent=2, allow_nan=False) + "\n")
    return out


# ----------------------------------------------------------------------------- Set A (D113), on the lead's go

def non_finite(value, path: str = "") -> list:
    if isinstance(value, dict):
        return [q for k, v in value.items() for q in non_finite(v, f"{path}.{k}" if path else str(k))]
    if isinstance(value, (list, tuple)):
        return [q for i, v in enumerate(value) for q in non_finite(v, f"{path}[{i}]")]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    return [path] if isinstance(value, float) and not math.isfinite(value) else []


def refused(reading_text: str, **known) -> dict:
    keep = {k: v for k, v in known.items() if not non_finite(v)}
    return {"verdict": {"refused": True, "reading": reading_text}, **keep}


def reading(dev_diff: float, lo: float, hi: float) -> str:
    """D113's reading of the 1.0 s bootstrap interval, in the fixed order."""
    if not (lo <= dev_diff <= hi):
        return "inconsistent"
    if (dev_diff < 0 and hi < 0) or (dev_diff > 0 and lo > 0):
        return "confirms"
    return "consistent"


def per_game_t(values) -> dict:
    """Report-only (D113, as D80): a 95 percent t interval on the per-game values."""
    import numpy as np
    from scipy import stats
    v = np.asarray(values, float)
    if len(v) < 2:
        return {"mean_of_game_values": float(v.mean()) if len(v) else float("nan"), "lo": float("nan"),
                "hi": float("nan"), "games": int(len(v))}
    half = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return {"mean_of_game_values": float(v.mean()), "lo": float(v.mean() - half), "hi": float(v.mean() + half),
            "games": int(len(v))}


def set_a_tables(root: Path, g: str):
    """A test game's cleaned frames, possessions, panel and feature frame by the frozen functions (research root)."""
    import numpy as np
    import pandas as pd
    F, _, V = _frozen()
    frames, _ = V.check_frames(pd.read_parquet(root / "sportvu_contract" / f"{g}.parquet"))
    poss, _ = V.check_possessions(pd.read_parquet(root / "analytics" / "possessions" / SEASON / f"{g}.parquet"), frames)
    panel, _ = F.build_panel_game(frames, poss, g)
    del frames
    panel["heldout"] = False
    panel["panel_row"] = np.arange(len(panel))
    return panel, F.feature_frame(panel)


def make_scorer(preds: dict, dev_diff: float, artifacts=None, provenance: dict = None, n: int = BOOT):
    """The scorer score_once runs (D113): the stand-in ("mlp") and the rollout network ("rollout"), both fitted on
    every development fit row before the score, rolled forward by the frozen roll_forward on every 5 Hz origin of
    the test games. The test data is reached only here, while score_once runs a tracking scorer. It refuses, with its
    reason and no number, unless every test game has frames, possessions and at least one 1.0 s origin; a summary
    with any non-finite number is a refused verdict, never a NaN."""
    import numpy as np
    import pandas as pd
    from analytics.eval import paired_bootstrap_by_game
    from tools import research_testset as RT
    from tools.research_xfg import research_data_root
    _, T, _ = _frozen()

    def scorer(test_games):
        if RT._SCORING is None or RT._SCORING[1] != RT.TRACKING:
            raise SystemExit(f"{WHO}: the test data is read only inside the scorer that score_once runs")
        head = {"experiment": EXPERIMENT, "stage": "D113 consistency check, scored once on the D71 test set",
                **(provenance or {})}
        root = research_data_root(artifacts)
        games = sorted(str(g) for g in test_games)
        missing = {g: [k for k, q in (("frames", root / "sportvu_contract" / f"{g}.parquet"),
                                      ("possessions", root / "analytics" / "possessions" / SEASON / f"{g}.parquet"))
                       if not q.exists()] for g in games}
        missing = {g: m for g, m in missing.items() if m}
        if missing:
            return {**head, **refused(f"{len(missing)} test game(s) lack a needed table", missing=missing)}
        rows, empty = [], []
        keys = ["sportvu_game", "possession", "frame", "def_player_id"]
        for g in games:
            panel, frame = set_a_tables(root, g)
            if not len(frame):
                empty.append(g)
                continue
            errs = []
            for c in ("mlp", "rollout"):
                e = T.roll_forward({"mlp": preds[c]}, frame, panel, fold=-3)
                e["candidate"] = c
                errs.append(e)
            err = pd.concat(errs, ignore_index=True)
            for h in (0.2, 1.0):
                w = err[err["horizon_s"] == h].pivot_table(index=keys, columns="candidate", values="error_ft",
                                                           aggfunc="first")
                if h == 1.0 and not len(w):
                    empty.append(g)
                rows.append({"sportvu_game": g, "horizon_s": h, "rows": int(len(w)),
                             **{f"sum_{c}": float(w[c].sum()) if c in w else 0.0 for c in ("mlp", "rollout")},
                             **{f"n_{c}": int(w[c].count()) if c in w else 0 for c in ("mlp", "rollout")}})
        if empty:
            return {**head, **refused(f"{len(set(empty))} test game(s) have no 1.0 s origin",
                                      games_without_origins=sorted(set(empty)))}
        S = pd.DataFrame(rows)
        uneven = S[(S["n_mlp"] != S["rows"]) | (S["n_rollout"] != S["rows"])]
        if len(uneven):
            return {**head, **refused(f"{len(uneven)} game-horizon(s) where the two ghosts do not cover the same "
                                      "origins", uneven=uneven[["sportvu_game", "horizon_s"]].astype(str).values.tolist())}
        out = {"games": len(games)}
        for h in (0.2, 1.0):
            d = S[S["horizon_s"] == h].reset_index(drop=True)
            pair = paired_bootstrap_by_game(d, "sportvu_game", _mean_of("rollout"), _mean_of("mlp"), n=n, seed=SEED)
            per_game = (d["sum_rollout"] / d["n_rollout"] - d["sum_mlp"] / d["n_mlp"]).to_numpy()
            out[str(h)] = {"rows": int(d["rows"].sum()), "mlp": float(d["sum_mlp"].sum() / d["n_mlp"].sum()),
                           "rollout": float(d["sum_rollout"].sum() / d["n_rollout"].sum()),
                           "rollout_minus_mlp": pair,
                           "report_only_t": per_game_t(per_game)}
        p1 = out["1.0"]["rollout_minus_mlp"]
        out["development_estimate_1s"] = dev_diff
        out["reading"] = reading(dev_diff, p1["lo"], p1["hi"])
        out["report_only_size_reading"] = {"rule": f"the 1.0 s interval's upper end below {DIFF_1S} ft (D115)",
                                           "upper_end": p1["hi"], "below": bool(p1["hi"] < DIFF_1S)}
        bad = non_finite(out)
        if bad:
            return {**head, **refused(f"{len(bad)} non-finite value(s) in the summary, for example {bad[0]}",
                                      games=len(games))}
        return {**head, **out}

    return scorer


def refit_problems(stand_in, rollout: dict, folds_desc: dict) -> list:
    """Why the all-development refit may not be scored (PLAN.md D115): a non-finite weight in the stand-in or the
    rollout network, or a final-epoch rollout loss outside the folds' range. Refuse only; never retrain to fit."""
    import numpy as np
    problems = []
    models = [c.cell_contents for c in (stand_in.__closure__ or ()) if hasattr(c.cell_contents, "coefs_")]
    if len(models) != 1:
        problems.append("the stand-in's fitted network was not found")
    elif not all(np.isfinite(w).all() for w in models[0].coefs_ + models[0].intercepts_):
        problems.append("a stand-in weight is not finite")
    if not all(np.isfinite(p.detach().cpu().numpy()).all() for p in rollout["net"].parameters()):
        problems.append("a rollout weight is not finite")
    epochs = rollout.get("epochs") or []
    hi = folds_desc["rollout_final_epoch_loss_range"][1]
    if not epochs:
        problems.append("the rollout's final-epoch loss is not recorded")
    elif not epochs[-1]["mean_loss"] <= hi:              # one-sided since D119: a diverged loss, not a lower one
        problems.append(f"the rollout's final-epoch loss {epochs[-1]['mean_loss']:.4f} is above the folds' "
                        f"maximum {hi:.4f}")
    return problems


def record_epochs_from_log(log: Path, folder: Path) -> Path:
    """PLAN.md D119: the saved all-development refit's epoch losses, transcribed from the log of the run that trained it
    (the refused e1-final run; train_net returns them only to the run that trains). Written once beside the weights,
    as rollout_epochs.json, with the log's hash."""
    import re
    rows = [{"epoch": int(m.group(1)), "horizon": int(m.group(2)), "mean_loss": float(m.group(3)),
             "seconds": float(m.group(4))}
            for m in re.finditer(r"\[research\] rollout epoch (\d+)/\d+ h=(\d+) loss ([0-9.]+) ([0-9.]+) s",
                                 log.read_text())]
    if [r["epoch"] for r in rows] != list(range(1, EPOCHS + 1)):
        raise SystemExit(f"{WHO}: the log does not hold exactly one record of each of the {EPOCHS} epochs")
    out = folder / "rollout_epochs.json"
    out.write_text(json.dumps({"source_log": log.name, "source_log_sha256": _sha256(log), "epochs": rows}, indent=2)
                   + "\n")
    return out


def clean_head() -> str:
    import subprocess
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                           check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{WHO}: the working tree is not clean; commit first:\n{dirty}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def final(artifacts=None) -> dict:
    head = clean_head()
    import gc
    import numpy as np
    import pandas as pd
    from analytics.eval import fit_threads
    from sklearn.preprocessing import StandardScaler
    from tools.research_testset import TRACKING, score_once
    from tools.research_xfg import test_root_ready
    _, T, _ = _frozen()
    dev_path = RESULTS / f"{EXPERIMENT}-dev.json"
    gate_path = RESULTS / f"{EXPERIMENT}-gate.json"
    if not dev_path.exists() or not gate_path.exists() or not (RESULTS / f"{EXPERIMENT}-folds.json").exists():
        raise SystemExit(f"{WHO}: run e1-dev first; the test set is scored after the development result exists")
    dev_out = json.loads(dev_path.read_text())
    if not dev_out["gate"]["passed"]:
        raise SystemExit(f"{WHO}: the gate did not pass; nothing is scored")
    if not dev_out["decision"]["adopted"]:
        raise SystemExit(f"{WHO}: development did not adopt, so Set A is not scored (D113)")
    if not test_root_ready(artifacts):
        raise SystemExit(f"{WHO}: the research data root has no ingested games; run tools.research_ingest first")
    root = Path(os.environ["DATA_ROOT"])
    wd = work_dir(artifacts)
    games = [g for g in dev_game_list(root)
             if json.loads((wd / "rows" / f"{g}.json").read_text())["feature_rows"] > 0]
    refuse_test(games, WHO)
    training_games_only(games, WHO)
    rows_of = lambda g: np.load(wd / "rows" / f"{g}.npy", mmap_mode="r")
    folder = wd / "all_development"
    folder.mkdir(parents=True, exist_ok=True)
    L = layout()
    n = sum(rows_of(g).shape[0] for g in games)
    mm_path = folder / "fit_rows.tmp.npy"
    X = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.float32, shape=(n, L["tgt"][1]))
    i = 0
    for g in games:
        r = rows_of(g)
        X[i:i + len(r)] = r[:, :L["tgt"][1]]
        i += len(r)
    frame = pd.DataFrame(X, columns=T.fit_columns(), copy=False)
    frame.insert(0, "sportvu_game", np.repeat(games, [rows_of(g).shape[0] for g in games]))
    frame["heldout"] = False
    with fit_threads(1):
        stand_in = T.fit_mlp(frame, seed=SEED)
    sc = StandardScaler()
    for j in range(0, n, T.SCALER_CHUNK):
        sc.partial_fit(X[j:j + T.SCALER_CHUNK, :L["win"][1]])
    del frame, X
    gc.collect()
    mm_path.unlink()
    import pickle
    models = [c.cell_contents for c in (stand_in.__closure__ or ())]
    with open(folder / "stand_in.pkl", "wb") as fh:                 # the deterministic stand-in refit, kept this time
        pickle.dump(models, fh)
    if not (folder / f"rollout_epoch{EPOCHS}.pt").exists() or not (folder / "rollout_epochs.json").exists():
        raise SystemExit(f"{WHO}: the saved all-development rollout refit or its epoch record is missing; "
                         "D119 scores the saved refit and never retrains it")
    rollout = train_net(rows_of, games, sc.mean_.astype(np.float32), sc.scale_.astype(np.float32), "rollout", folder)
    rollout["epochs"] = json.loads((folder / "rollout_epochs.json").read_text())["epochs"]
    problems = refit_problems(stand_in, rollout, json.loads((RESULTS / f"{EXPERIMENT}-folds.json").read_text()))
    if problems:                                     # refuse only; never retrain to fit (D115)
        raise SystemExit(f"{WHO}: the refit is refused before any score: {problems}")
    preds = predictors(stand_in, rollout["net"], rollout["net"])
    dev_diff = float(dev_out["paired_ft"]["1.0"]["rollout_minus_mlp"]["diff"])
    provenance = {"code_commit": head, "gate_sha256": _sha256(gate_path), "dev_json_sha256": _sha256(dev_path),
                  "rollout_weights_sha256": rollout["weights_sha256"],
                  "stand_in_pickle_sha256": _sha256(folder / "stand_in.pkl"),
                  "rollout_epochs_sha256": _sha256(folder / "rollout_epochs.json"),
                  "rollout_final_epoch_loss": rollout["epochs"][-1]["mean_loss"], "fit_rows": int(n),
                  "refit_note": "PLAN.md D119: the saved all-development rollout refit, scored as it is"}
    scorer = make_scorer({"mlp": preds["mlp"], "rollout": preds["rollout"]}, dev_diff, artifacts, provenance)
    out = score_once(EXPERIMENT, TRACKING, scorer)
    rule = {"development_decides": dev_out["decision"],
            "set_a": "D113 consistency check: inconsistent if the 1.0 s bootstrap interval excludes the development "
                     "estimate; otherwise confirms if it excludes zero on that side; otherwise consistent; only "
                     "inconsistent stops an adoption", "source": "PLAN.md D113"}
    record = {"experiment": EXPERIMENT, "code_commit": head, "scored": out,
              "setup": {"gate": json.loads(gate_path.read_text()), "rule": rule}}
    (RESULTS / f"{EXPERIMENT}-final.json").write_text(json.dumps(_plain(record), indent=2, allow_nan=False) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=WHO)
    ap.add_argument("command", choices=("e1-pass", "e1-fold", "e1-dev", "e1-folds", "e1-record-refit", "e1-final"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--max-minutes", type=float, default=25.0)
    a = ap.parse_args(argv)
    with heavy_job(f"research ghosting {a.command}" + (f" fold {a.fold}" if a.fold is not None else "")):
        if a.command == "e1-pass":
            out = frame_pass(a.artifacts, a.max_minutes)
        elif a.command == "e1-fold":
            if a.fold is None or not 0 <= a.fold < FOLDS:
                raise SystemExit("give --fold 0..4")
            out = run_fold(a.fold, a.artifacts)
        elif a.command == "e1-dev":
            out = dev(a.artifacts)
        elif a.command == "e1-folds":
            out = fold_description(a.artifacts)
        elif a.command == "e1-record-refit":
            wd = work_dir(a.artifacts)
            out = {"written": record_epochs_from_log(wd.parent.parent / "research_ghosting_final.log",
                                                     wd / "all_development").name}
        else:
            out = final(a.artifacts)
            out = {"reading": out.get("reading"), "verdict": out.get("verdict")}
    print(json.dumps(_plain(out), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
