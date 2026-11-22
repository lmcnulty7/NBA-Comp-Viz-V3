"""tools/p8_e3.py: tracking-E1 (Track B's E3), team labels from tracklet embeddings, as a replay on stored r23 tracks
(docs/research/tracking.md section 6, PLAN.md D153).

    <frozen CV python> -m tools.p8_e3 replay TAG    crops from the stored r23 boxes, the three team arms, new identities
    <frozen CV python> -m tools.p8_e3 score TAG     the pinned scorer's team rows per window for r23 and both arms
    python3 -m tools.p8_e3 report                  per game, pooled, paired intervals, the adoption verdict

Run replay and score under the heavy-job lock, one game at a time. Outputs: builds/e3_clip and builds/e3_dinov2_small
(the r23 sidecars linked, only team_by_track in <window>_identity.json replaced) and e3/ in the P8 workspace.

Boxes, tracks and court are identical in every arm: the replay reads r23's own <window>_frames.json (fragment ids and
boxes per processed frame) and identity (idmap, the canonical tracks), and changes only the team feature. No labels:
the k-means is fitted per window with no supervision, as the frozen detect/teams.py does. Nothing reads a D85 test
game, a spent held-out game or any 2025-26 game: every tag passes refuse_cv_test and tools.p8_frozen.game_of (the
spent held-out guard) before a frame is read. The rule details are in tracking.md, "tracking-E1 under D153".
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from tools.p8_common import atomic_write, p8_root
from tools.research_testset import refuse_cv_test

WHO = "tools.p8_e3"
POOLED = ("gsw_bkn_2015", "gsw_cha_2016", "gsw_sac_2015")
APART = {"gsw_ind_2016": "D98 (weak Mac camera truth)", "hou_cle_2016": "new arena, not in the adoption rule"}
ARMS = {"clip": "openai/clip-vit-base-patch32", "dinov2_small": "facebook/dinov2-small"}
N_CROPS = 8                       # crops embedded per fragment (tracking.md compute line), evenly spaced
BOOT_N, BOOT_SEED = 2000, 0
GAIN_PTS, MAX_DROP_PTS = 5.0, 1.0


# ── pure parts ──────────────────────────────────────────────────────────────────────────────────────────────────────
def pick_even(items: list, k: int = N_CROPS) -> list:
    """k items evenly spaced over the list (all of them when it is shorter)."""
    if len(items) <= k:
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idx]


def l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def embed_teams(frag_embs: dict, idmap: dict, keys: list, n_abstain: int, min_crops: int, seed: int) -> dict:
    """frag_embs: fragment id -> (n, D) per-crop embeddings. k-means (k = 2, n_init 10, seeded) on the L2-normed mean of
    each fragment with at least min_crops crops (too few fragments: no fit, every key abstains, as detect/teams.py);
    each canonical key gets the mean over its member fragments' crops and the nearest centroid; then the keys abstain
    so that exactly max(n_abstain, keys without a feature) of them carry None: those without a feature first, then the
    largest near/far distance ratio (rank-based, label-free) -> {key: 0 | 1 | None}."""
    from sklearn.cluster import KMeans
    fit = [l2(e.mean(0)) for e in frag_embs.values() if len(e) >= min_crops]
    if len(fit) < max(2 + 2, 4):
        return {k: None for k in keys}
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(np.stack(fit))
    pooled = {}
    for fid, e in frag_embs.items():
        pooled.setdefault(idmap.get(fid, fid), []).append(e)
    lab, ratio = {}, {}
    for k in keys:
        es = pooled.get(k)
        if not es or sum(len(e) for e in es) < min_crops:
            continue
        f = l2(np.concatenate(es).mean(0))
        d = np.linalg.norm(km.cluster_centers_ - f, axis=1)
        lab[k] = int(d.argmin())
        ratio[k] = float(d.min() / max(d.max(), 1e-6))
    out = {k: lab.get(k) for k in keys}
    extra = max(0, n_abstain - sum(v is None for v in out.values()))
    for k in sorted(ratio, key=lambda k: (-ratio[k], str(k)))[:extra]:
        out[k] = None
    return out


def window_team(rows: list) -> dict:
    """rows: (canonical id, label, SportVU team) -> n, correct under the frozen majority mapping per label
    (sportvu.bench.aggregate) and under the best one-to-one A/B mapping (fixed mapping)."""
    n = len(rows)
    maj = 0
    for lab in {r[1] for r in rows}:
        sub = [r[2] for r in rows if r[1] == lab]
        maj += max(sub.count(v) for v in set(sub))
    teams = sorted({r[2] for r in rows})
    fixed = 0
    if rows:
        cands = []
        for t0 in teams + [None]:
            for t1 in teams + [None]:
                if t0 is not None and t0 == t1:
                    continue
                m = {0: t0, 1: t1}
                cands.append(sum(1 for r in rows if m.get(r[1]) == r[2]))
        fixed = max(cands)
    return {"n": n, "majority": maj, "fixed": fixed}


def pooled_acc(wins: list, arm: str, kind: str = "majority"):
    n = sum(w[arm]["n"] for w in wins)
    return sum(w[arm][kind] for w in wins) / n if n else None


def paired_boot(groups: dict, arm: str, base: str = "r23", n: int = BOOT_N, seed: int = BOOT_SEED):
    """Cluster bootstrap over the groups (games or windows) of the pooled accuracy difference arm minus base."""
    keys = [k for k, v in groups.items() if v]
    if len(keys) < 2:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = [w for i in rng.choice(len(keys), len(keys), replace=True) for w in groups[keys[i]]]
        a, b = pooled_acc(pick, arm), pooled_acc(pick, base)
        if a is not None and b is not None:
            vals.append(a - b)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def verdict(pooled: dict, per_game: dict, arm: str) -> dict:
    d = pooled[arm]["diff_pts"]
    gi, wi = pooled[arm]["game_interval"], pooled[arm]["window_interval"]
    drops = {g: v[arm]["diff_pts"] for g, v in per_game.items() if v[arm]["diff_pts"] < -MAX_DROP_PTS}
    fixed_ok = pooled[arm]["fixed"] >= pooled["r23"]["fixed"]
    checks = {"gain_at_least_5_pts": d >= GAIN_PTS, "game_interval_above_zero": bool(gi and gi[0] > 0),
              "window_interval_above_zero": bool(wi and wi[0] > 0), "no_game_down_over_1_pt": not drops,
              "fixed_mapping_not_lower": fixed_ok}
    return {"adopt": all(checks.values()), "checks": checks, "games_down": drops}


def matched_abstain(lab: dict, ratio: dict, keys: list, n_abstain: int) -> dict:
    """{key: label} for the keys with a feature (lab, with its near/far ratio) -> labels where exactly
    max(n_abstain, keys without a feature) keys carry None: those without a feature first, then the largest ratio."""
    out = {k: lab.get(k) for k in keys}
    extra = max(0, n_abstain - sum(v is None for v in out.values()))
    for k in sorted(ratio, key=lambda k: (-ratio[k], str(k)))[:extra]:
        out[k] = None
    return out


def order_trace(native: dict, stored: dict) -> dict:
    """Agreement of one window's replayed labels with the stored ones as given and with 0 and 1 swapped: the cluster
    order is arbitrary and no metric reads it, so max(as given, swapped) is the agreement up to order."""
    sw = {0: 1, 1: 0, None: None}
    keys = list(stored)
    same = sum(native.get(k) == stored[k] for k in keys)
    swapped = sum(sw.get(native.get(k)) == stored[k] for k in keys)
    return {"keys": len(keys), "as_given": same, "up_to_order": max(same, swapped), "swapped_window": swapped > same}


CONFOUND_NEAR_PTS, CONFOUND_FAR_PTS = -2.0, -5.0


def confound_reading(colour_minus_r23_pts: float, arm_minus_colour: dict) -> str:
    """D154's reading, fixed before the run: "embeddings worse" when colour on the replayed crops stays within 2.0
    points of r23's stored rows (pooled) and every arm-minus-colour-replay interval (game and window) lies wholly below
    zero; "comparison confounded" when colour on the replayed crops itself drops 5.0 points or more; else
    "inconclusive"."""
    if colour_minus_r23_pts <= CONFOUND_FAR_PTS:
        return "comparison confounded"
    below = all(iv and iv[1] < 0 for a in arm_minus_colour.values() for iv in (a["game_interval"], a["window_interval"]))
    if colour_minus_r23_pts >= CONFOUND_NEAR_PTS and below:
        return "embeddings worse"
    return "inconclusive"


# ── replay (frozen CV python, inside the lock) ──────────────────────────────────────────────────────────────────────
def _guard(tag: str):
    from tools import p8_frozen as F
    refuse_cv_test([tag], WHO)
    return F.game_of(tag)


class _T:
    def __init__(self, tid, bbox):
        self.track_id, self.bbox = tid, bbox


def collect_crops(P: dict, stem: str, side: dict):
    """The frozen TorsoCropCollector over r23's stored boxes and fragment ids, frame by frame from the window clip."""
    import cv2
    from detect.reid import TorsoCropCollector
    coll = TorsoCropCollector()
    # decoded from frame 0 with grab(), never a seek: a seek in these clips can land short of the end (frame 978 of a
    # 990-frame window was unreadable after a seek in the smoke run), while the build itself read them sequentially
    cap = cv2.VideoCapture(str(P["label_build"] / f"{stem}.mp4"))
    want = sorted(side["frames"], key=lambda r: int(r["frame"]))
    pos = 0
    for fr in want:
        f = int(fr["frame"])
        while pos < f:
            if not cap.grab():
                raise SystemExit(f"{WHO}: {stem} ends at frame {pos}, before stored frame {f}")
            pos += 1
        ok, img = cap.read()
        if not ok:
            raise SystemExit(f"{WHO}: {stem} frame {f} unreadable")
        pos += 1
        coll.add(img, [_T(b["tid"], b["bbox"]) for b in fr["boxes"]])
    cap.release()
    return coll


def replay_window(P: dict, w: dict, models: dict) -> dict:
    """One window: the frozen TorsoCropCollector on r23's stored boxes, the frozen colour classifier (a reproduction
    check against the stored labels) and the embedding arms."""
    import config
    from detect.teams import TeamClassifier
    stem = w["window"]
    r23 = P["builds"] / "r23"
    side = json.loads((r23 / f"{stem}_frames.json").read_text())
    ident = json.loads((r23 / f"{stem}_identity.json").read_text())
    idmap = {int(k): int(v) for k, v in ident.get("idmap", {}).items()}
    stored = {int(k): v for k, v in ident.get("team_by_track", {}).items()}
    keys = sorted(stored)
    coll = collect_crops(P, stem, side)
    clf = TeamClassifier()
    colour = {}
    if keys and clf.fit(coll.crops):
        canon = {}
        for tid, lst in coll.crops.items():
            canon.setdefault(idmap.get(tid, tid), []).extend(lst)
        colour = {cid: t for cid, (t, _) in clf.assign(canon).items()}
    same = sum(colour.get(k) == stored[k] for k in keys)
    n_abst = sum(v is None for v in stored.values())
    out = {"window": stem, "keys": len(keys), "stored_abstain": n_abst,
           "colour_replay_agrees": same, "fragments_with_crops": len(coll.crops), "arms": {}}
    for arm, bb in models.items():
        sel = {fid: pick_even(lst) for fid, lst in coll.crops.items() if lst}
        flat = [c for lst in sel.values() for c in lst]
        embs = bb(flat) if flat else np.zeros((0, 1), np.float32)
        frag, i = {}, 0
        for fid, lst in sel.items():
            frag[fid] = embs[i:i + len(lst)]
            i += len(lst)
        teams = embed_teams(frag, idmap, keys, n_abst, config.TEAM_MIN_CROPS, config.SEED)
        out["arms"][arm] = {"abstain": sum(v is None for v in teams.values()), "teams": {str(k): v for k, v in teams.items()}}
        d = P["builds"] / f"e3_{arm}"
        d.mkdir(parents=True, exist_ok=True)
        for suffix in ("_frames.json", "_trajectories.json"):
            link = d / f"{stem}{suffix}"
            if not link.exists() and (r23 / f"{stem}{suffix}").exists():
                os.symlink(r23 / f"{stem}{suffix}", link)
        atomic_write(d / f"{stem}_identity.json", json.dumps({**ident, "team_by_track": {str(k): v for k, v in teams.items()},
                                                               "e3": {"arm": arm, "model": ARMS[arm], "rule": "D153"}}))
    return out


def colour_window(P: dict, w: dict) -> dict:
    """D154: the frozen colour classifier on the replayed crops, as an arm: fitted per window on the fragments as
    detect/teams.py fits, each canonical track given its nearest centroid, then the same abstention matching as the
    embedding arms; its native labels (the frozen abstention rule) kept for the cluster-order trace."""
    import config
    from detect.teams import TeamClassifier
    stem = w["window"]
    r23 = P["builds"] / "r23"
    side = json.loads((r23 / f"{stem}_frames.json").read_text())
    ident = json.loads((r23 / f"{stem}_identity.json").read_text())
    idmap = {int(k): int(v) for k, v in ident.get("idmap", {}).items()}
    stored = {int(k): v for k, v in ident.get("team_by_track", {}).items()}
    keys = sorted(stored)
    n_abst = sum(v is None for v in stored.values())
    coll = collect_crops(P, stem, side)
    clf = TeamClassifier()
    native, lab, ratio = {}, {}, {}
    if keys and clf.fit(coll.crops):
        canon = {}
        for tid, lst in coll.crops.items():
            canon.setdefault(idmap.get(tid, tid), []).extend(lst)
        native = {cid: t for cid, (t, _) in clf.assign(canon).items()}
        for k in keys:
            lst = canon.get(k, [])
            f = clf._track_feature(lst) if len(lst) >= config.TEAM_MIN_CROPS else None
            if f is None:
                continue
            d = np.linalg.norm(clf.km.cluster_centers_ - f, axis=1)
            lab[k] = int(d.argmin())
            ratio[k] = float(d.min() / max(d.max(), 1e-6))
    teams = matched_abstain(lab, ratio, keys, n_abst)
    d = P["builds"] / "e3_colour_replay"
    d.mkdir(parents=True, exist_ok=True)
    for suffix in ("_frames.json", "_trajectories.json"):
        link = d / f"{stem}{suffix}"
        if not link.exists() and (r23 / f"{stem}{suffix}").exists():
            os.symlink(r23 / f"{stem}{suffix}", link)
    atomic_write(d / f"{stem}_identity.json", json.dumps({**ident, "team_by_track": {str(k): v for k, v in teams.items()},
                                                           "e3": {"arm": "colour_replay", "rule": "D154"}}))
    return {"window": stem, "abstain": sum(v is None for v in teams.values()), "stored_abstain": n_abst,
            "trace": order_trace(native, stored)}


def cmd_replay_colour(args) -> dict:
    import time
    from tools import p8_frozen as F
    _guard(args.tag)
    P = F.install_paths()
    t0 = time.time()
    rows = [colour_window(P, w) for w in F.label_windows(P, args.tag)
            if (P["builds"] / "r23" / f"{w['window']}_frames.json").exists()]
    keys = sum(r["trace"]["keys"] for r in rows)
    given = sum(r["trace"]["as_given"] for r in rows)
    up = sum(r["trace"]["up_to_order"] for r in rows)
    out = {"tag": args.tag, "windows": len(rows), "keys": keys,
           "agree_as_given": round(given / keys, 4) if keys else None,
           "agree_up_to_order": round(up / keys, 4) if keys else None,
           "swapped_windows": sum(r["trace"]["swapped_window"] for r in rows),
           "disagreement_explained_by_order": round((up - given) / (keys - given), 4) if keys > given else None,
           "abstain_share": round(sum(r["abstain"] for r in rows) / keys, 4) if keys else None,
           "stored_abstain_share": round(sum(r["stored_abstain"] for r in rows) / keys, 4) if keys else None,
           "seconds": round(time.time() - t0, 1), "per_window": rows}
    atomic_write(p8_root() / "e3" / f"{args.tag}_replay_colour.json", json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k != "per_window"}, indent=1))
    return out


def cmd_score_colour(args) -> dict:
    import time
    from sportvu import bench as B
    from tools import p8_frozen as F
    from tools import p8_scorecard as S
    g = _guard(args.tag)
    P = F.install_paths()
    S.load_fixed_constants()
    mirror = F.mirror_of(P, args.tag)
    mom = S.Moments(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    events = S.pbp_events(g["sportvu"])
    refs = B.References()
    t0 = time.time()
    wins = {}
    for w in F.label_windows(P, args.tag):
        rec = S.score_window(P, args.tag, w, P["builds"] / "e3_colour_replay", "camera", mom, mirror, refs, events)[0]
        if "skipped" not in rec:
            wins[w["window"]] = window_team(rec["team_rows"])
    out = {"tag": args.tag, "truth": "camera", "arm": "colour_replay", "windows": wins, "seconds": round(time.time() - t0, 1)}
    atomic_write(p8_root() / "e3" / f"{args.tag}_teams_colour.json", json.dumps(out))
    print(json.dumps({"tag": args.tag, "windows": len(wins), "seconds": out["seconds"],
                      "colour_replay": pooled_acc([{"c": v} for v in wins.values()], "c")}, indent=1))
    return out


def load_models(device: str) -> dict:
    import torch
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    from PIL import Image
    from gate.backbones import as_tensor

    def runner(model, proc, clip: bool):
        def run(images, bs: int = 64):
            out = []
            with torch.no_grad():
                for i in range(0, len(images), bs):
                    batch = [Image.fromarray(im) for im in images[i:i + bs]]
                    inp = proc(images=batch, return_tensors="pt").to(device)
                    f = as_tensor(model.get_image_features(**inp) if clip else model(**inp))
                    out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
            return np.concatenate(out).astype(np.float32)
        return run
    clip_m = CLIPModel.from_pretrained(ARMS["clip"]).to(device).eval()
    dino_m = AutoModel.from_pretrained(ARMS["dinov2_small"]).to(device).eval()
    return {"clip": runner(clip_m, CLIPProcessor.from_pretrained(ARMS["clip"]), True),
            "dinov2_small": runner(dino_m, AutoImageProcessor.from_pretrained(ARMS["dinov2_small"]), False)}


def cmd_replay(args) -> dict:
    import time
    import config
    from tools import p8_frozen as F
    _guard(args.tag)
    P = F.install_paths()
    device = config.get_device()
    models = load_models(device)
    t0 = time.time()
    rows = []
    for w in F.label_windows(P, args.tag):
        if not (P["builds"] / "r23" / f"{w['window']}_frames.json").exists():
            continue
        rows.append(replay_window(P, w, models))
    keys = sum(r["keys"] for r in rows)
    out = {"tag": args.tag, "device": device, "windows": len(rows), "keys": keys,
           "colour_replay_agreement": round(sum(r["colour_replay_agrees"] for r in rows) / keys, 4) if keys else None,
           "stored_abstain_share": round(sum(r["stored_abstain"] for r in rows) / keys, 4) if keys else None,
           "arm_abstain_share": {a: round(sum(r["arms"][a]["abstain"] for r in rows) / keys, 4) if keys else None for a in ARMS},
           "seconds": round(time.time() - t0, 1), "per_window": rows}
    d = p8_root() / "e3"
    d.mkdir(parents=True, exist_ok=True)
    atomic_write(d / f"{args.tag}_replay.json", json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k != "per_window"}, indent=1))
    return out


def cmd_score(args) -> dict:
    import time
    from sportvu import bench as B
    from tools import p8_frozen as F
    from tools import p8_scorecard as S
    g = _guard(args.tag)
    P = F.install_paths()
    S.load_fixed_constants()
    mirror = F.mirror_of(P, args.tag)
    mom = S.Moments(json.loads((P["sportvu"] / f"{g['sportvu']}_moments.json").read_text())["moments"])
    events = S.pbp_events(g["sportvu"])
    refs = B.References()
    t0 = time.time()
    wins = {}
    for arm in ("r23", *(f"e3_{a}" for a in ARMS)):
        bd = P["builds"] / arm
        for w in F.label_windows(P, args.tag):
            rec = S.score_window(P, args.tag, w, bd, "camera", mom, mirror, refs, events)[0]
            if "skipped" in rec:
                continue
            wins.setdefault(w["window"], {})[arm.replace("e3_", "")] = window_team(rec["team_rows"])
    out = {"tag": args.tag, "truth": "camera", "windows": wins, "seconds": round(time.time() - t0, 1)}
    atomic_write(p8_root() / "e3" / f"{args.tag}_teams.json", json.dumps(out))
    print(json.dumps({"tag": args.tag, "windows": len(wins), "seconds": out["seconds"],
                      **{a: pooled_acc([v for v in wins.values() if a in v], a) for a in ("r23", *ARMS)}}, indent=1))
    return out


# ── report (no model) ───────────────────────────────────────────────────────────────────────────────────────────────
def game_rows(wins: list) -> dict:
    out = {}
    for a in ("r23", *ARMS):
        out[a] = {"n": sum(w[a]["n"] for w in wins), "acc": pooled_acc(wins, a), "fixed": pooled_acc(wins, a, "fixed")}
        if a != "r23":
            out[a]["diff_pts"] = round(100 * (out[a]["acc"] - out["r23"]["acc"]), 2)
    return out


def cmd_report(args) -> dict:
    d = p8_root() / "e3"
    refuse_cv_test(list(POOLED) + list(APART), WHO)
    games = {}
    for t in (*POOLED, *APART):
        p = d / f"{t}_teams.json"
        if p.exists():
            games[t] = [w for w in json.loads(p.read_text())["windows"].values() if all(a in w for a in ("r23", *ARMS))]
    replay = {t: {k: v for k, v in json.loads((d / f"{t}_replay.json").read_text()).items() if k != "per_window"}
              for t in games if (d / f"{t}_replay.json").exists()}
    per_game = {}
    for t, wins in games.items():
        per_game[t] = game_rows(wins)
        for a in ARMS:
            wg = {i: [w] for i, w in enumerate(wins)}
            per_game[t][a]["window_interval_pts"] = [round(100 * x, 2) for x in paired_boot(wg, a) or []]
    pooled_wins = {t: games[t] for t in POOLED if t in games}
    allw = [w for v in pooled_wins.values() for w in v]
    pooled = game_rows(allw) if allw else {}
    res = {"pooled_games": list(pooled_wins), "per_game": per_game, "replay": replay}
    if len(pooled_wins) == len(POOLED):
        win_groups = {f"{t}/{i}": [w] for t, v in pooled_wins.items() for i, w in enumerate(v)}
        for a in ARMS:
            gi, wi = paired_boot(pooled_wins, a), paired_boot(win_groups, a)
            pooled[a]["game_interval"], pooled[a]["window_interval"] = gi, wi
            pooled[a]["game_interval_pts"] = [round(100 * x, 2) for x in gi]
            pooled[a]["window_interval_pts"] = [round(100 * x, 2) for x in wi]
        res["pooled"] = pooled
        res["verdict"] = {a: verdict(pooled, {t: per_game[t] for t in POOLED}, a) for a in ARMS}
    atomic_write(d / "e3_result.json", json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def cmd_report_colour(args) -> dict:
    """D154: the colour-on-replay arm against r23, and both embedding arms against it, per game and pooled."""
    d = p8_root() / "e3"
    refuse_cv_test(list(POOLED) + list(APART), WHO)
    games = {}
    for t in (*POOLED, *APART):
        a, b = d / f"{t}_teams.json", d / f"{t}_teams_colour.json"
        if a.exists() and b.exists():
            base, col = json.loads(a.read_text())["windows"], json.loads(b.read_text())["windows"]
            games[t] = [{**base[k], "colour_replay": col[k]} for k in base
                        if k in col and all(x in base[k] for x in ("r23", *ARMS))]
    arms = ("r23", "colour_replay", *ARMS)

    def rows(wins):
        out = {a: {"n": sum(w[a]["n"] for w in wins), "acc": pooled_acc(wins, a), "fixed": pooled_acc(wins, a, "fixed")}
               for a in arms}
        out["colour_replay"]["minus_r23_pts"] = round(100 * (out["colour_replay"]["acc"] - out["r23"]["acc"]), 2)
        for a in ARMS:
            out[a]["minus_colour_replay_pts"] = round(100 * (out[a]["acc"] - out["colour_replay"]["acc"]), 2)
        return out

    def pts(iv):
        return [round(100 * x, 2) for x in iv] if iv else None

    per_game = {}
    for t, wins in games.items():
        r = rows(wins)
        wg = {i: [w] for i, w in enumerate(wins)}
        r["colour_replay"]["minus_r23_window_interval_pts"] = pts(paired_boot(wg, "colour_replay", "r23"))
        for a in ARMS:
            r[a]["minus_colour_replay_window_interval_pts"] = pts(paired_boot(wg, a, "colour_replay"))
        per_game[t] = r
    res = {"rule": "D154; tracking.md, colour-on-replay arm", "per_game": per_game}
    pw = {t: games[t] for t in POOLED if t in games}
    if len(pw) == len(POOLED):
        allw = [w for v in pw.values() for w in v]
        pooled = rows(allw)
        wg = {f"{t}/{i}": [w] for t, v in pw.items() for i, w in enumerate(v)}
        pooled["colour_replay"]["minus_r23_game_interval_pts"] = pts(paired_boot(pw, "colour_replay", "r23"))
        pooled["colour_replay"]["minus_r23_window_interval_pts"] = pts(paired_boot(wg, "colour_replay", "r23"))
        amc = {}
        for a in ARMS:
            gi, wi = paired_boot(pw, a, "colour_replay"), paired_boot(wg, a, "colour_replay")
            pooled[a]["minus_colour_replay_game_interval_pts"], pooled[a]["minus_colour_replay_window_interval_pts"] = pts(gi), pts(wi)
            amc[a] = {"game_interval": gi, "window_interval": wi}
        res["pooled"] = pooled
        res["reading"] = confound_reading(pooled["colour_replay"]["minus_r23_pts"], amc)
    res["trace"] = {t: {k: v for k, v in json.loads((d / f"{t}_replay_colour.json").read_text()).items() if k != "per_window"}
                    for t in games if (d / f"{t}_replay_colour.json").exists()}
    atomic_write(d / "e3b_result.json", json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tools.p8_e3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("replay", "score", "replay-colour", "score-colour"):
        s = sub.add_parser(c)
        s.add_argument("tag", choices=[*POOLED, *APART])
    sub.add_parser("report")
    sub.add_parser("report-colour")
    a = ap.parse_args(argv)
    {"replay": cmd_replay, "score": cmd_score, "report": cmd_report, "replay-colour": cmd_replay_colour,
     "score-colour": cmd_score_colour, "report-colour": cmd_report_colour}[a.cmd](a)


if __name__ == "__main__":
    main()
