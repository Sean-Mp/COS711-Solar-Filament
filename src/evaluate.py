"""Full-disk inference, baseline instance extraction, and validation against every annotation set.

Validation works per observation: the model runs once on the full 2048x2048 disk and the
predicted instances are scored against each annotator's set for that observation. Scores are
averaged over sets first, then over observations, so every observation counts equally.
"""
import itertools
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


@torch.no_grad()
def predict_probs(model, img, tta=False, amp=True, multiple=32):
    "img: (H, W) float32 in [0, 1] -> (C, H, W) float32 sigmoid probabilities."
    model.eval()
    dev = next(model.parameters()).device
    H, W = img.shape
    x = torch.from_numpy(np.ascontiguousarray(img, np.float32))[None, None].to(dev)
    x = F.pad(x, (0, -W % multiple, 0, -H % multiple))
    flips = [(), (3,), (2,), (2, 3)] if tta else [()]
    p = 0
    with torch.autocast(dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
        for f in flips:
            out = torch.sigmoid(model(x.flip(f) if f else x).float())
            p = p + (out.flip(f) if f else out)
    return (p / len(flips))[0, :, :H, :W].cpu().numpy()


def instances_cc(probs, disk, thr=0.5, min_area=50):
    "Baseline: threshold the fg channel inside the disk, connected components, drop small blobs."
    fg = ((probs[0] > thr) & disk).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    keep = np.zeros(n, np.int32)
    ids = np.nonzero(stats[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
    keep[ids] = np.arange(1, len(ids) + 1)
    return keep[lab]


def val_items(set_meta, split="val"):
    "One item per observation, listing all of its annotation sets."
    s = set_meta[set_meta.split == split]
    return [dict(obs=o, image_ids=list(g.image_id)) for o, g in s.groupby("obs")]


def load_obs(cache, obs):
    "Cached (img, disk) for an observation, as written by notebook 01."
    a = np.load(Path(cache) / "obs" / f"{obs}.npy")
    return a[0].astype(np.float32) / 255, a[1].astype(bool)


def load_gts(cache, image_ids):
    return [np.load(Path(cache) / "inst" / f"{i}.npy") for i in image_ids]


def score(pred, gts, image_ids, obs, pq_fn, split_merge_fn=None):
    "One row per annotation set: PQ/SQ/RQ/TP/FP/FN, pixel Dice, and split/merge counts."
    rows = []
    for iid, gt in zip(image_ids, gts):
        r = dict(obs=obs, image_id=iid, n_pred=int(pred.max()), **pq_fn(pred, gt))
        p, g = pred > 0, gt > 0
        r["dice"] = 2 * (p & g).sum() / max(p.sum() + g.sum(), 1)
        if split_merge_fn is not None:
            sm = split_merge_fn(pred, gt)
            r.update(one_to_many=sm["one_to_many"], many_to_one=sm["many_to_one"])
        rows.append(r)
    return rows


def summarise(df):
    "Mean/std over observations (each the mean over its sets) plus PQ pooled over all sets."
    per_obs = df.groupby("obs")[["PQ", "SQ", "RQ", "dice"]].mean()
    tp, fp, fn = df.TP.sum(), df.FP.sum(), df.FN.sum()
    out = dict(PQ=per_obs.PQ.mean(), PQ_std=per_obs.PQ.std(), SQ=per_obs.SQ.mean(),
               RQ=per_obs.RQ.mean(), dice=per_obs.dice.mean(),
               PQ_pooled=(df.SQ * df.TP).sum() / max(tp + 0.5 * fp + 0.5 * fn, 1e-9),
               TP=int(tp), FP=int(fp), FN=int(fn))
    if "one_to_many" in df:
        out.update(one_to_many=int(df.one_to_many.sum()), many_to_one=int(df.many_to_one.sum()))
    return {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in out.items()}


def evaluate(model, items, cache, pq_fn, split_merge_fn=None, postproc=instances_cc,
             pp_kwargs=None, tta=False):
    "Stream over the validation observations; returns (summary dict, per-set DataFrame)."
    rows = []
    for it in items:
        img, disk = load_obs(cache, it["obs"])
        pred = postproc(predict_probs(model, img, tta=tta), disk, **(pp_kwargs or {}))
        rows += score(pred, load_gts(cache, it["image_ids"]), it["image_ids"], it["obs"],
                      pq_fn, split_merge_fn)
    df = pd.DataFrame(rows)
    return summarise(df), df


def collect_probs(model, items, cache, tta=False):
    """Run the model once over the validation set and keep uint8 probabilities, disks and GT
    instance maps in memory, so post-processing can be tuned without re-running the network.
    (~40 MB per observation at 2048x2048, ~4 GB for the validation set.)"""
    out = []
    for it in items:
        img, disk = load_obs(cache, it["obs"])
        q = (predict_probs(model, img, tta=tta) * 255 + 0.5).astype(np.uint8)
        out.append(dict(it, q=q, disk=disk, gts=load_gts(cache, it["image_ids"])))
    return out


def score_collected(collected, pq_fn, split_merge_fn=None, postproc=instances_cc, **pp_kwargs):
    rows = []
    for c in collected:
        pred = postproc(c["q"].astype(np.float32) / 255, c["disk"], **pp_kwargs)
        rows += score(pred, c["gts"], c["image_ids"], c["obs"], pq_fn, split_merge_fn)
    df = pd.DataFrame(rows)
    return summarise(df), df


def grid_search(collected, pq_fn, grid, postproc=instances_cc):
    "Exhaustive search over post-processing parameters, e.g. grid=dict(thr=[...], min_area=[...])."
    rows = []
    for values in itertools.product(*grid.values()):
        kw = dict(zip(grid, values))
        rows.append(dict(kw, **score_collected(collected, pq_fn, postproc=postproc, **kw)[0]))
    return pd.DataFrame(rows).sort_values("PQ", ascending=False).reset_index(drop=True)


def benchmark(model, paths, preprocess, postproc=instances_cc, pp_kwargs=None, tta=False, warmup=2):
    """End-to-end seconds per image: raw file -> preprocess -> network -> instance map.
    preprocess(path) -> (img float32 [0,1], disk bool). The first `warmup` images are not timed."""
    times = []
    for i, p in enumerate(paths):
        t0 = time.perf_counter()
        img, disk = preprocess(p)
        t1 = time.perf_counter()
        probs = predict_probs(model, img, tta=tta)  # .cpu() inside synchronises the GPU
        t2 = time.perf_counter()
        postproc(probs, disk, **(pp_kwargs or {}))
        t3 = time.perf_counter()
        if i >= warmup:
            times.append(dict(preprocess_s=t1 - t0, network_s=t2 - t1, postproc_s=t3 - t2, total_s=t3 - t0))
    return pd.DataFrame(times).agg(["mean", "std"])
