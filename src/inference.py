"""Prediction variants for the enhancement study, and on-disk caching of their probability maps.

* test-time augmentation over the 8 rotations/reflections of the square (tta=8), or the 4 flips
  already used by evaluate.predict_probs (tta=4)
* ensembles: the probabilities of several models (e.g. the three training seeds) are averaged
* reduced-resolution inference (scale < 1): the network runs on a downscaled disk and the
  probabilities are upsampled again, trading accuracy for about 1/scale^2 less network compute
* amp=False runs the network in fp32 (the default is fp16 autocast on GPU, as in notebook 02)

Cached maps are stored as uint8 (probability * 255), one compressed .npz per observation, so the
post-processing experiments can run on CPU without the network.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import evaluate as ev

FLIPS4 = [(0, False), (0, True), (2, False), (2, True)]           # = evaluate.predict_probs(tta=True)
DIHEDRAL8 = [(k, f) for f in (False, True) for k in range(4)]


def _tfm(x, k, f):
    x = torch.rot90(x, k, (2, 3)) if k else x
    return x.flip(3) if f else x


def _inv(x, k, f):
    x = x.flip(3) if f else x
    return torch.rot90(x, -k, (2, 3)) if k else x


@torch.no_grad()
def predict_probs(models, img, tta=1, amp=True, scale=1.0, multiple=32):
    """img (H, W) float32 in [0, 1] -> (C, H, W) float32 mean sigmoid probability over the models
    and the test-time transforms. With one model, tta=1 and scale=1 this equals
    evaluate.predict_probs(model, img)."""
    models = models if isinstance(models, (list, tuple)) else [models]
    dev = next(models[0].parameters()).device
    H, W = img.shape
    src = img
    if scale != 1.0:
        src = cv2.resize(img, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA)
    h, w = src.shape
    x = torch.from_numpy(np.ascontiguousarray(src, np.float32))[None, None].to(dev)
    x = F.pad(x, (0, -w % multiple, 0, -h % multiple))
    tfms = {1: [(0, False)], 4: FLIPS4, 8: DIHEDRAL8}[tta]
    p, n = 0, 0
    with torch.autocast(dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
        for m in models:
            m.eval()
            for k, f in tfms:
                out = torch.sigmoid(m(_tfm(x, k, f)).float())
                p = p + _inv(out, k, f)
                n += 1
    p = (p / n)[0, :, :h, :w].cpu().numpy()
    if scale != 1.0:
        p = cv2.resize(p.transpose(1, 2, 0), (W, H), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        p = np.ascontiguousarray(p)
    return p.astype(np.float32)


def prob_path(prob_dir, obs):
    return Path(prob_dir) / f"{obs}.npz"


def load_probs(prob_dir, obs):
    return np.load(prob_path(prob_dir, obs))["q"].astype(np.float32) / 255


def cache_probs(predict_fn, items, cache, out_dir, progress=None):
    """Run predict_fn(img) -> (C, H, W) probabilities on every item's full disk and save uint8
    maps to out_dir. Observations already cached are skipped. Writes out_dir/timing.json with
    the mean network seconds per observation (GPU synchronised by the .cpu() inside)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    times = []
    todo = [it for it in items if not prob_path(out_dir, it["obs"]).exists()]
    for it in (progress(todo) if progress else todo):
        img, _ = ev.load_obs(cache, it["obs"])
        t0 = time.perf_counter()
        p = predict_fn(img)
        times.append(time.perf_counter() - t0)
        np.savez_compressed(prob_path(out_dir, it["obs"]), q=(p * 255 + 0.5).astype(np.uint8))
    tfile = out_dir / "timing.json"
    if times:
        old = json.loads(tfile.read_text()) if tfile.exists() else {}
        n_old = old.get("n", 0)
        mean = (old.get("network_s", 0) * n_old + sum(times)) / (n_old + len(times))
        tfile.write_text(json.dumps(dict(network_s=mean, n=n_old + len(times))))
    return json.loads(tfile.read_text()) if tfile.exists() else {}


def average_cached(src_dirs, out_dir, items):
    "Ensemble on CPU: average the cached uint8 maps of several variants (e.g. seeds)."
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for it in items:
        f = prob_path(out_dir, it["obs"])
        if f.exists():
            continue
        acc = sum(np.load(prob_path(d, it["obs"]))["q"].astype(np.float32) for d in src_dirs)
        np.savez_compressed(f, q=(acc / len(src_dirs) + 0.5).astype(np.uint8))
    t = [json.loads((Path(d) / "timing.json").read_text()).get("network_s", np.nan)
         for d in src_dirs if (Path(d) / "timing.json").exists()]
    if t:
        (out_dir / "timing.json").write_text(json.dumps(dict(network_s=float(np.sum(t)), n=len(items),
                                                             note="sum of the members' times")))
    return out_dir
