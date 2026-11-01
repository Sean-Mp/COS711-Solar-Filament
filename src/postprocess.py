"""Segmentation Enhancement: turning the U-Net's five output maps into filament instances.

Every public pipeline has the same signature as the baseline `evaluate.instances_cc`:

    fn(probs, disk, **kwargs) -> int32 instance map (0 = background, 1..n = filaments)

so it plugs into `evaluate.score_collected`, `evaluate.grid_search` and `evaluate.benchmark`.
`probs` is (5, H, W) float32 in [0, 1], in the channel order of `unet.TARGET_CHANNELS`:
fg, boundary, contact, dist, spine.

`enhanced` chains four optional stages, each aimed at one error type the baseline makes:

1. foreground   plain threshold, or hysteresis: keep low-confidence pixels only when they are
                connected to high-confidence ones (fewer faint false positives, fewer gaps)
2. join         group fragments of one filament: pieces bridged by low-confidence fg, by the
                predicted spine, or by a small morphological closing get the same label
                (targets fragmentation, the "one-to-many" error)
3. split        cut instances along predicted contact / boundary lines, or between distance-map
                peaks, with a marker-controlled watershed (targets merges, "many-to-one")
4. filter       drop instances by size, mean confidence, position on the disk, or by a small
                classifier trained on instance features (targets false positives)

With every option at its default, `enhanced(probs, disk, thr, min_area)` reproduces the baseline
connected-components extraction exactly.
"""
import cv2
import numpy as np
from skimage.segmentation import watershed

FG, BOUNDARY, CONTACT, DIST, SPINE = range(5)

FEATURES = ["log_area", "mean_fg", "max_fg", "frac_hi", "mean_spine", "max_spine", "mean_boundary",
            "mean_contact", "mean_dist", "max_dist", "elongation", "log_length", "mean_width", "rdist"]


# ----------------------------------------------------------------------------- helpers

def _kernel(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def label_cc(mask):
    "8-connected components of a boolean mask, as int32 labels."
    _, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    return lab


def relabel(lab, keep=None):
    """Consecutive labels 1..n. `keep` (bool per old label, index 0 ignored) drops instances."""
    m = int(lab.max())
    if m == 0:
        return lab.astype(np.int32)
    present = np.bincount(lab.ravel(), minlength=m + 1) > 0
    present[0] = False
    if keep is not None:
        present &= np.asarray(keep, bool)
    lut = np.zeros(m + 1, np.int32)
    lut[present] = np.arange(1, present.sum() + 1, dtype=np.int32)
    return lut[lab]


def disk_geometry(disk):
    "Centre (cx, cy) and radius of the solar disk mask, from its image moments."
    mo = cv2.moments(disk.astype(np.uint8), binaryImage=True)
    if mo["m00"] == 0:
        h, w = disk.shape
        return w / 2, h / 2, min(h, w) / 2
    return mo["m10"] / mo["m00"], mo["m01"] / mo["m00"], float(np.sqrt(mo["m00"] / np.pi))


# ----------------------------------------------------------------------------- stage 1: foreground

def foreground(probs, disk, thr=0.5, lo=None, min_seed=0):
    """Pixels that belong to some filament.

    lo=None: fg > thr (the baseline). lo < thr: hysteresis, i.e. connected regions of fg > lo
    that contain at least `min_seed` pixels (and at least one) with fg > thr."""
    fg = probs[FG]
    if lo is None or lo >= thr:
        return (fg > thr) & disk, None
    lab = label_cc((fg > lo) & disk)
    n = int(lab.max())
    strong = np.bincount(lab[(fg > thr) & disk], minlength=n + 1)
    keep = strong >= max(int(min_seed), 1)
    keep[0] = False
    return keep[lab], None


# ----------------------------------------------------------------------------- stage 2: join

def join_fragments(lab, probs, disk, join_r=0, join_thr=None, spine_thr=None):
    """Give fragments of one filament the same label, without adding pixels to the mask.

    Fragments are grouped if they are connected through (a) fg > join_thr, (b) spine > spine_thr,
    or (c) a morphological closing of radius join_r (gaps up to about 2 * join_r pixels)."""
    if not join_r and join_thr is None and spine_thr is None:
        return lab
    bridge = lab > 0
    if join_thr is not None:
        bridge |= (probs[FG] > join_thr) & disk
    if spine_thr is not None:
        bridge |= (probs[SPINE] > spine_thr) & disk
    if join_r:
        bridge = cv2.morphologyEx(bridge.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(int(join_r))) > 0
    groups = label_cc(bridge)
    return relabel(np.where(lab > 0, groups, 0))


# ----------------------------------------------------------------------------- stage 3: split

def split_instances(lab, probs, split=None, cut_thr=0.5, d_thr=0.5, min_part=50):
    """Split instances that the network marks as two touching filaments.

    split="contact" / "boundary": remove pixels with contact (boundary) > cut_thr, split="dist":
    keep only pixels with dist > d_thr. The remaining connected parts of at least `min_part`
    pixels become watershed markers; an instance is split only if it holds two or more markers,
    and its pixels are then assigned to the markers by a watershed on the boundary map."""
    if split is None:
        return lab
    mask = lab > 0
    if split == "contact":
        core = mask & (probs[CONTACT] <= cut_thr)
    elif split == "boundary":
        core = mask & (probs[BOUNDARY] <= cut_thr)
    elif split == "dist":
        core = mask & (probs[DIST] > d_thr)
    else:
        raise ValueError(f"unknown split mode {split!r}")
    markers = label_cc(core)
    nm = int(markers.max())
    if nm < 2:
        return lab
    size = np.bincount(markers.ravel(), minlength=nm + 1)
    ok = size >= min_part
    ok[0] = False
    markers = relabel(markers, ok)
    nm = int(markers.max())
    if nm < 2:
        return lab
    # markers per instance (each marker lies inside exactly one instance)
    sel = markers > 0
    pairs = np.unique(np.stack([lab[sel], markers[sel]]), axis=1)
    per_inst = np.bincount(pairs[0], minlength=int(lab.max()) + 1)
    to_split = per_inst >= 2
    if not to_split.any():
        return lab
    region = to_split[lab]
    elev = (probs[BOUNDARY] * 255).astype(np.uint8)
    ws = watershed(elev, np.where(region, markers, 0), mask=region)
    out = np.where(region & (ws > 0), ws + int(lab.max()), lab)
    return relabel(out)


# ----------------------------------------------------------------------------- stage 4: filter

def instance_features(lab, probs, disk):
    "One row of FEATURES per instance 1..n (row i-1 is instance i), plus the raw area."
    n = int(lab.max())
    if n == 0:
        return np.zeros((0, len(FEATURES)), np.float32), np.zeros(0)
    yy, xx = np.nonzero(lab)
    ll = lab[yy, xx]
    area = np.bincount(ll, minlength=n + 1)[1:].astype(np.float64)
    a = np.maximum(area, 1)

    def mean_of(ch):
        return np.bincount(ll, probs[ch][yy, xx], minlength=n + 1)[1:] / a

    def max_of(ch):
        out = np.zeros(n + 1)
        np.maximum.at(out, ll, probs[ch][yy, xx])
        return out[1:]

    fgv = probs[FG][yy, xx]
    frac_hi = np.bincount(ll, (fgv > 0.7).astype(np.float64), minlength=n + 1)[1:] / a
    # second moments -> elongation and length of the principal axis
    sx = np.bincount(ll, xx, minlength=n + 1)[1:] / a
    sy = np.bincount(ll, yy, minlength=n + 1)[1:] / a
    sxx = np.bincount(ll, xx.astype(np.float64) ** 2, minlength=n + 1)[1:] / a - sx ** 2
    syy = np.bincount(ll, yy.astype(np.float64) ** 2, minlength=n + 1)[1:] / a - sy ** 2
    sxy = np.bincount(ll, xx.astype(np.float64) * yy, minlength=n + 1)[1:] / a - sx * sy
    tr, det = sxx + syy, sxx * syy - sxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 0)
    length = 4 * np.sqrt(np.maximum(l1, 0)) + 1
    elong = np.sqrt((l1 + 1) / (l2 + 1))
    cx, cy, r = disk_geometry(disk)
    rdist = np.hypot(sx - cx, sy - cy) / max(r, 1)
    X = np.stack([np.log(a), mean_of(FG), max_of(FG), frac_hi, mean_of(SPINE), max_of(SPINE),
                  mean_of(BOUNDARY), mean_of(CONTACT), mean_of(DIST), max_of(DIST), elong,
                  np.log(length), area / length, rdist], 1).astype(np.float32)
    return X, area


def filter_instances(lab, probs, disk, min_area=0, min_mean=0.0, max_rdist=None, clf=None, clf_thr=0.5):
    """Drop instances smaller than min_area, with mean fg probability below min_mean, whose
    centroid lies beyond max_rdist disk radii, or that the classifier `clf` (anything with
    predict_proba on FEATURES) scores below clf_thr."""
    n = int(lab.max())
    if n == 0 or (not min_area and not min_mean and max_rdist is None and (clf is None or not clf_thr)):
        return relabel(lab)
    if not min_mean and max_rdist is None and (clf is None or not clf_thr):
        area = np.bincount(lab.ravel(), minlength=n + 1)[1:]     # size only: no features needed
        return relabel(lab, np.r_[False, area >= min_area])
    X, area = instance_features(lab, probs, disk)
    keep = np.ones(n, bool)
    if min_area:
        keep &= area >= min_area
    if min_mean:
        keep &= X[:, FEATURES.index("mean_fg")] >= min_mean
    if max_rdist is not None:
        keep &= X[:, FEATURES.index("rdist")] <= max_rdist
    if clf is not None and clf_thr:
        keep &= clf.predict_proba(X)[:, 1] >= clf_thr
    return relabel(lab, np.r_[False, keep])


# ----------------------------------------------------------------------------- full pipeline

FILTER_KEYS = ("min_area", "min_mean", "max_rdist", "clf", "clf_thr")


def upstream(probs, disk, thr=0.5, lo=None, min_seed=0, join_r=0, join_thr=None, spine_thr=None,
             split=None, cut_thr=0.5, d_thr=0.5, min_part=50):
    "Stages 1-3: foreground -> connected components -> join fragments -> split merges."
    fg, _ = foreground(probs, disk, thr, lo, min_seed)
    lab = label_cc(fg)
    lab = join_fragments(lab, probs, disk, join_r, join_thr, spine_thr)
    return split_instances(lab, probs, split, cut_thr, d_thr, min_part)


def enhanced(probs, disk, min_area=50, min_mean=0.0, max_rdist=None, clf=None, clf_thr=0.5, **kw):
    "The full pipeline: upstream stages, then the instance filter (stage 4)."
    return filter_instances(upstream(probs, disk, **kw), probs, disk, min_area, min_mean, max_rdist, clf, clf_thr)
