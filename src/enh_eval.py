"""Evaluation protocol for the Segmentation Enhancement section.

Problem with tuning post-processing on the validation set and reporting the same set (as in
notebook 02, 0.388 -> 0.405): the gain is optimistic. Here every post-processing setting is
**cross-fitted**: the 107 validation observations are split into two folds by whole 7-day
blocks (the same blocking as the train/validation split, so neighbouring days never straddle
folds). Settings are tuned on one fold and scored on the other, and the two held-out halves are
pooled. That out-of-fold (OOF) score is the number to report.

The ablation is a greedy chain of steps (threshold -> hysteresis -> join -> split -> filter ...).
Each step tunes only its own parameters, with the earlier steps fixed to what *that fold's*
chain chose, and each grid contains the "do nothing" setting, so a step that does not help can
be switched off. A third context, "all", tunes the same chain on every validation observation;
its parameters are the ones used on the test set.

Scores use notebook 01's protocol (evaluate.score / evaluate.summarise): each observation is
scored against every annotation set, averaged over sets, then over observations.
"""
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import evaluate as ev
from . import postprocess as pp
from .inference import load_probs
from .metrics import best_iou_per_pred, panoptic_quality, split_merge


# ----------------------------------------------------------------------------- folds

def make_folds(items, obs_meta, k=2, block_days=7, seed=0):
    "obs -> fold (0..k-1). Whole 7-day blocks go to one fold; folds get similar numbers of obs."
    obs = [it["obs"] for it in items]
    if "date" in obs_meta:
        dates = pd.to_datetime(obs_meta.set_index("obs").loc[obs, "date"]).reset_index(drop=True)
    else:
        dates = pd.to_datetime(pd.Series([o[:14] for o in obs]), format="%Y%m%d%H%M%S")
    block = ((dates - dates.min()).dt.days // block_days).to_numpy()
    size = pd.Series(block).value_counts()
    rng = np.random.default_rng(seed)
    count = np.zeros(k)
    fold_of_block = {}
    for b in rng.permutation(size.index.to_numpy()):
        f = int(np.argmin(count))
        fold_of_block[b] = f
        count[f] += size[b]
    return {o: fold_of_block[b] for o, b in zip(obs, block)}


# ----------------------------------------------------------------------------- scoring

def _eval_obs(prob_dir, cache, it, postproc, configs):
    probs = load_probs(prob_dir, it["obs"])
    _, disk = ev.load_obs(cache, it["obs"])
    gts = ev.load_gts(cache, it["image_ids"])
    any_gt = np.logical_or.reduce([g > 0 for g in gts])
    memo = {}   # pp.enhanced: configs that differ only in the filter reuse stages 1-3
    rows = []
    for cid, kw in configs:
        t0 = time.perf_counter()
        if postproc is pp.enhanced:
            up_kw = {k: v for k, v in kw.items() if k not in pp.FILTER_KEYS}
            key = tuple(sorted((k, repr(v)) for k, v in up_kw.items()))
            if key not in memo:
                t_up = time.perf_counter()
                memo[key] = (pp.upstream(probs, disk, **up_kw), time.perf_counter() - t_up)
            lab, up_s = memo[key]
            t0 = time.perf_counter() - up_s          # charge the shared stages to every config
            pred = pp.filter_instances(lab, probs, disk, **{k: kw[k] for k in pp.FILTER_KEYS if k in kw})
        else:
            pred = postproc(probs, disk, **kw)
        dt = time.perf_counter() - t0
        # Scoring only needs pixels that are filament in the prediction or in some annotation;
        # PQ, split/merge counts and Dice are unchanged, and it is ~50x faster than full images.
        sel = (pred > 0) | any_gt
        for r in ev.score(pred[sel], [g[sel] for g in gts], it["image_ids"], it["obs"],
                          panoptic_quality, split_merge):
            r.update(cid=cid, pp_s=dt)
            rows.append(r)
    return rows


def evaluate_configs(prob_dir, items, cache, configs_for, postproc=pp.enhanced, n_jobs=4):
    """configs_for(obs) -> list of (config id, kwargs). Each observation's probabilities are loaded
    once and every config is run on them. Returns one row per (config, annotation set)."""
    jobs = (delayed(_eval_obs)(str(prob_dir), str(cache), it, postproc, configs_for(it["obs"]))
            for it in items)
    rows = Parallel(n_jobs=n_jobs)(jobs) if n_jobs != 1 else [j[0](*j[1], **j[2]) for j in jobs]
    return pd.DataFrame([r for rs in rows for r in rs])


def score_params(prob_dir, items, cache, params_for, postproc=pp.enhanced, n_jobs=4):
    """Score each observation with params_for(obs) (a kwargs dict, or one dict for all).
    Returns (summary dict, per-set DataFrame) like evaluate.score_collected."""
    f = params_for if callable(params_for) else (lambda o: params_for)
    df = evaluate_configs(prob_dir, items, cache, lambda o: [(0, f(o))], postproc, n_jobs)
    return summary(df), df


def summary(df):
    s = ev.summarise(df.drop(columns=["cid"], errors="ignore"))
    if "pp_s" in df:
        s["pp_s"] = float(df.groupby("obs").pp_s.first().mean())
    return s


def per_obs_pq(df):
    return df.groupby("obs").PQ.mean()


# ----------------------------------------------------------------------------- grids and chain

def _valid(kw):
    lo, thr = kw.get("lo"), kw.get("thr", 0.5)
    if lo is not None and lo >= thr:
        return False
    if kw.get("join_thr") is not None and kw["join_thr"] >= thr:
        return False
    return True


KEEP = "keep"   # grid value meaning "leave this parameter as the earlier steps chose it"


def expand(grid, base):
    """Configs for one step, merged onto base (the earlier steps' choices). grid is either a dict
    of lists (every combination) or a list of dicts (exactly these). Invalid and duplicate
    configs are dropped; order is kept, so list the "do nothing" setting first."""
    combos = grid if isinstance(grid, list) else (
        [dict(zip(grid, v)) for v in itertools.product(*grid.values())] if grid else [{}])
    out, seen = [], set()
    for over in combos:
        kw = {**base, **{k: v for k, v in over.items() if not (isinstance(v, str) and v == KEEP)}}
        key = tuple(sorted((k, repr(v)) for k, v in kw.items() if k != "clf"))
        if _valid(kw) and key not in seen:
            seen.add(key)
            out.append(kw)
    return out


def _instances_obs(prob_dir, cache, it, postproc, kw):
    probs = load_probs(prob_dir, it["obs"])
    _, disk = ev.load_obs(cache, it["obs"])
    pred = postproc(probs, disk, **kw)
    X, area = pp.instance_features(pred, probs, disk)
    if len(X) == 0:
        return pd.DataFrame()
    gts = ev.load_gts(cache, it["image_ids"])
    ious = np.stack([best_iou_per_pred(pred, g) for g in gts])
    df = pd.DataFrame(X, columns=pp.FEATURES)
    df.insert(0, "inst", np.arange(1, len(X) + 1))
    df.insert(0, "obs", it["obs"])
    df["area"] = area
    df["match_frac"] = (ious > 0.5).mean(0)   # share of annotators for whom it is a true positive
    df["best_iou"] = ious.mean(0)
    return df


def instance_table(prob_dir, items, cache, params_for, postproc=pp.enhanced, n_jobs=4):
    "One row per predicted instance: FEATURES, area, and whether annotators count it as a match."
    f = params_for if callable(params_for) else (lambda o: params_for)
    dfs = Parallel(n_jobs=n_jobs)(delayed(_instances_obs)(str(prob_dir), str(cache), it, postproc, f(it["obs"]))
                                  for it in items)
    dfs = [d for d in dfs if len(d)]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["obs", "inst", *pp.FEATURES, "area", "match_frac", "best_iou"])


def fit_filter(table, seed=0):
    """Small gradient-boosted classifier: is this predicted instance a real filament (a match for
    at least half of the annotators)? Trained on instance FEATURES only."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    y = (table.match_frac >= 0.5).to_numpy() if len(table) else np.zeros(0, bool)
    if len(np.unique(y)) < 2:          # nothing to learn (no instances, or all of one kind)
        return None
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=seed)
    clf.fit(table[pp.FEATURES].to_numpy(np.float32), y)
    return clf


_NO_FILTER = dict(min_area=0, min_mean=0.0, clf=None)


def crossfit_chain(prob_dir, items, cache, folds, steps, postproc=pp.enhanced, n_jobs=4,
                   refit_all=True, log=print):
    """Greedy, cross-fitted post-processing chain.

    steps: list of dicts with keys name, grid (dict of lists), optional fixed (dict) and
    optional learn=True (train a fold-specific instance classifier first; its grid tunes clf_thr).
    Returns a list with one dict per step: params (context -> kwargs; contexts are the held-out
    folds 0..k-1 and "all"), oof (per-set DataFrame, each observation scored with the parameters
    tuned on the other folds), oof_summary, and grid (summary per config in the "all" context)."""
    k = max(folds.values()) + 1
    ctxs = list(range(k)) + (["all"] if refit_all else [])
    tune_obs = {c: [o for o in folds if (c == "all" or folds[o] != c)] for c in ctxs}
    base = {c: {} for c in ctxs}
    out = []
    for step in steps:
        t0 = time.time()
        fixed = step.get("fixed", {})
        extra = {c: {} for c in ctxs}
        if step.get("learn"):
            for c in ctxs:
                sub = [it for it in items if it["obs"] in set(tune_obs[c])]
                table = instance_table(prob_dir, sub, cache, {**base[c], **fixed, **_NO_FILTER}, postproc, n_jobs)
                extra[c]["clf"] = fit_filter(table)
        cfgs = {c: expand(step["grid"], {**base[c], **fixed, **extra[c]}) for c in ctxs}

        tune_sets = {c: set(v) for c, v in tune_obs.items()}

        def configs_for(o):
            return [((c, j), kw) for c in ctxs if o in tune_sets[c] for j, kw in enumerate(cfgs[c])]

        df = evaluate_configs(prob_dir, items, cache, configs_for, postproc, n_jobs)
        df["ctx"] = df.cid.map(lambda t: t[0])
        df["j"] = df.cid.map(lambda t: t[1])
        grid_tab = None
        for c in ctxs:
            d = df[df.ctx == c]
            score = d.groupby(["j", "obs"]).PQ.mean().groupby("j").mean()
            best = int(score.idxmax())           # first maximum: earlier (simpler) configs win ties
            base[c] = cfgs[c][best]
            if c == "all":
                grid_tab = pd.DataFrame([dict(_show(cfgs[c][j]), **summary(d[d.j == j])) for j in score.index])
                grid_tab = grid_tab.sort_values("PQ", ascending=False).reset_index(drop=True)
        # held-out evaluation: each observation scored with the chain tuned without its fold
        oof_s, oof = score_params(prob_dir, items, cache, lambda o: base[folds[o]], postproc, n_jobs)
        res = dict(name=step["name"], params={c: dict(base[c]) for c in ctxs}, oof=oof,
                   oof_summary=oof_s, grid=grid_tab, minutes=(time.time() - t0) / 60)
        out.append(res)
        log(f"{step['name']:<32} OOF PQ {oof_s['PQ']:.4f} (FP {oof_s['FP']}, FN {oof_s['FN']}, "
            f"split {oof_s.get('one_to_many')}, merge {oof_s.get('many_to_one')})  "
            f"chosen[all]: {_show(base.get('all', base[0]))}  [{res['minutes']:.1f} min]")
    return out


def _show(kw):
    "Parameters without the classifier object, for tables and logs."
    return {k: ("trained" if k == "clf" and v is not None else v) for k, v in kw.items()}


def apply_chain(prob_dir, items, cache, folds, step_result, postproc=pp.enhanced, n_jobs=4):
    "Score another set of probabilities (e.g. another seed) with a step's cross-fitted parameters."
    params = step_result["params"]
    return score_params(prob_dir, items, cache, lambda o: params[folds[o]], postproc, n_jobs)


def mean_std(summaries, keys=("PQ", "SQ", "RQ", "TP", "FP", "FN", "one_to_many", "many_to_one", "pp_s")):
    "Mean and std over seeds of the summary dicts."
    d = pd.DataFrame(summaries)
    out = {}
    for k in keys:
        if k in d:
            out[k] = d[k].mean()
            out[k + "_std"] = d[k].std(ddof=1) if len(d) > 1 else np.nan
    return out


def paired_bootstrap(df_a, df_b, n=5000, seed=0):
    "Change in per-observation PQ from a to b: mean and 95% bootstrap interval over observations."
    a, b = per_obs_pq(df_a), per_obs_pq(df_b)
    d = (b - a).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return dict(delta=d.mean(), lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5),
                better=int((d > 0).sum()), worse=int((d < 0).sum()), n_obs=len(d))


# ----------------------------------------------------------------------------- timing

def benchmark_pipeline(predict_fn, paths, preprocess, postproc=pp.enhanced, pp_kwargs=None, warmup=2):
    """End-to-end seconds per image (raw file -> preprocess -> network -> instances), like
    evaluate.benchmark but for any predict_fn(img) (TTA, ensembles, pruned models, ...)."""
    times = []
    for i, p in enumerate(paths):
        t0 = time.perf_counter()
        img, disk = preprocess(p)
        t1 = time.perf_counter()
        probs = predict_fn(img)
        t2 = time.perf_counter()
        postproc(probs, disk, **(pp_kwargs or {}))
        t3 = time.perf_counter()
        if i >= warmup:
            times.append(dict(preprocess_s=t1 - t0, network_s=t2 - t1, postproc_s=t3 - t2, total_s=t3 - t0))
    return pd.DataFrame(times).agg(["mean", "std"])


# ----------------------------------------------------------------------------- figures

def _colour(lab, seed=0):
    rng = np.random.default_rng(seed)
    pal = np.vstack([[0, 0, 0], rng.uniform(0.25, 1, (int(lab.max()) + 1, 3))])
    return pal[lab]


def show_examples(prob_dir, cache, it, pipelines, path=None, down=4, crop=None):
    """One row per observation: image, first annotator's filaments, then each pipeline's
    instances (each instance a random colour), with that pipeline's PQ against the annotator."""
    import matplotlib.pyplot as plt
    probs = load_probs(prob_dir, it["obs"])
    img, disk = ev.load_obs(cache, it["obs"])
    gt = ev.load_gts(cache, it["image_ids"][:1])[0]
    sl = (slice(None), slice(None)) if crop is None else (slice(crop[0], crop[1]), slice(crop[2], crop[3]))
    d = 1 if crop is not None else down
    panels = [("image", img[sl][::d, ::d], None), ("annotation", _colour(gt[sl][::d, ::d]), None)]
    for name, (fn, kw) in pipelines.items():
        pred = fn(probs, disk, **kw)
        q = panoptic_quality(pred, gt)
        panels.append((f"{name}\nPQ {q['PQ']:.2f}  TP {q['TP']} FP {q['FP']} FN {q['FN']}",
                       _colour(pred[sl][::d, ::d]), None))
    fig, ax = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.3))
    for a, (title, im, _) in zip(ax, panels):
        a.imshow(im, cmap="gray" if im.ndim == 2 else None)
        a.set_title(title, fontsize=9)
        a.axis("off")
    fig.suptitle(it["obs"], fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def plot_ablation(table, path=None, title="Cumulative post-processing ablation (out-of-fold, mean ± std over seeds)"):
    import matplotlib.pyplot as plt
    t = table.iloc[::-1]
    fig, ax = plt.subplots(1, 3, figsize=(15, 0.45 * len(t) + 1.5), sharey=True)
    ax[0].barh(t.step, t.PQ, xerr=t.get("PQ_std"), color="#2f6fb5", capsize=3)
    ax[0].set_xlabel("validation PQ")
    std = t.get("PQ_std", t.PQ * 0).fillna(0)
    for y, (v, s) in enumerate(zip(t.PQ, std)):
        ax[0].text(v + s + 0.003, y, f"{v:.3f}", va="center", fontsize=8)
    ax[0].set_xlim(max(0, t.PQ.min() - 0.05), (t.PQ + std).max() + 0.03)
    ax[1].barh(t.step, t.FP, color="#b8741a", label="false positives")
    ax[1].barh(t.step, -t.FN, color="#888", label="misses (left)")
    ax[1].set_xlabel("instances (seed mean)"); ax[1].legend(fontsize=8)
    ax[2].barh(t.step, t.one_to_many, color="#1f8a70", label="fragmented GT")
    ax[2].barh(t.step, -t.many_to_one, color="#a33", label="merged (left)")
    ax[2].set_xlabel("filaments (seed mean)"); ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3, axis="x")
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def plot_pareto(df, path=None, x="total_s", y="PQ", label="pipeline"):
    "PQ against end-to-end seconds per image; the non-dominated pipelines are joined by a line."
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(df[x], df[y], yerr=df.get(y + "_std"), fmt="o", capsize=3)
    for i, (_, r) in enumerate(df.sort_values(x).iterrows()):
        ax.annotate(r[label], (r[x], r[y]), textcoords="offset points", xytext=(6, 10 if i % 2 else -12),
                    fontsize=8, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    front = df.sort_values(x)
    front = front[front[y] >= front[y].cummax()]
    ax.plot(front[x], front[y], "--", color="gray", alpha=0.6)
    ax.set(xscale="log", xlabel="end-to-end seconds per image (log)", ylabel="validation PQ")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def plot_instance_errors(table, path=None):
    "Feature distributions of predicted instances that annotators accept (TP) or not (FP)."
    import matplotlib.pyplot as plt
    tp = table.match_frac >= 0.5
    cols = [("area", True), ("mean_fg", False), ("rdist", False), ("elongation", True),
            ("mean_spine", False), ("mean_width", True)]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for a, (c, logx) in zip(ax.ravel(), cols):
        v = table[c].clip(lower=1e-3) if logx else table[c]
        bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 40) if logx else 40
        a.hist(v[tp], bins=bins, alpha=0.6, label=f"matched ({tp.sum()})", color="#1f8a70")
        a.hist(v[~tp], bins=bins, alpha=0.6, label=f"false positive ({(~tp).sum()})", color="#b8741a")
        if logx:
            a.set_xscale("log")
        a.set_title(c); a.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig
