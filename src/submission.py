"""Write instance maps to a Kaggle submission CSV.

The exact column layout is set by the competition (see its Evaluation page and
sample_submission.csv), so it is a parameter here: `FORMAT` in notebook 03 says which column
holds the image id, whether there is one row per filament or one per image, and how a mask is
encoded. Check the first rows against the sample file before submitting.

Encoders:
* "rle"      Kaggle-style run-length string "start length start length ...", 1-indexed,
             pixels numbered row by row (order="C") or column by column (order="F")
* "coco_rle" pycocotools compressed RLE string (column-major), as stored in COCO files
"""
from pathlib import Path

import numpy as np
import pandas as pd


def rle_encode(mask, order="C"):
    px = np.concatenate([[0], np.asarray(mask, bool).ravel(order=order).astype(np.int8), [0]])
    runs = np.flatnonzero(px[1:] != px[:-1]) + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))


def rle_decode(s, shape, order="C"):
    v = np.asarray(str(s).split(), int) if isinstance(s, str) and s.strip() else np.zeros(0, int)
    m = np.zeros(shape[0] * shape[1], bool)
    for start, n in zip(v[0::2] - 1, v[1::2]):
        m[start:start + n] = True
    return m.reshape(shape, order=order)


def coco_rle_encode(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(np.asarray(mask, np.uint8)))
    return r["counts"].decode("ascii")


def encode(mask, encoding="rle", order="C"):
    return coco_rle_encode(mask) if encoding == "coco_rle" else rle_encode(mask, order)


def submission_rows(image_id, inst, fmt):
    """fmt keys: id_col, mask_col, per ("instance" or "image"), encoding, order, optional
    instance_col (1, 2, ... per image), optional extra (dict of constant columns),
    optional empty (value written for an image with no filaments)."""
    rows = []
    n = int(inst.max())
    if fmt.get("per", "instance") == "image":
        # one row per image; instances as separate RLE strings joined by fmt["sep"] or an id map
        enc = [encode(inst == k, fmt.get("encoding", "rle"), fmt.get("order", "C")) for k in range(1, n + 1)]
        rows.append({fmt["id_col"]: image_id, fmt["mask_col"]: fmt.get("sep", "|").join(enc) or fmt.get("empty", "")})
    else:
        for k in range(1, n + 1):
            r = {fmt["id_col"]: image_id, fmt["mask_col"]: encode(inst == k, fmt.get("encoding", "rle"), fmt.get("order", "C"))}
            if fmt.get("instance_col"):
                r[fmt["instance_col"]] = k
            rows.append(r)
        if n == 0 and "empty" in fmt:
            rows.append({fmt["id_col"]: image_id, fmt["mask_col"]: fmt["empty"]})
    for r in rows:
        r.update(fmt.get("extra", {}))
    return rows


def write_submission(pred_dir, out_csv, fmt, image_ids=None):
    """pred_dir holds <stem>.npz files with an `inst` array (notebooks 02/03). image_ids maps a
    file stem to the id Kaggle expects (default: the stem itself)."""
    rows = []
    for f in sorted(Path(pred_dir).glob("*.npz")):
        iid = image_ids.get(f.stem, f.stem) if image_ids else f.stem
        rows += submission_rows(iid, np.load(f)["inst"], fmt)
    cols = [fmt["id_col"]] + ([fmt["instance_col"]] if fmt.get("instance_col") else []) + [fmt["mask_col"]]
    df = pd.DataFrame(rows)
    df = df[cols + [c for c in df.columns if c not in cols]]
    if fmt.get("instance_col"):
        df[fmt["instance_col"]] = df[fmt["instance_col"]].astype("Int64")
    df.to_csv(out_csv, index=False)
    return df


def check_against_sample(sub, sample):
    "Same columns, and every image in the sample appears in the submission."
    problems = []
    if list(sub.columns) != list(sample.columns):
        problems.append(f"columns differ: submission {list(sub.columns)} vs sample {list(sample.columns)}")
    id_col = sample.columns[0]
    if id_col in sub:
        missing = set(sample[id_col].astype(str)) - set(sub[id_col].astype(str))
        if missing:
            problems.append(f"{len(missing)} sample ids missing, e.g. {sorted(missing)[:3]}")
    return problems
