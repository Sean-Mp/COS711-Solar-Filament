"""Instance metrics, identical to the ones defined in notebook 01 (panoptic_quality, split_merge).

They are copied here so worker processes and the enhancement modules can import them without
running notebook 01. Notebook 03 checks that both versions give the same scores.
"""
import numpy as np


def panoptic_quality(pred_inst, gt_inst, iou_thr=0.5):
    """PQ = SQ * RQ. IoU>0.5 guarantees unique matches"""
    P, G = pred_inst.max() + 1, gt_inst.max() + 1
    cont = np.bincount(gt_inst.ravel().astype(np.int64) * P + pred_inst.ravel(),
                       minlength=G * P).reshape(G, P)
    ga, pa = cont.sum(1), cont.sum(0)
    iou = cont / (ga[:, None] + pa[None, :] - cont + 1e-9)
    iou[0, :] = 0; iou[:, 0] = 0                   # ignore background
    gi, pi = np.nonzero(iou > iou_thr)
    tp = len(gi); ious = iou[gi, pi]
    n_gt, n_pr = len(np.unique(gt_inst)) - (0 in gt_inst), len(np.unique(pred_inst)) - (0 in pred_inst)
    fp, fn = n_pr - tp, n_gt - tp
    sq = ious.mean() if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 1.0
    return dict(PQ=sq * rq, SQ=sq, RQ=rq, TP=tp, FP=fp, FN=fn)


def split_merge(pred_inst, gt_inst, min_frac=0.1):
    """one-to-many: GT covered by >=2 preds; many-to-one: pred covering >=2 GTs.
    A link counts if the intersection is >= min_frac of the *smaller* object's area."""
    P, G = pred_inst.max() + 1, gt_inst.max() + 1
    cont = np.bincount(gt_inst.ravel().astype(np.int64) * P + pred_inst.ravel(),
                       minlength=G * P).reshape(G, P)[1:, 1:]
    if cont.size == 0: return dict(one_to_many=0, many_to_one=0, n_gt=G-1, n_pred=P-1)
    ga, pa = cont.sum(1), cont.sum(0)
    link = cont >= min_frac * np.minimum(ga[:, None], pa[None, :]).clip(1)
    link &= cont > 0
    return dict(one_to_many=int((link.sum(1) >= 2).sum()),
                many_to_one=int((link.sum(0) >= 2).sum()),
                n_gt=int((ga > 0).sum()), n_pred=int((pa > 0).sum()))


def best_iou_per_pred(pred_inst, gt_inst):
    "For each predicted instance 1..P-1, its highest IoU with any ground-truth instance."
    P, G = int(pred_inst.max()) + 1, int(gt_inst.max()) + 1
    cont = np.bincount(gt_inst.ravel().astype(np.int64) * P + pred_inst.ravel(),
                       minlength=G * P).reshape(G, P)
    ga, pa = cont.sum(1), cont.sum(0)
    iou = cont / (ga[:, None] + pa[None, :] - cont + 1e-9)
    iou[0, :] = 0
    return iou[:, 1:].max(0) if G > 1 else np.zeros(P - 1)
