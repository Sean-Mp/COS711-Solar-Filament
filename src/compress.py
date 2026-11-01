"""Model compression for the Segmentation Enhancement section (lecture: architecture selection
and compression, slides 16-24).

* unstructured magnitude pruning: zero the smallest-|w| convolution weights (global threshold).
  The tensors keep their shape, so the network is no faster on a GPU; the saved file only
  shrinks once it is compressed (we report gzip size to show this).
* structured pruning: remove whole convolution channels with the smallest L1 norm, using
  torch-pruning's dependency graph so skip connections and BatchNorm stay consistent. The model
  really gets smaller and faster, but needs fine-tuning afterwards (iterative pruning).
* knowledge distillation: a smaller student is trained on the ground truth and on the softened
  outputs of a teacher (here the ensemble of the three final seeds), per pixel and per channel.
"""
import gzip
import io
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import prune

from .unet import TARGET_CHANNELS


# ----------------------------------------------------------------------------- size

def count_params(model, nonzero=False):
    return sum(int((p != 0).sum()) if nonzero else p.numel() for p in model.parameters())


def stored_size_mb(model, half=True):
    """Size of the saved state_dict in MB, raw and gzip-compressed (fp16 if half). Zeros left by
    unstructured pruning only pay off in the compressed size."""
    sd = {k: (v.half() if half and v.is_floating_point() else v) for k, v in model.state_dict().items()}
    buf = io.BytesIO()
    torch.save(sd, buf)
    raw = buf.getvalue()
    return dict(file_MB=len(raw) / 2 ** 20, gzip_MB=len(gzip.compress(raw, 6)) / 2 ** 20)


def describe(model, example=None):
    "Parameters (total and non-zero), file sizes, and GFLOPs (MACs x 2) for one example input."
    d = dict(params_M=count_params(model) / 1e6, nonzero_params_M=count_params(model, True) / 1e6,
             **stored_size_mb(model))
    if example is not None:
        try:
            import torch_pruning as tp
            macs, _ = tp.utils.count_ops_and_params(model, example)
            d["GFLOPs"] = 2 * macs / 1e9
        except Exception as e:                      # counting is optional
            d["GFLOPs"] = float("nan")
            print("FLOP count failed:", e)
    return d


# ----------------------------------------------------------------------------- unstructured

def _prunable(model, skip=()):
    skip = set(id(m) for m in skip)
    return [(m, "weight") for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and id(m) not in skip]


def magnitude_prune(model, amount, skip=()):
    """Global unstructured L1 (magnitude) pruning of every conv weight except `skip` modules.
    The mask is folded into the weights (prune.remove), leaving plain zeros."""
    params = _prunable(model, skip)
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    for m, n in params:
        prune.remove(m, n)
    return model


def conv_sparsity(model):
    ws = [m.weight for m, _ in _prunable(model)]
    return float(sum(int((w == 0).sum()) for w in ws) / sum(w.numel() for w in ws))


# ----------------------------------------------------------------------------- structured

def head_modules(model):
    "Layers whose output channels must not be pruned: the final 1x1 conv producing the 5 maps."
    if hasattr(model, "net"):                       # SMPUNet
        return [model.net.segmentation_head]
    return [model.head]


def structured_prune(model, ratio, example, round_to=None):
    """Remove `ratio` of the channels of every prunable layer, ranked by the L1 norm of each
    channel group (lecture: magnitude criterion). Returns the same model object, now smaller."""
    import torch_pruning as tp
    model.eval()
    imp = tp.importance.GroupMagnitudeImportance(p=1)
    pruner = tp.pruner.MetaPruner(model, example, importance=imp, pruning_ratio=ratio,
                                  ignored_layers=head_modules(model), round_to=round_to)
    pruner.step()
    return model


def save_full(model, path):
    "Pruned models no longer match build_model's architecture, so save the whole module."
    torch.save(model, path)


def load_full(path, device):
    return torch.load(path, map_location=device, weights_only=False).eval()


# ----------------------------------------------------------------------------- distillation

class DistillLoss(nn.Module):
    """alpha * (the normal multi-task loss on the annotations)
       + (1 - alpha) * per-channel distillation towards the teacher ensemble.

    Binary channels: BCE between the student's and the teacher's sigmoids at temperature T,
    scaled by T^2 (Hinton et al., 2015; a sigmoid is the 2-class softmax). The `dist` channel is a
    regression target, so it is matched with MSE at T=1. Pixel weights (zero off the disk) and
    channel weights are the same as in the normal loss."""

    def __init__(self, base_loss, teachers, alpha=0.5, T=2.0):
        super().__init__()
        self.base, self.alpha, self.T = base_loss, alpha, T
        self.teachers = teachers
        for t in teachers:
            t.eval()
            for p in t.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def teacher_logits(self, x):
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.device.type == "cuda"):
            return sum(t(x).float() for t in self.teachers) / len(self.teachers)

    def forward(self, logits, batch):
        hard, parts = self.base(logits, batch)
        tl = self.teacher_logits(batch["image"])
        w = batch["weight"]
        wsum = w.sum().clamp_min(1.0)
        kd = logits.new_zeros((), dtype=torch.float32)
        for c, name in enumerate(TARGET_CHANNELS):
            lam = self.base.weights.get(name)
            if not lam:
                continue
            z, t = logits[:, c:c + 1].float(), tl[:, c:c + 1]
            if name == "dist":
                term = (w * (torch.sigmoid(z) - torch.sigmoid(t)) ** 2).sum() / wsum
            else:
                soft = torch.sigmoid(t / self.T)
                term = (w * F.binary_cross_entropy_with_logits(z / self.T, soft, reduction="none")).sum() / wsum
                term = term * self.T ** 2
            kd = kd + lam * term
        parts = {**parts, "kd": kd.detach()}
        return self.alpha * hard + (1 - self.alpha) * kd, parts


# ----------------------------------------------------------------------------- bookkeeping

def write_json(path, d):
    Path(path).write_text(json.dumps(d, indent=1, default=float))


def timed(fn, *a, **kw):
    t0 = time.time()
    out = fn(*a, **kw)
    return out, time.time() - t0
