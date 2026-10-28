"""Multi-task loss over the five U-Net output channels (see unet.TARGET_CHANNELS).

Binary channels (fg, boundary, contact, spine) use pixel-weighted BCE + soft Dice; `dist`
(per-instance normalised distance transform) uses pixel-weighted MSE on the sigmoid output.
The pixel weight from FilamentDS is 0 off the disk and ramps down towards the limb.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .unet import TARGET_CHANNELS


def soft_dice_loss(prob, target, weight, eps=1.0):
    "Batch-level soft Dice; eps keeps crops with no positives from dominating."
    inter = (weight * prob * target).sum()
    denom = (weight * (prob + target)).sum()
    return 1 - (2 * inter + eps) / (denom + eps)


class MultiTaskLoss(nn.Module):
    """weights: {channel: lambda}; channels with lambda 0 or missing are not trained.
    pos_weight: {channel: BCE positive weight} for the rare thin channels.
    soft_label_alpha: mix the annotator's fg mask with the multi-annotator consensus."""

    def __init__(self, weights, pos_weight=None, dice=True, soft_label_alpha=0.0):
        super().__init__()
        self.weights = {k: float(v) for k, v in weights.items() if v}
        self.pos_weight = pos_weight or {}
        self.dice, self.alpha = dice, soft_label_alpha

    def forward(self, logits, batch):
        w = batch["weight"]
        wsum = w.sum().clamp_min(1.0)
        parts = {}
        for c, name in enumerate(TARGET_CHANNELS):
            if name not in self.weights:
                continue
            z = logits[:, c:c + 1].float()
            t = batch["target"][:, c:c + 1]
            if name == "fg" and self.alpha > 0:
                t = (1 - self.alpha) * t + self.alpha * batch["soft_fg"]
            p = torch.sigmoid(z)
            if name == "dist":
                parts[name] = (w * (p - t) ** 2).sum() / wsum
                continue
            pw = torch.tensor(self.pos_weight.get(name, 1.0), device=z.device)
            bce = F.binary_cross_entropy_with_logits(z, t, pos_weight=pw, reduction="none")
            parts[name] = (w * bce).sum() / wsum + (soft_dice_loss(p, t, w) if self.dice else 0)
        total = sum(self.weights[k] * v for k, v in parts.items())
        return total, {k: v.detach() for k, v in parts.items()}
