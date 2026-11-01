"""U-Net models for multi-task filament segmentation.

Both models map a (B, 1, H, W) normalised disk image to (B, 5, H, W) logits, one channel per
target built in notebook 01 (same order as `TARGET_CHANNELS`). H and W must be divisible by 32.
"""
import math

import torch
from torch import nn

TARGET_CHANNELS = ("fg", "boundary", "contact", "dist", "spine")


class ConvBlock(nn.Sequential):
    "(conv3x3 -> BN -> ReLU) x 2, the standard U-Net stage."

    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """U-Net (Ronneberger et al., 2015) trained from scratch, with BatchNorm.

    `depth` resolution levels with widths base * 2**i (capped at max_ch). depth=6 gives five
    poolings (stride 32), the same as a ResNet encoder, so a 512 crop has a 16x16 bottleneck.
    """

    def __init__(self, in_ch=1, out_ch=len(TARGET_CHANNELS), base=32, depth=6, max_ch=512,
                 in_mean=0.5, in_std=0.3):
        super().__init__()
        chs = [min(base * 2 ** i, max_ch) for i in range(depth)]
        self.enc = nn.ModuleList(ConvBlock(cin, c) for cin, c in zip([in_ch] + chs[:-1], chs))
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2)
                                for i in reversed(range(depth - 1)))
        self.dec = nn.ModuleList(ConvBlock(2 * chs[i], chs[i]) for i in reversed(range(depth - 1)))
        self.head = nn.Conv2d(chs[0], out_ch, 1)
        self.register_buffer("in_mean", torch.tensor(in_mean))
        self.register_buffer("in_std", torch.tensor(in_std))
        init_head_bias(self.head)

    def forward(self, x):
        x = (x - self.in_mean) / self.in_std
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.head(x)


class SMPUNet(nn.Module):
    "U-Net with an ImageNet-pretrained encoder from segmentation_models_pytorch."

    def __init__(self, encoder="resnet34", encoder_weights="imagenet", in_ch=1,
                 out_ch=len(TARGET_CHANNELS), in_mean=0.5, in_std=0.3):
        super().__init__()
        import segmentation_models_pytorch as smp
        # smp sums the RGB filters of the first conv when in_channels=1
        self.net = smp.Unet(encoder, encoder_weights=encoder_weights, in_channels=in_ch, classes=out_ch)
        self.register_buffer("in_mean", torch.tensor(in_mean))
        self.register_buffer("in_std", torch.tensor(in_std))
        init_head_bias(self.net.segmentation_head[0])

    def forward(self, x):
        return self.net((x - self.in_mean) / self.in_std)


def init_head_bias(conv, prior=0.01):
    "Start every output at p=prior (Lin et al., 2017): targets are ~1% positive, so early BCE is stable."
    nn.init.constant_(conv.bias, -math.log((1 - prior) / prior))


def build_model(cfg):
    if cfg["model"] == "unet":
        return UNet(base=cfg["base_ch"], depth=cfg["depth"], in_mean=cfg["in_mean"], in_std=cfg["in_std"])
    if cfg["model"] == "smp_unet":
        return SMPUNet(cfg["encoder"], cfg["encoder_weights"], in_mean=cfg["in_mean"], in_std=cfg["in_std"])
    raise ValueError(f"unknown model {cfg['model']!r}")


def model_size(model):
    "Parameter count and in-memory size of the weights (fp32 and fp16)."
    n = sum(p.numel() for p in model.parameters())
    nbytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    return dict(params_M=n / 1e6, size_MB_fp32=nbytes / 2 ** 20, size_MB_fp16=nbytes / 2 ** 21)
