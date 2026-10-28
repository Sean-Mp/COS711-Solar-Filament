"""Training loop: AdamW, linear warmup + cosine decay, fp16 autocast, gradient clipping,
resumable checkpoints, and model selection by validation PQ."""
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine(total_steps, warmup_steps, min_lr_frac=0.01):
    "LR multiplier: linear warmup, then cosine decay to min_lr_frac of the peak."
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
        return min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))
    return f


def fit(model, train_dl, loss_fn, cfg, run_dir, val_fn=None, log=print):
    """Train for cfg["epochs"] epochs, resuming from run_dir/last.pt if it exists.

    val_fn(model) -> dict with at least "PQ"; it runs every cfg["val_every"] epochs and on the
    last epoch, and the best-PQ weights are saved to run_dir/best.pt. Returns the history table.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = next(model.parameters()).device
    amp = cfg["amp"] and dev.type == "cuda"
    steps_per_epoch = len(train_dl)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine(cfg["epochs"] * steps_per_epoch, cfg["warmup_epochs"] * steps_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start, best, history = 0, -1.0, []
    ckpt = run_dir / "last.pt"
    if ckpt.exists():
        s = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); scaler.load_state_dict(s["scaler"])
        start, best, history = s["epoch"] + 1, s["best"], s["history"]
        log(f"resumed after epoch {start}")

    for epoch in range(start, cfg["epochs"]):
        model.train()
        t0, sums = time.time(), {}
        for batch in train_dl:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                logits = model(batch["image"])
            loss, parts = loss_fn(logits, batch)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in {"loss": loss.detach(), **parts}.items():
                sums[k] = sums.get(k, 0.0) + float(v)

        row = dict(epoch=epoch + 1, lr=sched.get_last_lr()[0], train_s=time.time() - t0,
                   **{f"train_{k}": v / steps_per_epoch for k, v in sums.items()})

        last = epoch + 1 == cfg["epochs"]
        if val_fn is not None and ((epoch + 1) % cfg["val_every"] == 0 or last):
            t1 = time.time()
            metrics = val_fn(model)
            row.update({f"val_{k}": v for k, v in metrics.items()}, val_s=time.time() - t1)
            if metrics["PQ"] > best:
                best = metrics["PQ"]
                torch.save(model.state_dict(), run_dir / "best.pt")
        elif val_fn is None and last:
            torch.save(model.state_dict(), run_dir / "best.pt")

        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=epoch, best=best, history=history), ckpt)
        log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

    return pd.DataFrame(history)
