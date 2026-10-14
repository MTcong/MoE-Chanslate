"""Distill dense teacher -> upcycled MoE student from cached top-k teacher log-probs.

  python -m moe_chanslate.train.distill --student <moe_dir> --cache data/cache --out checkpoints/distill \
      --train mlp --lr 2e-5 --batch-size 2 --grad-accum 16 --epochs 1 [--optim adamw8bit]
"""
import argparse
import glob
import math
import random
from pathlib import Path

import torch

from moe_chanslate.train.losses import distill_loss


def load_cache(path):
    for f in sorted(glob.glob(f"{path}/shard_*.pt")):
        yield torch.load(f)


def batches(path, batch_size, seed=0, shuffle=True):
    rng = random.Random(seed)
    for shard in load_cache(path):
        n = shard["ids"].shape[0]
        order = list(range(n))
        if shuffle:
            rng.shuffle(order)
        for s in range(0, n - batch_size + 1, batch_size):
            sel = order[s:s + batch_size]
            yield {k: v[sel] for k, v in shard.items()}


def select_trainable(model, mode):
    for n, p in model.named_parameters():
        p.requires_grad = mode == "all" or ".mlp." in n       # experts, router, shared expert
    return [p for p in model.parameters() if p.requires_grad]


def make_optimizer(params, lr, kind):
    if kind == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=0.0)
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1))))


def train(model, cache, *, lr=2e-5, batch_size=2, grad_accum=16, epochs=1, warmup=50, mode="mlp",
          optim="adamw", ce_weight=0.1, aux_weight=1.0, max_steps=None, log_every=10,
          device=None, grad_ckpt=True, on_save=None, save_every=500):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    params = select_trainable(model, mode)
    opt = make_optimizer(params, lr, optim)
    n_batches = sum(s["ids"].shape[0] // batch_size for s in load_cache(cache)) * epochs
    total = max_steps or max(n_batches // grad_accum, 1)
    step, micro, hist = 0, 0, []
    for _ in range(epochs):
        for b in batches(cache, batch_size):
            ids = b["ids"].long().to(device)
            out = model(input_ids=ids, output_router_logits=True)
            labels = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -100)], dim=1)
            loss = distill_loss(out.logits, b["idx"].to(device), b["lp"].to(device), labels, ce_weight)
            aux = getattr(out, "aux_loss", None)
            if aux is not None:
                loss = loss + aux_weight * aux.to(loss.dtype)
            (loss / grad_accum).backward()
            micro += 1
            hist.append(loss.item())
            if micro % grad_accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, lr, warmup)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % log_every == 0:
                print(f"step {step}/{total} loss {sum(hist[-grad_accum * log_every:]) / len(hist[-grad_accum * log_every:]):.4f}", flush=True)
            if on_save and step % save_every == 0:
                on_save(step)
            if step >= total:
                return hist
    return hist


def main():
    from transformers import AutoModelForCausalLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", choices=["mlp", "all"], default="mlp")
    ap.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int)
    a = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16)
    out = Path(a.out)

    def save(step):
        model.save_pretrained(out / f"step{step}")

    train(model, a.cache, lr=a.lr, batch_size=a.batch_size, grad_accum=a.grad_accum, epochs=a.epochs,
          mode=a.train, optim=a.optim, max_steps=a.max_steps, on_save=save)
    model.save_pretrained(out / "final")


if __name__ == "__main__":
    main()
