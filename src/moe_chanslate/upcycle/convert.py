"""Granular upcycling: dense Qwen3.5 -> Qwen3_5Moe (HF `qwen3_5_moe`).

Each dense FFN (intermediate I) is partitioned along the intermediate dim into
slices of size `expert_dim`: `shared_slices` of them become the shared expert,
the remaining E = I/expert_dim - shared_slices become routed experts (no copies,
so total params ~= dense). Everything else is copied unchanged.

Output scale: routed top-k weights are renormalised to sum 1, so with a uniform
router the routed output is (1/E)*sum(all slices) in expectation -> routed
down_proj is multiplied by E. The shared expert is gated by sigmoid(0)=0.5 at
init -> shared down_proj is multiplied by 2. With top_k == E and router=0 the
converted model is numerically identical to the dense one (see tests).

Usage:
  python -m moe_chanslate.upcycle.convert --src <dense_dir> --dst <out_dir> \
      [--top-k 8] [--expert-dim 192] [--shared-slices 1]
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MLP_RE = re.compile(r"^(.*\.layers\.(\d+)\.mlp)\.(gate|up|down)_proj\.weight$")


def moe_config(dense_cfg: dict, top_k: int, expert_dim: int, shared_slices: int) -> dict:
    cfg = json.loads(json.dumps(dense_cfg))
    tc = cfg["text_config"]
    I = tc["intermediate_size"]
    assert I % expert_dim == 0, f"intermediate {I} % expert_dim {expert_dim} != 0"
    n_experts = I // expert_dim - shared_slices
    assert 1 <= top_k <= n_experts
    cfg["model_type"] = "qwen3_5_moe"
    cfg["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    tc["model_type"] = "qwen3_5_moe_text"
    tc.pop("intermediate_size")
    tc.update(num_experts=n_experts, num_experts_per_tok=top_k, moe_intermediate_size=expert_dim,
              shared_expert_intermediate_size=shared_slices * expert_dim,
              output_router_logits=False, router_aux_loss_coef=0.001)
    cfg["vision_config"]["model_type"] = "qwen3_5_moe"
    return cfg


def convert_layer(g, u, d, *, expert_dim, shared_slices, router_std, gen):
    """g,u: [I,H]  d: [H,I] -> dict of new tensors for one layer (dtype preserved)."""
    I, H = g.shape
    n_exp = I // expert_dim - shared_slices
    s = shared_slices * expert_dim            # first s neurons -> shared expert
    r = lambda t: t[s:]                        # routed neurons
    gr = r(g).reshape(n_exp, expert_dim, H)
    ur = r(u).reshape(n_exp, expert_dim, H)
    dr = d[:, s:].reshape(H, n_exp, expert_dim).permute(1, 0, 2)   # [E,H,e]
    out = {
        "experts.gate_up_proj": torch.cat([gr, ur], dim=1).contiguous(),     # [E,2e,H]
        "experts.down_proj": (dr * n_exp).to(d.dtype).contiguous(),          # [E,H,e]
        "gate.weight": (torch.randn(n_exp, H, generator=gen) * router_std).to(g.dtype),
        "shared_expert.gate_proj.weight": g[:s].contiguous(),
        "shared_expert.up_proj.weight": u[:s].contiguous(),
        "shared_expert.down_proj.weight": (d[:, :s] * 2).to(d.dtype).contiguous(),
        "shared_expert_gate.weight": torch.zeros(1, H, dtype=g.dtype),
    }
    return out


def convert_state(read, keys, *, expert_dim, shared_slices, router_std=0.0, seed=0):
    """Yield (new_key, tensor). `read(key)` returns a tensor."""
    gen = torch.Generator().manual_seed(seed)
    done = set()
    for k in keys:
        m = MLP_RE.match(k)
        if not m:
            yield k, read(k)
            continue
        base, _, _ = m.groups()
        if base in done:
            continue
        done.add(base)
        new = convert_layer(read(f"{base}.gate_proj.weight"), read(f"{base}.up_proj.weight"),
                            read(f"{base}.down_proj.weight"), expert_dim=expert_dim,
                            shared_slices=shared_slices, router_std=router_std, gen=gen)
        for nk, t in new.items():
            yield f"{base}.{nk}", t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--expert-dim", type=int, default=192)
    ap.add_argument("--shared-slices", type=int, default=1)
    ap.add_argument("--router-std", type=float, default=0.02)
    ap.add_argument("--shard-gb", type=float, default=4.0)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = moe_config(json.loads((src / "config.json").read_text()), a.top_k, a.expert_dim, a.shared_slices)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in src.iterdir():   # tokenizer / generation files etc.
        if f.suffix in {".json", ".txt", ".jinja", ".model", ".tiktoken"} and f.name not in {
                "config.json", "model.safetensors.index.json"}:
            shutil.copy(f, dst / f.name)

    files = sorted(src.glob("*.safetensors"))
    handles = {f: safe_open(f, "pt") for f in files}
    where = {k: f for f, h in handles.items() for k in h.keys()}
    read = lambda k: handles[where[k]].get_tensor(k)

    shard, size, idx, wmap = {}, 0, 0, {}
    def flush():
        nonlocal shard, size, idx
        if not shard:
            return
        name = f"model-{idx:05d}.safetensors"
        save_file(shard, dst / name, metadata={"format": "pt"})
        wmap.update({k: name for k in shard})
        shard, size, idx = {}, 0, idx + 1

    for k, t in convert_state(read, list(where), expert_dim=a.expert_dim,
                              shared_slices=a.shared_slices, router_std=a.router_std):
        shard[k] = t
        size += t.numel() * t.element_size()
        if size >= a.shard_gb * 2**30:
            flush()
    flush()
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": wmap}, indent=2))
    print(f"wrote {idx} shards to {dst}")


if __name__ == "__main__":
    main()
