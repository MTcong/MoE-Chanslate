"""Vocabulary trimming for byte-level BPE tokenizers (Qwen family).

Keeps: all 256 byte-level alphabet tokens, special/added tokens, the most frequent
tokens of a zh/vi/en corpus, plus the merge-derivation closure of those tokens so
every kept token is still reachable. Embedding / lm_head rows are sliced with the
resulting old->new id map and every token id in the config files is remapped.

  python -m moe_chanslate.vocab.trim --model <dir> --corpus data/processed/train.jsonl \
      --out <out_dir> --size 64000
"""
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROW_KEYS = re.compile(r"(^|\.)(embed_tokens|lm_head)\.weight$")


def bytes_alphabet() -> set[str]:
    """GPT-2 byte->unicode alphabet used by byte-level BPE."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c) for c in cs}


def _pair(m):
    return tuple(m.split(" ")) if isinstance(m, str) else tuple(m)


def count_tokens(tok, texts) -> Counter:
    c = Counter()
    for t in texts:
        c.update(tok.encode(t).ids)
    return c


def select_tokens(tj: dict, freq: Counter, size: int) -> list[str]:
    """Return kept *model-vocab* token strings (specials handled separately)."""
    vocab = tj["model"]["vocab"]
    id2tok = {i: t for t, i in vocab.items()}
    merges = [_pair(m) for m in tj["model"]["merges"]]
    made_by = {}                                  # token -> (a, b), earliest merge wins
    for a, b in merges:
        made_by.setdefault(a + b, (a, b))
    must = {t for t in vocab if t in bytes_alphabet()}
    n_special = len(tj.get("added_tokens", []))
    budget = size - n_special

    def closure(seed):
        keep, stack = set(must), list(seed)
        while stack:
            t = stack.pop()
            if t in keep:
                continue
            keep.add(t)
            if t in made_by:
                stack.extend(made_by[t])
        return keep

    ranked = [id2tok[i] for i, _ in freq.most_common() if i in id2tok]
    lo, hi = 0, len(ranked)
    while lo < hi:                                # largest prefix whose closure fits the budget
        mid = (lo + hi + 1) // 2
        if len(closure(ranked[:mid])) <= budget:
            lo = mid
        else:
            hi = mid - 1
    keep = closure(ranked[:lo])
    # fill remaining slots with next most-frequent tokens (closure-safe, one by one)
    for t in ranked[lo:]:
        if len(keep) >= budget:
            break
        extra = closure(list(keep) + [t])
        if len(extra) <= budget:
            keep = extra
    return sorted(keep, key=vocab.__getitem__)


def trim_tokenizer(tj: dict, keep: list[str]):
    """Return (new_tokenizer_json, old2new). Order: kept model tokens, then added tokens."""
    tj = json.loads(json.dumps(tj))
    old_vocab = tj["model"]["vocab"]
    kept = set(keep)
    old2new, new_vocab = {}, {}
    for t in keep:
        new_vocab[t] = len(new_vocab)
        old2new[old_vocab[t]] = new_vocab[t]
    as_str = isinstance(tj["model"]["merges"][0], str) if tj["model"]["merges"] else True
    merges = []
    for m in tj["model"]["merges"]:
        a, b = _pair(m)
        if a in kept and b in kept and a + b in kept:
            merges.append(f"{a} {b}" if as_str else [a, b])
    tj["model"]["vocab"], tj["model"]["merges"] = new_vocab, merges
    nxt = len(new_vocab)
    for at in tj.get("added_tokens", []):
        old2new[at["id"]] = nxt
        at["id"] = nxt
        nxt += 1
    return tj, old2new


_ID_KEYS = {"eos_token_id", "bos_token_id", "pad_token_id", "image_token_id", "video_token_id",
            "vision_start_token_id", "vision_end_token_id"}


def remap_ids(obj, old2new):
    """Remap *_token_id fields (int or list) anywhere in a json structure."""
    if isinstance(obj, dict):
        return {k: (_map(v, old2new) if k in _ID_KEYS else remap_ids(v, old2new))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [remap_ids(v, old2new) for v in obj]
    return obj


def _map(v, old2new):
    if isinstance(v, list):
        return [old2new.get(x, x) for x in v]
    return old2new.get(v, v) if isinstance(v, int) else v


def trim_rows(w: torch.Tensor, old2new: dict, new_size: int) -> torch.Tensor:
    out = w.mean(0, keepdim=True).repeat(new_size, 1).to(w.dtype)    # padding rows = mean
    src = torch.tensor(sorted(old2new, key=old2new.get))
    dst = torch.tensor([old2new[i] for i in src.tolist()])
    out[dst] = w[src]
    return out


def main():
    from tokenizers import Tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True, help="jsonl with zh/vi fields (and optional 'text')")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=64000)
    ap.add_argument("--max-lines", type=int, default=2_000_000)
    a = ap.parse_args()
    src, dst = Path(a.model), Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)

    tj = json.loads((src / "tokenizer.json").read_text())
    tok = Tokenizer.from_str(json.dumps(tj))

    def texts():
        with open(a.corpus, encoding="utf8") as f:
            for i, line in enumerate(f):
                if i >= a.max_lines:
                    return
                r = json.loads(line)
                for k in ("zh", "vi", "en", "text"):
                    if r.get(k):
                        yield r[k]

    keep = select_tokens(tj, count_tokens(tok, texts()), a.size)
    new_tj, old2new = trim_tokenizer(tj, keep)
    n = len(new_tj["model"]["vocab"]) + len(new_tj.get("added_tokens", []))
    new_size = (n + 127) // 128 * 128
    print(f"kept {n} tokens -> vocab_size {new_size}")

    (dst / "tokenizer.json").write_text(json.dumps(new_tj, ensure_ascii=False))
    for f in src.iterdir():
        if f.suffix in {".json", ".txt", ".jinja", ".model"} and f.name not in {
                "tokenizer.json", "model.safetensors.index.json"}:
            data = json.loads(f.read_text()) if f.suffix == ".json" else None
            if data is None:
                shutil.copy(f, dst / f.name)
                continue
            data = remap_ids(data, old2new)
            if f.name == "tokenizer_config.json" and "added_tokens_decoder" in data:
                data["added_tokens_decoder"] = {str(old2new[int(k)]): v
                                                for k, v in data["added_tokens_decoder"].items()
                                                if int(k) in old2new}
            if f.name == "config.json":
                (data.get("text_config") or data)["vocab_size"] = new_size
            (dst / f.name).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    wmap = {}
    for sf in sorted(src.glob("*.safetensors")):
        with safe_open(sf, "pt") as h:
            tensors = {k: h.get_tensor(k) for k in h.keys()}
        for k in list(tensors):
            if ROW_KEYS.search(k):
                tensors[k] = trim_rows(tensors[k], old2new, new_size)
        save_file(tensors, dst / sf.name, metadata={"format": "pt"})
        wmap.update({k: sf.name for k in tensors})
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": wmap}, indent=2))


if __name__ == "__main__":
    main()
