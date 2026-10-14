"""Cache the dense teacher's top-k log-probs for packed sequences.

Student ids are the (possibly vocab-trimmed) ids. The teacher is fed new2old[ids] and its
logits are restricted to the kept vocabulary (new-id order) and renormalised, so
teacher/student positions and vocab line up exactly.

  python -m moe_chanslate.train.cache_teacher --teacher Qwen/Qwen3.5-9B --tokenizer <trimmed_dir> \
      --data data/processed/train.jsonl --out data/cache --seq-len 1024 --k 16 --max-tokens 3e8
"""
import argparse
import json
from pathlib import Path

import torch

from moe_chanslate.data.prompts import to_messages


def pack(token_iter, seq_len):
    buf = []
    for ids in token_iter:
        buf.extend(ids)
        while len(buf) >= seq_len:
            yield buf[:seq_len]
            buf = buf[seq_len:]


@torch.no_grad()
def teacher_topk(model, ids, new2old, k):
    """ids [B,T] student ids -> (idx [B,T,k] student ids, logp [B,T,k] renormalised)."""
    t_ids = new2old[ids]
    logits = model(input_ids=t_ids).logits.float()[..., new2old]       # [B,T,Vnew_real]
    logp = torch.log_softmax(logits, dim=-1)
    top = logp.topk(k, dim=-1)
    return top.indices.int(), top.values.half()


def example_ids(tok, row):
    if "zh" in row and "vi" in row:
        text = tok.apply_chat_template(to_messages(row["zh"], row["vi"]), tokenize=False)
    else:
        text = row["text"]
    return tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--new2old", help="new2old.json (default: <tokenizer>/new2old.json or identity)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-tokens", type=float, default=3e8)
    ap.add_argument("--shard-seqs", type=int, default=2048)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.bfloat16, device_map="auto").eval()
    p = Path(a.new2old or Path(a.tokenizer) / "new2old.json")
    new2old = torch.tensor(json.loads(p.read_text()) if p.exists() else list(range(len(tok))),
                           device=model.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = (json.loads(l) for l in open(a.data, encoding="utf8"))
    seqs = pack((example_ids(tok, r) for r in rows), a.seq_len)
    shard, n_tok, n_shard = {"ids": [], "idx": [], "lp": []}, 0, 0
    batch = []

    def run(batch):
        ids = torch.tensor(batch, device=model.device)
        idx, lp = teacher_topk(model, ids, new2old, a.k)
        shard["ids"].append(ids.int().cpu()); shard["idx"].append(idx.cpu()); shard["lp"].append(lp.cpu())

    def flush():
        nonlocal shard, n_shard
        if shard["ids"]:
            torch.save({k: torch.cat(v) for k, v in shard.items()}, out / f"shard_{n_shard:05d}.pt")
            n_shard += 1
            shard = {"ids": [], "idx": [], "lp": []}

    for s in seqs:
        batch.append(s)
        if len(batch) == a.batch_size:
            run(batch); batch = []
            n_tok += a.batch_size * a.seq_len
            if sum(len(x) for x in shard["ids"]) * a.batch_size >= a.shard_seqs:
                flush()
            if n_tok >= a.max_tokens:
                break
    flush()
    print(f"cached {n_tok} tokens in {n_shard} shards -> {out}")


if __name__ == "__main__":
    main()
