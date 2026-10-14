"""Evaluate zh->vi translation: chrF++, BLEU (sacrebleu) and optional COMET.

  python -m moe_chanslate.eval.translate_eval --model Qwen/Qwen3.5-9B \
      --data flores | path.jsonl  [--limit 200] [--comet Unbabel/wmt22-comet-da] --out runs/base.json

`--data flores` loads FLORES-200 devtest from the HF dataset in configs/data_sources.yaml
(`eval` entry, unverified id); any jsonl with {"zh","vi"} per line also works.
"""
import argparse
import json
from pathlib import Path

from moe_chanslate.data.prompts import to_messages


def score(hyps: list[str], refs: list[str]) -> dict:
    import sacrebleu
    assert len(hyps) == len(refs)
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)       # chrF++
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return {"chrf++": round(chrf.score, 2), "bleu": round(bleu.score, 2), "n": len(hyps)}


def load_pairs(spec: str, limit: int | None):
    if spec == "flores":
        import yaml
        from datasets import load_dataset
        src = yaml.safe_load(open("configs/data_sources.yaml"))["eval"][0]
        ds = load_dataset(src["path"], split=src["split"], **src.get("name_args", {}))
        cols = ds.column_names
        zh = next(c for c in cols if "zho" in c and "Hans" in c or c in ("zh", "sentence_zho_Hans"))
        vi = next(c for c in cols if "vie" in c or c in ("vi",))
        pairs = [(r[zh], r[vi]) for r in ds]
    else:
        pairs = []
        for line in open(spec, encoding="utf8"):
            r = json.loads(line)
            pairs.append((r["zh"], r["vi"]))
    return pairs[:limit] if limit else pairs


def translate(model, tok, srcs, batch_size=8, max_new_tokens=512):
    import torch
    tok.padding_side = "left"
    outs = []
    order = sorted(range(len(srcs)), key=lambda i: len(srcs[i]))     # length-bucket for speed
    res = {}
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        prompts = [tok.apply_chat_template(to_messages(srcs[i]), tokenize=False,
                                           add_generation_prompt=True) for i in idx]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        for i, g in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
            res[i] = tok.decode(g, skip_special_tokens=True).strip()
    return [res[i] for i in range(len(srcs))]


def comet_score(srcs, hyps, refs, name):
    from comet import download_model, load_from_checkpoint
    m = load_from_checkpoint(download_model(name))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    return round(m.predict(data, batch_size=32).system_score, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="flores")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--comet")
    ap.add_argument("--out", default="runs/eval.json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto").eval()
    pairs = load_pairs(a.data, a.limit)
    srcs, refs = [p[0] for p in pairs], [p[1] for p in pairs]
    hyps = translate(model, tok, srcs, a.batch_size)
    res = score(hyps, refs)
    if a.comet:
        res["comet"] = comet_score(srcs, hyps, refs, a.comet)
    print(res)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": a.model, "metrics": res,
                               "samples": [{"zh": s, "hyp": h, "ref": r}
                                           for s, h, r in zip(srcs[:50], hyps, refs)]},
                              ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
