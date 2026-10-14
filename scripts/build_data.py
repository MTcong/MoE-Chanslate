"""Download + clean zh-vi parallel data -> data/processed/{train,valid}.jsonl

  python scripts/build_data.py --config configs/data_sources.yaml [--check] [--valid 5000]
"""
import argparse
import json
import random
from pathlib import Path

import yaml

from moe_chanslate.data.clean import clean_pairs


def iter_source(src):
    if src["kind"] == "jsonl":
        with open(src["path"], encoding="utf8") as f:
            for line in f:
                r = json.loads(line)
                yield r["zh"], r["vi"]
        return
    from datasets import load_dataset
    ds = load_dataset(src["path"], split=src.get("split", "train"), streaming=True,
                      **src.get("name_args", {}))
    for i, row in enumerate(ds):
        if i >= src.get("max_rows", 10**12):
            break
        t = row[src["translation_key"]] if "translation_key" in src else row
        yield t[src["zh_key"]], t[src["vi_key"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_sources.yaml")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--valid", type=int, default=5000)
    ap.add_argument("--check", action="store_true", help="only load 5 rows per source")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for src in cfg["parallel"]:
        if src["kind"] == "jsonl" and not Path(src["path"]).exists():
            print(f"[skip] {src['name']}: {src['path']} not found")
            continue
        it = iter_source(src)
        if a.check:
            for _, pair in zip(range(5), it):
                print(src["name"], pair)
            continue
        n = 0
        for zh, vi in clean_pairs(it):
            rows.append({"zh": zh, "vi": vi, "src": src["name"]})
            n += 1
        print(f"[ok] {src['name']}: kept {n}")
    if a.check:
        return
    random.Random(0).shuffle(rows)
    valid, train = rows[:a.valid], rows[a.valid:]
    for name, data in (("valid", valid), ("train", train)):
        with open(out / f"{name}.jsonl", "w", encoding="utf8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train={len(train)} valid={len(valid)} -> {out}")


if __name__ == "__main__":
    main()
