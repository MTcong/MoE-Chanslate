# MoE-Chanslate

Upcycle **Qwen3.5-9B** (dense) into a sparse MoE (~8–9B total params, **A1B/A2B** active) specialised for **zh → vi** translation.

## Roadmap (built step by step)
1. [x] Project scaffold + plan (this commit)
2. [x] Data cleaning + `scripts/build_data.py` (source ids in `configs/data_sources.yaml` still unverified)
3. [x] Granular upcycling to HF `qwen3_5_moe` (`upcycle/convert.py`, equivalence test passes)
   - [x] vocab trim (`vocab/trim.py`, tested on a synthetic BPE only; run on the real Qwen3.5 tokenizer in Colab)
4. [ ] Baseline eval of dense 9B on FLORES-200 / WMT (chrF++, COMET)
5. [ ] Distillation training (teacher = dense 9B) on Colab A100
6. [ ] SFT translation, optional preference stage
7. [ ] Final eval, export, model card

## Design notes
- **Granular upcycling**: each dense FFN (12288) is split into 64 slices of 192: 1 shared + 63 routed experts, router picks top-4/top-8. Attention + lm_head alone are ~3B active, so realistic target is **~7.4B total / ~A2.7–3B active** (with vocab trim), not A1B/A2B. See `python -m moe_chanslate.budget`. This is lossy → recovery via **distillation from the dense 9B teacher**, then SFT.
- Compute split: Colab A100 for conversion/training; local RTX 4060 Ti 16GB for data prep, eval and tiny-config debugging only (cannot train 9B).
- Architecture-specific code reads `config.json` of the base model at runtime; Qwen3.5 layout (e.g. hybrid linear attention) is verified in step 3.
