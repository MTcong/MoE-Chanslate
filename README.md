# MoE-Chanslate

Upcycle **Qwen3.5-9B** (dense) into a sparse MoE (~8–9B total params, **A1B/A2B** active) specialised for **zh → vi** translation.

## Roadmap (built step by step)
1. [x] Project scaffold + plan (this commit)
2. [ ] Data: download, clean, filter zh–vi parallel + mono data (`scripts/`, `src/moe_chanslate/data`)
3. [ ] Inspect base config + granular upcycling script (`src/moe_chanslate/upcycle`)
4. [ ] Baseline eval of dense 9B on FLORES-200 / WMT (chrF++, COMET)
5. [ ] Distillation training (teacher = dense 9B) on Colab A100
6. [ ] SFT translation, optional preference stage
7. [ ] Final eval, export, model card

## Design notes
- **Granular upcycling**: each dense FFN is split into 64 slices; router picks top-8 (A1B) or top-16 (A2B). Total params stay ≈ base (~9B, can prune/trim later); active params drop to ~1/8–1/4 of FFN. This is lossy → recovery via **distillation from the dense 9B teacher**, then SFT.
- Compute split: Colab A100 for conversion/training; local RTX 4060 Ti 16GB for data prep, eval and tiny-config debugging only (cannot train 9B).
- Architecture-specific code reads `config.json` of the base model at runtime; Qwen3.5 layout (e.g. hybrid linear attention) is verified in step 3.
