"""Parameter / active-parameter budget for granular upcycling of a dense Qwen3.5 text model.

Usage: python -m moe_chanslate.budget configs/qwen3_5_9b_config.json
"""
import json
import sys


def budget(tc: dict, num_experts=63, top_k=8, shared=192, vocab=None) -> dict:
    h, L, I = tc["hidden_size"], tc["num_hidden_layers"], tc["intermediate_size"]
    V = vocab or tc["vocab_size"]
    types = tc["layer_types"]
    n_full, n_lin = types.count("full_attention"), types.count("linear_attention")
    hd, nh, nkv = tc["head_dim"], tc["num_attention_heads"], tc["num_key_value_heads"]
    full = h * nh * hd * 2 + 2 * h * nkv * hd + nh * hd * h            # q(+gate), k, v, o
    kd = tc["linear_num_key_heads"] * tc["linear_key_head_dim"]
    vd = tc["linear_num_value_heads"] * tc["linear_value_head_dim"]
    lin = h * (2 * kd + vd) + h * vd + vd * h + 2 * h * tc["linear_num_value_heads"]
    attn = n_full * full + n_lin * lin
    emb = V * h
    routed = I - shared
    assert routed % num_experts == 0, f"{routed} not divisible by {num_experts}"
    e_dim = routed // num_experts
    ffn_total = L * 3 * h * I
    ffn_active = L * 3 * h * (shared + top_k * e_dim)
    total = attn + 2 * emb + ffn_total
    active = attn + emb + ffn_active          # embedding lookup is free, lm_head counted once
    return dict(expert_dim=e_dim, total_B=total / 1e9, active_B=active / 1e9,
                attn_B=attn / 1e9, lm_head_B=emb / 1e9,
                ffn_active_frac=(shared + top_k * e_dim) / I)


if __name__ == "__main__":
    tc = json.load(open(sys.argv[1]))["text_config"]
    print(f"{'config':<30}{'vocab':>8}{'total':>8}{'active':>8}{'ffn%':>7}")
    for vocab in (None, 64000):
        for E, shared, k in ((63, 192, 2), (63, 192, 4), (63, 192, 8), (56, 1536, 8)):
            r = budget(tc, num_experts=E, top_k=k, shared=shared, vocab=vocab)
            print(f"E{E} top{k:<2} shared{shared:<5}{'':<9}{vocab or tc['vocab_size']:>8}"
                  f"{r['total_B']:>7.2f}B{r['active_B']:>7.2f}B{r['ffn_active_frac']*100:>6.0f}%")
    r = budget(tc)
    print(f"\nfloor (non-FFN active): attn {r['attn_B']:.2f}B + lm_head {r['lm_head_B']:.2f}B (full vocab)")
