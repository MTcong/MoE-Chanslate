"""With top_k == num_experts and router == 0, the upcycled MoE must equal the dense model."""
import copy

import torch
from transformers import Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM

from moe_chanslate.upcycle.convert import convert_state, moe_config

TC = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
          num_key_value_heads=2, head_dim=16, vocab_size=128, linear_key_head_dim=16,
          linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
          layer_types=["linear_attention"] * 3 + ["full_attention"], max_position_embeddings=256)


def test_equivalence():
    torch.manual_seed(0)
    dense_cfg = Qwen3_5Config(text_config=TC).text_config
    dense = Qwen3_5ForCausalLM(dense_cfg).eval()
    sd = dense.state_dict()

    full = {"model_type": "qwen3_5", "text_config": dense_cfg.to_dict(),
            "vision_config": {"model_type": "qwen3_5"}}
    mcfg = moe_config(full, top_k=5, expert_dim=16, shared_slices=1)   # 96/16-1 = 5 experts
    tc = mcfg["text_config"]
    moe = Qwen3_5MoeForCausalLM(Qwen3_5MoeConfig(text_config=tc).text_config).eval()
    new = dict(convert_state(sd.__getitem__, list(sd), expert_dim=16, shared_slices=1, router_std=0.0))
    missing, unexpected = moe.load_state_dict(new, strict=False)
    assert not unexpected, unexpected
    assert not [m for m in missing if "mtp" not in m], missing

    ids = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        a, b = dense(ids).logits, moe(ids).logits
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
