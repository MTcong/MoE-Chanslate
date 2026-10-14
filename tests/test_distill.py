import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torch
from transformers import Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM

from moe_chanslate.train.cache_teacher import pack, teacher_topk
from moe_chanslate.train.distill import train
from moe_chanslate.train.losses import distill_loss
from moe_chanslate.upcycle.convert import convert_state, moe_config
from test_upcycle_equivalence import TC


def test_loss_zero_when_student_matches():
    logits = torch.randn(2, 5, 50)
    lp = torch.log_softmax(logits, -1)
    top = lp.topk(8, -1)
    assert distill_loss(logits, top.indices, top.values).item() < 1e-5
    assert distill_loss(torch.randn(2, 5, 50), top.indices, top.values).item() > 0.1


def test_pack():
    assert list(pack([[1, 2, 3], [4, 5, 6, 7]], 3)) == [[1, 2, 3], [4, 5, 6]]


def test_end_to_end_tiny(tmp_path):
    torch.manual_seed(0)
    dcfg = Qwen3_5Config(text_config=TC).text_config
    teacher = Qwen3_5ForCausalLM(dcfg).eval()
    sd = teacher.state_dict()
    full = {"model_type": "qwen3_5", "text_config": dcfg.to_dict(), "vision_config": {"model_type": "qwen3_5"}}
    mc = moe_config(full, top_k=2, expert_dim=16, shared_slices=1)
    student = Qwen3_5MoeForCausalLM(Qwen3_5MoeConfig(text_config=mc["text_config"]).text_config)
    student.load_state_dict(dict(convert_state(sd.__getitem__, list(sd), expert_dim=16, shared_slices=1,
                                               router_std=0.02)), strict=False)
    frozen_before = student.model.layers[0].linear_attn.out_proj.weight.clone()

    ids = torch.randint(0, 128, (16, 32))
    idx, lp = teacher_topk(teacher, ids, torch.arange(128), 8)
    torch.save({"ids": ids.int(), "idx": idx, "lp": lp}, tmp_path / "shard_00000.pt")

    hist = train(student, str(tmp_path), lr=5e-3, batch_size=4, grad_accum=1, epochs=8, warmup=1,
                 log_every=100, device="cpu", grad_ckpt=False)
    assert all(torch.isfinite(torch.tensor(hist)))
    assert sum(hist[-4:]) < sum(hist[:4])                      # loss goes down
    assert torch.equal(frozen_before, student.model.layers[0].linear_attn.out_proj.weight)
