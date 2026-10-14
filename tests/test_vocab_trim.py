import json

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from moe_chanslate.vocab.trim import (bytes_alphabet, count_tokens, remap_ids, select_tokens,
                                      trim_rows, trim_tokenizer)

CORPUS = ["今天天气很好，我们去公园玩。", "Hôm nay thời tiết rất đẹp, chúng ta đi công viên.",
          "The weather is nice today.", "我喜欢喝茶。Tôi thích uống trà."] * 20
EXTRA = ["日本語のテキストは使われない。", "Русский текст тоже лишний."] * 5


def _tok():
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=600, special_tokens=["<|im_end|>", "<|pad|>"],
                             initial_alphabet=sorted(bytes_alphabet()))
    t.train_from_iterator(CORPUS + EXTRA, tr)
    return t


def test_trim_roundtrip_and_weights():
    tok = _tok()
    tj = json.loads(tok.to_str())
    n_old = tok.get_vocab_size()
    keep = select_tokens(tj, count_tokens(tok, CORPUS), size=300)
    new_tj, old2new = trim_tokenizer(tj, keep)
    new = Tokenizer.from_str(json.dumps(new_tj))
    assert new.get_vocab_size() <= 300 < n_old
    for s in CORPUS[:4] + EXTRA[:2]:            # byte-level => everything stays encodable
        ids = new.encode(s).ids
        assert max(ids) < new.get_vocab_size()
        assert new.decode(ids) == s
    # special tokens keep working and are remapped
    assert new.token_to_id("<|im_end|>") == old2new[tok.token_to_id("<|im_end|>")]
    # embedding rows follow the map
    w = torch.randn(n_old + 10, 8)
    nw = trim_rows(w, old2new, 384)
    for o, n in list(old2new.items())[:20]:
        assert torch.equal(nw[n], w[o])
    assert remap_ids({"text_config": {"eos_token_id": 0}}, {0: 7}) == {"text_config": {"eos_token_id": 7}}
