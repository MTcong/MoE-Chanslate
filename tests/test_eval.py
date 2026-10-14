from moe_chanslate.eval.translate_eval import score


def test_score_perfect_and_bad():
    refs = ["Hôm nay thời tiết rất đẹp.", "Tôi thích uống trà."]
    assert score(refs, refs)["chrf++"] == 100.0
    bad = score(["xyz", "abc"], refs)
    assert bad["chrf++"] < 20 and bad["n"] == 2
