from moe_chanslate.data.clean import clean_pairs, keep_pair, normalize


def test_good_pair():
    assert keep_pair("今天天气很好。", "Hôm nay thời tiết rất đẹp.")


def test_rejects():
    assert not keep_pair("今天天气很好。", "今天天气很好。")                  # copy
    assert not keep_pair("今天天气很好。", "Hôm nay 天气 đẹp.")               # CJK in vi
    assert not keep_pair("Hello world", "Xin chào thế giới")                  # zh side not Chinese
    assert not keep_pair("我有3个苹果", "Tôi có 5 quả táo")                    # number mismatch
    assert not keep_pair("你好", "x" * 200)                                    # length ratio
    assert not keep_pair("请访问 http://a.com", "Vui lòng truy cập http://a.com")


def test_dedup_and_normalize():
    p = [("你好，世界！", "Xin chào thế giới!")] * 3
    assert len(list(clean_pairs(p))) == 1
    assert normalize("a​  b\n c") == "a b c"
