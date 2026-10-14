"""Prompt formatting shared by SFT and eval."""
SYSTEM = "You are a professional Chinese-to-Vietnamese translator."
TEMPLATE = "Dịch đoạn văn sau từ tiếng Trung sang tiếng Việt, chỉ trả về bản dịch.\n\n{zh}"


def to_messages(zh: str, vi: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TEMPLATE.format(zh=zh)}]
    if vi is not None:
        msgs.append({"role": "assistant", "content": vi})
    return msgs
