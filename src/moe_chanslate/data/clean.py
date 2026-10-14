"""Rule-based cleaning / filtering of zh-vi sentence pairs (no model deps)."""
import hashlib
import re
import unicodedata

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_LATIN = re.compile(r"[A-Za-zÀ-ỹ]")
_VI_DIACRITIC = re.compile(
    r"[ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ]", re.I)
_URL = re.compile(r"https?://|www\.|@\w+\.\w+")
_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f​-‏﻿]")
_NUM = re.compile(r"\d+")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = _CTRL.sub("", s)
    return _WS.sub(" ", s).strip()


def _frac(rx, s):
    letters = [c for c in s if not c.isspace()]
    return len(rx.findall(s)) / max(len(letters), 1)


def keep_pair(zh: str, vi: str, *, max_zh=300, max_vi=600, max_ratio=4.0) -> bool:
    """Both strings must already be normalized."""
    if not zh or not vi or zh == vi:
        return False
    if len(zh) > max_zh or len(vi) > max_vi:
        return False
    if _URL.search(zh) or _URL.search(vi):
        return False
    if _frac(_CJK, zh) < 0.3:                 # zh side must be mostly Chinese
        return False
    if _CJK.search(vi):                       # vi side must not contain Chinese
        return False
    if _frac(_LATIN, vi) < 0.5:               # vi side must be mostly Latin letters
        return False
    if len(vi) >= 25 and not _VI_DIACRITIC.search(vi):   # long text w/o diacritics ~ not Vietnamese
        return False
    # Chinese is ~2-3x shorter than Vietnamese in chars; reject extreme length ratios
    r = len(vi) / max(len(zh), 1)
    if r > max_ratio * 3 or r < 1 / max_ratio:
        return False
    if sorted(_NUM.findall(zh)) != sorted(_NUM.findall(vi)) and len(zh) < 30:
        return False                          # short lines should agree on numbers
    return True


def pair_key(zh: str, vi: str) -> str:
    return hashlib.md5(f"{zh}\t{vi}".encode()).hexdigest()


def clean_pairs(pairs):
    """Iterable of (zh, vi) -> generator of cleaned, de-duplicated pairs."""
    seen = set()
    for zh, vi in pairs:
        zh, vi = normalize(zh), normalize(vi)
        if not keep_pair(zh, vi):
            continue
        k = pair_key(zh, vi)
        if k in seen:
            continue
        seen.add(k)
        yield zh, vi
