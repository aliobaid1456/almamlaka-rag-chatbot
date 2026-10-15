"""Language detection, Arabic/English normalization and number extraction."""
from __future__ import annotations

import re

_ARABIC_CHAR = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN_CHAR = re.compile(r"[A-Za-z]")
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # harakat + tatweel
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def detect_language(text: str) -> str:
    """Return 'ar' or 'en' for the *current* message, by script majority.

    Arabic wins when Arabic letters are at least 30% of the letters, so a
    mostly-Arabic question containing an English name ("ما هو دور Omar Fayez؟")
    is still answered in Arabic.
    """
    ar = len(_ARABIC_CHAR.findall(text))
    en = len(_LATIN_CHAR.findall(text))
    if ar == 0 and en == 0:
        return "en"
    return "ar" if ar / (ar + en) >= 0.3 else "en"


def arabic_ratio(text: str) -> float:
    ar = len(_ARABIC_CHAR.findall(text))
    en = len(_LATIN_CHAR.findall(text))
    return ar / (ar + en) if (ar + en) else 0.0


def normalize_digits(text: str) -> str:
    return text.translate(_ARABIC_INDIC).replace("٬", ",").replace("٫", ".")


def normalize_arabic(text: str) -> str:
    text = _DIACRITICS.sub("", text)
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return text


_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "and", "or", "is", "are", "was",
    "were", "be", "been", "it", "its", "this", "that", "these", "those", "what", "which", "who", "whom",
    "when", "where", "how", "why", "do", "does", "did", "as", "from", "about", "into", "will", "would",
    "can", "could", "should", "me", "tell", "please", "i", "you", "we", "they", "there", "their", "has",
    "have", "had", "any", "all", "also",
    "في", "من", "على", "الى", "إلى", "عن", "ما", "ماذا", "هل", "هو", "هي", "متى", "كم", "من", "التي", "الذي",
    "و", "او", "أو", "مع", "هذا", "هذه", "ذلك", "تلك", "كان", "كانت",
}

_TOKEN = re.compile(r"\$?\d[\d,\.]*%?|\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lexical tokens for BM25: lowercase, Arabic-normalized, numbers without separators."""
    text = normalize_arabic(normalize_digits(text.lower()))
    tokens: list[str] = []
    for tok in _TOKEN.findall(text):
        if tok[0] == "$" or tok[0].isdigit():
            tok = tok.lstrip("$").rstrip("%").replace(",", "").rstrip(".")
            if tok:
                tokens.append(tok)
            continue
        if tok.endswith("'s"):
            tok = tok[:-2]
        if tok in _STOPWORDS or len(tok) < 2:
            continue
        # Arabic definite article "ال" — index both forms
        if tok.startswith("ال") and len(tok) > 4:
            tokens.append(tok[2:])
        # crude English plural folding
        if re.fullmatch(r"[a-z]+s", tok) and len(tok) > 4 and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def extract_numbers(text: str) -> set[str]:
    """Numbers with 2+ significant characters, normalized (no thousands separators).

    Single-digit numbers are ignored because they are too common to be a
    meaningful grounding signal (list markers, "3 platforms" vs "section 3").
    """
    out: set[str] = set()
    for m in _NUMBER.findall(normalize_digits(text)):
        value = m.replace(",", "").rstrip(".")
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        if len(value.replace(".", "")) >= 2:
            out.add(value)
    return out
