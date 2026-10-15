"""Input validation, prompt-injection heuristics, evidence sanitization and
log redaction.

These are defense-in-depth layers around the main defense, which is
architectural: the model has no tools, no network, no file access; it only
sees a server-controlled system prompt, a question, and evidence wrapped as
data; and every answer is validated in code before it reaches the user.
"""
from __future__ import annotations

import re
import unicodedata

from .text_utils import normalize_arabic

# Random marker embedded in the system prompts. If it ever appears in an
# output, the prompt leaked and the answer is withheld.
CANARY = "ALM-7Q2X-CANARY"


class InputRejected(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


_CONTROL = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F​-‏‪-‮⁦-⁩]")


def sanitize_user_input(text: str, max_chars: int) -> str:
    if not isinstance(text, str):
        raise InputRejected("invalid")
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub("", text).strip()
    if not text:
        raise InputRejected("empty")
    if len(text) > max_chars:
        raise InputRejected("too_long")
    return text


_EN_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instruction|instructions|rules|prompt|guidelines|context|directions)\b",
    r"\b(system|developer|hidden|initial|original)\s+(prompt|message|instructions)\b",
    r"\breveal\b.{0,30}\b(prompt|instructions|rules|configuration|secrets?)\b",
    r"\byou are now\b|\bpretend (to be|you are)\b|\broleplay\b|\bfrom now on,? you\b",
    r"\b(jailbreak|dan mode|developer mode|god mode)\b",
    r"\b(environment variables?|env vars?|\.env\b|api[_ -]?keys?|secret keys?|access tokens?)\b",
    r"\b(show|reveal|print|give|list|tell|leak|dump)\b.{0,30}\b(passwords?|credentials?|secrets?|tokens?)\b",
    r"\b(execute|run)\b.{0,20}\b(command|shell|code|script|bash|python)\b|\brm -rf\b|\bos\.system\b|\bsubprocess\b",
    r"\b(other|another) (user|users|session|sessions)('s)?\b",
    r"\b(tell|write|give) me (a|an) (joke|poem|story|song)\b",
]
_AR_PATTERNS = [
    r"(تجاهل|انس|انسى|تخط)\S*.{0,40}(التعليمات|الاوامر|القواعد|التوجيهات)",
    r"(موجه|تعليمات|رساله)\s+النظام",
    r"(اكشف|اظهر|اعرض|اطبع)\S*.{0,30}(التعليمات|الموجه|الاعدادات|الاسرار|المفتاح)",
    r"(انت الان|تصرف ك|تظاهر)",
    r"(متغيرات البييه|مفتاح ال?api|مفاتيح ال?api|كلمه المرور|كلمات المرور)",
    r"(نفذ|شغل)\S*.{0,20}(امر|اوامر|كود|سكربت)",
    r"(اكتب|احك|قل|اعطني|اخبرني)\S*.{0,15}(نكته|قصيده|قصه|شعر)",
]
_EN_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _EN_PATTERNS]
_AR_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _AR_PATTERNS]  # patterns are in normalize_arabic() form


def detect_injection(text: str) -> bool:
    """Heuristic detector for direct injection / scope-escape attempts.

    Deliberately conservative: it flags requests to change role, reveal
    instructions/secrets, run code, reach other sessions, or produce off-scope
    creative content. Subtler attempts are caught by the query classifier and
    by output validation.
    """
    if any(r.search(text) for r in _EN_RE):
        return True
    ar = normalize_arabic(text)
    return any(r.search(ar) for r in _AR_RE)


_EVIDENCE_TAGS = re.compile(r"</?\s*(evidence|passage|system|instructions?|user|assistant)[^>]*>", re.IGNORECASE)


def sanitize_evidence(text: str) -> str:
    """Neutralize tag-like sequences so a passage cannot close its own data block."""
    return _EVIDENCE_TAGS.sub(lambda m: m.group(0).replace("<", "‹").replace(">", "›"), text)


def evidence_contains_instructions(text: str) -> bool:
    return detect_injection(text)


_SECRET_PATTERNS = [
    re.compile(r"sk-(?:ant-)?[A-Za-z0-9_\-]{12,}"),
    re.compile(r"gsk_[A-Za-z0-9]{12,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{12,}"),
    re.compile(r"(?i)(api[_-]?key|authorization|bearer)\s*[:=]\s*\S+"),
]


def redact(text: str) -> str:
    for p in _SECRET_PATTERNS:
        text = p.sub("[REDACTED]", text)
    return text


def output_leaks_prompt(text: str) -> bool:
    return CANARY in text
