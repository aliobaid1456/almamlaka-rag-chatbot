"""Central configuration. Values come from environment variables (or Streamlit
secrets when running on Streamlit Cloud). No secret has a default value."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

ABSTAIN_EN = "I couldn't find sufficient information to answer this question in the three supplied PDF documents."
ABSTAIN_AR = "لم أجد معلومات كافية للإجابة عن هذا السؤال ضمن ملفات PDF الثلاثة المقدّمة."

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai_compatible": "",  # must be set explicitly (depends on the provider)
}


def _get(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is not None and value != "":
        return value
    try:  # Streamlit Cloud / HF Spaces secrets, only if streamlit is running
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    llm_provider: str = "anthropic"
    llm_api_key: str = field(default="", repr=False)
    llm_model: str = ""
    llm_base_url: str = ""
    llm_timeout_s: float = 45.0
    llm_max_retries: int = 2
    llm_json_mode: bool = True
    llm_reasoning_effort: str = ""  # e.g. "low" for reasoning models such as gpt-oss

    embedding_backend: str = "fastembed"  # fastembed | none
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    pdf_dir: Path = ROOT_DIR / "data" / "pdfs"
    index_dir: Path = ROOT_DIR / "index"

    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 80
    top_k: int = 6
    max_input_chars: int = 1000
    history_turns: int = 6

    @classmethod
    def from_env(cls) -> "Settings":
        provider = _get("LLM_PROVIDER", "anthropic").strip().lower()
        return cls(
            llm_provider=provider,
            llm_api_key=_get("LLM_API_KEY"),
            llm_model=_get("LLM_MODEL", DEFAULT_MODELS.get(provider, "")),
            llm_base_url=_get("LLM_BASE_URL"),
            llm_timeout_s=float(_get("LLM_TIMEOUT_S", "45")),
            llm_max_retries=int(_get("LLM_MAX_RETRIES", "2")),
            llm_json_mode=_bool(_get("LLM_JSON_MODE", "true")),
            llm_reasoning_effort=_get("LLM_REASONING_EFFORT", "").strip().lower(),
            embedding_backend=_get("EMBEDDING_BACKEND", "fastembed").strip().lower(),
            embedding_model=_get(
                "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ),
            pdf_dir=Path(_get("PDF_DIR", str(ROOT_DIR / "data" / "pdfs"))),
            index_dir=Path(_get("INDEX_DIR", str(ROOT_DIR / "index"))),
            top_k=int(_get("TOP_K", "6")),
            max_input_chars=int(_get("MAX_INPUT_CHARS", "1000")),
            history_turns=int(_get("HISTORY_TURNS", "6")),
        )
