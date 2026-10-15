"""Almamlaka TV Document Assistant — Streamlit UI."""
from __future__ import annotations

import base64
import html
import logging
import uuid
from pathlib import Path

import streamlit as st

from backend.config import Settings
from backend.ingestion import IngestionError
from backend.query_rewriting import Turn
from backend.rag import ChatResponse, RAGPipeline
from backend.text_utils import detect_language

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
AVATAR_BOT = str(ASSETS / "avatar_bot.png")
AVATAR_USER = str(ASSETS / "avatar_user.png")

st.set_page_config(page_title="Almamlaka TV Document Assistant", page_icon=AVATAR_BOT, layout="centered")

UI = {
    "en": {
        "dir": "ltr",
        "title": "Document Assistant",
        "subtitle": "Digital Expansion Initiative",
        "hero": "Ask the initiative's documents",
        "hero_sub": "Every answer comes from three project documents and shows the page it came from. "
                    "If the documents don't say, the assistant tells you so.",
        "new_chat": "New chat",
        "sources_heading": "Source documents",
        "scope_heading": "What it answers",
        "scope": "Goals, scope, budget, timeline, team, governance and policies of the Digital Expansion "
                 "Initiative, in English or Arabic.",
        "examples_heading": "Try a question",
        "input": "Ask about the documents in English or Arabic",
        "thinking": "Reading the documents…",
        "sources": "Sources",
        "page": "page",
        "conflict": "The documents disagree",
        "conflict_hint": "Both versions are shown with their sources. The assistant does not decide which is correct.",
        "pages": "pages",
        "disclaimer": "Independent technical demo built for an Almamlaka TV assessment. Not an official Almamlaka TV service.",
        "boot_error": "The document index could not be loaded",
        "retrieval": "Retrieval",
        "excerpt": "Show passage",
    },
    "ar": {
        "dir": "rtl",
        "title": "مساعد الوثائق",
        "subtitle": "مبادرة التوسع الرقمي",
        "hero": "اسأل وثائق المبادرة",
        "hero_sub": "كل إجابة مأخوذة من وثائق المشروع الثلاث مع ذكر الصفحة التي وردت فيها. "
                    "وإذا لم تتضمن الوثائق الإجابة، يخبرك المساعد بذلك.",
        "new_chat": "محادثة جديدة",
        "sources_heading": "الوثائق المصدر",
        "scope_heading": "ما الذي يجيب عنه",
        "scope": "أهداف مبادرة التوسع الرقمي ونطاقها وميزانيتها وجدولها الزمني وفريقها وحوكمتها وسياساتها، "
                 "بالعربية أو الإنجليزية.",
        "examples_heading": "جرّب سؤالاً",
        "input": "اسأل عن الوثائق بالعربية أو الإنجليزية",
        "thinking": "جارٍ قراءة الوثائق…",
        "sources": "المصادر",
        "page": "صفحة",
        "conflict": "الوثائق متعارضة",
        "conflict_hint": "تُعرض الروايتان مع مصدر كل منهما، ولا يحسم المساعد أيهما الصحيحة.",
        "pages": "صفحات",
        "disclaimer": "عرض تقني مستقل أُعدّ لتقييم لدى قناة المملكة، وليس خدمة رسمية من قناة المملكة.",
        "boot_error": "تعذّر تحميل فهرس الوثائق",
        "retrieval": "الاسترجاع",
        "excerpt": "عرض المقطع",
    },
}

EXAMPLES = {
    "en": [
        "When is the public launch date?",
        "What is the total budget of the project?",
        "Who is on the project team?",
        "Who approves budget changes above $50,000?",
    ],
    "ar": [
        "ما هو موعد الإطلاق العام للمشروع؟",
        "ما هي الميزانية الإجمالية للمبادرة؟",
        "من هو قائد المشروع؟",
        "هل تتم مشاركة بيانات المستخدمين مع أطراف ثالثة؟",
    ],
}

DOCS = {
    "Almamlaka_Project_Overview.pdf": {
        "en": ("Project Overview", "Goals, platforms, languages, stakeholders, success metrics"),
        "ar": ("نظرة عامة على المشروع", "الأهداف والمنصات واللغات والجهات المعنية ومؤشرات النجاح"),
    },
    "Almamlaka_Budget_Timeline.pdf": {
        "en": ("Budget & Timeline", "Quarterly budget, revision, milestones, risks"),
        "ar": ("الميزانية والجدول الزمني", "الميزانية الربعية والتعديل والمراحل والمخاطر"),
    },
    "Almamlaka_Team_Governance.pdf": {
        "en": ("Team & Governance", "Team, steering committee, launch decision, policies"),
        "ar": ("الفريق والحوكمة", "الفريق ولجنة التوجيه وقرار الإطلاق والسياسات"),
    },
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=Reem+Kufi:wght@500;600;700&display=swap');
:root {
  --crimson: #9D0B0E;      /* sampled from the supplied logo */
  --wine: #3F0A0D;
  --wine-2: #5A1216;
  --brass: #B08D57;
  --ink: #241A1C;
  --muted: #6E6264;
  --rule: #E9E0E0;
  --mist: #F7F3F3;
  --paper: #FFFFFF;
  --amber: #8A5300;
  --amber-bg: #FFF6E6;
  --display: 'Reem Kufi', 'IBM Plex Sans Arabic', system-ui, sans-serif;
  --body: 'IBM Plex Sans Arabic', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
html, body, p, li, h1, h2, h3, h4, label, textarea, input, button, summary, .stMarkdown { font-family: var(--body) !important; }
#MainMenu, footer, header[data-testid="stHeader"] [data-testid="stToolbar"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; padding-bottom: 7rem; max-width: 860px; }
.stApp { background: var(--paper); }

/* ---- masthead: the one bold element -------------------------------------- */
.masthead { position: relative; display: flex; align-items: center; gap: 22px; padding: 22px 28px 20px;
  background: var(--crimson); color: #fff; border-radius: 4px; overflow: hidden; }
.masthead::before { content: ""; position: absolute; inset: 0; pointer-events: none;
  background: repeating-linear-gradient(115deg, rgba(255,255,255,0.045) 0 2px, transparent 2px 14px); }
.masthead::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 3px; background: var(--brass); }
.masthead img { height: 46px; position: relative; flex: none; }
.masthead .sep { width: 1px; align-self: stretch; background: rgba(255,255,255,0.35); position: relative; }
.masthead .titles { position: relative; }
.masthead h1 { font-family: var(--display) !important; font-weight: 600; font-size: 1.75rem; line-height: 1.15;
  margin: 0; padding: 0; color: #fff; letter-spacing: 0; }
.masthead .sub { font-size: 0.95rem; opacity: 0.88; margin-top: 4px; }

/* ---- empty state ------------------------------------------------------------ */
.hero { padding: 2.6rem 0 1.4rem; }
.hero h2 { font-family: var(--display) !important; font-weight: 600; font-size: 2.6rem; line-height: 1.15;
  color: var(--ink); margin: 0 0 0.7rem; padding: 0; }
.hero h2, .hero p { text-align: start; }
.hero p { color: var(--muted); font-size: 1.05rem; line-height: 1.7; max-width: 62ch; margin: 0; }
.dossiers { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; margin: 1.6rem 0 1.8rem;
  border-top: 1px solid var(--ink); border-bottom: 1px solid var(--rule); }
.dossier { padding: 14px 16px 16px; border-inline-end: 1px solid var(--rule); }
.dossier:last-child { border-inline-end: none; }
.dossier .t { font-family: var(--display) !important; font-weight: 600; font-size: 1.08rem; color: var(--crimson); }
.dossier .d { font-size: 0.86rem; color: var(--muted); line-height: 1.55; margin-top: 4px; }
.dossier .f { font-size: 0.76rem; color: var(--muted); margin-top: 8px; direction: ltr; unicode-bidi: isolate; }
.try { font-family: var(--body) !important; font-weight: 600; color: var(--ink); font-size: 1.05rem; margin: 0 0 0.6rem; }
@media (max-width: 640px) {
  .dossiers { grid-template-columns: 1fr; }
  .dossier { border-inline-end: none; border-bottom: 1px solid var(--rule); }
  .hero h2 { font-size: 2rem; }
  .masthead { padding: 18px; gap: 14px; } .masthead img { height: 34px; } .masthead h1 { font-size: 1.35rem; }
}

/* suggestion tiles (main-area buttons) */
div[data-testid="stMainBlockContainer"] .stButton button, .block-container .stButton button {
  min-height: 64px; width: 100%; justify-content: flex-start; text-align: start; padding: 12px 16px;
  background: var(--paper); color: var(--ink); border: 1px solid var(--rule); border-inline-start: 3px solid var(--crimson);
  border-radius: 3px; font-size: 0.97rem; transition: background .15s ease, border-color .15s ease; }
.block-container .stButton button:hover { background: var(--mist); border-color: var(--crimson); color: var(--ink); }
.block-container .stButton button p, section[data-testid="stSidebar"] .stButton button p { text-align: start; }
.stButton button > div { justify-content: flex-start; width: 100%; }
section[data-testid="stSidebar"] .stButton button[kind="primary"] > div { justify-content: center; }
.masthead h1, .hero h2, .dossier .t { word-spacing: 0.12em; }
.try, .side-h, .cite-list .head, .conflict .title { font-family: var(--body) !important; }

/* ---- conversation --------------------------------------------------------- */
[data-testid="stChatMessage"] { background: transparent; padding: 0.9rem 0; gap: 14px; }
[data-testid="stChatMessage"] + [data-testid="stChatMessage"] { border-top: 1px solid var(--rule); }
[data-testid="stChatMessageAvatarCustom"] img, [data-testid="stChatMessage"] img { border-radius: 8px; }
.msg { line-height: 1.8; font-size: 1.02rem; color: var(--ink); }
.msg.user { font-weight: 500; }
.msg[dir="rtl"] { text-align: right; }
.cite-list { margin-top: 0.9rem; }
.cite-list .head { font-family: var(--body) !important; font-weight: 600; color: var(--muted); font-size: 0.9rem; margin-bottom: 0.35rem; }
.cite { display: flex; gap: 10px; align-items: baseline; padding: 5px 0; font-size: 0.9rem; border-bottom: 1px dashed var(--rule); }
.cite:last-child { border-bottom: none; }
.cite .n { flex: none; min-width: 1.55em; height: 1.55em; line-height: 1.55em; text-align: center;
  background: var(--crimson); color: #fff; font-weight: 600; font-size: 0.76rem; border-radius: 2px; }
.cite .doc { font-weight: 600; color: var(--ink); }
.cite .where { color: var(--muted); }
.conflict { border: 1px solid #EBD3A8; border-inline-start: 4px solid var(--amber); background: var(--amber-bg);
  padding: 12px 16px; margin: 0.9rem 0 0.2rem; border-radius: 3px; }
.conflict .title { font-family: var(--body) !important; font-weight: 600; color: var(--amber); font-size: 1.02rem; display: flex; gap: 8px; align-items: center; }
.conflict .hint { font-size: 0.84rem; color: var(--muted); margin-top: 2px; }
.conflict .topic { font-weight: 600; margin-top: 8px; }
.conflict ul { margin: 0.3rem 0 0; padding-inline-start: 1.2rem; }
.conflict li { margin: 2px 0; }
.status { border-inline-start: 3px solid var(--muted); padding: 10px 14px; background: var(--mist); border-radius: 2px; line-height: 1.7; }
.status.error { border-color: var(--crimson); }
.excerpt { white-space: pre-wrap; font-size: 0.88rem; color: var(--ink); background: var(--mist);
  border-inline-start: 3px solid var(--brass); padding: 10px 14px; direction: ltr; text-align: left; line-height: 1.6; }
.excerpt-meta { font-size: 0.8rem; color: var(--muted); margin-bottom: 6px; }
[data-testid="stExpander"] details { border: 1px solid var(--rule); border-radius: 3px; }
[data-testid="stExpander"] summary { font-size: 0.86rem; color: var(--muted); }

/* chat input */
[data-testid="stChatInput"] { border-radius: 4px; }
[data-testid="stChatInput"] > div { border: 1px solid var(--ink) !important; border-radius: 4px !important; background: var(--paper) !important; }
[data-testid="stChatInput"] button { background: var(--crimson) !important; color: #fff !important; border-radius: 3px !important; }
[data-testid="stBottomBlockContainer"], [data-testid="stBottom"] > div { background: var(--paper); }

/* ---- sidebar: deep wine ----------------------------------------------------- */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, var(--wine) 0%, #2B0709 100%); }
section[data-testid="stSidebar"] * { color: #F3E9E9; }
section[data-testid="stSidebar"] .side-h { font-family: var(--body) !important; font-weight: 600; font-size: 0.98rem;
  color: #fff; margin: 1.3rem 0 0.4rem; padding-bottom: 6px; border-bottom: 1px solid rgba(176,141,87,0.45); }
section[data-testid="stSidebar"] .side-p { font-size: 0.9rem; line-height: 1.65; opacity: 0.88; }
section[data-testid="stSidebar"] .doc { margin: 0.55rem 0; font-size: 0.9rem; line-height: 1.45; }
section[data-testid="stSidebar"] .doc b { color: #fff; }
section[data-testid="stSidebar"] .doc span { opacity: 0.7; font-size: 0.8rem; direction: ltr; unicode-bidi: isolate; }
section[data-testid="stSidebar"] .stButton button { width: 100%; background: transparent; border: 1px solid rgba(255,255,255,0.22);
  color: #fff; border-radius: 3px; justify-content: flex-start; text-align: start; font-size: 0.9rem; min-height: 0; }
section[data-testid="stSidebar"] .stButton button:hover { border-color: var(--brass); background: rgba(255,255,255,0.06); }
section[data-testid="stSidebar"] .stButton button[kind="primary"] { background: var(--brass); border-color: var(--brass); color: var(--wine); font-weight: 600; justify-content: center; }
section[data-testid="stSidebar"] .stButton button[kind="primary"] * { color: var(--wine); }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"], section[data-testid="stSidebar"] .foot { opacity: 0.62; font-size: 0.78rem; line-height: 1.55; }
section[data-testid="stSidebar"] [role="radiogroup"] label div:first-child { border-color: var(--brass) !important; }

button:focus-visible, a:focus-visible, summary:focus-visible { outline: 3px solid var(--brass) !important; outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""


@st.cache_resource(show_spinner="Indexing the three PDF documents…")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline.from_settings(Settings.from_env())


@st.cache_data
def data_uri(path: str) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def md_block(text: str, lang: str, cls: str = "") -> None:
    """Render model markdown inside a direction-aware wrapper. Raw HTML from the
    model is escaped; markdown (bold, bullets) still renders."""
    direction = "rtl" if lang == "ar" else "ltr"
    safe = esc(text).replace("&quot;", '"').replace("&#x27;", "'")
    st.markdown(f'<div class="msg {cls}" dir="{direction}" lang="{lang}">\n\n{safe}\n\n</div>', unsafe_allow_html=True)


SCALE_ICON = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
              'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v18M7 21h10M5 7h14"/>'
              '<path d="M5 7l-3 6a3 3 0 0 0 6 0zM19 7l-3 6a3 3 0 0 0 6 0z"/></svg>')


def render_response(resp: ChatResponse) -> None:
    lang = resp.language
    t = UI[lang]
    direction = t["dir"]
    if resp.kind in ("error", "abstain", "refused", "out_of_scope"):
        cls = "status error" if resp.kind == "error" else "status"
        st.markdown(f'<div class="{cls}" dir="{direction}">{esc(resp.text)}</div>', unsafe_allow_html=True)
        return
    md_block(resp.text, lang)

    if resp.conflicts:
        items = []
        for c in resp.conflicts:
            sides = "".join(
                f"<li>{esc(claim)} <b>{''.join(f'[{n}]' for n in nums)}</b></li>" for claim, nums in c.sides
            )
            note = f'<div class="hint">{esc(c.note)}</div>' if c.note else ""
            items.append(f'<div class="topic">{esc(c.topic)}</div><ul>{sides}</ul>{note}')
        st.markdown(
            f'<div class="conflict" dir="{direction}"><div class="title">{SCALE_ICON}{t["conflict"]}</div>'
            f'<div class="hint">{t["conflict_hint"]}</div>{"".join(items)}</div>',
            unsafe_allow_html=True,
        )

    if resp.citations:
        rows = []
        for c in resp.citations:
            title = DOCS.get(c.document_name, {}).get(lang, (c.document_name, ""))[0]
            rows.append(
                f'<div class="cite"><span class="n">{c.number}</span><span><span class="doc">{esc(title)}</span> '
                f'<span class="where">— {t["page"]} {c.page_number}, <bdi>{esc(c.section_title)}</bdi></span></span></div>'
            )
        st.markdown(
            f'<div class="cite-list" dir="{direction}"><div class="head">{t["sources"]}</div>{"".join(rows)}</div>',
            unsafe_allow_html=True,
        )
        for c in resp.citations:
            with st.expander(f"[{c.number}] {t['excerpt']}: {c.document_name}, {t['page']} {c.page_number}"):
                st.markdown(
                    f'<div class="excerpt-meta">{esc(c.document_title)} — {esc(c.section_title)}</div>'
                    f'<div class="excerpt">{esc(c.excerpt)}</div>',
                    unsafe_allow_html=True,
                )


def history_turns() -> list[Turn]:
    return [
        Turn(m["role"], m["response"].text if m["role"] == "assistant" else m["content"])
        for m in st.session_state.messages
    ]


# ---- page ---------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.messages = []
    st.session_state.ui_lang = "en"

with st.sidebar:
    choice = st.radio("Language / اللغة", ["English", "العربية"], horizontal=True,
                      index=0 if st.session_state.ui_lang == "en" else 1, label_visibility="collapsed")
    st.session_state.ui_lang = "en" if choice == "English" else "ar"
ui = st.session_state.ui_lang
t = UI[ui]

try:
    pipeline = get_pipeline()
except IngestionError as exc:
    st.error(f"{t['boot_error']}: {exc}")
    st.stop()

st.markdown(
    f"""<div class="masthead" dir="{t['dir']}">
      <img src="{data_uri(str(ASSETS / 'logo_white.png'))}" alt="Almamlaka">
      <div class="sep"></div>
      <div class="titles"><h1>{t['title']}</h1><div class="sub">{t['subtitle']}</div></div>
    </div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    if st.button(t["new_chat"], type="primary", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.markdown(f'<div class="side-h" dir="{t["dir"]}">{t["scope_heading"]}</div>'
                f'<div class="side-p" dir="{t["dir"]}">{t["scope"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="side-h" dir="{t["dir"]}">{t["examples_heading"]}</div>', unsafe_allow_html=True)
    for i, q in enumerate(EXAMPLES[ui]):
        if st.button(q, key=f"side-ex-{ui}-{i}", width="stretch"):
            st.session_state.pending = q
    st.markdown(f'<div class="side-h" dir="{t["dir"]}">{t["sources_heading"]}</div>', unsafe_allow_html=True)
    docs = pipeline.manifest.get("documents", [])
    st.markdown(
        f'<div dir="{t["dir"]}">' + "".join(
            f'<div class="doc"><b>{esc(DOCS.get(d["name"], {}).get(ui, (d["name"], ""))[0])}</b><br>'
            f'<span>{esc(d["name"])}, {d["pages"]} {t["pages"]}</span></div>' for d in docs)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"{t['retrieval']}: {pipeline.retriever.mode}, {pipeline.manifest.get('chunk_count', '?')} chunks")
    st.markdown(f'<div class="foot" dir="{t["dir"]}">{t["disclaimer"]}</div>', unsafe_allow_html=True)

if not st.session_state.messages and not st.session_state.get("pending"):
    dossiers = "".join(
        f'<div class="dossier"><div class="t">{esc(DOCS[name][ui][0])}</div><div class="d">{esc(DOCS[name][ui][1])}</div>'
        f'<div class="f">{esc(name)}</div></div>'
        for name in DOCS
    )
    st.markdown(
        f'<div class="hero" dir="{t["dir"]}"><h2>{t["hero"]}</h2><p>{t["hero_sub"]}</p></div>'
        f'<div class="dossiers" dir="{t["dir"]}">{dossiers}</div>'
        f'<div class="try" dir="{t["dir"]}">{t["examples_heading"]}</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES[ui]):
        with cols[i % 2]:
            if st.button(q, key=f"tile-{ui}-{i}", width="stretch"):
                st.session_state.pending = q
                st.rerun()

for m in st.session_state.messages:
    if m["role"] == "user":
        with st.chat_message("user", avatar=AVATAR_USER):
            md_block(m["content"], m["lang"], "user")
    else:
        with st.chat_message("assistant", avatar=AVATAR_BOT):
            render_response(m["response"])

prompt = st.chat_input(t["input"], max_chars=pipeline.settings.max_input_chars)
if st.session_state.get("pending"):
    prompt = st.session_state.pop("pending")

if prompt:
    lang = detect_language(prompt)
    history = history_turns()
    st.session_state.messages.append({"role": "user", "content": prompt, "lang": lang})
    with st.chat_message("user", avatar=AVATAR_USER):
        md_block(prompt, lang, "user")
    with st.chat_message("assistant", avatar=AVATAR_BOT):
        with st.spinner(UI[lang]["thinking"]):
            try:
                response = pipeline.answer(prompt, history)
            except Exception:  # never show a stack trace to the user
                logging.exception("Unhandled error while answering")
                response = ChatResponse("error", lang, "An unexpected error occurred. Please try again."
                                        if lang == "en" else "حدث خطأ غير متوقع. يُرجى المحاولة مرة أخرى.")
        render_response(response)
    st.session_state.messages.append({"role": "assistant", "response": response, "lang": lang})
