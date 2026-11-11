# ── AI Meeting Assistant ── app.py (Light SaaS UI) ────────────────────────────
from dotenv import load_dotenv

load_dotenv()

import html
import time
import tempfile
from pathlib import Path
from io import BytesIO

import streamlit as st

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (light theme, dark text, SaaS look) ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg:#f5f6fb; --surface:#ffffff; --border:#e4e7f0; --text:#0f172a;
    --muted:#475569; --brand:#4f46e5; --brand-2:#7c3aed; --brand-soft:#eef2ff;
    --ok:#059669; --ok-soft:#d1fae5; --warn:#b45309; --warn-soft:#fef3c7;
}

/* Force readable dark text everywhere */
html, body, .stApp, [class*="css"] { font-family:'Plus Jakarta Sans',sans-serif; }
.stApp { background: radial-gradient(1200px 500px at 85% -10%, #e0e7ff 0%, transparent 60%), var(--bg) !important; }
.stApp, .stApp p, .stApp span, .stApp li, .stApp label, .stApp div,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stMarkdown, [data-testid="stCaptionContainer"] { color: var(--text); }
[data-testid="stCaptionContainer"], .muted { color: var(--muted) !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1200px; }

[data-testid="stSidebar"] { background: var(--surface) !important; border-right:1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text); }

/* Inputs */
.stTextInput input, .stTextArea textarea {
    background:#fff !important; color:var(--text) !important; border:1.5px solid var(--border) !important;
    border-radius:10px !important; padding:.65rem .8rem !important;
}
.stTextInput input:focus { border-color: var(--brand) !important; box-shadow:0 0 0 3px rgba(79,70,229,.15) !important; }
.stTextInput input::placeholder { color:#94a3b8 !important; }
div[data-baseweb="select"] > div { background:#fff !important; border:1.5px solid var(--border) !important; border-radius:10px !important; }
div[data-baseweb="select"] * { color: var(--text) !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li { background:#fff !important; color:var(--text) !important; }
[data-testid="stFileUploaderDropzone"] { background:#f8faff !important; border:1.5px dashed #a5b4fc !important; border-radius:12px !important; }
[data-testid="stFileUploaderDropzone"] * { color: var(--text) !important; }
[data-testid="stFileUploaderDropzone"] button { background:#fff !important; border:1px solid var(--border) !important; }

/* Buttons */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, var(--brand), var(--brand-2)) !important;
    color:#fff !important; border:none !important; border-radius:10px !important;
    font-weight:700 !important; padding:.6rem 1rem !important;
    box-shadow:0 4px 14px rgba(79,70,229,.25); transition:all .2s ease;
}
.stButton > button *, .stDownloadButton > button *, .stFormSubmitButton > button * { color:#fff !important; }
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-2px); box-shadow:0 10px 24px rgba(79,70,229,.35);
}
.stButton > button:active { transform: scale(.98); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:6px; background:#fff; padding:6px; border-radius:12px; border:1px solid var(--border); }
.stTabs [data-baseweb="tab"] { border-radius:8px; padding:.5rem 1.1rem; font-weight:600; color:var(--muted) !important; }
.stTabs [aria-selected="true"] { background: var(--brand-soft) !important; color: var(--brand) !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

/* Brand + hero */
.brand { display:flex; align-items:center; gap:.6rem; font-weight:800; font-size:1.15rem; }
.brand-logo { width:36px; height:36px; border-radius:10px; display:grid; place-items:center;
    background:linear-gradient(135deg,var(--brand),var(--brand-2)); font-size:1.1rem; }
.hero-title { font-size:clamp(1.8rem,4vw,2.6rem); font-weight:800; letter-spacing:-.02em; line-height:1.15; }
.hero-sub { color:var(--muted); font-size:1rem; margin-top:.4rem; }
.pill { display:inline-block; padding:.25rem .7rem; border-radius:999px; font-size:.75rem; font-weight:700; }
.pill-brand { background:var(--brand-soft); color:var(--brand) !important; }
.pill-ok { background:var(--ok-soft); color:var(--ok) !important; }
.pill-warn { background:var(--warn-soft); color:var(--warn) !important; }

/* Cards */
.card { background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:1.4rem;
    margin-bottom:1rem; box-shadow:0 1px 3px rgba(15,23,42,.05); }
.card-title { font-weight:700; font-size:.95rem; margin-bottom:.7rem; display:flex; align-items:center; gap:.5rem; }
.card-content { font-size:.93rem; line-height:1.75; white-space:pre-wrap; overflow-wrap:anywhere; }
.stat { background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:1rem 1.2rem; }
.stat-num { font-size:1.7rem; font-weight:800; color:var(--brand) !important; }
.stat-label { font-size:.8rem; color:var(--muted) !important; font-weight:500; }
.transcript-box { background:#f8fafc; border:1px solid var(--border); border-radius:12px; padding:1.2rem;
    font-size:.88rem; line-height:1.8; max-height:480px; overflow-y:auto; white-space:pre-wrap; overflow-wrap:anywhere; }

/* Pipeline steps */
.step { display:flex; align-items:center; gap:.75rem; padding:.7rem 1rem; background:#fff;
    border:1px solid var(--border); border-radius:12px; margin-bottom:.5rem; font-size:.9rem; font-weight:500; }
.step .st { margin-left:auto; font-size:.78rem; font-weight:700; }
.dot { width:10px; height:10px; border-radius:50%; background:#cbd5e1; }
.dot.active { background:var(--brand); animation:pulse 1s infinite; }
.dot.done { background:var(--ok); }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(79,70,229,.5)} 100%{box-shadow:0 0 0 10px rgba(79,70,229,0)} }

/* CTA banner */
.cta { background:linear-gradient(135deg,#4f46e5,#7c3aed); border-radius:18px; padding:1.4rem 1.6rem; margin-bottom:1rem; }
.cta, .cta * { color:#fff !important; }
.cta h3 { margin:0 0 .25rem 0; font-size:1.2rem; font-weight:800; }
.cta p { margin:0; opacity:.9; font-size:.9rem; }

/* Chat dialog */
div[role="dialog"] { background:#fff !important; border-radius:20px !important; }
div[role="dialog"] * { color: var(--text); }
.chat-head { display:flex; align-items:center; gap:.75rem; padding-bottom:.8rem; border-bottom:1px solid var(--border); margin-bottom:.8rem; }
.avatar { width:40px; height:40px; border-radius:12px; display:grid; place-items:center; font-size:1.2rem;
    background:linear-gradient(135deg,var(--brand),var(--brand-2)); }
.online { width:8px; height:8px; border-radius:50%; background:#10b981; display:inline-block; margin-right:6px; animation:pulse 1.6s infinite; }
.chat-scroll { display:flex; flex-direction:column-reverse; height:380px; overflow-y:auto;
    background:#f8faff; border:1px solid var(--border); border-radius:14px; padding:1rem; }
.chat-inner { display:flex; flex-direction:column; gap:.9rem; }
.msg { display:flex; gap:.6rem; align-items:flex-end; animation:slideIn .35s cubic-bezier(.2,.8,.2,1); }
.msg.user { flex-direction:row-reverse; }
.msg .ava { width:28px; height:28px; border-radius:50%; display:grid; place-items:center; font-size:.85rem; flex-shrink:0; background:#e0e7ff; }
.msg.user .ava { background:#ddd6fe; }
.bubble { padding:.7rem 1rem; border-radius:16px; max-width:80%; font-size:.9rem; line-height:1.65;
    white-space:pre-wrap; overflow-wrap:anywhere; }
.msg.bot .bubble { background:#fff; border:1px solid var(--border); border-bottom-left-radius:4px; box-shadow:0 2px 6px rgba(15,23,42,.05); }
.msg.user .bubble { background:linear-gradient(135deg,var(--brand),var(--brand-2)); border-bottom-right-radius:4px; }
.msg.user .bubble, .msg.user .bubble * { color:#fff !important; }
@keyframes slideIn { from{opacity:0; transform:translateY(14px) scale(.97)} to{opacity:1; transform:none} }
.typing { display:inline-flex; gap:5px; padding:.4rem .2rem; }
.typing i { width:8px; height:8px; border-radius:50%; background:var(--brand); opacity:.4; animation:bounce 1.2s infinite; }
.typing i:nth-child(2){ animation-delay:.15s } .typing i:nth-child(3){ animation-delay:.3s }
@keyframes bounce { 0%,60%,100%{transform:translateY(0);opacity:.4} 30%{transform:translateY(-6px);opacity:1} }
.cursor { display:inline-block; width:2px; height:1em; background:var(--brand); margin-left:2px; vertical-align:text-bottom; animation:blink .8s infinite; }
@keyframes blink { 50%{opacity:0} }
.empty-chat { text-align:center; margin:auto; padding:1rem; color:var(--muted) !important; font-size:.9rem; }
@media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "result": None, "chat_history": [], "pipeline_done": False, "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

PIPELINE_STEPS = [
    ("audio", "🔊", "Audio Processing"),
    ("transcript", "🎙️", "Whisper Transcription"),
    ("title", "🏷️", "Title Generation"),
    ("summary", "📋", "Meeting Summarization"),
    ("extract", "🔍", "Information Extraction"),
    ("rag", "🧠", "RAG Pipeline"),
]
LANGUAGE_OPTIONS = {"English": "en", "Hindi": "hi", "Auto-detect": None}
SUGGESTIONS = ["Summarise in 3 points", "What were the decisions?", "Who owns which task?"]

steps_placeholder = None  # filled while the pipeline runs


# ── Helpers ───────────────────────────────────────────────────────────────────
def render_steps_html() -> str:
    labels = {"active": "Processing…", "done": "Done", "pending": "Waiting"}
    rows = []
    for key, icon, label in PIPELINE_STEPS:
        s = st.session_state.pipeline_steps.get(key, "pending")
        rows.append(
            f'<div class="step"><div class="dot {s}"></div><span>{icon} {label}</span>'
            f'<span class="st">{labels[s]}</span></div>'
        )
    return "".join(rows)


def update_step(key: str, state: str):
    st.session_state.pipeline_steps[key] = state
    if steps_placeholder is not None:
        steps_placeholder.markdown(render_steps_html(), unsafe_allow_html=True)


def save_uploaded_file(uploaded_file) -> str:
    upload_dir = Path("downloads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=upload_dir) as f:
        f.write(uploaded_file.getbuffer())
        return f.name


def build_export_text(r: dict) -> str:
    line = "-" * 65
    return (
        f"AI MEETING ASSISTANT\n{'=' * 65}\n\nMEETING TITLE\n{line}\n{r['title']}\n\n"
        f"MEETING SUMMARY\n{line}\n{r['summary']}\n\nACTION ITEMS\n{line}\n{r['action_items']}\n\n"
        f"KEY DECISIONS\n{line}\n{r['key_decisions']}\n\nOPEN QUESTIONS\n{line}\n{r['open_questions']}\n\n"
        f"FULL TRANSCRIPT\n{line}\n{r['transcript']}\n\n{'=' * 65}\nGenerated using AI Meeting Assistant\n"
    )


def generate_pdf(r: dict) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("MeetingTitle", parent=styles["Title"], fontSize=20, leading=25,
                                 textColor=colors.HexColor("#4f46e5"), spaceAfter=20)
    heading_style = ParagraphStyle("SectionHeading", parent=styles["Heading2"], fontSize=13, leading=18,
                                   textColor=colors.HexColor("#3730a3"), spaceBefore=14, spaceAfter=8)
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9, leading=14, spaceAfter=8)
    story = [Paragraph("AI Meeting Assistant Report", title_style),
             Paragraph(html.escape(str(r["title"])), styles["Heading1"])]
    for heading, content in [
        ("Meeting Summary", r["summary"]), ("Action Items", r["action_items"]),
        ("Key Decisions", r["key_decisions"]), ("Open Questions", r["open_questions"]),
        ("Full Transcript", r["transcript"]),
    ]:
        story.append(Paragraph(heading, heading_style))
        story.append(Preformatted(str(content), body_style, maxLineLength=95))
        story.append(Spacer(1, 10))
    doc.build(story)
    return buffer.getvalue()


def card(title: str, body: str) -> str:
    return (f'<div class="card"><div class="card-title">{title}</div>'
            f'<div class="card-content">{html.escape(str(body))}</div></div>')


def count_lines(text) -> int:
    return len([l for l in str(text).splitlines() if l.strip()])


# ── Chat dialog (temporary popup) ─────────────────────────────────────────────
def chat_html(history, typing=False, streaming_text=None) -> str:
    """Build the animated chat transcript. Only the newest message re-animates."""
    parts = []
    for m in history:
        role = "user" if m["role"] == "user" else "bot"
        ava = "🙂" if role == "user" else "🤖"
        parts.append(f'<div class="msg {role}"><div class="ava">{ava}</div>'
                     f'<div class="bubble">{html.escape(m["content"])}</div></div>')
    if typing:
        parts.append('<div class="msg bot"><div class="ava">🤖</div><div class="bubble">'
                     '<span class="typing"><i></i><i></i><i></i></span></div></div>')
    if streaming_text is not None:
        parts.append(f'<div class="msg bot"><div class="ava">🤖</div><div class="bubble">'
                     f'{html.escape(streaming_text)}<span class="cursor"></span></div></div>')
    if not parts:
        parts.append('<div class="empty-chat">👋 Ask anything about this meeting.<br>'
                     'Try a suggestion below to get started.</div>')
    return f'<div class="chat-scroll"><div class="chat-inner">{"".join(parts)}</div></div>'


def _clear_chat():
    st.session_state.chat_history = []


def _queue_suggestion(text: str):
    st.session_state.pending_q = text


_dialog = getattr(st, "dialog", None) or getattr(st, "experimental_dialog")


@_dialog("Meeting Chat", width="large")
def chat_dialog():
    result = st.session_state.result
    st.markdown(
        '<div class="chat-head"><div class="avatar">🤖</div><div>'
        '<div style="font-weight:700">Meeting Assistant</div>'
        '<div class="muted" style="font-size:.8rem"><span class="online"></span>'
        'Answers from your transcript</div></div></div>',
        unsafe_allow_html=True,
    )

    chat_area = st.empty()
    chat_area.markdown(chat_html(st.session_state.chat_history), unsafe_allow_html=True)

    # Suggestion chips
    cols = st.columns(len(SUGGESTIONS))
    for col, s in zip(cols, SUGGESTIONS):
        col.button(s, key=f"sg_{s}", on_click=_queue_suggestion, args=(s,), use_container_width=True)

    with st.form("meeting_chat_form", clear_on_submit=True, border=False):
        c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
        typed = c1.text_input("Your question", placeholder="Ask about decisions, owners, deadlines…",
                              label_visibility="collapsed")
        sent = c2.form_submit_button("Send ➤", use_container_width=True)

    st.button("🗑️ Clear conversation", on_click=_clear_chat, key="clear_chat")

    question = st.session_state.pop("pending_q", None) or (typed.strip() if sent else "")
    if not question:
        return

    history = st.session_state.chat_history
    history.append({"role": "user", "content": question})

    # 1) message-sent animation, then typing indicator
    chat_area.markdown(chat_html(history, typing=True), unsafe_allow_html=True)
    try:
        answer = str(ask_question(result["rag_chain"], question))
    except Exception as e:
        history.pop()
        chat_area.markdown(chat_html(history), unsafe_allow_html=True)
        st.error(f"Could not generate an answer: {e}")
        return

    # 2) typewriter reveal of the reply
    words = answer.split(" ")
    step = max(1, len(words) // 60)
    for i in range(step, len(words) + step, step):
        chat_area.markdown(chat_html(history, streaming_text=" ".join(words[:i])), unsafe_allow_html=True)
        time.sleep(0.025)

    history.append({"role": "assistant", "content": answer})
    chat_area.markdown(chat_html(history), unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-logo">🎙️</div>Meeting AI</div>', unsafe_allow_html=True)
    st.caption("Turn recordings into clear next steps.")
    st.markdown("---")

    source = st.text_input("YouTube link", placeholder="https://youtube.com/watch?v=...")
    uploaded_file = st.file_uploader(
        "Or upload audio / video",
        type=["mp3", "wav", "m4a", "mp4", "mpeg", "mpga", "webm", "ogg", "flac"],
    )
    language_label = st.selectbox("Language", options=list(LANGUAGE_OPTIONS.keys()), index=0)
    language = LANGUAGE_OPTIONS[language_label]

    run_btn = st.button("⚡ Analyse meeting", use_container_width=True)

    if st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<span class="pill pill-ok">✓ Analysis complete</span>', unsafe_allow_html=True)
        st.markdown(render_steps_html(), unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-title">AI Meeting Assistant</div>'
    '<div class="hero-sub">Transcribe, summarise, extract and chat with any meeting.</div>',
    unsafe_allow_html=True,
)
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# ── Pipeline ──────────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip() and uploaded_file is None:
        st.error("Add a YouTube link or upload a file to continue.")
    elif source.strip() and uploaded_file is not None:
        st.warning("Use either a YouTube link or a file, not both.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        st.markdown('<div class="card-title">Processing your meeting</div>', unsafe_allow_html=True)
        steps_placeholder = st.empty()
        steps_placeholder.markdown(render_steps_html(), unsafe_allow_html=True)
        progress_bar = st.progress(0)

        try:
            update_step("audio", "active")
            input_source = save_uploaded_file(uploaded_file) if uploaded_file is not None else source.strip()
            chunks = process_input(input_source)
            if not chunks:
                raise ValueError("No audio chunks were generated.")
            update_step("audio", "done"); progress_bar.progress(15)

            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language=language)
            if not transcript or not transcript.strip():
                raise ValueError("Whisper returned an empty transcript.")
            update_step("transcript", "done"); progress_bar.progress(35)

            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done"); progress_bar.progress(45)

            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done"); progress_bar.progress(60)

            update_step("extract", "active")
            action_items = extract_action_items(transcript)
            decisions = extract_key_decisions(transcript)
            questions = extract_questions(transcript)
            update_step("extract", "done"); progress_bar.progress(80)

            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done"); progress_bar.progress(100)

            st.session_state.result = {
                "title": title, "transcript": transcript, "summary": summary,
                "action_items": action_items, "key_decisions": decisions,
                "open_questions": questions, "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            st.rerun()

        except Exception as e:
            for key, _, _ in PIPELINE_STEPS:
                if st.session_state.pipeline_steps.get(key) == "active":
                    update_step(key, "pending")
            st.error(f"Processing failed: {e}")


# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.result:
    result = st.session_state.result

    st.markdown(
        f'<div class="card"><span class="pill pill-brand">Meeting title</span>'
        f'<div style="font-size:1.6rem;font-weight:800;margin-top:.6rem;letter-spacing:-.01em">'
        f'{html.escape(str(result["title"]))}</div></div>',
        unsafe_allow_html=True,
    )

    # Stats row
    s1, s2, s3, s4 = st.columns(4, gap="medium")
    for col, num, label in [
        (s1, len(str(result["transcript"]).split()), "Words transcribed"),
        (s2, count_lines(result["action_items"]), "Action items"),
        (s3, count_lines(result["key_decisions"]), "Key decisions"),
        (s4, count_lines(result["open_questions"]), "Open questions"),
    ]:
        col.markdown(f'<div class="stat"><div class="stat-num">{num:,}</div>'
                     f'<div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Chat call-to-action → opens temporary dialog
    cta_text, cta_btn = st.columns([4, 1.3], vertical_alignment="center")
    cta_text.markdown(
        '<div class="cta"><h3>💬 Chat with this meeting</h3>'
        '<p>Ask follow-up questions and get answers grounded in the transcript.</p></div>',
        unsafe_allow_html=True,
    )
    if cta_btn.button("Open chat", use_container_width=True, key="open_chat"):
        chat_dialog()

    tab_overview, tab_insights, tab_transcript, tab_export = st.tabs(
        ["📋 Overview", "💡 Insights", "📝 Transcript", "📥 Export"]
    )

    with tab_overview:
        st.markdown(card("📋 Meeting summary", result["summary"]), unsafe_allow_html=True)

    with tab_insights:
        c1, c2, c3 = st.columns(3, gap="medium")
        c1.markdown(card("✅ Action items", result["action_items"]), unsafe_allow_html=True)
        c2.markdown(card("🔑 Key decisions", result["key_decisions"]), unsafe_allow_html=True)
        c3.markdown(card("❓ Open questions", result["open_questions"]), unsafe_allow_html=True)

    with tab_transcript:
        st.markdown(
            f'<div class="transcript-box">{html.escape(str(result["transcript"]))}</div>',
            unsafe_allow_html=True,
        )

    with tab_export:
        st.markdown('<div class="card-title">Download your report</div>', unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        e1.download_button("📄 Download TXT", data=build_export_text(result),
                           file_name="meeting_report.txt", mime="text/plain", use_container_width=True)
        e2.download_button("📕 Download PDF", data=generate_pdf(result),
                           file_name="meeting_report.pdf", mime="application/pdf", use_container_width=True)

# ── Empty state ───────────────────────────────────────────────────────────────
elif not run_btn:
    st.markdown(
        """
        <div class="card" style="text-align:center;padding:3.5rem 2rem">
            <div style="font-size:3.5rem">🎙️</div>
            <div style="font-size:1.5rem;font-weight:800;margin:.6rem 0 .4rem">Ready when you are</div>
            <div class="muted" style="max-width:440px;margin:0 auto;line-height:1.7">
                Paste a YouTube link or upload a recording in the sidebar,
                choose a language, then click <b>Analyse meeting</b>.
            </div>
            <div style="margin-top:1.4rem;display:flex;gap:.6rem;justify-content:center;flex-wrap:wrap">
                <span class="pill pill-brand">Whisper transcription</span>
                <span class="pill pill-ok">Gemini summaries</span>
                <span class="pill pill-warn">RAG chat</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )