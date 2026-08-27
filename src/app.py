import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/summarize"
CHAT_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="AI Research Paper Summarizer", page_icon="📄", layout="wide")

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tabler-icons/2.44.0/iconfont/tabler-icons.min.css">
<style>
    :root {
        --accent: #185FA5;
        --accent-bg: #E6F1FB;
        --accent-text: #0C447C;
        --success-bg: #EAF3DE;
        --success-text: #3B6D11;
        --surface-1: #F7F8FA;
        --surface-2: #FFFFFF;
        --border: #E5E7EB;
        --text-secondary: #5F5E5A;
        --text-muted: #888780;
        --radius: 8px;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; max-width: 1100px; }

    .app-title { font-size: 1.9rem; font-weight: 700; margin-bottom: 0.1rem; }
    .app-subtitle { color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 1.5rem; }

    .sidebar-card {
        background: var(--surface-1);
        border-radius: 12px;
        padding: 1rem;
    }
    .sidebar-heading { font-weight: 600; font-size: 0.95rem; margin-bottom: 6px; }
    .sidebar-text { font-size: 0.82rem; color: var(--text-secondary); line-height: 1.6; }
    .sidebar-recent {
        font-size: 0.78rem; padding: 6px 8px; border-radius: 6px;
        background: var(--surface-2); margin-bottom: 4px; color: var(--text-secondary);
    }
    .sidebar-footer { font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; }

    .paper-title { font-size: 1.4rem; font-weight: 700; margin: 0 0 6px; }
    .tag {
        display: inline-block; background: var(--surface-1); color: var(--text-secondary);
        padding: 3px 12px; border-radius: 12px; font-size: 0.78rem; margin-right: 6px;
    }

    .result-card {
        background: var(--surface-2); border: 1px solid var(--border);
        border-radius: 12px; padding: 14px 18px; margin-bottom: 12px;
    }
    .result-card-header {
        display: flex; align-items: center; gap: 6px;
        font-size: 0.95rem; font-weight: 600; margin-bottom: 6px; color: var(--accent-text);
    }
    .result-card-body { font-size: 0.88rem; color: var(--text-secondary); line-height: 1.65; }

    div[data-testid="stMetric"] {
        background: var(--surface-1); border-radius: var(--radius); padding: 0.8rem 1rem;
    }
    div[data-testid="stMetric"]:nth-of-type(3) {
        background: var(--success-bg);
    }

    .stTabs [data-baseweb="tab"] { padding: 8px 20px; font-weight: 500; }
    div.stButton > button[kind="primary"] {
        background: #1a1a1a; border: none; font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-heading"><i class="ti ti-file-description"></i>&nbsp; About</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-text">Upload a research paper or paste a DOI to get a structured summary '
        'covering contribution, methodology, gaps, findings, and future directions.</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.history:
        st.markdown('<hr style="margin:12px 0; border-color:var(--border);">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-heading" style="font-size:0.8rem;">Recent papers</div>', unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-5:]):
            short = item[:38] + "..." if len(item) > 38 else item
            st.markdown(f'<div class="sidebar-recent"><i class="ti ti-file-text"></i>&nbsp; {short}</div>', unsafe_allow_html=True)
    st.markdown('<hr style="margin:12px 0; border-color:var(--border);">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-footer">Built for NUST Research Directorate</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-footer">Backend: FastAPI · Llama 3.3 70B</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="app-title"><i class="ti ti-file-text" style="color:var(--accent);"></i>&nbsp; AI Research Paper Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Upload a PDF or paste a DOI to get a structured summary.</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Upload PDF", "Enter DOI"])
result = None

with tab1:
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")
    if uploaded_file:
        st.caption(f"Selected: **{uploaded_file.name}** ({uploaded_file.size / 1024:.0f} KB)")
    if uploaded_file and st.button("Summarize PDF", type="primary"):
        with st.spinner("Reading and summarizing your paper... this can take 30-60 seconds."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                result = response.json()
                st.session_state.chat_history = []
                st.session_state.history.append(result.get("title") or uploaded_file.name)
            else:
                try:
                    detail = response.json().get("detail", "Unknown error")
                except Exception:
                    detail = response.text or "(empty response from server)"
                st.error(f"Error {response.status_code}: {detail}")

with tab2:
    doi_input = st.text_input("Enter a DOI", placeholder="e.g. 10.1371/journal.pone.0121283", label_visibility="collapsed")
    if doi_input and st.button("Summarize DOI", type="primary"):
        with st.spinner("Resolving DOI and summarizing... this can take 30-60 seconds."):
            response = requests.post(API_URL, data={"doi": doi_input})
            if response.status_code == 200:
                result = response.json()
                st.session_state.chat_history = []
                st.session_state.history.append(result.get("title") or doi_input)
            else:
                try:
                    detail = response.json().get("detail", "Unknown error")
                except Exception:
                    detail = response.text or "(empty response from server)"
                st.error(f"Error {response.status_code}: {detail}")

if result:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<div class="paper-title">{result["title"] or "Untitled paper"}</div>', unsafe_allow_html=True)

    original_words = len(result.get("full_text", "").split())
    summary_words = sum(len(result[k].split()) for k in ["summary", "methodology", "research_gaps", "findings", "future_work"])
    reduction_pct = round((1 - summary_words / original_words) * 100) if original_words else 0
    read_minutes = max(1, round(summary_words / 200))

    st.markdown(
        f'<span class="tag"><i class="ti ti-file"></i> Source: {result["source"]}</span>'
        f'<span class="tag"><i class="ti ti-clock"></i> ~{read_minutes} min read</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3 = st.columns(3)
    c1.metric("Original length", f"{original_words:,} words")
    c2.metric("Summary length", f"{summary_words:,} words")
    c3.metric("Time saved", f"~{reduction_pct}%")

    st.write("")
    col1, col2 = st.columns(2)

    def card(icon, title, body):
        st.markdown(f'''
        <div class="result-card">
            <div class="result-card-header"><i class="ti {icon}"></i> {title}</div>
            <div class="result-card-body">{body}</div>
        </div>
        ''', unsafe_allow_html=True)

    with col1:
        card("ti-notes", "Summary", result["summary"])
        card("ti-flask", "Methodology", result["methodology"])
        card("ti-search", "Research Gaps", result["research_gaps"])

    with col2:
        card("ti-chart-bar", "Findings", result["findings"])
        card("ti-rocket", "Future Work", result["future_work"])

    full_report = f"""# {result['title']}

## Summary
{result['summary']}

## Methodology
{result['methodology']}

## Research Gaps
{result['research_gaps']}

## Findings
{result['findings']}

## Future Work
{result['future_work']}
"""
    st.download_button("Download summary as text", data=full_report, file_name="paper_summary.txt", mime="text/plain")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-heading" style="font-size:1.05rem; margin-bottom:10px;"><i class="ti ti-message-circle"></i>&nbsp; Ask about this paper</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "paper_full_text" not in st.session_state:
        st.session_state.paper_full_text = result.get("full_text", result["summary"])

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    question = st.chat_input("Ask a question about this paper...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            chat_response = requests.post(
                CHAT_URL,
                json={
                    "paper_text": st.session_state.paper_full_text,
                    "question": question,
                    "chat_history": st.session_state.chat_history,
                },
            )
            if chat_response.status_code == 200:
                answer = chat_response.json()["answer"]
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()
            else:
                st.error(f"Chat failed: {chat_response.status_code}")
else:
    st.info("Upload a PDF or enter a DOI above to get started.")
