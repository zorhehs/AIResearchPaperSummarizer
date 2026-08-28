import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/summarize"
CHAT_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="AI Research Paper Summarizer", page_icon="📄", layout="wide")

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tabler-icons/2.44.0/iconfont/tabler-icons.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-main: #F8FAFC;
        --text-main: #0F172A;
        --text-muted: #64748B;
        --card-bg: #FFFFFF;
        --border-color: #E2E8F0;
        --radius: 12px;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: var(--bg-main);
    }
    
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; max-width: 1100px; }

    /* Headers */
    .app-title { font-size: 2.2rem; font-weight: 700; color: var(--text-main); margin-bottom: 0.2rem; letter-spacing: -0.02em; }
    .app-subtitle { color: var(--text-muted); font-size: 1.05rem; margin-bottom: 2rem; }
    .paper-title { font-size: 1.6rem; font-weight: 700; color: var(--text-main); margin: 1rem 0 10px 0; line-height: 1.3;}
    
    /* Sidebar */
    .sidebar-card {
        background: var(--card-bg); border: 1px solid var(--border-color);
        border-radius: var(--radius); padding: 1.2rem; margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .sidebar-heading { font-weight: 600; font-size: 0.95rem; color: var(--text-main); margin-bottom: 8px; }
    .sidebar-text { font-size: 0.85rem; color: var(--text-muted); line-height: 1.6; }
    .sidebar-recent {
        font-size: 0.8rem; padding: 8px 10px; border-radius: 6px;
        background: var(--bg-main); margin-bottom: 6px; color: var(--text-main);
        border: 1px solid var(--border-color);
    }
    .sidebar-footer { font-size: 0.75rem; color: #94A3B8; margin-top: 8px; }

    /* Tags & Metrics */
    .tag {
        display: inline-flex; align-items: center; gap: 4px;
        background: var(--card-bg); color: var(--text-muted); border: 1px solid var(--border-color);
        padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 500; margin-right: 8px;
    }
    div[data-testid="stMetric"] {
        background: var(--card-bg); border: 1px solid var(--border-color);
        border-radius: var(--radius); padding: 1.2rem 1.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    div[data-testid="stMetric"] label { font-weight: 500 !important; color: var(--text-muted) !important; }

    /* Custom Result Cards */
    .result-card {
        background: var(--card-bg); border: 1px solid var(--border-color);
        border-radius: 16px; padding: 24px; margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .result-card-header {
        display: flex; align-items: center; gap: 8px;
        font-size: 1.15rem; font-weight: 600; margin-bottom: 14px; color: var(--text-main);
        border-bottom: 1px solid #F1F5F9; padding-bottom: 12px;
    }
    .result-card-body { font-size: 0.95rem; color: #334155; line-height: 1.7; }
    
    div.stButton > button[kind="primary"] {
        background: var(--text-main); border: none; font-weight: 500; border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-heading"><i class="ti ti-file-description"></i>&nbsp; About</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-text">Upload a research paper or paste a DOI to get a structured summary covering core contributions, methodology, gaps, findings, and future directions.</div>', unsafe_allow_html=True)
    
    if st.session_state.history:
        st.markdown('<hr style="margin:16px 0; border-color:var(--border-color);">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-heading" style="font-size:0.85rem;">Recent Papers</div>', unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-5:]):
            short = item[:38] + "..."if len(item) > 38 else item
            st.markdown(f'<div class="sidebar-recent"><i class="ti ti-file-text"></i>&nbsp; {short}</div>', unsafe_allow_html=True)
            
    st.markdown('<hr style="margin:16px 0; border-color:var(--border-color);">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-footer">Built for NUST Research Directorate</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="app-title">AI Research Paper Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Upload a PDF or paste a DOI to get a structured summary.</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Upload PDF", "Enter DOI"])
result = None

with tab1:
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")
    if uploaded_file and st.button("Summarize PDF", type="primary"):
        with st.status("🧠 Processing Document...", expanded=True) as status:
            st.write("Extracting and condensing text from PDF...")
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(),"application/pdf")}
            st.write("Running local AI inference...")
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                st.write("Parsing structured output...")
                result = response.json()
                st.session_state.chat_history = []
                st.session_state.history.append(result.get("title") or uploaded_file.name)
                status.update(label="Analysis Complete!", state="complete", expanded=False)
            else:
                status.update(label="Generation Failed", state="error", expanded=True)
                st.error("Processing failed.")

with tab2:
    doi_input = st.text_input("Enter a DOI", placeholder="e.g. 10.1371/journal.pone.0121283", label_visibility="collapsed")
    if doi_input and st.button("Summarize DOI", type="primary"):
        with st.status("🔍 Resolving DOI...", expanded=True) as status:
            st.write("Fetching paper content from DOI...")
            response = requests.post(API_URL, data={"doi": doi_input})
            if response.status_code == 200:
                result = response.json()
                st.session_state.chat_history = []
                st.session_state.history.append(result.get("title") or doi_input)
                status.update(label="Analysis Complete!", state="complete", expanded=False)
            else:
                status.update(label="Generation Failed", state="error", expanded=True)
                st.error("Processing failed.")

if result:
    st.markdown("<hr style='border-color: var(--border-color); margin-top: 2rem;'>", unsafe_allow_html=True)
    st.markdown(f'<div class="paper-title">{result.get("title", "Untitled paper")}</div>', unsafe_allow_html=True)

    original_words = len(result.get("full_text", "").split())
    summary_words = sum(len(str(result[k]).split()) for k in ["summary", "methodology", "research_gaps", "findings", "future_work"])
    reduction_pct = round((1 - summary_words / original_words) * 100) if original_words else 0
    read_minutes = max(1, round(summary_words / 200))

    st.markdown(
        f'<span class="tag"><i class="ti ti-file"></i> Source: {result.get("source", "PDF")}</span>'
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
        card("ti-notes", "Summary", result.get("summary", ""))
        card("ti-flask", "Methodology", result.get("methodology", ""))
        card("ti-search", "Research Gaps", result.get("research_gaps", ""))

    with col2:
        card("ti-chart-bar", "Findings", result.get("findings", ""))
        card("ti-rocket", "Future Work", result.get("future_work", ""))

    st.markdown("<hr style='border-color: var(--border-color); margin-top: 1rem;'>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-heading" style="font-size:1.1rem; margin-bottom:14px;"><i class="ti ti-message-circle"></i> Ask about this paper</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "paper_full_text" not in st.session_state:
        st.session_state.paper_full_text = result.get("full_text", result.get("summary", ""))

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

    if question := st.chat_input("Ask a question about this paper..."):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            chat_response = requests.post(CHAT_URL, json={
                "paper_text": st.session_state.paper_full_text,
                "question": question,
                "chat_history": st.session_state.chat_history,
            })
            if chat_response.status_code == 200:
                st.session_state.chat_history.append({"role": "assistant", "content": chat_response.json()["answer"]})
                st.rerun()
else:
    st.info("Upload a PDF or enter a DOI above to get started.")
