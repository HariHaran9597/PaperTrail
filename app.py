"""
PaperTrail — Streamlit Frontend
Beautiful, tabbed interface showcasing all agents' output.
Features: arXiv URL input, PDF upload fallback, HITL novelty review,
          Research Thread multi-paper synthesis, PDF export.
"""

import streamlit as st
from graph.pipeline import analyze_paper, analyze_pdf
from utils.pdf_report import generate_pdf_report
from agents.research_thread import ResearchThreadAgent
from config import GROQ_API_KEY, GROQ_MODEL, MISSING_INDEX_MESSAGE
import streamlit.components.v1 as components
import os
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ─── Page Config ───
st.set_page_config(
    page_title="PaperTrail — Research Paper Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for Premium Look ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp { font-family: 'Inter', sans-serif; }
    
    .hero-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        border-radius: 20px;
        padding: 40px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.3) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.5rem; font-weight: 800; color: #ffffff;
        margin: 0 0 8px 0; position: relative; z-index: 1;
    }
    .hero-subtitle {
        font-size: 1.1rem; color: #c7d2fe; margin: 0;
        position: relative; z-index: 1; font-weight: 400;
    }
    
    .paper-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border-radius: 16px; padding: 28px; color: white; margin: 16px 0;
        border: 1px solid rgba(139, 92, 246, 0.3);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    }
    .paper-card h2 { font-size: 1.4rem; font-weight: 700; margin-bottom: 8px; line-height: 1.3; }
    .paper-card .authors { color: #c7d2fe; font-size: 0.95rem; margin-bottom: 12px; }
    .paper-card .meta { color: #a5b4fc; font-size: 0.85rem; }
    
    .explanation-card {
        background: #f8fafc; border-left: 4px solid;
        border-radius: 0 12px 12px 0; padding: 20px; margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .eli5-card { border-color: #f59e0b; background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); }
    .undergrad-card { border-color: #3b82f6; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); }
    .expert-card { border-color: #8b5cf6; background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%); }
    .explanation-card h4 { margin: 0 0 12px 0; font-weight: 700; }
    .explanation-card p { margin: 0; line-height: 1.6; color: #334155; }
    
    .novelty-badge {
        display: inline-block; padding: 4px 14px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 700; margin-right: 8px; letter-spacing: 0.5px;
    }
    .badge-novel { background: #dcfce7; color: #166534; }
    .badge-incremental { background: #fef3c7; color: #92400e; }
    .badge-builds { background: #e0e7ff; color: #3730a3; }
    
    .score-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border-radius: 16px; padding: 20px 24px; text-align: center; margin: 8px 0 16px 0;
    }
    .score-number { font-size: 3rem; font-weight: 800; color: #a5b4fc; }
    .score-label { color: #c7d2fe; font-size: 0.9rem; }
    
    .insight-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac; border-radius: 12px; padding: 16px 20px; margin: 12px 0;
    }
    .insight-card p { margin: 0; color: #166534; font-weight: 500; }
    
    .question-item {
        background: #f8fafc; border-radius: 10px; padding: 12px 16px;
        margin: 8px 0; border-left: 3px solid #6366f1;
    }
    
    .thread-card {
        background: linear-gradient(135deg, #fdf4ff 0%, #fae8ff 100%);
        border: 1px solid #d946ef; border-radius: 12px; padding: 20px; margin: 12px 0;
    }
    .thread-card h4 { color: #86198f; margin: 0 0 8px 0; }
    
    .hitl-box {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
        border: 1px solid #fb923c; border-radius: 12px; padding: 16px 20px; margin: 16px 0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .sidebar-header { font-size: 1.1rem; font-weight: 700; color: #6366f1; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


# ─── Hero Header ───
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🔬 PaperTrail</div>
    <div class="hero-subtitle">Understand any research paper in 30 seconds — powered by 6 AI agents</div>
</div>
""", unsafe_allow_html=True)


# ─── Session State ───
if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "thread_result" not in st.session_state:
    st.session_state.thread_result = None


# ─── Sidebar ───
with st.sidebar:
    st.markdown('<div class="sidebar-header">📚 Quick Examples</div>', unsafe_allow_html=True)
    st.caption("Click to analyze a famous paper:")

    example_papers = {
        "Attention Is All You Need": "1706.03762",
        "BERT": "1810.04805",
        "Vision Transformer": "2010.11929",
    }

    for name, arxiv_id in example_papers.items():
        if st.button(f"📄 {name}", key=f"ex_{arxiv_id}", use_container_width=True):
            st.session_state.selected_url = f"https://arxiv.org/abs/{arxiv_id}"

    st.divider()

    # History
    if st.session_state.history:
        st.markdown('<div class="sidebar-header">📜 Analysis History</div>', unsafe_allow_html=True)
        for i, item in enumerate(reversed(st.session_state.history)):
            if st.button(
                f"🔬 {item['title'][:40]}...",
                key=f"hist_{i}",
                use_container_width=True,
            ):
                st.session_state.current_result = item["result"]

    st.divider()
    if not GROQ_API_KEY:
        st.warning("Set GROQ_API_KEY in .env before analyzing papers.")
    st.caption("Built with LangGraph, Groq, FAISS, and Streamlit")
    st.caption(f"Groq model: {GROQ_MODEL}")


# ═══════════════════════════════════════════════════
# INPUT SECTION — arXiv URL + PDF Upload Fallback
# ═══════════════════════════════════════════════════

input_mode = st.radio(
    "Input method:",
    ["🔗 arXiv URL", "📄 Upload PDF"],
    horizontal=True,
    label_visibility="collapsed",
)

if input_mode == "🔗 arXiv URL":
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        default_url = st.session_state.get("selected_url", "")
        url = st.text_input(
            "Paste an arXiv URL or paper ID:",
            value=default_url,
            placeholder="https://arxiv.org/abs/1706.03762",
            label_visibility="collapsed",
        )
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

    # arXiv analysis
    if analyze_clicked and url:
        if "selected_url" in st.session_state:
            del st.session_state.selected_url

        with st.spinner("🤖 5 AI agents analyzing your paper..."):
            try:
                result = analyze_paper(url)
                if result.get("error"):
                    st.error(f"❌ Error: {result['error']}")
                    st.info("💡 If arXiv is down, try uploading the PDF directly using the '📄 Upload PDF' option above.")
                else:
                    st.session_state.current_result = result
                    st.session_state.history.append({
                        "title": result["parsed_paper"]["parsed"]["title"],
                        "url": url,
                        "result": result,
                    })
                    st.rerun()
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("💡 Try uploading the PDF directly, or check your GROQ_API_KEY.")

else:
    # PDF Upload fallback
    st.caption("📄 Upload any research paper PDF — works even when arXiv is down!")
    uploaded_file = st.file_uploader(
        "Upload a research paper PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file and st.button("🔍 Analyze PDF", type="primary", use_container_width=True):
        with st.spinner("🤖 5 AI agents analyzing your uploaded paper..."):
            try:
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                result = analyze_pdf(tmp_path, uploaded_file.name)

                if result.get("error"):
                    st.error(f"❌ Error: {result['error']}")
                else:
                    st.session_state.current_result = result
                    st.session_state.history.append({
                        "title": result["parsed_paper"]["parsed"]["title"],
                        "url": f"uploaded: {uploaded_file.name}",
                        "result": result,
                    })
                    st.rerun()
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
            finally:
                # Clean up temp file
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)


# ═══════════════════════════════════════════════════
# DISPLAY RESULTS
# ═══════════════════════════════════════════════════

result = st.session_state.current_result

if result and not result.get("error"):
    parsed = result["parsed_paper"]["parsed"]
    metadata = result["parsed_paper"]["metadata"]

    # ─── Paper Info Card ───
    authors_list = metadata.get("authors", ["Unknown"])
    if isinstance(authors_list, list):
        authors_str = ", ".join(authors_list[:5])
        if len(authors_list) > 5:
            authors_str += f" + {len(authors_list) - 5} more"
    else:
        authors_str = str(authors_list)

    entry_url = metadata.get("entry_url", "")
    arxiv_link = f' &nbsp;|&nbsp; 🔗 <a href="{entry_url}" target="_blank" style="color: #a5b4fc;">arXiv</a>' if entry_url else ""
    source_label = "📤 Uploaded" if metadata.get("source") == "pdf_upload" else f"📅 {metadata.get('published', 'N/A')[:10]}"

    st.markdown(f"""
    <div class="paper-card">
        <h2>{parsed['title']}</h2>
        <div class="authors">{authors_str}</div>
        <div class="meta">
            {source_label} &nbsp;|&nbsp; 
            🏷️ {', '.join(metadata.get('categories', ['N/A']))} &nbsp;|&nbsp; 
            📝 {parsed['paper_type'].capitalize()}{arxiv_link}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Tabs ───
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📖 Explanations",
        "🆕 What's New",
        "🕸️ Concept Map",
        "❓ Questions",
        "📥 Export",
    ])

    # ═══════════════════════════════════════
    # TAB 1: Explanations
    # ═══════════════════════════════════════
    with tab1:
        exp = result.get("explanations", {})
        if exp:
            st.markdown(f"**💡 In one sentence:** {exp.get('one_sentence', 'N/A')}")
            st.markdown(f"""
            <div class="insight-card">
                <p>🔑 <strong>Key Insight:</strong> {exp.get('key_insight', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
            st.divider()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="explanation-card eli5-card">
                    <h4>🧒 ELI5</h4>
                    <p>{exp.get('eli5', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="explanation-card undergrad-card">
                    <h4>🎓 Undergrad</h4>
                    <p>{exp.get('undergrad', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="explanation-card expert-card">
                    <h4>🔬 Expert</h4>
                    <p>{exp.get('expert', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)

    # ═══════════════════════════════════════
    # TAB 2: Novelty Analysis + HITL
    # ═══════════════════════════════════════
    with tab2:
        nov = result.get("novelty_analysis", {})
        if nov:
            if nov.get("requires_index"):
                st.warning(MISSING_INDEX_MESSAGE)
                st.info("After building the index, rerun the paper analysis to retrieve related papers and calculate a novelty score.")
            else:
                # AI Score display
                ai_score = nov.get("novelty_score", 5)
                col_s1, col_s2 = st.columns([1, 3])

                with col_s1:
                    st.markdown(f"""
                    <div class="score-container">
                        <div class="score-number">{ai_score}/10</div>
                        <div class="score-label">AI Novelty Score</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_s2:
                    st.markdown(f"**Summary:** {nov.get('novelty_summary', 'N/A')}")
                    if ai_score >= 8:
                        st.success("🌟 Highly novel — introduces significant new ideas")
                    elif ai_score >= 5:
                        st.info("📊 Solid contribution — meaningful improvements with some new ideas")
                    else:
                        st.warning("📈 Incremental — builds upon existing work with minor improvements")

                # ─── HITL: Human-in-the-Loop Novelty Review ───
                st.markdown("""
                <div class="hitl-box">
                    <h4 style="color: #9a3412; margin: 0 0 4px 0;">👤 Human Review (Optional)</h4>
                    <p style="color: #78350f; margin: 0; font-size: 0.85rem;">
                        Disagree with the AI's assessment? Adjust the score and add your notes below.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                col_h1, col_h2 = st.columns([1, 3])
                with col_h1:
                    human_score = st.slider(
                        "Your novelty score",
                        min_value=1, max_value=10,
                        value=ai_score,
                        key="human_novelty_score",
                    )
                with col_h2:
                    human_notes = st.text_area(
                        "Your review notes (optional)",
                        placeholder="e.g., 'The attention mechanism was truly novel at the time, but the architecture borrows heavily from...'",
                        height=80,
                        key="human_novelty_notes",
                    )

                if human_score != ai_score:
                    st.info(f"📝 Your score: **{human_score}/10** (AI said {ai_score}/10). Noted for your records.")

                st.divider()

                # Novel contributions
                st.markdown("### ✨ Novel Contributions")
                for c in nov.get("novel_contributions", []):
                    st.markdown(f'<span class="novelty-badge badge-novel">NEW</span> {c}', unsafe_allow_html=True)

                st.markdown("### 📈 Incremental Improvements")
                for c in nov.get("incremental_improvements", []):
                    st.markdown(f'<span class="novelty-badge badge-incremental">IMPROVED</span> {c}', unsafe_allow_html=True)

                st.markdown("### 📚 Builds Upon")
                for c in nov.get("builds_upon", []):
                    st.markdown(f'<span class="novelty-badge badge-builds">PRIOR</span> {c}', unsafe_allow_html=True)

                related = result.get("related_papers", [])
                if related:
                    st.divider()
                    st.markdown("### 🔗 Related Papers from Literature")
                    for p in related:
                        score_val = p.get("similarity_score", 0)
                        year = p.get("published", "N/A")[:4]
                        st.markdown(f"- **{p['title']}** ({year}) - similarity: `{score_val:.3f}`")

    # ═══════════════════════════════════════
    # TAB 3: Concept Map
    # ═══════════════════════════════════════
    with tab3:
        graph_path = result.get("graph_html_path")
        concept_map = result.get("concept_map", {})

        if graph_path and os.path.exists(graph_path):
            st.markdown("### 🕸️ Interactive Knowledge Graph")
            st.caption("Drag nodes to rearrange • Hover for details • Scroll to zoom")

            with open(graph_path, "r", encoding="utf-8") as f:
                graph_html = f.read()
            components.html(graph_html, height=580, scrolling=True)

            st.markdown("**Legend:**")
            legend_cols = st.columns(6)
            legend_items = [
                ("🔷 Method", "#6366f1"), ("🟡 Dataset", "#f59e0b"),
                ("🟢 Metric", "#10b981"), ("🔵 Concept", "#3b82f6"),
                ("🔴 Result", "#ef4444"), ("🟣 Problem", "#8b5cf6"),
            ]
            for col, (label, color) in zip(legend_cols, legend_items):
                col.markdown(f"<span style='color:{color}'>●</span> {label.split(' ', 1)[1]}", unsafe_allow_html=True)
        elif concept_map.get("nodes"):
            st.warning("Graph visualization file not found. Showing raw concept data:")
            st.json(concept_map)
        else:
            st.info("No concept map was generated for this paper.")

    # ═══════════════════════════════════════
    # TAB 4: Questions
    # ═══════════════════════════════════════
    with tab4:
        q = result.get("questions", {})
        if q:
            st.markdown("### ✅ Questions This Paper Answers")
            for question in q.get("questions_answered", []):
                st.markdown(f'<div class="question-item">✅ {question}</div>', unsafe_allow_html=True)

            st.markdown("### 🔮 Questions Left Open")
            for question in q.get("questions_left_open", []):
                st.markdown(f'<div class="question-item" style="border-color: #f59e0b;">🔮 {question}</div>', unsafe_allow_html=True)

            st.markdown("### 📖 Suggested Follow-Up Reading")
            for item in q.get("follow_up_reading", []):
                st.markdown(f"- 📗 {item}")

            st.divider()
            st.markdown("### 💬 Discussion Questions")
            for question in q.get("discussion_questions", []):
                st.markdown(f'<div class="question-item" style="border-color: #8b5cf6;">💬 {question}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════
    # TAB 5: Export
    # ═══════════════════════════════════════
    with tab5:
        st.markdown("### 📥 Download Analysis Report")
        st.caption("Get a professional PDF report with all analysis results.")

        if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                try:
                    pdf_path = generate_pdf_report(result)
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=f"PaperTrail_{parsed['title'][:30].replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.success("✅ PDF generated successfully!")
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")

        st.divider()
        st.markdown("### 📊 Raw Analysis Data")
        with st.expander("View raw JSON output"):
            st.json(result)

elif result and result.get("error"):
    st.error(f"❌ Analysis failed: {result['error']}")

else:
    # Empty state
    st.markdown("---")
    col_empty1, col_empty2, col_empty3 = st.columns(3)
    with col_empty1:
        st.markdown("### 📖 Layered Explanations")
        st.caption("ELI5 → Undergrad → Expert level explanations")
    with col_empty2:
        st.markdown("### 🆕 Novelty Detection")
        st.caption("RAG-powered analysis of what's genuinely new")
    with col_empty3:
        st.markdown("### 🕸️ Concept Maps")
        st.caption("Interactive knowledge graph visualization")
    st.markdown("---")
    st.markdown(
        "<center><p style='color: #94a3b8;'>Paste an arXiv URL or upload a PDF above to get started!</p></center>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════
# RESEARCH THREAD — Multi-Paper Synthesis (Agent 6)
# ═══════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div class="thread-card">
    <h4>🧵 Research Thread — Multi-Paper Synthesis</h4>
    <p style="color: #86198f; margin: 0; font-size: 0.9rem;">
        Analyze 2-5 papers together to find common themes, contradictions, 
        idea evolution, and gaps for future work.
    </p>
</div>
""", unsafe_allow_html=True)

# Check if we have enough papers in history
history_papers = st.session_state.history
if len(history_papers) >= 2:
    st.caption(f"You have {len(history_papers)} papers in your history. Select 2-5 to synthesize:")

    # Paper selection checkboxes
    selected_indices = []
    for i, item in enumerate(history_papers):
        if st.checkbox(f"📄 {item['title'][:60]}", key=f"thread_select_{i}"):
            selected_indices.append(i)

    num_selected = len(selected_indices)

    if num_selected >= 2:
        if num_selected > 5:
            st.warning("Please select at most 5 papers for synthesis.")
        else:
            if st.button(f"🧵 Synthesize {num_selected} Papers", type="primary", use_container_width=True):
                with st.spinner(f"🤖 Agent 6 synthesizing {num_selected} papers into a research thread..."):
                    try:
                        # Get parsed papers from selected history items
                        selected_papers = [
                            history_papers[i]["result"]["parsed_paper"]
                            for i in selected_indices
                        ]

                        thread_agent = ResearchThreadAgent()
                        thread_result = thread_agent.synthesize(selected_papers)
                        st.session_state.thread_result = thread_result

                    except Exception as e:
                        st.error(f"❌ Research Thread synthesis failed: {str(e)}")

    elif num_selected == 1:
        st.info("Select at least 2 papers to generate a Research Thread.")

    # Display thread result
    thread = st.session_state.thread_result
    if thread:
        st.markdown(f"## 🧵 {thread.get('thread_title', 'Research Thread')}")

        st.markdown(f"### 📝 State of the Field")
        st.markdown(f"> {thread.get('field_summary', 'N/A')}")

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown("### 🔗 Common Themes")
            for theme in thread.get("common_themes", []):
                st.markdown(f"- 🔗 {theme}")

            st.markdown("### ✅ Consensus Findings")
            for finding in thread.get("consensus_findings", []):
                st.markdown(f"- ✅ {finding}")

            st.markdown("### 📈 Idea Evolution")
            for evolution in thread.get("idea_evolution", []):
                st.markdown(f"- 📈 {evolution}")

        with col_t2:
            st.markdown("### ⚡ Contradictions")
            for contradiction in thread.get("contradictions", []):
                st.markdown(f"- ⚡ {contradiction}")
            if not thread.get("contradictions"):
                st.caption("No major contradictions found.")

            st.markdown("### 🔮 Open Gaps")
            for gap in thread.get("open_gaps", []):
                st.markdown(f"- 🔮 {gap}")

            st.markdown("### 🚀 Recommended Next Steps")
            for step in thread.get("recommended_next_steps", []):
                st.markdown(f"- 🚀 {step}")

else:
    st.caption("💡 Analyze 2+ papers first, then select them here to generate a Research Thread synthesis.")
