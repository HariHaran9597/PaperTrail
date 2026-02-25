"""
PaperTrail — Streamlit Frontend
Beautiful, tabbed interface showcasing all 5 agents' output.
"""

import streamlit as st
from graph.pipeline import analyze_paper
from utils.pdf_report import generate_pdf_report
import streamlit.components.v1 as components
import os
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
    
    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Header */
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
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.3) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 8px 0;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #c7d2fe;
        margin: 0;
        position: relative;
        z-index: 1;
        font-weight: 400;
    }
    
    /* Paper Info Card */
    .paper-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border-radius: 16px;
        padding: 28px;
        color: white;
        margin: 16px 0;
        border: 1px solid rgba(139, 92, 246, 0.3);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    }
    
    .paper-card h2 {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 8px;
        line-height: 1.3;
    }
    
    .paper-card .authors {
        color: #c7d2fe;
        font-size: 0.95rem;
        margin-bottom: 12px;
    }
    
    .paper-card .meta {
        color: #a5b4fc;
        font-size: 0.85rem;
    }
    
    /* Explanation Cards */
    .explanation-card {
        background: #f8fafc;
        border-left: 4px solid;
        border-radius: 0 12px 12px 0;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .eli5-card { border-color: #f59e0b; background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); }
    .undergrad-card { border-color: #3b82f6; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); }
    .expert-card { border-color: #8b5cf6; background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%); }
    
    .explanation-card h4 {
        margin: 0 0 12px 0;
        font-weight: 700;
    }
    
    .explanation-card p {
        margin: 0;
        line-height: 1.6;
        color: #334155;
    }
    
    /* Novelty Badges */
    .novelty-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 8px;
        letter-spacing: 0.5px;
    }
    
    .badge-novel { background: #dcfce7; color: #166534; }
    .badge-incremental { background: #fef3c7; color: #92400e; }
    .badge-builds { background: #e0e7ff; color: #3730a3; }
    
    /* Novelty Score meter */
    .score-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        margin: 8px 0 16px 0;
    }
    
    .score-number {
        font-size: 3rem;
        font-weight: 800;
        color: #a5b4fc;
    }
    
    .score-label {
        color: #c7d2fe;
        font-size: 0.9rem;
    }
    
    /* Insight Card */
    .insight-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    
    .insight-card p {
        margin: 0;
        color: #166534;
        font-weight: 500;
    }
    
    /* Quick select buttons */
    .example-btn {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 10px 14px;
        color: #475569;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .example-btn:hover {
        background: #e2e8f0;
        border-color: #6366f1;
        color: #4338ca;
    }
    
    /* Question list styling */
    .question-item {
        background: #f8fafc;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 3px solid #6366f1;
    }
    
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #6366f1;
        margin-bottom: 8px;
    }
    
    /* Progress styling */
    .agent-progress {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Hero Header ───
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🔬 PaperTrail</div>
    <div class="hero-subtitle">Understand any research paper in 30 seconds — powered by 5 AI agents</div>
</div>
""", unsafe_allow_html=True)


# ─── Session State ───
if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None


# ─── Sidebar: History + Example Papers ───
with st.sidebar:
    st.markdown('<div class="sidebar-header">📚 Quick Examples</div>', unsafe_allow_html=True)
    st.caption("Click to analyze a famous paper:")

    example_papers = {
        "Attention Is All You Need": "1706.03762",
        "BERT": "1810.04805",
        "GPT-3": "2005.14165",
        "ResNet": "1512.03385",
        "Diffusion Models Beat GANs": "2105.05233",
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
    st.caption("Built with LangGraph, Groq, FAISS, and Streamlit")
    st.caption("🔑 Uses Kimi K2 Instruct via Groq (free)")


# ─── Main Input ───
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


# ─── Analysis Pipeline ───
if analyze_clicked and url:
    # Clear any previously selected URL
    if "selected_url" in st.session_state:
        del st.session_state.selected_url

    progress_container = st.container()

    with progress_container:
        progress_bar = st.progress(0, text="🤖 Initializing 5 AI agents...")

        try:
            # Run analysis with progress updates
            progress_bar.progress(10, text="📥 Agent 1: Fetching & parsing paper...")
            result = analyze_paper(url)

            if result.get("error"):
                st.error(f"❌ Error: {result['error']}")
            else:
                progress_bar.progress(100, text="✅ Analysis complete!")
                st.session_state.current_result = result

                # Add to history
                st.session_state.history.append({
                    "title": result["parsed_paper"]["parsed"]["title"],
                    "url": url,
                    "result": result,
                })

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("💡 Make sure your GROQ_API_KEY is set in the .env file.")


# ─── Display Results ───
result = st.session_state.current_result

if result and not result.get("error"):
    parsed = result["parsed_paper"]["parsed"]
    metadata = result["parsed_paper"]["metadata"]

    # ─── Paper Info Card ───
    authors_str = ", ".join(metadata["authors"][:5])
    if len(metadata["authors"]) > 5:
        authors_str += f" + {len(metadata['authors']) - 5} more"

    st.markdown(f"""
    <div class="paper-card">
        <h2>{parsed['title']}</h2>
        <div class="authors">{authors_str}</div>
        <div class="meta">
            📅 {metadata['published'][:10]} &nbsp;|&nbsp; 
            🏷️ {', '.join(metadata['categories'])} &nbsp;|&nbsp; 
            📝 {parsed['paper_type'].capitalize()} &nbsp;|&nbsp;
            🔗 <a href="{metadata['entry_url']}" target="_blank" style="color: #a5b4fc;">arXiv</a>
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
            # One sentence + Key insight
            st.markdown(f"**💡 In one sentence:** {exp.get('one_sentence', 'N/A')}")

            st.markdown(f"""
            <div class="insight-card">
                <p>🔑 <strong>Key Insight:</strong> {exp.get('key_insight', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # 3-level explanations in columns
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
    # TAB 2: Novelty Analysis
    # ═══════════════════════════════════════
    with tab2:
        nov = result.get("novelty_analysis", {})

        if nov:
            # Score display
            score = nov.get("novelty_score", 5)
            col_s1, col_s2 = st.columns([1, 3])

            with col_s1:
                st.markdown(f"""
                <div class="score-container">
                    <div class="score-number">{score}/10</div>
                    <div class="score-label">Novelty Score</div>
                </div>
                """, unsafe_allow_html=True)

            with col_s2:
                st.markdown(f"**Summary:** {nov.get('novelty_summary', 'N/A')}")

                # Score interpretation
                if score >= 8:
                    st.success("🌟 Highly novel — introduces significant new ideas")
                elif score >= 5:
                    st.info("📊 Solid contribution — meaningful improvements with some new ideas")
                else:
                    st.warning("📈 Incremental — builds upon existing work with minor improvements")

            st.divider()

            # Novel contributions
            st.markdown("### ✨ Novel Contributions")
            for c in nov.get("novel_contributions", []):
                st.markdown(
                    f'<span class="novelty-badge badge-novel">NEW</span> {c}',
                    unsafe_allow_html=True,
                )

            st.markdown("### 📈 Incremental Improvements")
            for c in nov.get("incremental_improvements", []):
                st.markdown(
                    f'<span class="novelty-badge badge-incremental">IMPROVED</span> {c}',
                    unsafe_allow_html=True,
                )

            st.markdown("### 📚 Builds Upon")
            for c in nov.get("builds_upon", []):
                st.markdown(
                    f'<span class="novelty-badge badge-builds">PRIOR</span> {c}',
                    unsafe_allow_html=True,
                )

            # Related papers
            related = result.get("related_papers", [])
            if related:
                st.divider()
                st.markdown("### 🔗 Related Papers from Literature")
                for p in related:
                    score_val = p.get("similarity_score", 0)
                    st.markdown(f"- **{p['title']}** (similarity: `{score_val:.3f}`)")

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

            # Legend
            st.markdown("**Legend:**")
            legend_cols = st.columns(6)
            legend_items = [
                ("🔷 Method", "#6366f1"),
                ("🟡 Dataset", "#f59e0b"),
                ("🟢 Metric", "#10b981"),
                ("🔵 Concept", "#3b82f6"),
                ("🔴 Result", "#ef4444"),
                ("🟣 Problem", "#8b5cf6"),
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
                st.markdown(f"""
                <div class="question-item">✅ {question}</div>
                """, unsafe_allow_html=True)

            st.markdown("### 🔮 Questions Left Open")
            for question in q.get("questions_left_open", []):
                st.markdown(f"""
                <div class="question-item" style="border-color: #f59e0b;">🔮 {question}</div>
                """, unsafe_allow_html=True)

            st.markdown("### 📖 Suggested Follow-Up Reading")
            for item in q.get("follow_up_reading", []):
                st.markdown(f"- 📗 {item}")

            st.divider()

            st.markdown("### 💬 Discussion Questions")
            for question in q.get("discussion_questions", []):
                st.markdown(f"""
                <div class="question-item" style="border-color: #8b5cf6;">💬 {question}</div>
                """, unsafe_allow_html=True)

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

        # Raw data export
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
        "<center><p style='color: #94a3b8;'>Paste an arXiv URL above and click Analyze to get started!</p></center>",
        unsafe_allow_html=True,
    )
