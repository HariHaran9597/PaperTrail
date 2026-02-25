# 🔬 PaperTrail — Research Paper Understanding Engine

> **Paste any arXiv paper URL → get layered explanations, novelty analysis, interactive concept maps, and research questions — in 30 seconds.**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://papertrail.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent_Pipeline-00C853?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)

---

## 🎯 Problem

Research papers are **dense, jargon-heavy, and time-consuming**. Reading a single paper properly takes 2-4 hours. Most people give up after the abstract. Researchers waste hours figuring out if a paper is actually novel or just repackaging known ideas.

**PaperTrail solves this** by deploying 5 specialized AI agents to analyze any paper and deliver:
- Multi-level explanations (ELI5 → Expert)
- Honest novelty assessment against 5,000+ papers
- Interactive knowledge graphs
- Research questions for deeper understanding

---

## 🏗️ Architecture

```
User Input (arXiv URL)
         ↓
   ┌──────────────────────────────┐
   │  Agent 1: Paper Parser       │  PDF → Structured JSON
   │  (arXiv API + PyMuPDF + LLM) │  
   └──────────┬───────────────────┘
              ↓
   ┌──────────────────────────────┐
   │  Agent 2: Layered Explainer  │  ELI5 / Undergrad / Expert
   │  (Groq LLM)                  │  
   └──────────┬───────────────────┘
              ↓
   ┌──────────────────────────────┐
   │  Agent 3: Novelty Detector   │  RAG over 5,000 papers
   │  (FAISS + Groq LLM)          │  What's new vs. incremental
   └──────────┬───────────────────┘
              ↓
   ┌──────────────────────────────┐
   │  Agent 4: Concept Mapper     │  Knowledge graph extraction
   │  (Groq LLM + Pyvis)          │  Interactive visualization
   └──────────┬───────────────────┘
              ↓
   ┌──────────────────────────────┐
   │  Agent 5: Question Generator │  What it answers +
   │  (Groq LLM)                  │  what it leaves open
   └──────────┬───────────────────┘
              ↓
   Beautiful Streamlit UI with tabs
```

All 5 agents are orchestrated via **LangGraph** with conditional error handling and graceful fallbacks.

---

## 🛠️ Tech Stack

| Component           | Tool                              | Why                                      |
|---------------------|-----------------------------------|------------------------------------------|
| Agent Orchestration | LangGraph                         | Industry standard, conditional routing   |
| LLM                 | Groq + Kimi K2 Instruct           | Free, fast, high quality                 |
| Vector DB           | FAISS                             | Local, free, no cloud dependency         |
| Embeddings          | sentence-transformers (all-MiniLM) | Free, runs locally, fast                |
| Paper Fetching      | arXiv API + PyMuPDF               | Free, reliable                           |
| Knowledge Graph     | Pyvis (vis.js wrapper)            | Interactive, browser-native              |
| Frontend            | Streamlit                         | Fast to build, free deployment           |
| PDF Reports         | ReportLab                         | Free, professional output                |

**Total cost: $0**

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/papertrail.git
cd papertrail
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API key
```bash
copy .env.example .env
# Edit .env and add your GROQ_API_KEY (free from https://console.groq.com/)
```

### 5. Build the seed index (one-time, ~10 min)
```bash
python scripts/build_seed_index.py
```

### 6. Run the app
```bash
streamlit run app.py
```

---

## 📸 Features

### 📖 Layered Explanations
Three complexity levels for every paper:
- **🧒 ELI5** — Uses everyday analogies, no jargon
- **🎓 Undergrad** — CS student level with basic ML terms
- **🔬 Expert** — Full technical depth for researchers

### 🆕 Novelty Detection (RAG-powered)
- Searches 5,000+ ML papers via FAISS semantic search
- Identifies **genuinely novel contributions** vs. **incremental improvements**
- Provides a 1-10 novelty score with honest assessment

### 🕸️ Interactive Concept Maps
- Extracts 10-15 key concepts and their relationships
- Color-coded by category (method, dataset, metric, concept, result, problem)
- Drag, zoom, and hover for details

### ❓ Research Questions
- What the paper answers
- What it leaves open
- Suggested follow-up reading
- Discussion questions for reading groups

### 📥 PDF Report Export
- Professional branded PDF report
- All analysis results in one downloadable document

---

## 📁 Project Structure

```
papertrail/
├── agents/
│   ├── paper_parser.py          # Agent 1: PDF → structured JSON
│   ├── layered_explainer.py     # Agent 2: 3-level explanations
│   ├── novelty_detector.py      # Agent 3: RAG novelty analysis
│   ├── concept_mapper.py        # Agent 4: Knowledge graph
│   └── question_generator.py    # Agent 5: Q&A generation
├── graph/
│   ├── state.py                 # LangGraph state definition
│   └── pipeline.py              # LangGraph orchestration
├── utils/
│   ├── arxiv_fetcher.py         # arXiv paper download
│   ├── pdf_extractor.py         # PyMuPDF text extraction
│   ├── embeddings.py            # FAISS + sentence-transformers
│   ├── graph_visualizer.py      # Pyvis knowledge graph
│   └── pdf_report.py            # ReportLab PDF generation
├── scripts/
│   ├── build_seed_index.py      # Build FAISS index
│   └── test_pipeline.py         # End-to-end tests
├── data/
│   ├── papers_cache/            # Downloaded PDFs
│   └── faiss_index/             # Vector store
├── app.py                       # Streamlit frontend
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧪 Testing

Run the end-to-end test suite:

```bash
python scripts/test_pipeline.py
```

This tests the full pipeline on 3 papers (Attention, BERT, ViT) and verifies all 5 agents produce valid outputs.

---

## 🚢 Deployment (Streamlit Cloud)

1. Push to GitHub (public repo)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `app.py`
4. Add `GROQ_API_KEY` in Streamlit Secrets
5. Deploy!

**Note:** For Streamlit Cloud, the FAISS index should be pre-built and committed (or built at startup).

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- [Groq](https://groq.com/) for free, fast LLM inference
- [LangChain](https://langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/) for agent orchestration
- [arXiv](https://arxiv.org/) for open access to research papers
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
