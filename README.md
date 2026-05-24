# PaperTrail

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Workflow-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-LLM%20Inference-F55036)](https://groq.com/)

PaperTrail is an AI research-paper analysis app that turns an arXiv paper or uploaded PDF into a structured, demo-ready research brief. It parses the paper, explains it at multiple levels, compares it against related prior work, builds an interactive concept map, generates research questions, and exports a PDF report.

The project is intentionally scoped for a recruiter or technical interview demo: it focuses on a complete, defensible paper-understanding workflow instead of a large set of unrelated AI features.

## What It Does

Given an arXiv URL, arXiv ID, or PDF upload, PaperTrail produces:

- Structured paper metadata and analysis
- ELI5, undergraduate, and expert-level explanations
- Prior-work retrieval with FAISS semantic search
- Novelty analysis with a 1-10 score and human override
- Interactive concept graph of important ideas and relationships
- Research questions, open problems, and follow-up reading paths
- Downloadable PDF report
- Optional multi-paper synthesis for 2-5 analyzed papers

## Demo Inputs

The app includes three built-in examples:

- `1706.03762` - Attention Is All You Need
- `1810.04805` - BERT
- `2010.11929` - Vision Transformer

You can also paste any valid arXiv URL or upload a local research-paper PDF.

## Architecture

```text
Input: arXiv URL, arXiv ID, or PDF
        |
        v
Paper Parser
  - arXiv metadata
  - PDF text extraction
  - structured paper fields
        |
        v
Layered Explainer
  - ELI5
  - undergraduate
  - expert
        |
        v
Novelty Detector
  - FAISS retrieval
  - related papers
  - novelty score
        |
        v
Concept Mapper
  - nodes
  - relationships
  - interactive graph
        |
        v
Question Generator
  - answered questions
  - open questions
  - follow-up reading
        |
        v
Streamlit UI + PDF export
```

LangGraph coordinates the main analysis pipeline. Each agent has a focused responsibility, and shared runtime configuration lives in `config.py`.

## Tech Stack

| Area | Technology |
| --- | --- |
| Frontend | Streamlit |
| Agent orchestration | LangGraph |
| LLM provider | Groq |
| Default model | `qwen/qwen3-32b` |
| PDF extraction | PyMuPDF |
| Embeddings | sentence-transformers |
| Vector search | FAISS |
| Graph visualization | Pyvis / vis.js |
| Report export | ReportLab |

## Quickstart

```bash
git clone https://github.com/HariHaran9597/PaperTrail
cd PaperTrail
pip install -r requirements.txt
copy .env.example .env
python scripts/build_seed_index.py
streamlit run app.py
```

On macOS or Linux, use:

```bash
cp .env.example .env
```

Edit `.env` before running the app:

```env
GROQ_API_KEY=replace_with_your_groq_api_key
GROQ_MODEL=qwen/qwen3-32b
```

## Local Paper Index

Novelty analysis depends on a local FAISS index of related papers. Build it with:

```bash
python scripts/build_seed_index.py
```

If the index is missing, PaperTrail does not pretend to perform prior-work analysis. It shows this clear message instead:

```text
Novelty analysis needs the local paper index. Run python scripts/build_seed_index.py.
```

The rest of the app, including parsing, explanation, concept mapping, questions, and PDF export, can still run without the index.

## Project Structure

```text
PaperTrail/
├── app.py
├── config.py
├── agents/
│   ├── paper_parser.py
│   ├── layered_explainer.py
│   ├── novelty_detector.py
│   ├── concept_mapper.py
│   ├── question_generator.py
│   └── research_thread.py
├── graph/
│   ├── pipeline.py
│   └── state.py
├── utils/
│   ├── arxiv_fetcher.py
│   ├── pdf_extractor.py
│   ├── embeddings.py
│   ├── graph_visualizer.py
│   └── pdf_report.py
├── scripts/
│   ├── build_seed_index.py
│   └── test_pipeline.py
├── requirements.txt
└── .env.example
```

## Configuration

Runtime configuration is centralized in `config.py` and environment variables:

| Variable | Required | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | Yes | Groq API key used by all LLM-backed agents |
| `GROQ_MODEL` | No | Groq model name. Defaults to `qwen/qwen3-32b` |

Do not commit `.env`. It is ignored by `.gitignore`.

## Testing

Run the smoke test:

```bash
python scripts/test_pipeline.py
```

The test runs one known arXiv paper through the pipeline and verifies the important output fields. Full novelty scoring requires the FAISS index; if the index is missing, the test accepts the intentional skip state.

## Streamlit Deployment

To deploy on Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Create a new app at `share.streamlit.io`.
3. Select this repository, branch `main`, and entry file `app.py`.
4. Add secrets in Streamlit:

```toml
GROQ_API_KEY="your_groq_api_key"
GROQ_MODEL="qwen/qwen3-32b"
```

5. Deploy the app.

For a full novelty-analysis demo on Streamlit Cloud, the FAISS index must be available to the deployed app. Without it, the app still runs but shows the explicit missing-index message in the novelty tab.


