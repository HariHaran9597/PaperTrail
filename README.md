# PaperTrail

PaperTrail is a research-paper understanding engine built for fast demos and defensible interviews. Paste an arXiv URL, an arXiv ID such as `1706.03762`, or upload a PDF to get structured paper parsing, layered explanations, prior-work novelty analysis, a concept map, research questions, and a downloadable PDF report.

## Quickstart

```bash
git clone https://github.com/HariHaran9597/PaperTrail
cd PaperTrail
pip install -r requirements.txt
copy .env.example .env
python scripts/build_seed_index.py
streamlit run app.py
```

Edit `.env` before running the app:

```bash
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3-32b
```

`scripts/build_seed_index.py` builds the local FAISS paper index used for novelty analysis. If you skip it, the app will still parse and explain papers, but novelty analysis will show:

> Novelty analysis needs the local paper index. Run python scripts/build_seed_index.py.

## Core Features

### Paper Input

- arXiv URL input, for example `https://arxiv.org/abs/1706.03762`
- arXiv ID input, for example `1706.03762`
- PDF upload fallback for non-arXiv papers

### Paper Parsing

PaperTrail extracts:

- Title
- Authors
- Abstract
- Problem statement
- Methodology
- Key results
- Limitations
- Important technical terms

### Layered Explanation

Each paper gets:

- ELI5 explanation
- Undergraduate explanation
- Expert explanation
- One-sentence summary
- Key insight

### Novelty And Prior Work

When the local FAISS index is available, PaperTrail:

- Retrieves top related papers
- Shows title, year, and similarity score
- Explains what is genuinely new
- Explains what is incremental
- Gives a 1-10 novelty score
- Allows a human override of the score in the UI

### Concept Map

PaperTrail extracts 10-15 important concepts and directed relationships. Invalid LLM-generated edges are filtered so the graph does not break when an edge references a missing node.

### Research Questions

The app generates:

- Questions answered by the paper
- Open questions
- Follow-up research ideas
- Suggested reading path
- Discussion questions

### Export

The PDF report includes the paper summary, layered explanations, novelty analysis, concept-map data, related papers, and research questions.

## Demo Examples

The sidebar includes three preloaded examples:

- Attention Is All You Need
- BERT
- Vision Transformer

## Optional Feature

If you analyze 2-5 papers in one session, the Research Thread section can synthesize:

- Common themes
- Contradictions
- Evolution of ideas
- Open gaps
- Recommended next research steps

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
│   ├── state.py
│   └── pipeline.py
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

## Test

Run the smoke test on one known paper:

```bash
python scripts/test_pipeline.py
```

The test checks that the pipeline returns the expected top-level fields. It requires `GROQ_API_KEY`; full novelty scoring also requires the local FAISS index.
