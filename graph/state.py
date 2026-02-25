"""
LangGraph State Definition
Defines the shared state that flows through the PaperTrail agent pipeline.
"""

from typing import TypedDict, Optional


class PaperTrailState(TypedDict):
    """Shared state for the PaperTrail LangGraph pipeline."""

    # ─── Input ───
    arxiv_url: str

    # ─── Agent 1: Paper Parser output ───
    parsed_paper: Optional[dict]

    # ─── Agent 2: Layered Explainer output ───
    explanations: Optional[dict]

    # ─── Agent 3: Novelty Detector output ───
    novelty_analysis: Optional[dict]
    related_papers: Optional[list]

    # ─── Agent 4: Concept Mapper output ───
    concept_map: Optional[dict]
    graph_html_path: Optional[str]

    # ─── Agent 5: Question Generator output ───
    questions: Optional[dict]

    # ─── Control flow ───
    error: Optional[str]
    status: str  # "processing", "complete", "error"
