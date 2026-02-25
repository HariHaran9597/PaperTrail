"""
LangGraph Pipeline Orchestration
Wires all 5 agents into a sequential pipeline with error handling.

Flow:
    parse_paper → explain_paper → detect_novelty → map_concepts → generate_questions → END
    (error at any stage) → END
"""

from langgraph.graph import StateGraph, END
from graph.state import PaperTrailState
from utils.graph_visualizer import build_knowledge_graph
import logging
import time

logger = logging.getLogger(__name__)

# Lazy agent initialization — only loaded when pipeline is first invoked
_agents = {}


def _get_agents():
    """Lazily initialize agents on first use to avoid import-time API key errors."""
    if not _agents:
        from agents.paper_parser import PaperParserAgent
        from agents.layered_explainer import LayeredExplainerAgent
        from agents.novelty_detector import NoveltyDetectorAgent
        from agents.concept_mapper import ConceptMapperAgent
        from agents.question_generator import QuestionGeneratorAgent

        _agents["parser"] = PaperParserAgent()
        _agents["explainer"] = LayeredExplainerAgent()
        _agents["novelty_detector"] = NoveltyDetectorAgent()
        _agents["concept_mapper"] = ConceptMapperAgent()
        _agents["question_gen"] = QuestionGeneratorAgent()
        logger.info("All 5 agents initialized successfully")
    return _agents


# ─── Node Functions ───

def parse_paper_node(state: PaperTrailState) -> dict:
    """Agent 1: Parse the paper from arXiv URL or uploaded PDF."""
    try:
        agents = _get_agents()
        logger.info("=" * 50)
        logger.info("PIPELINE NODE: parse_paper")
        start = time.time()

        # Route: PDF upload path vs arXiv URL path
        if state.get("pdf_path"):
            logger.info("  → Using PDF upload path")
            result = agents["parser"].parse_from_pdf(
                state["pdf_path"],
                state.get("pdf_filename", "Uploaded Paper"),
            )
        else:
            logger.info("  → Using arXiv URL path")
            result = agents["parser"].parse(state["arxiv_url"])

        logger.info(f"  Completed in {time.time() - start:.1f}s")
        return {"parsed_paper": result, "status": "processing"}
    except Exception as e:
        logger.error(f"  ❌ Paper parsing failed: {e}")
        return {"error": str(e), "status": "error"}


def explain_paper_node(state: PaperTrailState) -> dict:
    """Agent 2: Generate layered explanations."""
    try:
        agents = _get_agents()
        logger.info("=" * 50)
        logger.info("PIPELINE NODE: explain_paper")
        start = time.time()
        result = agents["explainer"].explain(state["parsed_paper"])
        logger.info(f"  Completed in {time.time() - start:.1f}s")
        return {"explanations": result}
    except Exception as e:
        logger.error(f"  ❌ Explanation failed: {e}")
        return {"explanations": {
            "eli5": "Unable to generate explanation.",
            "undergrad": "Unable to generate explanation.",
            "expert": "Unable to generate explanation.",
            "one_sentence": "Unable to generate summary.",
            "key_insight": "Unable to generate insight.",
        }}


def detect_novelty_node(state: PaperTrailState) -> dict:
    """Agent 3: Detect novelty using RAG."""
    try:
        agents = _get_agents()
        logger.info("=" * 50)
        logger.info("PIPELINE NODE: detect_novelty")
        start = time.time()
        result = agents["novelty_detector"].detect(state["parsed_paper"])
        logger.info(f"  Completed in {time.time() - start:.1f}s")
        return {
            "novelty_analysis": result["novelty"],
            "related_papers": result["related_papers"],
        }
    except Exception as e:
        logger.error(f"  ❌ Novelty detection failed: {e}")
        return {
            "novelty_analysis": {
                "novel_contributions": ["Analysis unavailable"],
                "incremental_improvements": [],
                "builds_upon": [],
                "novelty_score": 5,
                "novelty_summary": "Novelty analysis could not be completed.",
            },
            "related_papers": [],
        }


def map_concepts_node(state: PaperTrailState) -> dict:
    """Agent 4: Build concept knowledge graph."""
    try:
        agents = _get_agents()
        logger.info("=" * 50)
        logger.info("PIPELINE NODE: map_concepts")
        start = time.time()
        result = agents["concept_mapper"].map_concepts(state["parsed_paper"])
        graph_path = build_knowledge_graph(result)
        logger.info(f"  Completed in {time.time() - start:.1f}s")
        return {"concept_map": result, "graph_html_path": graph_path}
    except Exception as e:
        logger.error(f"  ❌ Concept mapping failed: {e}")
        return {
            "concept_map": {"nodes": [], "edges": []},
            "graph_html_path": None,
        }


def generate_questions_node(state: PaperTrailState) -> dict:
    """Agent 5: Generate research questions."""
    try:
        agents = _get_agents()
        logger.info("=" * 50)
        logger.info("PIPELINE NODE: generate_questions")
        start = time.time()
        result = agents["question_gen"].generate(
            state["parsed_paper"],
            {"novelty": state["novelty_analysis"]},
        )
        logger.info(f"  Completed in {time.time() - start:.1f}s")
        return {"questions": result, "status": "complete"}
    except Exception as e:
        logger.error(f"  ❌ Question generation failed: {e}")
        return {
            "questions": {
                "questions_answered": ["Analysis unavailable"],
                "questions_left_open": [],
                "follow_up_reading": [],
                "discussion_questions": [],
            },
            "status": "complete",
        }


def should_continue(state: PaperTrailState) -> str:
    """Conditional edge: check if parsing succeeded."""
    if state.get("error"):
        return "error"
    return "continue"


# ─── Build the Pipeline ───

def build_pipeline():
    """Construct the LangGraph pipeline."""
    workflow = StateGraph(PaperTrailState)

    # Add nodes
    workflow.add_node("parse_paper", parse_paper_node)
    workflow.add_node("explain_paper", explain_paper_node)
    workflow.add_node("detect_novelty", detect_novelty_node)
    workflow.add_node("map_concepts", map_concepts_node)
    workflow.add_node("generate_questions", generate_questions_node)

    # Define edges
    workflow.set_entry_point("parse_paper")

    # After parsing, check for errors
    workflow.add_conditional_edges(
        "parse_paper",
        should_continue,
        {"continue": "explain_paper", "error": END},
    )

    # Sequential flow after successful parsing
    workflow.add_edge("explain_paper", "detect_novelty")
    workflow.add_edge("detect_novelty", "map_concepts")
    workflow.add_edge("map_concepts", "generate_questions")
    workflow.add_edge("generate_questions", END)

    return workflow.compile()


# Compile the pipeline (done once at import time — this is safe, no API keys needed)
pipeline = build_pipeline()


def analyze_paper(arxiv_url: str) -> dict:
    """
    Run the full PaperTrail pipeline via arXiv URL.
    
    Args:
        arxiv_url: arXiv URL or paper ID
        
    Returns:
        Complete analysis result dict
    """
    logger.info("🔬 Starting PaperTrail analysis pipeline")
    logger.info(f"   Input: {arxiv_url}")
    start = time.time()

    result = pipeline.invoke({
        "arxiv_url": arxiv_url,
        "status": "processing",
    })

    elapsed = time.time() - start
    logger.info(f"🏁 Pipeline complete in {elapsed:.1f}s — Status: {result.get('status')}")

    return result


def analyze_pdf(pdf_path: str, filename: str = "Uploaded Paper") -> dict:
    """
    Run the full PaperTrail pipeline via direct PDF upload.
    Fallback when arXiv API is unavailable or for non-arXiv papers.
    
    Args:
        pdf_path: Path to the uploaded PDF
        filename: Original filename
        
    Returns:
        Complete analysis result dict
    """
    logger.info("🔬 Starting PaperTrail analysis pipeline (PDF upload)")
    logger.info(f"   Input: {filename}")
    start = time.time()

    result = pipeline.invoke({
        "arxiv_url": "",
        "pdf_path": pdf_path,
        "pdf_filename": filename,
        "status": "processing",
    })

    elapsed = time.time() - start
    logger.info(f"🏁 Pipeline complete in {elapsed:.1f}s — Status: {result.get('status')}")

    return result
