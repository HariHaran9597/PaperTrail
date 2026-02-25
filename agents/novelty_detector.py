"""
Agent 3: Novelty Detector
Uses RAG over a FAISS index of ~5,000 papers to determine what's
genuinely novel vs. incremental in a given paper.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from utils.embeddings import PaperIndex
import logging

logger = logging.getLogger(__name__)


class NoveltyAnalysis(BaseModel):
    """Analysis of what's novel vs. incremental in a paper."""
    novel_contributions: list[str] = Field(
        description="What this paper introduces that didn't exist before. Be specific about the actual contribution."
    )
    incremental_improvements: list[str] = Field(
        description="What this paper improves upon existing work. Mention what it improves and by how much."
    )
    builds_upon: list[str] = Field(
        description="Key prior work this paper extends or builds upon. Include paper names or techniques."
    )
    novelty_score: int = Field(
        description="1-10 rating of how novel this paper is. 1=purely incremental, 5=solid contribution, 10=paradigm-shifting.",
        ge=1, le=10
    )
    novelty_summary: str = Field(
        description="2-3 sentence summary of what's actually new and why it matters."
    )


class NoveltyDetectorAgent:
    """Agent 3: Detects what's genuinely novel using RAG over related papers."""

    def __init__(self):
        self.llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0)
        self.paper_index = PaperIndex()
        try:
            self.paper_index.load_index()
            logger.info("Novelty Detector: FAISS index loaded successfully")
        except FileNotFoundError:
            logger.warning(
                "Novelty Detector: FAISS index not found. "
                "Run 'python scripts/build_seed_index.py' to build it. "
                "Novelty detection will work with limited context."
            )
            self.paper_index = None

    def detect(self, parsed_paper: dict) -> dict:
        """
        Analyze novelty of a paper against the existing literature.
        
        Args:
            parsed_paper: Output from PaperParserAgent.parse()
            
        Returns:
            dict with 'novelty' (NoveltyAnalysis) and 'related_papers' (top 5)
        """
        logger.info(f"Agent 3 (Novelty Detector): Analyzing '{parsed_paper['parsed']['title']}'")

        # Step 1: Find related papers via FAISS
        related_papers = []
        related_context = "No related papers index available."

        if self.paper_index is not None:
            query = (
                f"{parsed_paper['parsed']['title']}. "
                f"{parsed_paper['parsed']['problem_statement']}. "
                f"{parsed_paper['parsed']['methodology_summary']}"
            )
            related_papers = self.paper_index.search(query, top_k=10)

            # Build context string from related papers
            related_context = "\n\n".join([
                f"- **{p['title']}** ({p.get('published', 'N/A')[:4]}): "
                f"{p['abstract'][:300]}..."
                for p in related_papers
            ])
            logger.info(f"  Found {len(related_papers)} related papers")

        # Step 2: LLM novelty analysis
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior ML researcher reviewing a paper for novelty.
            Compare it against related prior work and identify:
            
            1. What's genuinely NEW (a novel contribution that didn't exist before)
            2. What's an incremental improvement over existing work (better numbers, minor modifications)
            3. What prior work it clearly builds upon
            
            Be brutally honest — if the contribution is incremental, say so.
            If the paper is genuinely groundbreaking, acknowledge that too.
            
            Consider: Does this paper introduce a new architecture? A new training method? 
            A new dataset? Or does it primarily combine existing techniques?"""),
            ("user", """Paper under review:
            Title: {title}
            Problem: {problem}
            Methodology: {methodology}
            Key Results: {results}
            Key Terms: {key_terms}
            
            Related prior work found in the literature:
            {related_papers}
            
            Analyze the novelty of this paper.""")
        ])

        structured_llm = self.llm.with_structured_output(NoveltyAnalysis)

        result = structured_llm.invoke(
            prompt.format_messages(
                title=parsed_paper["parsed"]["title"],
                problem=parsed_paper["parsed"]["problem_statement"],
                methodology=parsed_paper["parsed"]["methodology_summary"],
                results="\n".join(parsed_paper["parsed"]["key_results"]),
                key_terms=", ".join(parsed_paper["parsed"]["key_terms"]),
                related_papers=related_context,
            )
        )

        logger.info(f"  ✅ Novelty score: {result.novelty_score}/10")

        return {
            "novelty": result.model_dump(),
            "related_papers": related_papers[:5],  # Top 5 for display
        }
