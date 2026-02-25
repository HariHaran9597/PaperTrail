"""
Agent 6: Research Thread — Multi-Paper Synthesis
Analyzes 3-5 papers together to extract:
  - Common themes across papers
  - Contradictions and disagreements
  - Evolution of ideas over time
  - Gaps and opportunities for future work
  - "State of the field" summary

This is the HIGH-IMPACT differentiating feature of PaperTrail.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class ResearchThread(BaseModel):
    """Multi-paper synthesis output."""
    thread_title: str = Field(
        description="A descriptive title for this research thread, e.g. 'Transformer Efficiency: From Attention to Linear Complexity'"
    )
    common_themes: list[str] = Field(
        description="3-5 themes that appear across multiple papers. Be specific about which papers share each theme."
    )
    contradictions: list[str] = Field(
        description="Points where papers disagree or present conflicting evidence. Include which papers conflict."
    )
    idea_evolution: list[str] = Field(
        description="How ideas evolved chronologically across these papers. Show the progression of thought."
    )
    consensus_findings: list[str] = Field(
        description="Findings that all or most papers agree on."
    )
    open_gaps: list[str] = Field(
        description="Research gaps that none of the papers address — opportunities for future work."
    )
    field_summary: str = Field(
        description="A comprehensive 'state of the field' paragraph (8-12 sentences) synthesizing all papers into a cohesive narrative."
    )
    recommended_next_steps: list[str] = Field(
        description="3-5 concrete research directions suggested by the collective gaps in these papers."
    )


class ResearchThreadAgent:
    """Agent 6: Multi-paper synthesis for research thread analysis."""

    def __init__(self):
        self.llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0.3)

    def synthesize(self, parsed_papers: list[dict]) -> dict:
        """
        Synthesize multiple parsed papers into a research thread.
        
        Args:
            parsed_papers: List of outputs from PaperParserAgent.parse()
                          (each with 'metadata' and 'parsed' keys)
            
        Returns:
            dict with thread analysis
        """
        num_papers = len(parsed_papers)
        logger.info(f"Agent 6 (Research Thread): Synthesizing {num_papers} papers")

        if num_papers < 2:
            raise ValueError("Research Thread requires at least 2 papers.")

        # Build context from all papers
        papers_context = ""
        for i, paper in enumerate(parsed_papers, 1):
            p = paper["parsed"]
            m = paper["metadata"]
            papers_context += f"""
--- PAPER {i}: {p['title']} ---
Authors: {', '.join(m.get('authors', ['Unknown'])[:5])}
Published: {m.get('published', 'N/A')[:10]}
Problem: {p['problem_statement']}
Methodology: {p['methodology_summary']}
Key Results: {'; '.join(p['key_results'])}
Limitations: {'; '.join(p['limitations'])}
Key Terms: {', '.join(p['key_terms'])}

"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior research scientist performing a literature synthesis.
            You have been given {num_papers} papers on a related topic.
            
            Your job is to find the CONNECTIONS between these papers:
            
            1. **Common Themes**: What ideas appear in multiple papers? Be specific.
            2. **Contradictions**: Where do papers disagree? Conflicting results? Different conclusions?
            3. **Idea Evolution**: How have ideas progressed chronologically? What builds on what?
            4. **Consensus**: What do most/all papers agree on?
            5. **Open Gaps**: What does NONE of the papers address? What's missing?
            6. **Field Summary**: Write a cohesive narrative that a PhD student could use to 
               understand the current state of this research area.
            7. **Next Steps**: What concrete research should be done next, given the collective gaps?
            
            Be analytical, not descriptive. Don't just list what each paper says — 
            show how they RELATE to each other."""),
            ("user", """Here are the {num_papers} papers to synthesize:

{papers_context}

Provide a comprehensive research thread synthesis.""")
        ])

        structured_llm = self.llm.with_structured_output(ResearchThread)

        result = structured_llm.invoke(
            prompt.format_messages(
                num_papers=num_papers,
                papers_context=papers_context,
            )
        )

        logger.info(f"  ✅ Research Thread: '{result.thread_title}'")
        logger.info(f"     {len(result.common_themes)} themes, {len(result.contradictions)} contradictions, {len(result.open_gaps)} gaps")

        return result.model_dump()
