"""
Agent 2: Layered Explainer
Generates explanations at 3 complexity levels:
  - ELI5 (a child could understand)
  - Undergrad (CS student level)
  - Expert (ML researcher level)
Plus a one-sentence summary and key insight.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class LayeredExplanation(BaseModel):
    """Multi-level explanation of a research paper."""
    eli5: str = Field(
        description="Explanation a 5-year-old could understand. Use everyday analogies. No jargon. Max 4 sentences."
    )
    undergrad: str = Field(
        description="Explanation for a CS undergrad. Can use basic ML terms like 'neural network', 'loss function', 'training'. Assumes linear algebra and probability knowledge. 6-8 sentences."
    )
    expert: str = Field(
        description="Technical explanation for an ML researcher. Include specific techniques, comparisons to prior work, and mathematical intuitions. 8-10 sentences."
    )
    one_sentence: str = Field(
        description="The paper summarized in exactly one clear sentence."
    )
    key_insight: str = Field(
        description="The single most important insight or takeaway from this paper in 1-2 sentences."
    )


class LayeredExplainerAgent:
    """Agent 2: Generates 3-level explanations of research papers."""

    def __init__(self):
        self.llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0.3)

    def explain(self, parsed_paper: dict) -> dict:
        """
        Generate layered explanations for a parsed paper.
        
        Args:
            parsed_paper: Output from PaperParserAgent.parse()
            
        Returns:
            dict with eli5, undergrad, expert, one_sentence, key_insight
        """
        logger.info(f"Agent 2 (Layered Explainer): Explaining '{parsed_paper['parsed']['title']}'")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a world-class science communicator who can explain any 
            research paper at multiple complexity levels. Your explanations are:
            
            - ELI5: Uses everyday analogies, no jargon whatsoever, a curious child could understand.
              Example style: "Imagine you have a huge pile of LEGO blocks..."
            
            - Undergrad: Uses basic CS/ML terminology. Assumes the reader knows what neural networks, 
              gradient descent, and attention mechanisms are. Explains the paper's specific approach.
            
            - Expert: Uses precise technical language, references specific architectures and techniques,
              discusses computational complexity, and situates the work in the broader research landscape.
            
            CRITICAL: Be specific to THIS paper — don't give generic explanations. 
            Reference actual methods, datasets, and results from the paper."""),
            ("user", """Paper: {title}
            
            Abstract: {abstract}
            
            Problem Statement: {problem}
            
            Methodology: {methodology}
            
            Key Results: {results}
            
            Key Terms: {key_terms}
            
            Generate layered explanations for this paper.""")
        ])

        structured_llm = self.llm.with_structured_output(LayeredExplanation)

        result = structured_llm.invoke(
            prompt.format_messages(
                title=parsed_paper["parsed"]["title"],
                abstract=parsed_paper["metadata"]["abstract"],
                problem=parsed_paper["parsed"]["problem_statement"],
                methodology=parsed_paper["parsed"]["methodology_summary"],
                results="\n".join(parsed_paper["parsed"]["key_results"]),
                key_terms=", ".join(parsed_paper["parsed"]["key_terms"]),
            )
        )

        logger.info("  ✅ Generated 3-level explanations")
        return result.model_dump()
