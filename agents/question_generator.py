"""
Agent 5: Question Generator
Generates insightful research questions:
  - What the paper answers
  - What it leaves open
  - Suggested follow-up reading
  - Discussion questions for reading groups
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class PaperQuestions(BaseModel):
    """Research questions generated from a paper analysis."""
    questions_answered: list[str] = Field(
        description="5-7 specific research questions this paper answers. Be concrete and specific to the paper."
    )
    questions_left_open: list[str] = Field(
        description="3-5 questions this paper does NOT answer or explicitly leaves for future work."
    )
    follow_up_reading: list[str] = Field(
        description="3-5 suggested topics or specific papers to read next to deepen understanding."
    )
    discussion_questions: list[str] = Field(
        description="3 thought-provoking questions for a reading group or seminar discussion."
    )


class QuestionGeneratorAgent:
    """Agent 5: Generates research questions for deeper paper understanding."""

    def __init__(self):
        self.llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0.3)

    def generate(self, parsed_paper: dict, novelty: dict) -> dict:
        """
        Generate research questions based on paper analysis and novelty detection.
        
        Args:
            parsed_paper: Output from PaperParserAgent.parse()
            novelty: Output from NoveltyDetectorAgent.detect() — the 'novelty' sub-dict
            
        Returns:
            dict with questions_answered, questions_left_open, follow_up_reading, discussion_questions
        """
        logger.info(f"Agent 5 (Question Generator): Generating questions for '{parsed_paper['parsed']['title']}'")

        # Build novelty context
        novel_contributions = "\n".join(
            novelty.get("novelty", {}).get("novel_contributions", ["Not available"])
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research mentor helping a student deeply understand a paper.
            Generate insightful questions that:
            
            1. **Questions Answered**: What specific research questions does this paper answer? 
               Frame them as actual questions (e.g., "Can attention mechanisms replace recurrence entirely?")
            
            2. **Questions Left Open**: What does this paper NOT address? 
               What gaps remain? What did the authors explicitly mention as future work?
            
            3. **Follow-up Reading**: What should someone read NEXT to understand this area better?
               Suggest specific topics, techniques, or landmark papers.
            
            4. **Discussion Questions**: What are the deeper implications? 
               What assumptions could be challenged? What would happen if a key result didn't hold?
            
            CRITICAL: All questions must be specific to THIS paper, not generic research questions."""),
            ("user", """Paper: {title}
            
            Problem: {problem}
            
            Methodology: {methodology}
            
            Key Results: {results}
            
            Limitations: {limitations}
            
            Novel Contributions: {novel}
            
            Generate specific, insightful questions.""")
        ])

        structured_llm = self.llm.with_structured_output(PaperQuestions)

        result = structured_llm.invoke(
            prompt.format_messages(
                title=parsed_paper["parsed"]["title"],
                problem=parsed_paper["parsed"]["problem_statement"],
                methodology=parsed_paper["parsed"]["methodology_summary"],
                results="\n".join(parsed_paper["parsed"]["key_results"]),
                limitations="\n".join(parsed_paper["parsed"]["limitations"]),
                novel=novel_contributions,
            )
        )

        logger.info(
            f"  ✅ Generated {len(result.questions_answered)} answered, "
            f"{len(result.questions_left_open)} open, "
            f"{len(result.discussion_questions)} discussion questions"
        )
        return result.model_dump()
