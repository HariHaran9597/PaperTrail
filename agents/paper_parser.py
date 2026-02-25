"""
Agent 1: Paper Parser
Combines arXiv fetcher + PDF extractor into a clean pipeline.
Uses LLM to enhance section detection and extract structured fields.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from utils.arxiv_fetcher import ArxivFetcher
from utils.pdf_extractor import PDFExtractor
import logging

logger = logging.getLogger(__name__)


class ParsedPaper(BaseModel):
    """Structured representation of a parsed research paper."""
    title: str
    authors: list[str]
    abstract: str
    problem_statement: str = Field(description="The core problem this paper addresses")
    methodology_summary: str = Field(description="Key methodology in 3-5 sentences")
    key_results: list[str] = Field(description="Top 3-5 results/findings")
    limitations: list[str] = Field(description="Acknowledged limitations")
    key_terms: list[str] = Field(description="Important technical terms used")
    paper_type: str = Field(description="One of: theoretical, empirical, survey, benchmark, system")


class PaperParserAgent:
    """Agent 1: Parses arXiv papers into structured JSON using PDF extraction + LLM."""

    def __init__(self):
        self.fetcher = ArxivFetcher()
        self.extractor = PDFExtractor()
        self.llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0)

    def parse(self, arxiv_url: str) -> dict:
        """
        Full paper parsing pipeline.
        
        Args:
            arxiv_url: arXiv URL or paper ID
            
        Returns:
            dict with metadata, extracted_text, and parsed (LLM-enhanced) fields
        """
        logger.info(f"Agent 1 (Paper Parser): Processing {arxiv_url}")

        # Step 1: Fetch metadata + PDF from arXiv
        logger.info("  Step 1: Fetching from arXiv...")
        metadata = self.fetcher.fetch(arxiv_url)

        # Step 2: Extract text from PDF
        logger.info("  Step 2: Extracting text from PDF...")
        extracted = self.extractor.extract(metadata["pdf_path"])

        # Step 3: LLM-enhanced structured parsing
        logger.info("  Step 3: LLM-enhanced parsing...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research paper analyst. Extract structured information 
            from this paper. Be precise and factual — only state what the paper actually claims.
            
            Important:
            - problem_statement: What specific problem does this paper solve? Be specific.
            - methodology_summary: Describe the actual method/approach in 3-5 sentences.
            - key_results: List the top 3-5 concrete results with numbers if available.
            - limitations: What do the authors acknowledge as limitations?
            - key_terms: List 5-10 important technical terms specific to this paper.
            - paper_type: Choose ONE of: theoretical, empirical, survey, benchmark, system"""),
            ("user", """Paper Title: {title}
            
            Authors: {authors}
            
            Abstract: {abstract}
            
            Full Text (first 6000 chars): {text_excerpt}
            
            Extract the structured information as requested.""")
        ])

        structured_llm = self.llm.with_structured_output(ParsedPaper)

        # Use the best available text — prefer sections, fall back to full text
        text_for_llm = ""
        sections = extracted.get("sections", {})
        for section_name in ["introduction", "methodology", "results", "conclusion"]:
            if section_name in sections:
                text_for_llm += f"\n\n--- {section_name.upper()} ---\n{sections[section_name][:1500]}"

        if not text_for_llm:
            text_for_llm = extracted["full_text"][:6000]
        else:
            text_for_llm = text_for_llm[:6000]

        result = structured_llm.invoke(
            prompt.format_messages(
                title=metadata["title"],
                authors=", ".join(metadata["authors"][:10]),
                abstract=metadata["abstract"],
                text_excerpt=text_for_llm,
            )
        )

        logger.info(f"  ✅ Parsed: {result.title} ({result.paper_type})")

        return {
            "metadata": metadata,
            "extracted_text": extracted,
            "parsed": result.model_dump(),
        }
