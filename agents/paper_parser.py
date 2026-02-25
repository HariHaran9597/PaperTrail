"""
Agent 1: Paper Parser
Combines arXiv fetcher + PDF extractor into a clean pipeline.
Uses LLM to enhance section detection and extract structured fields.
Supports both arXiv URLs AND direct PDF upload as fallback.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from utils.arxiv_fetcher import ArxivFetcher
from utils.pdf_extractor import PDFExtractor
import os
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
    """Agent 1: Parses papers into structured JSON. Supports arXiv URL or direct PDF."""

    def __init__(self):
        self.fetcher = ArxivFetcher()
        self.extractor = PDFExtractor()
        self.llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0)

    def parse(self, arxiv_url: str) -> dict:
        """
        Full paper parsing pipeline via arXiv URL.
        
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
        return self._llm_parse(metadata, extracted)

    def parse_from_pdf(self, pdf_path: str, filename: str = "Uploaded Paper") -> dict:
        """
        Parse a directly uploaded PDF (fallback when arXiv is unavailable).
        
        Args:
            pdf_path: Local path to the PDF file
            filename: Original filename for display
            
        Returns:
            dict with metadata, extracted_text, and parsed (LLM-enhanced) fields
        """
        logger.info(f"Agent 1 (Paper Parser): Processing uploaded PDF: {filename}")

        # Step 1: Extract text from the uploaded PDF
        logger.info("  Step 1: Extracting text from uploaded PDF...")
        extracted = self.extractor.extract(pdf_path)

        # Step 2: Build placeholder metadata (no arXiv data available)
        metadata = {
            "arxiv_id": "uploaded",
            "title": filename.replace(".pdf", ""),
            "authors": ["Unknown (uploaded PDF)"],
            "abstract": "",
            "categories": ["uploaded"],
            "published": "N/A",
            "pdf_path": pdf_path,
            "pdf_url": "",
            "entry_url": "",
            "source": "pdf_upload",
        }

        # Try to extract abstract from the PDF text
        sections = extracted.get("sections", {})
        if "abstract" in sections:
            metadata["abstract"] = sections["abstract"][:1000]
        elif "preamble" in sections:
            # Often the abstract is in the preamble
            metadata["abstract"] = sections["preamble"][:1000]
        else:
            metadata["abstract"] = extracted["full_text"][:500]

        # Step 3: LLM-enhanced structured parsing
        return self._llm_parse(metadata, extracted)

    def _llm_parse(self, metadata: dict, extracted: dict) -> dict:
        """Shared LLM parsing logic for both arXiv and PDF upload paths."""
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
                authors=", ".join(metadata["authors"][:10]) if isinstance(metadata["authors"], list) else metadata["authors"],
                abstract=metadata["abstract"],
                text_excerpt=text_for_llm,
            )
        )

        # Update metadata with LLM-extracted title if from upload
        if metadata.get("source") == "pdf_upload":
            metadata["title"] = result.title
            metadata["authors"] = result.authors

        logger.info(f"  ✅ Parsed: {result.title} ({result.paper_type})")

        return {
            "metadata": metadata,
            "extracted_text": extracted,
            "parsed": result.model_dump(),
        }
