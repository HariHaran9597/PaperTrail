"""
PDF Text Extractor
Extracts structured text from academic PDFs using PyMuPDF.
Handles section detection, text cleaning, and multi-column layouts.
"""

import fitz  # PyMuPDF
import re
import logging

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extracts and structures text content from academic PDFs."""

    # Section heading patterns — ordered by typical paper structure
    SECTION_PATTERNS = [
        (r'(?i)^\s*(?:abstract)\s*$', 'abstract'),
        (r'(?i)^\s*(?:\d+[\.\s]+)?introduction', 'introduction'),
        (r'(?i)^\s*(?:\d+[\.\s]+)?(?:related\s+work|background|prior\s+work|literature\s+review)', 'related_work'),
        (r'(?i)^\s*(?:\d+[\.\s]+)?(?:method(?:ology|s)?|approach|(?:proposed\s+)?model|(?:our\s+)?framework|architecture|system\s+design)', 'methodology'),
        (r'(?i)^\s*(?:\d+[\.\s]+)?(?:experiment(?:s|al)?(?:\s+(?:setup|results))?|result(?:s)?|evaluation|empirical|analysis)', 'results'),
        (r'(?i)^\s*(?:\d+[\.\s]+)?(?:discussion)', 'discussion'),
        (r'(?i)^\s*(?:\d+[\.\s]+)?(?:conclusion(?:s)?|summary|concluding\s+remarks)', 'conclusion'),
        (r'(?i)^\s*(?:\d+[\.\s]+)?(?:future\s+work|limitations)', 'future_work'),
        (r'(?i)^\s*(?:references|bibliography)', 'references'),
        (r'(?i)^\s*(?:appendix|supplementary)', 'appendix'),
    ]

    def extract(self, pdf_path: str) -> dict:
        """
        Extract structured text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            dict with keys: full_text, sections, page_count, figures_mentioned
        """
        logger.info(f"Extracting text from: {pdf_path}")

        doc = fitz.open(pdf_path)
        full_text = ""
        page_count = len(doc)

        for page in doc:
            # Extract text with better handling of layouts
            text = page.get_text("text")
            full_text += text + "\n"

        doc.close()

        # Clean the extracted text
        full_text = self._clean_text(full_text)

        # Detect sections
        sections = self._detect_sections(full_text)

        # Extract figure references
        figures_mentioned = self._extract_figure_refs(full_text)

        logger.info(f"Extracted {len(full_text)} chars, {len(sections)} sections, {page_count} pages")

        return {
            "full_text": full_text,
            "sections": sections,
            "page_count": page_count,
            "figures_mentioned": figures_mentioned,
        }

    def _clean_text(self, text: str) -> str:
        """Remove noise from extracted PDF text."""
        # Remove page numbers (standalone numbers on a line)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # Remove headers/footers that repeat (common in papers)
        lines = text.split('\n')
        if len(lines) > 20:
            # Check first and last lines of pages for repeating content
            line_counts = {}
            for line in lines:
                stripped = line.strip()
                if stripped and len(stripped) < 80:
                    line_counts[stripped] = line_counts.get(stripped, 0) + 1

            # Remove lines that appear more than 3 times (likely headers/footers)
            repeated_lines = {line for line, count in line_counts.items() if count > 3}
            if repeated_lines:
                lines = [l for l in lines if l.strip() not in repeated_lines]
                text = '\n'.join(lines)

        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Fix hyphenation at line breaks
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

        # Remove common artifacts
        text = re.sub(r'arXiv:\d+\.\d+v?\d*\s*\[.*?\].*?\d{4}', '', text)

        return text.strip()

    def _detect_sections(self, text: str) -> dict:
        """
        Split text into sections based on heading patterns.
        Falls back to full_text if no sections are detected.
        """
        sections = {}
        lines = text.split('\n')
        current_section = 'preamble'
        current_content = []

        for line in lines:
            matched = False
            stripped = line.strip()

            # Skip very short or very long lines for section matching
            if 2 < len(stripped) < 100:
                for pattern, section_name in self.SECTION_PATTERNS:
                    if re.match(pattern, stripped):
                        # Save the current section
                        if current_content:
                            content = '\n'.join(current_content).strip()
                            if content:  # Only save non-empty sections
                                sections[current_section] = content
                        current_section = section_name
                        current_content = []
                        matched = True
                        break

            if not matched:
                current_content.append(line)

        # Save the last section
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections[current_section] = content

        # If we only got preamble, section detection failed
        if list(sections.keys()) == ['preamble']:
            logger.warning("Section detection failed — returning full text only")
            sections = {"full_text": text}

        return sections

    def _extract_figure_refs(self, text: str) -> list:
        """Extract references to figures and tables."""
        figure_refs = []

        # Match "Figure X" or "Fig. X" or "Table X"
        patterns = [
            r'(?:Figure|Fig\.?)\s+(\d+)(?:\s*[:.]?\s*([^\n]{10,80}))?',
            r'(?:Table)\s+(\d+)(?:\s*[:.]?\s*([^\n]{10,80}))?',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                ref_type = "Figure" if "Fig" in pattern else "Table"
                num = match.group(1)
                caption = match.group(2).strip() if match.group(2) else ""
                ref = f"{ref_type} {num}"
                if caption:
                    ref += f": {caption}"
                if ref not in figure_refs:
                    figure_refs.append(ref)

        return figure_refs
