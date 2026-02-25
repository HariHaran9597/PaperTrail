"""
ArXiv Paper Fetcher
Downloads papers and extracts metadata from arXiv using the arxiv Python library.
Handles both abs/ and pdf/ URLs, as well as direct arXiv IDs.
"""

import arxiv
import os
import re
import logging

logger = logging.getLogger(__name__)


class ArxivFetcher:
    """Fetches paper metadata and PDFs from arXiv."""

    def __init__(self, cache_dir: str = "data/papers_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def parse_arxiv_id(self, url_or_id: str) -> str:
        """
        Extract arXiv ID from various URL formats or direct ID input.
        
        Supports:
            - https://arxiv.org/abs/1706.03762
            - https://arxiv.org/pdf/1706.03762
            - https://arxiv.org/abs/1706.03762v1
            - 1706.03762
            - 1706.03762v1
        """
        url_or_id = url_or_id.strip()

        patterns = [
            r'arxiv\.org/abs/([\d.]+(?:v\d+)?)',
            r'arxiv\.org/pdf/([\d.]+(?:v\d+)?)',
            r'^([\d]{4}\.[\d]{4,5}(?:v\d+)?)$',
        ]
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)

        raise ValueError(
            f"Could not parse arXiv ID from: '{url_or_id}'. "
            f"Please provide a valid arXiv URL (e.g., https://arxiv.org/abs/1706.03762) "
            f"or arXiv ID (e.g., 1706.03762)."
        )

    def fetch(self, url_or_id: str) -> dict:
        """
        Fetch paper metadata and download the PDF.
        
        Args:
            url_or_id: arXiv URL or paper ID
            
        Returns:
            dict with keys: arxiv_id, title, authors, abstract, categories,
            published, pdf_path, pdf_url, entry_url
        """
        arxiv_id = self.parse_arxiv_id(url_or_id)
        logger.info(f"Fetching paper: {arxiv_id}")

        # Search for the paper
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])

        try:
            paper = next(client.results(search))
        except StopIteration:
            raise ValueError(f"Paper not found on arXiv: {arxiv_id}")

        # Download PDF if not already cached
        safe_filename = f"{arxiv_id.replace('/', '_')}.pdf"
        pdf_path = os.path.join(self.cache_dir, safe_filename)

        if not os.path.exists(pdf_path):
            logger.info(f"Downloading PDF to {pdf_path}")
            paper.download_pdf(dirpath=self.cache_dir, filename=safe_filename)
        else:
            logger.info(f"PDF already cached at {pdf_path}")

        return {
            "arxiv_id": arxiv_id,
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "abstract": paper.summary,
            "categories": paper.categories,
            "published": str(paper.published),
            "pdf_path": pdf_path,
            "pdf_url": paper.pdf_url,
            "entry_url": paper.entry_id,
        }
