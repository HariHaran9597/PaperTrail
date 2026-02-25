"""
Seed Index Builder
Fetches ~5,000 CS/ML paper abstracts from arXiv and builds a FAISS index
for the Novelty Detector agent to search against.

Usage:
    python scripts/build_seed_index.py
"""

import arxiv
import json
import os
import sys
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embeddings import PaperIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ArXiv categories to fetch
CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE"]
PAPERS_PER_CATEGORY = 1000


def fetch_papers() -> list[dict]:
    """Fetch paper abstracts from arXiv across multiple CS categories."""
    all_papers = []
    seen_ids = set()
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3.0,  # Be respectful to arXiv API
        num_retries=5,
    )

    for category in CATEGORIES:
        logger.info(f"Fetching {PAPERS_PER_CATEGORY} papers from {category}...")
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=PAPERS_PER_CATEGORY,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        count = 0
        try:
            for paper in client.results(search):
                if paper.entry_id not in seen_ids:
                    seen_ids.add(paper.entry_id)
                    all_papers.append({
                        "arxiv_id": paper.entry_id,
                        "title": paper.title,
                        "abstract": paper.summary,
                        "authors": [a.name for a in paper.authors[:10]],  # Limit authors
                        "categories": paper.categories,
                        "published": str(paper.published),
                    })
                    count += 1

                if count % 100 == 0 and count > 0:
                    logger.info(f"  {category}: fetched {count} papers...")

        except Exception as e:
            logger.warning(f"Error fetching {category}: {e}. Continuing with {count} papers.")
            time.sleep(5)  # Wait before continuing to next category

        logger.info(f"  {category}: total {count} papers fetched")

    logger.info(f"Total unique papers: {len(all_papers)}")
    return all_papers


def main():
    # Create data directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/papers_cache", exist_ok=True)

    # Step 1: Fetch papers from arXiv
    logger.info("=" * 60)
    logger.info("Step 1: Fetching papers from arXiv")
    logger.info("=" * 60)

    seed_path = "data/seed_papers.json"

    if os.path.exists(seed_path):
        logger.info(f"Seed papers already exist at {seed_path}. Loading...")
        with open(seed_path, "r", encoding="utf-8") as f:
            papers = json.load(f)
        logger.info(f"Loaded {len(papers)} papers from cache.")
    else:
        papers = fetch_papers()
        with open(seed_path, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(papers)} papers to {seed_path}")

    # Step 2: Build FAISS index
    logger.info("=" * 60)
    logger.info("Step 2: Building FAISS index")
    logger.info("=" * 60)

    indexer = PaperIndex()
    indexer.build_index(papers)

    # Step 3: Verify
    logger.info("=" * 60)
    logger.info("Step 3: Verification")
    logger.info("=" * 60)

    stats = indexer.get_stats()
    logger.info(f"Index stats: {stats}")

    # Test search
    test_query = "transformer self-attention mechanism for natural language processing"
    results = indexer.search(test_query, top_k=5)

    logger.info(f"\nTest search: '{test_query}'")
    for i, r in enumerate(results, 1):
        logger.info(f"  {i}. [{r['similarity_score']:.3f}] {r['title']}")

    logger.info("\n✅ Seed index built successfully!")
    logger.info(f"  Index: data/faiss_index/papers.index")
    logger.info(f"  Metadata: data/faiss_index/metadata.json")


if __name__ == "__main__":
    main()
