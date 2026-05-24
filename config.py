"""Central runtime configuration for PaperTrail."""

import os

from dotenv import load_dotenv


load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MISSING_INDEX_MESSAGE = (
    "Novelty analysis needs the local paper index. "
    "Run python scripts/build_seed_index.py."
)
