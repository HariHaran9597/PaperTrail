"""
PaperTrail smoke test.

Runs the full pipeline on one known arXiv paper and checks that the
recruiter-demo output fields are present.

Usage:
    python scripts/test_pipeline.py
"""

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from graph.pipeline import analyze_paper


TEST_PAPER = "1706.03762"
REQUIRED_TOP_LEVEL_FIELDS = [
    "parsed_paper",
    "explanations",
    "novelty_analysis",
    "concept_map",
    "questions",
]


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def run_test() -> int:
    print("=" * 60)
    print("PaperTrail smoke test")
    print("=" * 60)

    start = time.time()
    errors: list[str] = []

    try:
        result = analyze_paper(TEST_PAPER)
    except Exception as exc:
        print(f"FAIL: pipeline raised {exc}")
        traceback.print_exc()
        return 1

    elapsed = time.time() - start

    require(result.get("status") == "complete", f"status={result.get('status')}", errors)
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        require(field in result and result.get(field) is not None, f"missing {field}", errors)

    parsed = result.get("parsed_paper", {}).get("parsed", {})
    require(bool(parsed.get("title")), "missing parsed title", errors)
    require(bool(parsed.get("authors")), "missing parsed authors", errors)
    require(bool(parsed.get("abstract")), "missing parsed abstract", errors)
    require(bool(parsed.get("problem_statement")), "missing problem statement", errors)
    require(bool(parsed.get("methodology_summary")), "missing methodology summary", errors)
    require(bool(parsed.get("key_results")), "missing key results", errors)
    require("limitations" in parsed, "missing limitations field", errors)
    require(bool(parsed.get("key_terms")), "missing key terms", errors)

    explanations = result.get("explanations", {})
    for field in ["eli5", "undergrad", "expert", "one_sentence", "key_insight"]:
        require(bool(explanations.get(field)), f"missing explanation field: {field}", errors)

    novelty = result.get("novelty_analysis", {})
    if novelty.get("requires_index"):
        print("WARN: local FAISS index is missing; novelty scoring was correctly skipped.")
    else:
        score = novelty.get("novelty_score")
        require(isinstance(score, int) and 1 <= score <= 10, f"invalid novelty score: {score}", errors)
        require(bool(result.get("related_papers")), "missing related papers", errors)

    concept_map = result.get("concept_map", {})
    require(len(concept_map.get("nodes", [])) >= 3, "too few concept nodes", errors)

    questions = result.get("questions", {})
    require(bool(questions.get("questions_answered")), "missing answered questions", errors)
    require(bool(questions.get("questions_left_open")), "missing open questions", errors)
    require(bool(questions.get("follow_up_reading")), "missing follow-up reading", errors)

    if errors:
        print(f"FAIL in {elapsed:.1f}s")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"PASS in {elapsed:.1f}s")
    print(f"Title: {parsed.get('title', 'N/A')}")
    if novelty.get("requires_index"):
        print("Novelty: skipped until local FAISS index is built")
    else:
        print(f"Novelty: {novelty.get('novelty_score')}/10")
    print(f"Concepts: {len(concept_map.get('nodes', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_test())
