"""
PaperTrail End-to-End Test Script
Tests the full pipeline against multiple papers to verify reliability.

Usage:
    python scripts/test_pipeline.py
"""

import sys
import os
import time
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from graph.pipeline import analyze_paper


# Test papers covering different types
TEST_PAPERS = [
    ("1706.03762", "Attention Is All You Need"),       # Seminal ML paper
    ("1810.04805", "BERT"),                             # NLP landmark
    ("2010.11929", "Vision Transformer (ViT)"),        # Computer vision
]


def run_tests():
    """Run the full pipeline on test papers and verify outputs."""
    print("=" * 60)
    print("🔬 PaperTrail — End-to-End Pipeline Test")
    print("=" * 60)
    
    results = []
    
    for arxiv_id, name in TEST_PAPERS:
        print(f"\n{'─' * 50}")
        print(f"📄 Testing: {name} ({arxiv_id})")
        print(f"{'─' * 50}")
        
        start = time.time()
        
        try:
            result = analyze_paper(arxiv_id)
            elapsed = time.time() - start
            
            # Run assertions
            errors = []
            
            if result.get("status") != "complete":
                errors.append(f"Status: {result.get('status')} (expected 'complete')")
                if result.get("error"):
                    errors.append(f"Error: {result['error']}")
            
            # Check Agent 1: Parser
            if not result.get("parsed_paper"):
                errors.append("Missing: parsed_paper")
            else:
                p = result["parsed_paper"]["parsed"]
                if not p.get("title"):
                    errors.append("Missing: parsed title")
                if not p.get("problem_statement"):
                    errors.append("Missing: problem_statement")
                if not p.get("key_results"):
                    errors.append("Missing: key_results")
            
            # Check Agent 2: Explainer
            if not result.get("explanations"):
                errors.append("Missing: explanations")
            else:
                exp = result["explanations"]
                if len(exp.get("eli5", "")) < 30:
                    errors.append("ELI5 explanation too short")
                if len(exp.get("expert", "")) < 50:
                    errors.append("Expert explanation too short")
            
            # Check Agent 3: Novelty
            if not result.get("novelty_analysis"):
                errors.append("Missing: novelty_analysis")
            else:
                nov = result["novelty_analysis"]
                score = nov.get("novelty_score", 0)
                if not (1 <= score <= 10):
                    errors.append(f"Invalid novelty_score: {score}")
            
            # Check Agent 4: Concept Map
            if not result.get("concept_map"):
                errors.append("Missing: concept_map")
            else:
                cm = result["concept_map"]
                if len(cm.get("nodes", [])) < 3:
                    errors.append(f"Too few concept nodes: {len(cm.get('nodes', []))}")
            
            # Check Agent 5: Questions
            if not result.get("questions"):
                errors.append("Missing: questions")
            else:
                q = result["questions"]
                if len(q.get("questions_answered", [])) < 2:
                    errors.append("Too few questions_answered")
            
            # Report
            if errors:
                status = "⚠️ PARTIAL"
                for err in errors:
                    print(f"  ⚠️  {err}")
            else:
                status = "✅ PASS"
            
            print(f"\n  {status} | {name}")
            print(f"  ⏱️  {elapsed:.1f}s")
            print(f"  📝 Title: {result.get('parsed_paper', {}).get('parsed', {}).get('title', 'N/A')}")
            print(f"  🆕 Novelty: {result.get('novelty_analysis', {}).get('novelty_score', 'N/A')}/10")
            print(f"  🕸️ Concepts: {len(result.get('concept_map', {}).get('nodes', []))} nodes")
            
            results.append({"name": name, "status": status, "time": elapsed, "errors": errors})
            
        except Exception as e:
            elapsed = time.time() - start
            print(f"\n  ❌ FAIL | {name}")
            print(f"  Error: {e}")
            traceback.print_exc()
            results.append({"name": name, "status": "❌ FAIL", "time": elapsed, "errors": [str(e)]})
        
        # Rate limiting — wait between papers
        print("  ⏳ Waiting 5s for rate limits...")
        time.sleep(5)
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📊 TEST SUMMARY")
    print(f"{'=' * 60}")
    
    passed = sum(1 for r in results if r["status"] == "✅ PASS")
    partial = sum(1 for r in results if r["status"] == "⚠️ PARTIAL")
    failed = sum(1 for r in results if r["status"] == "❌ FAIL")
    
    for r in results:
        print(f"  {r['status']} {r['name']} ({r['time']:.1f}s)")
    
    print(f"\n  Total: {len(results)} | ✅ {passed} | ⚠️ {partial} | ❌ {failed}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_tests()
