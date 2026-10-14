"""Run Gemini-assisted, claim-level grounding evaluation.

From the project root:
    python -m evaluation.run_semantic_audit
    python -m evaluation.run_semantic_audit --query "What are the risks of combining warfarin with aspirin?"
    python -m evaluation.run_semantic_audit --self-test

The evaluator's judgments are provisional, not clinical verification.
Each claim may require a separate Gemini request, so this incurs API usage.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_QUERY = "What are the risks of combining warfarin with aspirin?"
OUTPUT_DIR = Path(__file__).resolve().parent / "reports"


def run_audit(query: str, output: Path | None = None) -> dict[str, Any]:
    from retrieval.retriever import hybrid_retrieve
    from generation.llm_generator import generate_from_retrieval

    print("\nRetrieving evidence and generating an answer...")
    documents = hybrid_retrieve(query)
    print(f"Retrieved {len(documents)} documents")
    if not documents:
        raise RuntimeError("No evidence was retrieved; semantic audit cannot proceed")

    # Existing integration handles citation validation before any grounding check.
    result = generate_from_retrieval(
        query=query,
        retrieved_documents=documents,
        grounding_use_llm=True,
    )

    print("\n" + "=" * 72)
    print("GENERATION STATUS:", result.get("status"))
    print("=" * 72)
    print(result.get("answer", ""))

    report = result.get("grounding_report")
    if not report:
        print("\nGrounding not run:", result.get("grounding_error") or result.get("status"))
        return result

    print("\n" + "=" * 72)
    print("GEMINI-ASSISTED GROUNDING AUDIT (PROVISIONAL)")
    print("=" * 72)
    print("Method:", report.get("method"))
    print("Claims:", report.get("total_claims"))
    print("Counts:", json.dumps(report.get("counts", {}), indent=2))
    for index, item in enumerate(report.get("claims", []), start=1):
        print(f"\n[{index}] {item.get('status', 'unknown').upper()}")
        print("Claim:", item.get("claim", ""))
        print("Citations:", ", ".join(item.get("citations", [])) or "none")
        print("Reason:", item.get("reason", ""))

    output = output or OUTPUT_DIR / f"semantic_audit_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "generation_status": result.get("status"),
        "answer": result.get("answer"),
        "citation_validation": result.get("citation_validation"),
        "grounding_report": report,
        "warning": "Model-generated judgments are not clinically verified. Human review is required.",
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\nAudit saved to:", output)
    return result


def self_test() -> None:
    from evaluation.grounding_evaluator import evaluate_answer
    docs = [{"source": "PubMed", "pmid": "12345", "text": "Major bleeding RR 1.58."}]
    result = evaluate_answer("Major bleeding RR 9.99 (PMID:12345).", docs, use_llm=False)
    assert result["counts"]["unsupported"] == 1, result
    print("Semantic audit runner self-test passed (offline; no Gemini call).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run provisional Gemini claim-grounding audit")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Clinical evidence question")
    parser.add_argument("--output", type=Path, help="JSON report destination")
    parser.add_argument("--self-test", action="store_true", help="Run offline smoke test")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    run_audit(args.query, args.output)


if __name__ == "__main__":
    main()
