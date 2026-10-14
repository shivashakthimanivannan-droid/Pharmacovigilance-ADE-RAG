"""Reproducible synthetic grounding benchmark for the Pharmacovigilance RAG project.

Commands:
    python -m evaluation.benchmark --self-test
    python -m evaluation.benchmark --mode deterministic
    python -m evaluation.benchmark --mode llm

WARNING: Synthetic labels are authored test expectations, not clinician annotations.
LLM judgments are provisional. Do not report these scores as clinical accuracy.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.grounding_evaluator import evaluate_answer

HERE = Path(__file__).resolve().parent
LABELS = (
    "supported", "contradicted", "insufficient_evidence", "invalid_citation",
    "unsupported", "uncited", "methodological_disclaimer", "needs_review",
)


def is_methodological_disclaimer(claim: str) -> bool:
    """Narrow, transparent rule; not an entailment or clinical accuracy check."""
    text = " ".join(claim.lower().split())
    return (
        ("faers" in text or "spontaneous report" in text)
        and ("do not establish causality" in text or "cannot establish causality" in text)
        and ("incidence" in text)
        and not any(token in text for token in ("pmid:", "faers-case:", "rr ", "hr "))
    )


def load_cases(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"]
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate benchmark case IDs")
    for case in cases:
        if case["expected"] not in LABELS:
            raise ValueError(f"Invalid expected label in {case['id']}")
        if not case.get("documents") or not case.get("claim"):
            raise ValueError(f"Missing claim or evidence in {case['id']}")
    return data, cases


def classify(case: dict[str, Any], mode: str) -> dict[str, Any]:
    claim = case["claim"]
    if is_methodological_disclaimer(claim):
        return {"predicted": "methodological_disclaimer", "reason": "Recognized standard FAERS methodological disclaimer"}
    result = evaluate_answer(claim, case["documents"], use_llm=(mode == "llm"))
    items = result["claims"]
    if len(items) != 1:
        raise ValueError(f"Expected exactly one parsed claim in {case['id']}, got {len(items)}")
    return {"predicted": items[0]["status"], "reason": items[0]["reason"]}


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for row in rows:
        matrix[row["expected"]][row["predicted"]] += 1
    by_label = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label][other] for other in LABELS if other != label)
        support = sum(matrix[label].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_label[label] = {"support": support, "precision": round(precision, 4),
                           "recall": round(recall, 4), "f1": round(f1, 4)}
    total = len(rows)
    correct = sum(r["expected"] == r["predicted"] for r in rows)
    observed = [label for label in LABELS if by_label[label]["support"]]
    return {"total": total, "correct": correct, "accuracy": round(correct / total, 4) if total else 0,
            "macro_f1_present_labels": round(sum(by_label[k]["f1"] for k in observed) / len(observed), 4) if observed else 0,
            "labels": list(LABELS), "by_label": by_label, "confusion_matrix": matrix,
            "prediction_counts": dict(Counter(row["predicted"] for row in rows))}


def run(mode: str, path: Path, output: Path | None = None) -> dict[str, Any]:
    metadata, cases = load_cases(path)
    rows = []
    for case in cases:
        outcome = classify(case, mode)
        row = {"id": case["id"], "expected": case["expected"], **outcome}
        rows.append(row)
        print(f"{row['id']:<27} expected={row['expected']:<27} predicted={row['predicted']}")
    result = {"dataset": metadata["dataset_name"], "provenance": metadata["provenance"],
              "mode": mode, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "metrics": metrics(rows), "cases": rows,
              "warning": "Synthetic smoke-test scores are NOT medical validity, real-world accuracy, or independently annotated evaluation."}
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report saved to: {output}")
    print(json.dumps({k: result["metrics"][k] for k in ("total", "correct", "accuracy", "macro_f1_present_labels")}, indent=2))
    return result


def self_test() -> None:
    data, cases = load_cases(HERE / "benchmark_cases.json")
    assert len(cases) == 9
    assert data["dataset_name"]
    assert is_methodological_disclaimer(cases[-1]["claim"])
    assert not is_methodological_disclaimer(cases[0]["claim"])
    m = metrics([{"expected": "supported", "predicted": "supported"},
                 {"expected": "contradicted", "predicted": "insufficient_evidence"}])
    assert m["total"] == 2 and m["correct"] == 1 and m["accuracy"] == 0.5
    result = run("deterministic", HERE / "benchmark_cases.json")
    assert result["metrics"]["total"] == 9
    assert result["metrics"]["prediction_counts"].get("methodological_disclaimer") == 1
    print("Benchmark self-test passed (no API calls).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["deterministic", "llm"], default="deterministic")
    parser.add_argument("--cases", type=Path, default=HERE / "benchmark_cases.json")
    parser.add_argument("--output", type=Path, help="Optional JSON report destination")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    output = args.output or (HERE / "reports" / f"benchmark_{args.mode}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
    run(args.mode, args.cases, output)


if __name__ == "__main__":
    main()
