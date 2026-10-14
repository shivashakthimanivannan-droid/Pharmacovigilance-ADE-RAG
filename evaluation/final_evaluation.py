import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from retrieval.retriever import hybrid_retrieve
from generation.llm_generator import generate_from_retrieval


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "final_evaluation_results.json"
)

EXPECTED_TOTAL_DOCUMENTS = 10
EXPECTED_PUBMED_DOCUMENTS = 5
EXPECTED_FAERS_DOCUMENTS = 5


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

EVALUATION_QUERIES = [
    {
        "id": "Q1",
        "query": "What are the risks of combining warfarin with aspirin?"
    },
    {
        "id": "Q2",
        "query": "What adverse events are associated with anticoagulant therapy?"
    },
    {
        "id": "Q3",
        "query": "What are the safety concerns related to gastrointestinal bleeding?"
    }
]


# ============================================================
# SAFE TEXT
# ============================================================

def safe_text(value: Any) -> str:

    if value is None:
        return ""

    return str(value).strip()


# ============================================================
# SOURCE DETECTION
# ============================================================

def get_source(document: Dict[str, Any]) -> str:

    source = safe_text(
        document.get("source", "")
    ).lower()

    if source:
        return source

    if document.get("pmid") or document.get("PMID"):
        return "pubmed"

    if document.get("case_id") or document.get("CASEID"):
        return "faers"

    return "unknown"


# ============================================================
# CITATION EXTRACTION
# ============================================================

def extract_citations(answer: str) -> Set[str]:

    citations = set()

    # PubMed citations
    pubmed_matches = re.findall(
        r"PMID[:\s-]*(\d+)",
        answer,
        flags=re.IGNORECASE
    )

    for pmid in pubmed_matches:

        citations.add(
            f"PMID:{pmid}"
        )


    # FAERS citations
    faers_matches = re.findall(
        r"FAERS[-\s]*CASE[:\s-]*(\d+)",
        answer,
        flags=re.IGNORECASE
    )

    for case_id in faers_matches:

        citations.add(
            f"FAERS-CASE:{case_id}"
        )


    return citations


# ============================================================
# DOCUMENT CITATIONS
# ============================================================

def get_document_citation(
    document: Dict[str, Any]
) -> str:

    source = get_source(document)

    if source == "pubmed":

        pmid = safe_text(
            document.get(
                "pmid",
                document.get("PMID", "")
            )
        )

        if pmid:

            return f"PMID:{pmid}"


    if source == "faers":

        case_id = safe_text(
            document.get(
                "case_id",
                document.get("CASEID", "")
            )
        )

        if case_id:

            return f"FAERS-CASE:{case_id}"


    return ""


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def evaluate_retrieval(
    documents: List[Dict[str, Any]]
) -> Dict[str, Any]:

    total = len(documents)

    pubmed_count = sum(
        1
        for document in documents
        if get_source(document) == "pubmed"
    )

    faers_count = sum(
        1
        for document in documents
        if get_source(document) == "faers"
    )


    total_success = (
        total == EXPECTED_TOTAL_DOCUMENTS
    )

    pubmed_success = (
        pubmed_count == EXPECTED_PUBMED_DOCUMENTS
    )

    faers_success = (
        faers_count == EXPECTED_FAERS_DOCUMENTS
    )


    retrieval_score = (
        (
            int(total_success)
            + int(pubmed_success)
            + int(faers_success)
        )
        / 3
    ) * 100


    return {
        "total_documents": total,
        "pubmed_documents": pubmed_count,
        "faers_documents": faers_count,
        "total_success": total_success,
        "pubmed_balance_success": pubmed_success,
        "faers_balance_success": faers_success,
        "retrieval_score": round(
            retrieval_score,
            2
        )
    }


# ============================================================
# CITATION VALIDITY
# ============================================================

def evaluate_citations(
    answer: str,
    documents: List[Dict[str, Any]]
) -> Dict[str, Any]:

    generated_citations = extract_citations(
        answer
    )


    retrieved_citations = set()

    for document in documents:

        citation = get_document_citation(
            document
        )

        if citation:

            retrieved_citations.add(
                citation
            )


    if not generated_citations:

        return {
            "generated_citations": [],
            "retrieved_citations": sorted(
                retrieved_citations
            ),
            "valid_citations": [],
            "invalid_citations": [],
            "citation_validity": 0.0
        }


    valid_citations = (
        generated_citations
        & retrieved_citations
    )

    invalid_citations = (
        generated_citations
        - retrieved_citations
    )


    validity = (
        len(valid_citations)
        / len(generated_citations)
    ) * 100


    return {
        "generated_citations": sorted(
            generated_citations
        ),
        "retrieved_citations": sorted(
            retrieved_citations
        ),
        "valid_citations": sorted(
            valid_citations
        ),
        "invalid_citations": sorted(
            invalid_citations
        ),
        "citation_validity": round(
            validity,
            2
        )
    }


# ============================================================
# CITATION COVERAGE / GROUNDING
# ============================================================

def evaluate_grounding(
    answer: str,
    documents: List[Dict[str, Any]]
) -> Dict[str, Any]:

    citation_result = evaluate_citations(
        answer,
        documents
    )

    generated_citations = set(
        citation_result[
            "generated_citations"
        ]
    )

    retrieved_citations = set(
        citation_result[
            "retrieved_citations"
        ]
    )


    if not generated_citations:

        return {
            "grounded_citations": 0,
            "total_generated_citations": 0,
            "grounding_score": 0.0
        }


    grounded = (
        generated_citations
        & retrieved_citations
    )


    grounding_score = (
        len(grounded)
        / len(generated_citations)
    ) * 100


    return {
        "grounded_citations": len(
            grounded
        ),
        "total_generated_citations": len(
            generated_citations
        ),
        "grounding_score": round(
            grounding_score,
            2
        )
    }


# ============================================================
# SOURCE COVERAGE
# ============================================================

def evaluate_source_coverage(
    answer: str,
    documents: List[Dict[str, Any]]
) -> Dict[str, Any]:

    answer_lower = answer.lower()


    has_pubmed_evidence = any(
        get_source(document) == "pubmed"
        for document in documents
    )

    has_faers_evidence = any(
        get_source(document) == "faers"
        for document in documents
    )


    has_pubmed_citation = bool(
        re.search(
            r"PMID[:\s-]*\d+",
            answer,
            flags=re.IGNORECASE
        )
    )

    has_faers_citation = bool(
        re.search(
            r"FAERS[-\s]*CASE[:\s-]*\d+",
            answer,
            flags=re.IGNORECASE
        )
    )


    pubmed_score = (
        100
        if (
            not has_pubmed_evidence
            or has_pubmed_citation
        )
        else 0
    )


    faers_score = (
        100
        if (
            not has_faers_evidence
            or has_faers_citation
        )
        else 0
    )


    overall = (
        pubmed_score
        + faers_score
    ) / 2


    return {
        "pubmed_evidence_available": has_pubmed_evidence,
        "faers_evidence_available": has_faers_evidence,
        "pubmed_citation_present": has_pubmed_citation,
        "faers_citation_present": has_faers_citation,
        "source_coverage_score": round(
            overall,
            2
        )
    }


# ============================================================
# GENERATION SUCCESS
# ============================================================

def evaluate_generation(
    answer: str
) -> Dict[str, Any]:

    answer = safe_text(answer)


    success = (
        len(answer) > 50
        and "unable to generate" not in answer.lower()
        and "generation error" not in answer.lower()
    )


    return {
        "generation_success": success,
        "answer_length": len(answer),
        "generation_score": 100.0 if success else 0.0
    }


# ============================================================
# ANSWER QUALITY SIGNALS
# ============================================================

def evaluate_answer_structure(
    answer: str
) -> Dict[str, Any]:

    answer_lower = answer.lower()


    summary_present = (
        "summary" in answer_lower
    )

    evidence_present = (
        "evidence" in answer_lower
    )

    limitations_present = (
        "limitation" in answer_lower
    )

    faers_section_present = (
        "faers signals" in answer_lower
        or "faers" in answer_lower
    )

    structure_count = sum(
        [
            summary_present,
            evidence_present,
            limitations_present,
            faers_section_present
        ]
    )


    structure_score = (
        structure_count / 4
    ) * 100


    return {
        "summary_present": summary_present,
        "evidence_present": evidence_present,
        "limitations_present": limitations_present,
        "faers_section_present": faers_section_present,
        "structure_score": round(
            structure_score,
            2
        )
    }


# ============================================================
# OVERALL SCORE
# ============================================================

def calculate_overall_score(
    retrieval_score: float,
    generation_score: float,
    citation_score: float,
    grounding_score: float,
    source_coverage_score: float,
    structure_score: float
) -> float:

    overall = (
        retrieval_score * 0.25
        + generation_score * 0.15
        + citation_score * 0.20
        + grounding_score * 0.20
        + source_coverage_score * 0.10
        + structure_score * 0.10
    )


    return round(
        overall,
        2
    )


# ============================================================
# EVALUATE ONE QUERY
# ============================================================

def evaluate_query(
    query_id: str,
    query: str
) -> Dict[str, Any]:

    print("\n")
    print("=" * 80)
    print(f"{query_id}: {query}")
    print("=" * 80)


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    print("\n[1/4] Running hybrid retrieval...")


    try:

        documents = hybrid_retrieve(
            query
        )

    except Exception as error:

        print("\nRetrieval failed:")
        print(error)

        return {
            "query_id": query_id,
            "query": query,
            "status": "FAILED",
            "error_stage": "retrieval",
            "error": str(error)
        }


    retrieval_metrics = evaluate_retrieval(
        documents
    )


    print(
        f"Retrieved: "
        f"{retrieval_metrics['total_documents']} "
        f"documents"
    )

    print(
        f"PubMed: "
        f"{retrieval_metrics['pubmed_documents']}"
    )

    print(
        f"FAERS: "
        f"{retrieval_metrics['faers_documents']}"
    )


    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    print("\n[2/4] Generating answer...")


    try:

        result = generate_from_retrieval(
            query=query,
            retrieved_documents=documents
        )

        answer = safe_text(
            result.get("answer", "")
        )

    except Exception as error:

        print("\nGeneration failed:")
        print(error)

        return {
            "query_id": query_id,
            "query": query,
            "status": "FAILED",
            "error_stage": "generation",
            "error": str(error),
            "retrieval": retrieval_metrics
        }


    generation_metrics = evaluate_generation(
        answer
    )


    # --------------------------------------------------------
    # CITATION EVALUATION
    # --------------------------------------------------------

    print("\n[3/4] Evaluating citations...")


    citation_metrics = evaluate_citations(
        answer,
        documents
    )


    grounding_metrics = evaluate_grounding(
        answer,
        documents
    )


    source_metrics = evaluate_source_coverage(
        answer,
        documents
    )


    structure_metrics = evaluate_answer_structure(
        answer
    )


    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    overall_score = calculate_overall_score(
        retrieval_score=retrieval_metrics[
            "retrieval_score"
        ],
        generation_score=generation_metrics[
            "generation_score"
        ],
        citation_score=citation_metrics[
            "citation_validity"
        ],
        grounding_score=grounding_metrics[
            "grounding_score"
        ],
        source_coverage_score=source_metrics[
            "source_coverage_score"
        ],
        structure_score=structure_metrics[
            "structure_score"
        ]
    )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\n[4/4] Evaluation complete.")

    print(
        f"\nRetrieval Score: "
        f"{retrieval_metrics['retrieval_score']:.2f}%"
    )

    print(
        f"Generation Score: "
        f"{generation_metrics['generation_score']:.2f}%"
    )

    print(
        f"Citation Validity: "
        f"{citation_metrics['citation_validity']:.2f}%"
    )

    print(
        f"Grounding Score: "
        f"{grounding_metrics['grounding_score']:.2f}%"
    )

    print(
        f"Source Coverage: "
        f"{source_metrics['source_coverage_score']:.2f}%"
    )

    print(
        f"Answer Structure: "
        f"{structure_metrics['structure_score']:.2f}%"
    )

    print(
        f"\nOVERALL SCORE: "
        f"{overall_score:.2f}/100"
    )


    return {
        "query_id": query_id,
        "query": query,
        "status": "SUCCESS",
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
        "citations": citation_metrics,
        "grounding": grounding_metrics,
        "source_coverage": source_metrics,
        "answer_structure": structure_metrics,
        "overall_score": overall_score,
        "answer": answer,
        "retrieved_documents": [
            {
                "source": get_source(document),
                "citation": get_document_citation(
                    document
                ),
                "title": safe_text(
                    document.get("title", "")
                ),
                "score": document.get(
                    "score",
                    document.get(
                        "final_score",
                        document.get(
                            "rerank_score",
                            None
                        )
                    )
                )
            }
            for document in documents
        ]
    }


# ============================================================
# AGGREGATE RESULTS
# ============================================================

def aggregate_results(
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:

    successful_results = [
        result
        for result in results
        if result.get("status") == "SUCCESS"
    ]


    total_queries = len(results)

    successful_queries = len(
        successful_results
    )


    if not successful_results:

        return {
            "queries_evaluated": total_queries,
            "successful_queries": 0,
            "retrieval_success_rate": 0.0,
            "generation_success_rate": 0.0,
            "average_retrieval_score": 0.0,
            "average_generation_score": 0.0,
            "average_citation_validity": 0.0,
            "average_grounding_score": 0.0,
            "average_source_coverage": 0.0,
            "average_structure_score": 0.0,
            "overall_score": 0.0
        }


    retrieval_scores = [
        result["retrieval"]["retrieval_score"]
        for result in successful_results
    ]

    generation_scores = [
        result["generation"]["generation_score"]
        for result in successful_results
    ]

    citation_scores = [
        result["citations"]["citation_validity"]
        for result in successful_results
    ]

    grounding_scores = [
        result["grounding"]["grounding_score"]
        for result in successful_results
    ]

    source_scores = [
        result["source_coverage"][
            "source_coverage_score"
        ]
        for result in successful_results
    ]

    structure_scores = [
        result["answer_structure"][
            "structure_score"
        ]
        for result in successful_results
    ]

    overall_scores = [
        result["overall_score"]
        for result in successful_results
    ]


    return {
        "queries_evaluated": total_queries,
        "successful_queries": successful_queries,

        "retrieval_success_rate": round(
            (
                successful_queries
                / total_queries
            ) * 100,
            2
        ),

        "generation_success_rate": round(
            (
                successful_queries
                / total_queries
            ) * 100,
            2
        ),

        "average_retrieval_score": round(
            sum(retrieval_scores)
            / len(retrieval_scores),
            2
        ),

        "average_generation_score": round(
            sum(generation_scores)
            / len(generation_scores),
            2
        ),

        "average_citation_validity": round(
            sum(citation_scores)
            / len(citation_scores),
            2
        ),

        "average_grounding_score": round(
            sum(grounding_scores)
            / len(grounding_scores),
            2
        ),

        "average_source_coverage": round(
            sum(source_scores)
            / len(source_scores),
            2
        ),

        "average_structure_score": round(
            sum(structure_scores)
            / len(structure_scores),
            2
        ),

        "overall_score": round(
            sum(overall_scores)
            / len(overall_scores),
            2
        )
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results: List[Dict[str, Any]],
    aggregate: Dict[str, Any]
) -> None:

    output = {
        "project": (
            "Pharmacovigilance & "
            "Adverse Drug Event RAG System"
        ),
        "evaluation_type": (
            "End-to-end RAG evaluation"
        ),
        "evaluation_queries": EVALUATION_QUERIES,
        "aggregate_metrics": aggregate,
        "query_results": results
    }


    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("PHARMACOVIGILANCE RAG - FINAL EVALUATION")
    print("=" * 80)

    print(
        "\nEvaluation queries: "
        f"{len(EVALUATION_QUERIES)}"
    )


    results = []


    for item in EVALUATION_QUERIES:

        result = evaluate_query(
            query_id=item["id"],
            query=item["query"]
        )

        results.append(
            result
        )


    # --------------------------------------------------------
    # AGGREGATE
    # --------------------------------------------------------

    aggregate = aggregate_results(
        results
    )


    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("FINAL EVALUATION REPORT")
    print("=" * 80)


    print(
        f"\nQueries evaluated: "
        f"{aggregate['queries_evaluated']}"
    )

    print(
        f"Successful queries: "
        f"{aggregate['successful_queries']}"
    )

    print(
        f"\nRetrieval success rate: "
        f"{aggregate['retrieval_success_rate']:.2f}%"
    )

    print(
        f"Generation success rate: "
        f"{aggregate['generation_success_rate']:.2f}%"
    )

    print(
        f"\nAverage retrieval score: "
        f"{aggregate['average_retrieval_score']:.2f}%"
    )

    print(
        f"Average citation validity: "
        f"{aggregate['average_citation_validity']:.2f}%"
    )

    print(
        f"Average grounding score: "
        f"{aggregate['average_grounding_score']:.2f}%"
    )

    print(
        f"Average source coverage: "
        f"{aggregate['average_source_coverage']:.2f}%"
    )

    print(
        f"Average answer structure: "
        f"{aggregate['average_structure_score']:.2f}%"
    )

    print(
        f"\nOVERALL SYSTEM SCORE: "
        f"{aggregate['overall_score']:.2f}/100"
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_results(
        results=results,
        aggregate=aggregate
    )


    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_PATH
    )


    print("\n")
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()