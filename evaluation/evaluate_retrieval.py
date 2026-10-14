# ============================================================
# PHARMACOVIGILANCE RAG
# RETRIEVAL EVALUATION
# ============================================================

import os
import sys
import json
from typing import List, Dict, Any
# ============================================================
# PROJECT PATH
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from retrieval.retriever import hybrid_retrieve


# ============================================================
# EVALUATION QUERIES
# ============================================================

TEST_QUERIES = [

    {
        "query": "What are the risks of combining warfarin with aspirin?",

        "expected_terms": [
            "warfarin",
            "aspirin",
            "bleeding",
            "hemorrhage",
            "anticoagulant",
            "antiplatelet"
        ]
    },

    {
        "query": "What adverse events are associated with anticoagulant therapy?",

        "expected_terms": [
            "bleeding",
            "hemorrhage",
            "anticoagulant",
            "thrombosis"
        ]
    },

    {
        "query": "What are the safety concerns related to gastrointestinal bleeding?",

        "expected_terms": [
            "gastrointestinal",
            "bleeding",
            "hemorrhage"
        ]
    }
]


# ============================================================
# SAFE TEXT
# ============================================================

def safe_text(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, list):

        return " ".join(
            safe_text(item)
            for item in value
        )

    if isinstance(value, dict):

        return " ".join(
            f"{safe_text(key)} {safe_text(value)}"
            for key, value in value.items()
        )

    return str(value)


# ============================================================
# CALCULATE RELEVANCE
# ============================================================

def calculate_relevance(
    documents: List[Dict[str, Any]],
    expected_terms: List[str]
) -> Dict[str, Any]:

    if not documents:

        return {
            "relevant_documents": 0,
            "total_documents": 0,
            "relevance_rate": 0.0,
            "matched_terms": []
        }

    relevant_documents = 0

    matched_terms = set()

    for document in documents:

        text = safe_text(
            document.get("text", "")
        ).lower()

        title = safe_text(
            document.get("title", "")
        ).lower()

        combined_text = (
            title
            + " "
            + text
        )

        document_matches = []

        for term in expected_terms:

            if term.lower() in combined_text:

                document_matches.append(
                    term
                )

                matched_terms.add(
                    term
                )

        if document_matches:

            relevant_documents += 1

    total_documents = len(
        documents
    )

    relevance_rate = (
        relevant_documents
        / total_documents
    )

    return {

        "relevant_documents":
            relevant_documents,

        "total_documents":
            total_documents,

        "relevance_rate":
            round(
                relevance_rate,
                3
            ),

        "matched_terms":
            sorted(
                matched_terms
            )
    }


# ============================================================
# SOURCE COVERAGE
# ============================================================

def calculate_source_coverage(
    documents: List[Dict[str, Any]]
) -> Dict[str, int]:

    pubmed = 0
    faers = 0
    other = 0

    for document in documents:

        source = safe_text(
            document.get(
                "source",
                ""
            )
        ).lower()

        if "pubmed" in source:

            pubmed += 1

        elif "faers" in source:

            faers += 1

        else:

            other += 1

    return {

        "PubMed": pubmed,

        "FAERS": faers,

        "Other": other
    }


# ============================================================
# EVALUATE ONE QUERY
# ============================================================

def evaluate_query(
    test_case: Dict[str, Any],
    top_k: int = 10
) -> Dict[str, Any]:

    query = test_case["query"]

    expected_terms = test_case[
        "expected_terms"
    ]

    print("\n" + "=" * 70)

    print(
        f"Evaluating query:\n{query}"
    )

    print("=" * 70)

    try:

        documents = hybrid_retrieve(
            query,
            top_k=top_k
        )

    except Exception as e:

        print(
            f"Retrieval failed: {e}"
        )

        return {

            "query": query,

            "status": "FAILED",

            "error": str(e)
        }

    relevance = calculate_relevance(
        documents,
        expected_terms
    )

    source_coverage = (
        calculate_source_coverage(
            documents
        )
    )

    result = {

        "query": query,

        "status": "SUCCESS",

        "total_retrieved":
            len(documents),

        "relevant_documents":
            relevance[
                "relevant_documents"
            ],

        "relevance_rate":
            relevance[
                "relevance_rate"
            ],

        "matched_terms":
            relevance[
                "matched_terms"
            ],

        "source_coverage":
            source_coverage
    }

    print(
        f"\nRetrieved: "
        f"{result['total_retrieved']}"
    )

    print(
        f"Relevant: "
        f"{result['relevant_documents']}"
    )

    print(
        f"Relevance rate: "
        f"{result['relevance_rate']:.2%}"
    )

    print(
        f"Sources: "
        f"{source_coverage}"
    )

    return result


# ============================================================
# OVERALL EVALUATION
# ============================================================

def run_evaluation(
    top_k: int = 10
):

    print("\n")
    print("=" * 70)
    print(
        "PHARMACOVIGILANCE RAG RETRIEVAL EVALUATION"
    )
    print("=" * 70)

    results = []

    for test_case in TEST_QUERIES:

        result = evaluate_query(
            test_case,
            top_k=top_k
        )

        results.append(
            result
        )

    successful_results = [

        result
        for result in results

        if result.get(
            "status"
        ) == "SUCCESS"
    ]

    if successful_results:

        average_relevance = sum(

            result[
                "relevance_rate"
            ]

            for result in successful_results

        ) / len(
            successful_results
        )

    else:

        average_relevance = 0.0

    summary = {

        "total_test_queries":
            len(TEST_QUERIES),

        "successful_queries":
            len(successful_results),

        "average_relevance_rate":
            round(
                average_relevance,
                3
            ),

        "results":
            results
    }

    print("\n")
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Test queries: "
        f"{summary['total_test_queries']}"
    )

    print(
        f"Successful queries: "
        f"{summary['successful_queries']}"
    )

    print(
        f"Average relevance rate: "
        f"{average_relevance:.2%}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    evaluation_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    output_file = os.path.join(
        evaluation_dir,
        "evaluation_results.json"
    )

    try:

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                summary,
                file,
                indent=4,
                ensure_ascii=False
            )

        print(
            f"\nEvaluation results saved to:"
        )

        print(
            output_file
        )

    except Exception as e:

        print(
            f"Could not save evaluation results: {e}"
        )

    return summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_evaluation(
        top_k=10
    )