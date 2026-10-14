import os
import sys

# ============================================================
# FIX PROJECT IMPORT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from retrieval.retriever import search_pubmed
from retrieval.reranker import rerank_documents
from detection.ade_detector import detect_adverse_events
from generation.llm_generator import generate_safety_analysis


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("PHARMACOVIGILANCE & ADE DETECTION PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # USER QUERY
    # --------------------------------------------------------

    query = input(
        "\nEnter your medical query: "
    ).strip()

    if not query:

        print("Query cannot be empty.")

        return

    # --------------------------------------------------------
    # STEP 1 — RETRIEVAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 1: RETRIEVING PUBMED + FAERS EVIDENCE")
    print("=" * 70)

    documents = search_pubmed(
        query,
        limit=10
    )

    if not documents:

        print("\nNo evidence retrieved.")

        return

    print(
        f"\nRetrieved {len(documents)} candidates."
    )

    # --------------------------------------------------------
    # STEP 2 — RERANKING
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 2: SEMANTIC RERANKING")
    print("=" * 70)

    reranked_documents = rerank_documents(
        query,
        documents
    )

    if not reranked_documents:

        print("\nNo documents available after reranking.")

        return

    # --------------------------------------------------------
    # DISPLAY TOP EVIDENCE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TOP RERANKED EVIDENCE")
    print("=" * 70)

    for index, document in enumerate(
        reranked_documents[:5],
        start=1
    ):

        print(
            f"\nRESULT {index}"
        )

        print("-" * 70)

        print(
            f"Source: "
            f"{document.get('source', 'Unknown')}"
        )

        print(
            f"PMID/ID: "
            f"{document.get('pmid', 'Unknown')}"
        )

        print(
            f"Title: "
            f"{document.get('title', 'Unknown')}"
        )

        print(
            f"Journal: "
            f"{document.get('journal', 'Unknown')}"
        )

        print(
            f"Year: "
            f"{document.get('year', 'Unknown')}"
        )

        if "reranker_score" in document:

            print(
                f"Reranker Score: "
                f"{document['reranker_score']}"
            )

        print(
            f"\nText:\n"
            f"{document.get('text', '')[:2000]}"
        )

    # --------------------------------------------------------
    # STEP 3 — ADE DETECTION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 3: ADVERSE DRUG EVENT DETECTION")
    print("=" * 70)

    try:

        ade_results = detect_adverse_events(
            reranked_documents
        )

    except TypeError:

        # Fallback if detector expects text instead
        combined_text = "\n\n".join(
            [
                document.get("text", "")
                for document in reranked_documents
            ]
        )

        ade_results = detect_adverse_events(
            combined_text
        )

    print("\n")
    print("=" * 70)
    print("PHARMACOVIGILANCE FINDINGS")
    print("=" * 70)

    if ade_results:

        print(ade_results)

    else:

        print(
            "No predefined adverse-event terms "
            "detected in the retrieved evidence."
        )

    # --------------------------------------------------------
    # STEP 4 — LLM SAFETY ANALYSIS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 4: AI SAFETY ANALYSIS")
    print("=" * 70)

    try:

        analysis = generate_safety_analysis(
            query=query,
            documents=reranked_documents,
            ade_results=ade_results
        )

    except TypeError:

        # Fallback for simpler generator function
        analysis = generate_safety_analysis(
            query,
            reranked_documents,
            ade_results
        )

    print("\n")
    print("=" * 70)
    print("AI-GENERATED SAFETY ANALYSIS")
    print("=" * 70)

    print(
        analysis
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()