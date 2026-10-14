from __future__ import annotations

from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: Optional[CrossEncoder] = None


# ============================================================
# LOAD MODEL
# ============================================================

def load_reranker() -> CrossEncoder:
    global _model

    if _model is None:
        print("\nLoading reranker model...")
        print(f"Model: {MODEL_NAME}")

        _model = CrossEncoder(
            MODEL_NAME
        )

        print("Reranker model loaded successfully.")

    return _model


# ============================================================
# RERANK DOCUMENTS
# ============================================================

def rerank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:

    if not documents:
        return []

    model = load_reranker()

    pairs = []

    valid_documents = []

    for document in documents:

        if not isinstance(document, dict):
            continue

        text = str(
            document.get(
                "text",
                ""
            )
        ).strip()

        if not text:
            continue

        pairs.append(
            [
                query,
                text
            ]
        )

        valid_documents.append(
            document
        )

    if not pairs:
        return []

    print(
        f"Reranking {len(pairs)} documents..."
    )

    scores = model.predict(
        pairs,
        show_progress_bar=False
    )

    reranked = []

    for document, score in zip(
        valid_documents,
        scores
    ):

        updated = document.copy()

        updated[
            "reranker_score"
        ] = float(score)

        reranked.append(
            updated
        )

    reranked.sort(
        key=lambda x: x.get(
            "reranker_score",
            float("-inf")
        ),
        reverse=True
    )

    return reranked[:top_k]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Reranker module loaded successfully."
    )