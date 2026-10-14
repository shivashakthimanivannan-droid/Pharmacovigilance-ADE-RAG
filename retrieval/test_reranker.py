from retriever import search_pubmed
from reranker import rerank_documents


def main():

    query = input(
        "\nEnter your medical query: "
    )

    print("\nRetrieving candidates...")

    documents = search_pubmed(
        query,
        limit=10
    )

    print(
        f"Retrieved {len(documents)} candidates."
    )

    ranked_documents = rerank_documents(
        query,
        documents,
        top_k=3
    )

    print("\n")
    print("=" * 70)
    print("RERANKED PUBMED EVIDENCE")
    print("=" * 70)

    for i, item in enumerate(
        ranked_documents,
        start=1
    ):

        document = item["document"]
        score = item["score"]

        properties = document.properties

        print(f"\nRESULT {i}")
        print("-" * 70)

        print(
            f"Reranker Score: {score:.4f}"
        )

        print(
            f"PMID: {properties.get('pmid')}"
        )

        print(
            f"Title: {properties.get('title')}"
        )

        print(
            f"Journal: {properties.get('journal')}"
        )

        print(
            f"Year: {properties.get('year')}"
        )

        print(
            f"\nText:\n{properties.get('text')}"
        )


if __name__ == "__main__":
    main()