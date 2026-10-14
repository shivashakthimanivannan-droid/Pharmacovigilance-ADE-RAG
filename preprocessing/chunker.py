import json
from pathlib import Path


INPUT_FILE = Path("data/processed/pubmed_clean.json")
OUTPUT_FILE = Path("data/processed/pubmed_chunks.json")


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def create_chunks(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping word-based chunks.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        # Move forward while maintaining overlap
        start += chunk_size - overlap

    return chunks


def chunk_articles():

    # Load cleaned PubMed articles
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        articles = json.load(file)

    chunked_documents = []

    for article in articles:

        text = article.get("text", "")

        if not text:
            continue

        chunks = create_chunks(
            text,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        for index, chunk in enumerate(chunks):

            chunked_documents.append({
                "pmid": article.get("pmid", ""),
                "title": article.get("title", ""),
                "journal": article.get("journal", ""),
                "year": article.get("year", ""),
                "chunk_id": index,
                "text": chunk
            })

    # Make sure output directory exists
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save chunks
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunked_documents,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"Articles processed: {len(articles)}"
    )

    print(
        f"Total chunks created: {len(chunked_documents)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    chunk_articles()