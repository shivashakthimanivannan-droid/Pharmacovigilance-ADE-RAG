import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


INPUT_FILE = Path("data/processed/pubmed_chunks.json")
OUTPUT_FILE = Path("data/processed/pubmed_embeddings.json")

MODEL_NAME = "NeuML/pubmedbert-base-embeddings"

_model = None


def load_model():

    global _model

    if _model is None:

        print("Loading biomedical embedding model...")

        _model = SentenceTransformer(
            MODEL_NAME
        )

        print("Model loaded successfully!")

    return _model


def create_embedding(text):

    model = load_model()

    embedding = model.encode(
        text,
        convert_to_numpy=True
    )

    return embedding.tolist()


def load_chunks():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def generate_embeddings():

    model = load_model()

    chunks = load_chunks()

    print(
        f"Chunks found: {len(chunks)}"
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    documents = []

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        documents.append({

            "pmid": chunk["pmid"],
            "title": chunk["title"],
            "journal": chunk["journal"],
            "year": chunk["year"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "embedding": embedding.tolist()

        })

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\nEmbedding generation completed!"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print(
        f"Embedding dimension: {len(embeddings[0])}"
    )


if __name__ == "__main__":

    generate_embeddings()