import os
import sys
import pandas as pd
import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(
    os.path.join(PROJECT_ROOT, ".env")
)

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")


# ============================================================
# FAERS FILE
# ============================================================

FAERS_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "faers_rag_documents.csv"
)


# ============================================================
# EXISTING WEAVIATE COLLECTION
# ============================================================

COLLECTION_NAME = "PubMedArticle"


# ============================================================
# EMBEDDING MODEL
# ============================================================

MODEL_NAME = (
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)


# ============================================================
# CONNECT TO WEAVIATE
# ============================================================

def connect_to_weaviate():

    print("\nConnecting to Weaviate...")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=wvc.init.Auth.api_key(
            WEAVIATE_API_KEY
        ),
        skip_init_checks=True
    )

    print("Connected to Weaviate.")

    return client


# ============================================================
# LOAD FAERS DATA
# ============================================================

def load_faers():

    print("\nLoading FAERS documents...")

    df = pd.read_csv(
        FAERS_FILE,
        dtype=str
    )

    df = df.fillna("")

    print(
        f"Total FAERS documents available: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # FIRST TEST: ONLY 5,000 DOCUMENTS
    # --------------------------------------------------------

    df = df.head(5000)

    print(
        f"Documents selected for this test: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# INSERT FAERS DOCUMENTS
# ============================================================

def insert_faers(
    collection,
    df,
    model
):

    print("\nPreparing FAERS documents...")

    texts = []

    case_ids = []

    for _, row in df.iterrows():

        case_id = str(
            row.iloc[0]
        )

        text = (
            "SOURCE: FDA FAERS\n"
            "TYPE: Post-market adverse event report\n"
            "DATABASE: FDA Adverse Event Reporting System\n\n"
            + str(row["RAG_TEXT"])
        )

        case_ids.append(case_id)

        texts.append(text)

    # --------------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------------

    print(
        f"\nCreating biomedical embeddings "
        f"for {len(texts):,} documents..."
    )

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print("\nEmbeddings created successfully.")

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    print("\nUploading FAERS documents to Weaviate...")

    success = 0

    with collection.batch.dynamic() as batch:

        for index, text in enumerate(texts):

            case_id = case_ids[index]

            properties = {
                "pmid": f"FAERS-{case_id}",
                "title": "FAERS Adverse Event Report",
                "journal": "FDA FAERS",
                "year": "2026",
                "chunk_id": index,
                "text": text
            }

            batch.add_object(
                properties=properties,
                vector=embeddings[index].tolist()
            )

            success += 1

            if success % 500 == 0:

                print(
                    f"Uploaded {success:,} / "
                    f"{len(texts):,}"
                )

    print(
        f"\nSuccessfully uploaded "
        f"{success:,} FAERS documents."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FAERS → EXISTING WEAVIATE COLLECTION")
    print("=" * 70)

    client = None

    try:

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        print(
            "\nLoading biomedical embedding model..."
        )

        model = SentenceTransformer(
            MODEL_NAME
        )

        print(
            "Embedding model loaded."
        )

        # ----------------------------------------------------
        # LOAD FAERS
        # ----------------------------------------------------

        df = load_faers()

        # ----------------------------------------------------
        # CONNECT
        # ----------------------------------------------------

        client = connect_to_weaviate()

        # ----------------------------------------------------
        # GET EXISTING COLLECTION
        # ----------------------------------------------------

        print(
            f"\nUsing existing collection: "
            f"{COLLECTION_NAME}"
        )

        collection = client.collections.get(
            COLLECTION_NAME
        )

        # ----------------------------------------------------
        # INSERT
        # ----------------------------------------------------

        insert_faers(
            collection,
            df,
            model
        )

        print("\n" + "=" * 70)
        print("FAERS INGESTION COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:

        print("\n" + "=" * 70)
        print("ERROR")
        print("=" * 70)

        print(
            f"\n{type(e).__name__}: {e}"
        )

    finally:

        if client is not None:

            client.close()

            print(
                "\nWeaviate connection closed."
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()