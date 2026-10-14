import os
import json

import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

INPUT_FILE = "data/processed/pubmed_embeddings.json"

COLLECTION_NAME = "PubMedArticle"


def connect_to_weaviate():

    if not WEAVIATE_URL or not WEAVIATE_API_KEY:
        raise ValueError(
            "WEAVIATE_URL or WEAVIATE_API_KEY is missing from .env"
        )

    print("Connecting to Weaviate...")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(
            WEAVIATE_API_KEY
        ),
        skip_init_checks=True
    )

    print("Weaviate client created.")

    return client


def create_collection(client):

    if client.collections.exists(COLLECTION_NAME):

        print(
            f"Collection '{COLLECTION_NAME}' already exists."
        )

        return client.collections.use(
            COLLECTION_NAME
        )

    collection = client.collections.create(

        name=COLLECTION_NAME,

        vector_config=(
            Configure.Vectors.self_provided()
        ),

        properties=[

            Property(
                name="pmid",
                data_type=DataType.TEXT
            ),

            Property(
                name="title",
                data_type=DataType.TEXT
            ),

            Property(
                name="journal",
                data_type=DataType.TEXT
            ),

            Property(
                name="year",
                data_type=DataType.TEXT
            ),

            Property(
                name="chunk_id",
                data_type=DataType.INT
            ),

            Property(
                name="text",
                data_type=DataType.TEXT
            )
        ]
    )

    print(
        f"Created collection: {COLLECTION_NAME}"
    )

    return collection


def load_embeddings():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def upload_documents(collection, documents):

    print(
        f"Uploading {len(documents)} documents..."
    )

    with collection.batch.dynamic() as batch:

        for document in documents:

            properties = {

                "pmid": document["pmid"],

                "title": document["title"],

                "journal": document["journal"],

                "year": document["year"],

                "chunk_id": document["chunk_id"],

                "text": document["text"]
            }

            batch.add_object(

                properties=properties,

                vector=document["embedding"]
            )

    print("Upload completed!")


def main():

    client = connect_to_weaviate()

    try:

        print("Checking Weaviate connection...")

        print(
            "Weaviate ready:",
            client.is_ready()
        )

        collection = create_collection(
            client
        )

        documents = load_embeddings()

        upload_documents(
            collection,
            documents
        )

        print()
        print("=" * 60)
        print("WEAVIATE SETUP COMPLETED")
        print("=" * 60)

    finally:

        client.close()

        print("Connection closed.")


if __name__ == "__main__":

    main()