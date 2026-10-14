import requests
import json
from pathlib import Path
import xml.etree.ElementTree as ET


BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(query, max_results=20):
    """Search PubMed and return PubMed IDs."""

    url = f"{BASE_URL}/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    return data["esearchresult"]["idlist"]


def fetch_pubmed_articles(pubmed_ids):
    """Fetch PubMed article metadata and abstracts."""

    if not pubmed_ids:
        return []

    url = f"{BASE_URL}/efetch.fcgi"

    params = {
        "db": "pubmed",
        "id": ",".join(pubmed_ids),
        "retmode": "xml"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    root = ET.fromstring(response.text)

    articles = []

    for article in root.findall(".//PubmedArticle"):

        pmid = article.findtext(
            ".//PMID",
            default=""
        )

        title_element = article.find(".//ArticleTitle")

        title = ""

        if title_element is not None:
            title = "".join(
                title_element.itertext()
            )

        abstract_parts = []

        for abstract_text in article.findall(
            ".//AbstractText"
        ):
            text = "".join(
                abstract_text.itertext()
            )

            abstract_parts.append(text)

        abstract = " ".join(
            abstract_parts
        )

        journal = article.findtext(
            ".//Journal/Title",
            default=""
        )

        year = ""

        year_element = article.find(
            ".//PubDate/Year"
        )

        if year_element is not None:
            year = year_element.text

        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year
        })

    return articles


def save_articles(articles):

    output_path = Path(
        "data/raw/pubmed.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            articles,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"Saved {len(articles)} articles"
    )

    print(
        f"File: {output_path}"
    )


if __name__ == "__main__":

    query = (
        '"adverse drug event"[Title/Abstract] '
        'OR "drug-drug interaction"[Title/Abstract] '
        'OR "adverse drug reaction"[Title/Abstract]'
    )

    print("Searching PubMed...")

    pubmed_ids = search_pubmed(
        query,
        max_results=20
    )

    print(
        f"Found {len(pubmed_ids)} PubMed articles"
    )

    print("Fetching article data...")

    articles = fetch_pubmed_articles(
        pubmed_ids
    )

    save_articles(articles)

    print("\nPubMed ingestion completed!")
    