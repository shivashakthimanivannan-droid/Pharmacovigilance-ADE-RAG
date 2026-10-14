import json
import re
from pathlib import Path


INPUT_FILE = Path("data/raw/pubmed.json")
OUTPUT_FILE = Path("data/processed/pubmed_clean.json")


def clean_text(text):
    """Clean text by removing unnecessary whitespace."""

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_articles():

    # Load raw PubMed data
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        articles = json.load(file)

    cleaned_articles = []

    for article in articles:

        title = clean_text(
            article.get("title", "")
        )

        abstract = clean_text(
            article.get("abstract", "")
        )

        # Ignore articles without title and abstract
        if not title and not abstract:
            continue

        # Combine title and abstract
        combined_text = (
            f"Title: {title}\n\n"
            f"Abstract: {abstract}"
        )

        cleaned_articles.append({
            "pmid": article.get("pmid", ""),
            "title": title,
            "abstract": abstract,
            "journal": article.get("journal", ""),
            "year": article.get("year", ""),
            "text": combined_text
        })

    # Create processed directory
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save cleaned data
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            cleaned_articles,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Original articles: {len(articles)}")
    print(f"Clean articles: {len(cleaned_articles)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    clean_articles()