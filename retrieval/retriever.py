
"""Query-aware PubMed + FDA FAERS retrieval for pharmacovigilance research."""

from __future__ import annotations

import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from sentence_transformers import CrossEncoder
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FAERS_PATH = Path(
    os.getenv("FAERS_PATH", "data/processed/faers_rag_documents.csv")
)
if not FAERS_PATH.is_absolute():
    FAERS_PATH = PROJECT_ROOT / FAERS_PATH

PUBMED_CANDIDATE_LIMIT = 60
FAERS_CANDIDATE_LIMIT = 40
PUBMED_FINAL = 5
FAERS_FINAL = 5
FINAL_LIMIT = 10

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

DRUG_ALIASES = {
    "warfarin": ["warfarin", "warfarin sodium", "coumadin"],
    "aspirin": ["aspirin", "acetylsalicylic acid"],
    "apixaban": ["apixaban", "eliquis"],
    "rivaroxaban": ["rivaroxaban", "xarelto"],
    "heparin": ["heparin", "unfractionated heparin"],
    "enoxaparin": ["enoxaparin", "lovenox"],
    "dabigatran": ["dabigatran", "pradaxa"],
    "edoxaban": ["edoxaban", "savaysa"],
    "clopidogrel": ["clopidogrel", "plavix"],
    "ibuprofen": ["ibuprofen", "advil", "motrin"],
    "naproxen": ["naproxen", "aleve"],
    "diclofenac": ["diclofenac", "voltaren"],
    "amiodarone": ["amiodarone", "cordarone"],

    # Additional common drugs
    "metformin": ["metformin", "metformin hydrochloride", "glucophage"],
    "insulin": ["insulin"],
    "glimepiride": ["glimepiride", "amaryl"],
    "gliclazide": ["gliclazide"],
    "empagliflozin": ["empagliflozin", "jardiance"],
    "dapagliflozin": ["dapagliflozin", "farxiga", "forxiga"],
    "semaglutide": ["semaglutide", "ozempic", "rybelsus", "wegovy"],
    "sitagliptin": ["sitagliptin", "januvia"],
    "atorvastatin": ["atorvastatin", "lipitor"],
    "rosuvastatin": ["rosuvastatin", "crestor"],
    "amlodipine": ["amlodipine", "norvasc"],
    "losartan": ["losartan", "cozaar"],
    "lisinopril": ["lisinopril", "zestril", "prinivil"],
    "omeprazole": ["omeprazole", "prilosec"],
    "pantoprazole": ["pantoprazole", "protonix"],
    "paracetamol": ["paracetamol", "acetaminophen", "tylenol"],
    "amoxicillin": ["amoxicillin", "amoxil"],
    "azithromycin": ["azithromycin", "zithromax"],
}

DRUG_CLASS_ALIASES = {
    "anticoagulant": [
        "anticoagulant", "anticoagulants", "anticoagulation"
    ],
    "antiplatelet": [
        "antiplatelet", "antiplatelets", "antiplatelet therapy"
    ],
    "statin": ["statin", "statins"],
}

DRUG_CLASS_TO_DRUGS = {
    "anticoagulant": [
        "warfarin", "apixaban", "rivaroxaban", "heparin",
        "enoxaparin", "dabigatran", "edoxaban"
    ],
    "antiplatelet": ["aspirin", "clopidogrel"],
    "statin": ["atorvastatin", "rosuvastatin"],
}

EVENT_ALIASES = {
    "gastrointestinal bleeding": [
        "gastrointestinal bleeding", "gastrointestinal bleed",
        "gastrointestinal haemorrhage", "gastrointestinal hemorrhage",
        "gi bleeding", "gi bleed", "hematochezia", "melena"
    ],
    "bleeding": [
        "bleeding", "bleed", "haemorrhage", "hemorrhage",
        "haemorrhagic", "hemorrhagic"
    ],
    "stroke": [
        "stroke", "cerebrovascular accident",
        "ischemic stroke", "ischaemic stroke"
    ],
    "thrombosis": [
        "thrombosis", "thromboembolism", "thromboembolic",
        "pulmonary embolism", "deep vein thrombosis"
    ],
    "coagulation abnormality": [
        "inr increased", "supratherapeutic inr",
        "international normalized ratio increased"
    ],
    "drug interaction": [
        "drug interaction", "drug-drug interaction"
    ],
    "lactic acidosis": [
        "lactic acidosis", "hyperlactatemia"
    ],
    "hypoglycemia": [
        "hypoglycemia", "hypoglycaemia", "low blood sugar"
    ],
    "nausea": ["nausea", "nauseous"],
    "vomiting": ["vomiting", "emesis"],
    "diarrhea": ["diarrhea", "diarrhoea"],
    "kidney injury": [
        "acute kidney injury", "renal injury", "renal failure"
    ],
    "hepatotoxicity": [
        "hepatotoxicity", "liver injury", "drug-induced liver injury"
    ],
}

SAFETY_TERMS = [
    "adverse event", "adverse events", "adverse effect",
    "adverse effects", "adverse reaction", "side effect",
    "side effects", "drug safety", "safety", "toxicity",
    "complication", "risk", "drug interaction",
    "bleeding", "mortality", "hospitalization",
    "lactic acidosis", "hypoglycemia"
]

COMPARATOR_TERMS = [
    "versus", "vs.", "compared with", "compared to",
    "either warfarin or aspirin", "warfarin or aspirin"
]

COMBINATION_TERMS = [
    "combination", "combined", "concomitant",
    "coadministration", "co-administered", "dual therapy",
    "plus", "addition of", "added to", "together"
]


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", safe_text(value).lower()).strip()


def alias_pattern(alias: str) -> str:
    return rf"(?<!\w){re.escape(normalize_text(alias))}(?!\w)"


def term_present(text: Any, term: str) -> bool:
    return bool(
        term and re.search(alias_pattern(term), normalize_text(text))
    )


def contains_any(text: Any, terms: List[str]) -> bool:
    return any(term_present(text, term) for term in terms)


def get_term_positions(text: Any, term: str) -> List[int]:
    return [
        match.start()
        for match in re.finditer(alias_pattern(term), normalize_text(text))
    ]


def drug_present(text: Any, drug: str) -> bool:
    return contains_any(text, DRUG_ALIASES.get(drug, [drug]))


def drugs_in_text(text: Any) -> List[str]:
    return [
        drug for drug in DRUG_ALIASES
        if drug_present(text, drug)
    ]


def split_sentences(text: Any) -> List[str]:
    value = safe_text(text)
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", value)
        if sentence.strip()
    ]


def detect_query_drugs(query: str) -> List[str]:
    return drugs_in_text(query)


def detect_query_drug_classes(query: str) -> List[str]:
    return [
        name for name, aliases in DRUG_CLASS_ALIASES.items()
        if contains_any(query, aliases)
    ]


def detect_query_events(query: str) -> List[str]:
    return [
        event for event, aliases in EVENT_ALIASES.items()
        if contains_any(query, aliases)
    ]


def expand_query(query: str) -> str:
    aliases = [
        alias
        for drug in detect_query_drugs(query)
        for alias in DRUG_ALIASES[drug]
    ]
    return " ".join(
        [query] + list(dict.fromkeys(aliases))
    ).strip()


def pubmed_or_block(terms: List[str]) -> str:
    unique = list(dict.fromkeys(term for term in terms if term))
    if not unique:
        return ""
    return "(" + " OR ".join(
        f'"{term}"[Title/Abstract]' for term in unique
    ) + ")"


def remove_duplicate_queries(queries: List[str]) -> List[str]:
    return list(dict.fromkeys(
        query.strip() for query in queries if query.strip()
    ))


def build_pubmed_queries(query: str) -> List[str]:
    drugs = detect_query_drugs(query)
    classes = detect_query_drug_classes(query)
    events = detect_query_events(query)

    event_terms = [
        alias
        for event in events
        for alias in EVENT_ALIASES[event]
    ]
    event_block = pubmed_or_block(event_terms)
    safety_block = pubmed_or_block(SAFETY_TERMS)

    queries = []

    if len(drugs) >= 2:
        blocks = [
            pubmed_or_block(DRUG_ALIASES[drug])
            for drug in drugs
        ]
        base = " AND ".join(blocks)

        combination = pubmed_or_block([
            "combination", "combined", "concomitant",
            "coadministration", "dual therapy",
            "addition", "plus"
        ])

        queries.extend([
            f"{base} AND {combination} AND {event_block or safety_block}",
            f"{base} AND {combination}",
            f"{base} AND {event_block or safety_block}",
            base
        ])

    elif drugs:
        base = " AND ".join(
            pubmed_or_block(DRUG_ALIASES[drug])
            for drug in drugs
        )

        if event_block:
            queries.append(f"{base} AND {event_block}")

        queries.extend([
            f"{base} AND {safety_block}",
            base
        ])

    elif classes:
        terms = []

        for drug_class in classes:
            terms.extend(DRUG_CLASS_ALIASES[drug_class])
            for drug in DRUG_CLASS_TO_DRUGS[drug_class]:
                terms.extend(DRUG_ALIASES[drug])

        base = pubmed_or_block(terms)

        if event_block:
            queries.append(f"{base} AND {event_block}")

        queries.extend([
            f"{base} AND {safety_block}",
            base
        ])

    elif event_block:
        queries.append(event_block)

    else:
        # Do not silently retrieve generic safety documents
        # for an unrecognized medication name.
        print(
            "No recognized drug, class, or event. "
            "Skipping broad PubMed fallback."
        )

    return remove_duplicate_queries(queries)


@lru_cache(maxsize=1)
def create_pubmed_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount(
        "https://", HTTPAdapter(max_retries=retry)
    )
    session.headers.update({
        "User-Agent": "Pharmacovigilance-RAG/3.1"
    })
    return session


def ncbi_params() -> dict:
    result = {}
    if os.getenv("NCBI_EMAIL"):
        result["email"] = os.getenv("NCBI_EMAIL")
    if os.getenv("NCBI_API_KEY"):
        result["api_key"] = os.getenv("NCBI_API_KEY")
    return result


def search_pubmed(
    query: str,
    max_results: int = 30
) -> List[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        "sort": "relevance",
        **ncbi_params()
    }

    response = create_pubmed_session().get(
        f"{NCBI_BASE}/esearch.fcgi",
        params=params,
        timeout=30
    )
    response.raise_for_status()

    return response.json().get(
        "esearchresult", {}
    ).get("idlist", [])


def fetch_pubmed_articles(
    pubmed_ids: List[str]
) -> List[Dict[str, Any]]:
    if not pubmed_ids:
        return []

    documents = []

    for offset in range(0, len(pubmed_ids), 100):
        batch = pubmed_ids[offset:offset + 100]

        response = create_pubmed_session().get(
            f"{NCBI_BASE}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(map(str, batch)),
                "retmode": "xml",
                **ncbi_params()
            },
            timeout=60
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)

        for article in root.findall(".//PubmedArticle"):
            pmid = safe_text(
                article.findtext(".//MedlineCitation/PMID")
            )

            title_node = article.find(".//ArticleTitle")
            title = (
                "".join(title_node.itertext()).strip()
                if title_node is not None else ""
            )

            journal = safe_text(
                article.findtext(".//Journal/Title")
            )

            year = safe_text(
                article.findtext(".//PubDate/Year")
            )

            if not year:
                medline = safe_text(
                    article.findtext(".//PubDate/MedlineDate")
                )
                match = re.search(
                    r"\b(?:19|20)\d{2}\b", medline
                )
                year = match.group() if match else ""

            abstract_parts = []

            for node in article.findall(
                ".//Abstract/AbstractText"
            ):
                content = "".join(node.itertext()).strip()
                label = safe_text(node.get("Label"))

                if content:
                    abstract_parts.append(
                        f"{label}: {content}"
                        if label else content
                    )

            abstract = " ".join(abstract_parts)
            text = f"{title}. {abstract}".strip()

            if not pmid or not text:
                continue

            documents.append({
                "source": "PubMed",
                "pmid": pmid,
                "document_id": f"PMID:{pmid}",
                "title": title,
                "journal": journal,
                "year": year,
                "abstract": abstract,
                "text": text,
                "content": text,
                "citation": f"PMID:{pmid}",
                "retrieval_score": 0.0
            })

    return documents


def hard_filter_single(text: str, query: str) -> bool:
    drugs = detect_query_drugs(query)
    events = detect_query_events(query)
    classes = detect_query_drug_classes(query)

    if drugs and not all(
        drug_present(text, drug) for drug in drugs
    ):
        return False

    if events and not any(
        contains_any(text, EVENT_ALIASES[event])
        for event in events
    ):
        return False

    if classes and not any(
        contains_any(text, DRUG_CLASS_ALIASES[c])
        or any(
            drug_present(text, drug)
            for drug in DRUG_CLASS_TO_DRUGS[c]
        )
        for c in classes
    ):
        return False

    return True


def hard_filter_document(
    document: Dict[str, Any],
    query: str
) -> bool:
    return hard_filter_single(
        document.get("text")
        or document.get("content", ""),
        query
    )


def drug_pair_patterns(drugs: List[str]) -> List[str]:
    if len(drugs) != 2:
        return []

    a = "(?:" + "|".join(
        alias_pattern(alias)
        for alias in DRUG_ALIASES[drugs[0]]
    ) + ")"

    b = "(?:" + "|".join(
        alias_pattern(alias)
        for alias in DRUG_ALIASES[drugs[1]]
    ) + ")"

    connector = (
        r"\s*(?:\+|plus|and|with|combined with|"
        r"in combination with|together with|"
        r"coadministered with|added to)\s*"
    )

    return [
        a + connector + b,
        b + connector + a
    ]


def classify_pubmed_evidence(
    document: Dict[str, Any],
    query: str
) -> Dict[str, Any]:
    drugs = detect_query_drugs(query)
    title = safe_text(document.get("title"))
    abstract = safe_text(document.get("abstract"))

    sentences = split_sentences(abstract)
    same = [
        sentence for sentence in sentences
        if len(drugs) >= 2 and all(
            drug_present(sentence, drug)
            for drug in drugs
        )
    ]

    explicit = []
    comparison = []
    patterns = drug_pair_patterns(drugs)

    for sentence in [title] + sentences:
        low = normalize_text(sentence)

        if len(drugs) < 2 or not all(
            drug_present(low, drug) for drug in drugs
        ):
            continue

        is_comparison = contains_any(
            low, COMPARATOR_TERMS
        ) or bool(
            re.search(r"\bvs\.?\b", low)
        )

        strong = bool(re.search(
            r"\b(?:concomitant(?:ly)?|"
            r"coadministration|coadministered|"
            r"co-administered|combined therapy|"
            r"combined treatment|combination therapy|"
            r"combination treatment|dual therapy|"
            r"addition of|added to|treated with both|"
            r"receiving both)\b",
            low
        ))

        plus_pair = any(
            re.search(
                pattern.replace(
                    "plus|and|with|combined",
                    "plus|combined"
                ),
                low
            )
            for pattern in patterns
        ) or bool(re.search(r"\s\+\s", low))

        co_treatment = bool(re.search(
            r"\b(?:received|receiving|administered|"
            r"co-treated|treated with|taking both|"
            r"concurrent use|simultaneous use)\b",
            low
        ))

        pair = any(
            re.search(pattern, low)
            for pattern in patterns
        )

        if is_comparison:
            comparison.append(sentence)
            if strong or plus_pair:
                explicit.append(sentence)

        elif strong or plus_pair or (
            pair and co_treatment
        ):
            explicit.append(sentence)

    category = (
        "combination" if explicit
        else "comparison" if comparison
        else "context"
    )

    return {
        "evidence_type": category,
        "combination_sentences": explicit[:4],
        "comparison_sentences": comparison[:4],
        "same_sentence": bool(same),
        "safety_sentence": any(
            contains_any(sentence, SAFETY_TERMS)
            for sentence in same
        ),
        "combination_sentence": bool(explicit),
        "title_drug_matches": sum(
            drug_present(title, drug)
            for drug in drugs
        )
    }


def pubmed_combination_features(
    document: Dict[str, Any],
    query: str
) -> Dict[str, Any]:
    return classify_pubmed_evidence(document, query)


def context_relevance_score(
    document: Dict[str, Any],
    query: str
) -> float:
    drugs = detect_query_drugs(query)
    events = detect_query_events(query)

    text = safe_text(document.get("text"))
    title = safe_text(document.get("title"))

    if drugs and not all(
        drug_present(text, drug) for drug in drugs
    ):
        return 0.0

    features = classify_pubmed_evidence(
        document, query
    )

    score = 12.0 * len(drugs)

    score += 12.0 * sum(
        drug_present(title, drug)
        for drug in drugs
    )

    if len(drugs) >= 2:
        if features["evidence_type"] == "combination":
            score += 110
        elif features["evidence_type"] == "comparison":
            score += 12
        elif features["same_sentence"]:
            score += 8

    if contains_any(title, SAFETY_TERMS):
        score += 12

    if contains_any(text, SAFETY_TERMS):
        score += 8

    for event in events:
        if contains_any(text, EVENT_ALIASES[event]):
            score += 18

    return float(max(0, score))


def calculate_domain_score(
    document: Dict[str, Any],
    query: str
) -> float:
    return context_relevance_score(
        document, query
    )


def retrieve_pubmed_candidates(
    query: str,
    limit: int = PUBMED_CANDIDATE_LIMIT
) -> List[Dict[str, Any]]:
    queries = build_pubmed_queries(query)

    print("\nPubMed retrieval queries:")
    for i, item in enumerate(queries, 1):
        print(f"  [{i}] {item}")

    if not queries:
        return []

    pmids = []
    seen = set()

    for item in queries:
        try:
            results = search_pubmed(
                item,
                max_results=max(35, limit)
            )
        except requests.RequestException as exc:
            print(f"PubMed search failed: {exc}")
            continue

        for pmid in results:
            if pmid not in seen:
                seen.add(pmid)
                pmids.append(pmid)

        if len(pmids) >= limit * 4:
            break

    pmids = pmids[:limit * 4]

    print(
        f"PubMed unique PMIDs collected: {len(pmids)}"
    )

    try:
        articles = fetch_pubmed_articles(pmids)
    except (
        requests.RequestException,
        ET.ParseError
    ) as exc:
        print(f"PubMed fetch failed: {exc}")
        return []

    filtered = [
        doc for doc in articles
        if hard_filter_document(doc, query)
    ]

    for doc in filtered:
        features = classify_pubmed_evidence(
            doc, query
        )
        doc["combination_features"] = features
        doc["evidence_type"] = features["evidence_type"]
        doc["domain_score"] = context_relevance_score(
            doc, query
        )

    filtered.sort(
        key=lambda doc: doc["domain_score"],
        reverse=True
    )

    print(
        f"PubMed after filtering: {len(filtered)}"
    )

    return filtered[:limit]


def extract_faers_field(
    text: str,
    field: str
) -> str:
    match = re.search(
        rf"(?im)^\s*{re.escape(field)}\s*:\s*(.*)$",
        safe_text(text)
    )
    return match.group(1).strip() if match else ""


def parse_faers_document(
    text: str,
    case_id: str = ""
) -> Dict[str, Any]:
    drugs_text = (
        extract_faers_field(text, "DRUGNAME")
        or extract_faers_field(text, "PROD_AI")
    )

    reactions_text = extract_faers_field(
        text, "PT"
    )

    case_id = (
        case_id
        or extract_faers_field(text, "CASEID")
    )

    drugs = list(dict.fromkeys(
        item.strip()
        for item in drugs_text.split(";")
        if item.strip()
    ))

    reactions = list(dict.fromkeys(
        item.strip()
        for item in reactions_text.split(";")
        if item.strip()
    ))

    return {
        "case_id": safe_text(case_id),
        "reported_drugs": drugs,
        "reported_reactions": reactions,
        "drug_count": len(drugs),
        "reaction_count": len(reactions),
        "drug_text": drugs_text,
        "reaction_text": reactions_text
    }


@lru_cache(maxsize=1)
def load_faers_data() -> pd.DataFrame:
    print(f"\nLoading FAERS data from:\n{FAERS_PATH}")

    df = pd.read_csv(
        FAERS_PATH,
        dtype=str,
        low_memory=False,
        usecols=lambda col: col in {
            "CASEID", "RAG_TEXT"
        }
    ).fillna("")

    if "RAG_TEXT" not in df.columns:
        raise ValueError(
            "FAERS CSV must contain RAG_TEXT."
        )

    if "CASEID" not in df.columns:
        df["CASEID"] = ""

    df["_DRUGNAME"] = df[
        "RAG_TEXT"
    ].str.extract(
        r"(?im)^\s*DRUGNAME\s*:\s*([^\r\n]*)",
        expand=False
    ).fillna("")

    df["_PT"] = df[
        "RAG_TEXT"
    ].str.extract(
        r"(?im)^\s*PT\s*:\s*([^\r\n]*)",
        expand=False
    ).fillna("")

    print(
        f"Loaded {len(df):,} FAERS documents."
    )

    return df


def get_faers_search_text(document: Any) -> str:
    if isinstance(document, dict):
        return safe_text(
            document.get("RAG_TEXT")
            or document.get("text")
            or document.get("content")
        )
    return safe_text(document)


def calculate_faers_relevance_score(
    document: Dict[str, Any],
    query: str
) -> float:
    drugs = detect_query_drugs(query)
    events = detect_query_events(query)
    classes = detect_query_drug_classes(query)

    structured = document.get(
        "faers_structured"
    )

    if not isinstance(structured, dict):
        structured = parse_faers_document(
            get_faers_search_text(document),
            safe_text(document.get("case_id"))
        )

    drug_text = structured.get(
        "drug_text", ""
    )
    reaction_text = structured.get(
        "reaction_text", ""
    )

    if drugs and not all(
        drug_present(drug_text, drug)
        for drug in drugs
    ):
        return 0.0

    if events and not any(
        contains_any(
            reaction_text,
            EVENT_ALIASES[event]
        )
        for event in events
    ):
        return 0.0

    if classes and not any(
        any(
            drug_present(drug_text, drug)
            for drug in DRUG_CLASS_TO_DRUGS[c]
        )
        for c in classes
    ):
        return 0.0

    score = 30 * len(drugs)

    if len(drugs) >= 2:
        score += 25

    score += sum(
        35
        for event in events
        if contains_any(
            reaction_text,
            EVENT_ALIASES[event]
        )
    )

    if contains_any(
        reaction_text,
        ["drug interaction"]
    ):
        score += 10

    n_drugs = structured.get(
        "drug_count", 0
    )
    n_reactions = structured.get(
        "reaction_count", 0
    )

    if n_drugs:
        score -= 8 * math.log2(
            max(1, n_drugs)
        )

    if n_reactions:
        score -= 3 * math.log2(
            max(1, n_reactions)
        )

    if n_drugs > 15:
        score -= 15

    if n_drugs > 30:
        score -= 20

    return float(max(0, score))


def search_faers(
    query: str,
    candidate_limit: int = FAERS_CANDIDATE_LIMIT
) -> List[Dict[str, Any]]:
    df = load_faers_data()

    drugs = detect_query_drugs(query)
    events = detect_query_events(query)
    classes = detect_query_drug_classes(query)

    if not (drugs or events or classes):
        print(
            "No recognized drug, class, or event. "
            "Skipping FAERS scan."
        )
        return []

    mask = pd.Series(
        True, index=df.index
    )

    for drug in drugs:
        pattern = "|".join(
            alias_pattern(alias)
            for alias in DRUG_ALIASES[drug]
        )

        mask &= df[
            "_DRUGNAME"
        ].str.contains(
            pattern,
            regex=True,
            case=False,
            na=False
        )

    if events:
        event_mask = pd.Series(
            False, index=df.index
        )

        for event in events:
            pattern = "|".join(
                alias_pattern(alias)
                for alias in EVENT_ALIASES[event]
            )

            event_mask |= df[
                "_PT"
            ].str.contains(
                pattern,
                regex=True,
                case=False,
                na=False
            )

        mask &= event_mask

    if classes:
        class_mask = pd.Series(
            False, index=df.index
        )

        for drug_class in classes:
            for drug in DRUG_CLASS_TO_DRUGS[
                drug_class
            ]:
                pattern = "|".join(
                    alias_pattern(alias)
                    for alias in DRUG_ALIASES[drug]
                )

                class_mask |= df[
                    "_DRUGNAME"
                ].str.contains(
                    pattern,
                    regex=True,
                    case=False,
                    na=False
                )

        mask &= class_mask

    candidates = df.loc[mask]

    print(
        f"FAERS strict candidates: {len(candidates):,}"
    )

    documents = []

    for index, row in candidates.iterrows():
        case_id = safe_text(
            row["CASEID"]
        )
        text = safe_text(
            row["RAG_TEXT"]
        )

        structured = parse_faers_document(
            text, case_id
        )
        case_id = (
            case_id
            or structured["case_id"]
        )

        doc = {
            "source": "FAERS",
            "case_id": case_id,
            "CASEID": case_id,
            "document_id": f"FAERS-{index}",
            "faers_index": str(index),
            "text": text,
            "content": text,
            "RAG_TEXT": text,
            "faers_structured": structured,
            "reported_drugs": structured[
                "reported_drugs"
            ],
            "reported_reactions": structured[
                "reported_reactions"
            ],
            "drug_count": structured[
                "drug_count"
            ],
            "reaction_count": structured[
                "reaction_count"
            ],
            "citation": f"FAERS-CASE:{case_id}"
        }

        score = calculate_faers_relevance_score(
            doc, query
        )

        if score <= 0:
            continue

        doc["domain_score"] = score
        doc["retrieval_score"] = score
        documents.append(doc)

    documents.sort(
        key=lambda doc: doc["domain_score"],
        reverse=True
    )

    unique = []
    seen = set()

    for doc in documents:
        key = (
            doc["case_id"]
            or doc["document_id"]
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(doc)

        if len(unique) >= candidate_limit:
            break

    print(
        f"FAERS candidates sent to reranker: "
        f"{len(unique)}"
    )

    return unique


@lru_cache(maxsize=1)
def get_cross_encoder():
    return CrossEncoder(
        CROSS_ENCODER_MODEL
    )


def build_rerank_text(
    document: Dict[str, Any]
) -> str:
    if document.get("source") == "FAERS":
        structured = document.get(
            "faers_structured", {}
        )
        drugs = structured.get(
            "reported_drugs", []
        )
        reactions = structured.get(
            "reported_reactions", []
        )

        return (
            "FDA FAERS spontaneous report. "
            "Reported drugs: "
            + "; ".join(drugs[:25])
            + ". Reported reactions: "
            + "; ".join(reactions[:30])
            + ". Co-report only; causality not established."
        )

    features = document.get(
        "combination_features", {}
    )
    relevant = " ".join(
        features.get(
            "combination_sentences", []
        )[:2]
    )

    return (
        safe_text(document.get("title"))
        + ". "
        + relevant
        + " "
        + safe_text(
            document.get("abstract")
            or document.get("text")
        )
    )[:3500]


def rerank_documents(
    query: str,
    documents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not documents:
        return []

    pairs = [
        (query, build_rerank_text(doc))
        for doc in documents
    ]

    try:
        scores = get_cross_encoder().predict(
            pairs,
            show_progress_bar=False
        )
    except Exception as exc:
        print(
            f"Cross-encoder failed: {exc}. "
            "Using domain scores."
        )
        scores = [
            doc.get("domain_score", 0.0)
            for doc in documents
        ]

    result = []

    for doc, score in zip(
        documents, scores
    ):
        item = dict(doc)
        item["cross_encoder_score"] = float(
            score
        )
        result.append(item)

    return sorted(
        result,
        key=lambda doc: doc[
            "cross_encoder_score"
        ],
        reverse=True
    )


def document_dedup_key(
    document: Dict[str, Any]
):
    if document.get("source") == "PubMed":
        return (
            "pubmed",
            document.get("pmid")
            or document.get("document_id")
        )

    return (
        "faers",
        document.get("case_id")
        or document.get("document_id")
    )


def combine_results(
    pubmed_documents: List[Dict[str, Any]],
    faers_documents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    result = []
    seen = set()

    for doc in (
        pubmed_documents + faers_documents
    ):
        key = document_dedup_key(doc)

        if key in seen:
            continue

        seen.add(key)
        result.append(doc)

    return result


def min_max_normalize(
    documents: List[Dict[str, Any]],
    score_key: str
) -> List[Dict[str, Any]]:
    if not documents:
        return documents

    scores = [
        float(doc.get(score_key, 0.0))
        for doc in documents
    ]

    lo = min(scores)
    hi = max(scores)

    for doc, score in zip(
        documents, scores
    ):
        doc[
            f"{score_key}_normalized"
        ] = (
            (score - lo) / (hi - lo)
            if hi > lo else 0.5
        )

    return documents


def source_aware_rerank(
    query: str,
    documents: List[Dict[str, Any]],
    pubmed_final: int = PUBMED_FINAL,
    faers_final: int = FAERS_FINAL
) -> List[Dict[str, Any]]:
    if not documents:
        return []

    pubmed = rerank_documents(
        query,
        [
            doc for doc in documents
            if doc.get("source") == "PubMed"
        ]
    )

    faers = rerank_documents(
        query,
        [
            doc for doc in documents
            if doc.get("source") == "FAERS"
        ]
    )

    for group in (pubmed, faers):
        min_max_normalize(
            group, "cross_encoder_score"
        )
        min_max_normalize(
            group, "domain_score"
        )

        for doc in group:
            semantic = doc[
                "cross_encoder_score_normalized"
            ]
            domain = doc[
                "domain_score_normalized"
            ]

            doc["source_aware_score"] = (
                0.40 * semantic
                + 0.60 * domain
            )
            doc["retrieval_score"] = doc[
                "source_aware_score"
            ]

    if len(detect_query_drugs(query)) >= 2:
        pubmed.sort(
            key=lambda doc: (
                {
                    "combination": 2,
                    "comparison": 1,
                    "context": 0
                }.get(
                    doc.get(
                        "evidence_type", "context"
                    ), 0
                ),
                doc["source_aware_score"]
            ),
            reverse=True
        )
    else:
        pubmed.sort(
            key=lambda doc: doc[
                "source_aware_score"
            ],
            reverse=True
        )

    faers.sort(
        key=lambda doc: doc[
            "source_aware_score"
        ],
        reverse=True
    )

    selected = (
        pubmed[:pubmed_final]
        + faers[:faers_final]
    )

    return sorted(
        selected,
        key=lambda doc: doc[
            "source_aware_score"
        ],
        reverse=True
    )


def hybrid_retrieve(
    query: str,
    final_limit: int = FINAL_LIMIT
) -> List[Dict[str, Any]]:
    print(
        "\n"
        + "=" * 70
        + "\nHYBRID RETRIEVAL\nQuery: "
        + query
        + "\n"
        + "=" * 70
    )

    if not safe_text(query) or final_limit <= 0:
        return []

    print(
        "Expanded query:", expand_query(query)
    )

    drugs = detect_query_drugs(query)
    classes = detect_query_drug_classes(query)
    events = detect_query_events(query)

    print(
        "Drugs:", drugs,
        "Classes:", classes,
        "Events:", events
    )

    if not (drugs or classes or events):
        print(
            "Unrecognized query context. "
            "Returning no evidence instead of "
            "unrelated generic results."
        )
        return []

    pubmed = retrieve_pubmed_candidates(
        query, PUBMED_CANDIDATE_LIMIT
    )

    faers = search_faers(
        query, FAERS_CANDIDATE_LIMIT
    )

    combined = combine_results(
        pubmed, faers
    )

    pubmed_quota = min(
        PUBMED_FINAL,
        math.ceil(final_limit / 2)
    )

    faers_quota = min(
        FAERS_FINAL,
        final_limit - pubmed_quota
    )

    selected = source_aware_rerank(
        query,
        combined,
        pubmed_quota,
        faers_quota
    )

    if len(selected) < final_limit:
        keys = {
            document_dedup_key(doc)
            for doc in selected
        }

        remaining = [
            doc for doc in combined
            if document_dedup_key(doc)
            not in keys
        ]

        if remaining:
            extra = source_aware_rerank(
                query,
                remaining,
                final_limit,
                final_limit
            )

            selected.extend(
                extra[
                    :final_limit - len(selected)
                ]
            )

    selected = selected[:final_limit]

    print(
        "\nFINAL RETRIEVAL:",
        len(selected)
    )

    for i, doc in enumerate(
        selected, 1
    ):
        if doc.get("source") == "PubMed":
            print(
                f"[{i}] PubMed "
                f"PMID:{doc.get('pmid')} "
                f"type={doc.get('evidence_type')} "
                f"title={doc.get('title')}"
            )
        else:
            print(
                f"[{i}] FAERS "
                f"case={doc.get('case_id')}"
            )

    return selected


retrieve = hybrid_retrieve
hybrid_retrieval = hybrid_retrieve


def run_local_tests():
    warfarin_query = (
        "What are the risks of combining "
        "warfarin with aspirin?"
    )

    metformin_query = (
        "What are the adverse effects "
        "of metformin?"
    )

    assert detect_query_drugs(
        warfarin_query
    ) == ["warfarin", "aspirin"]

    assert detect_query_drugs(
        metformin_query
    ) == ["metformin"]

    metformin_queries = build_pubmed_queries(
        metformin_query
    )

    assert metformin_queries
    assert all(
        "metformin" in item.lower()
        for item in metformin_queries
    )

    assert not build_pubmed_queries(
        "What are the adverse effects of an unknown medicine?"
    )

    def doc(text):
        return {
            "title": text,
            "abstract": text,
            "text": text
        }

    combo = classify_pubmed_evidence(
        doc(
            "Patients receiving warfarin plus "
            "aspirin experienced bleeding."
        ),
        warfarin_query
    )

    comparison = classify_pubmed_evidence(
        doc(
            "Warfarin versus aspirin was "
            "compared for stroke prevention."
        ),
        warfarin_query
    )

    context = classify_pubmed_evidence(
        doc(
            "Warfarin is an anticoagulant. "
            "Aspirin is an antiplatelet."
        ),
        warfarin_query
    )

    assert combo["evidence_type"] == "combination"
    assert comparison["evidence_type"] == "comparison"
    assert context["evidence_type"] == "context"

    print(
        "Local retrieval tests passed."
    )


if __name__ == "__main__":
    run_local_tests()

    for question in [
        "What are the risks of combining warfarin with aspirin?",
        "What are the adverse effects of metformin?"
    ]:
        print("\nTesting:", question)

        results = hybrid_retrieve(question)

        print(
            "Documents returned:",
            len(results)
        )

        print(
            "PubMed:",
            sum(
                doc.get("source") == "PubMed"
                for doc in results
            )
        )

        print(
            "FAERS:",
            sum(
                doc.get("source") == "FAERS"
                for doc in results
            )
        )
