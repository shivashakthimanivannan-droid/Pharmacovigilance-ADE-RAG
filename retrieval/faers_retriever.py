# ============================================================
# FAERS LOCAL RETRIEVER
# Pharmacovigilance RAG System
#
# Improved Evidence-Quality Retrieval
# ============================================================

import os
import re
import pandas as pd


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FAERS_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "faers_rag_documents.csv"
)


# ============================================================
# CACHE
# ============================================================

_FAERS_DATA = None


# ============================================================
# DRUG ALIASES
# ============================================================

DRUG_ALIASES = {

    "warfarin": [
        "warfarin",
        "warfarin sodium",
        "coumadin",
    ],

    "aspirin": [
        "aspirin",
        "acetylsalicylic acid",
    ],

    "heparin": [
        "heparin",
    ],

    "apixaban": [
        "apixaban",
        "eliquis",
    ],

    "rivaroxaban": [
        "rivaroxaban",
        "xarelto",
    ],

    "clopidogrel": [
        "clopidogrel",
        "plavix",
    ],

    "dabigatran": [
        "dabigatran",
        "pradaxa",
    ],
}


# ============================================================
# SAFETY / ADE TERMS
# ============================================================

SAFETY_TERMS = [

    # Bleeding
    "bleeding",
    "hemorrhage",
    "haemorrhage",
    "hemorrhagic",
    "haemorrhagic",
    "gastrointestinal hemorrhage",
    "gastrointestinal haemorrhage",
    "intracranial hemorrhage",
    "intracranial haemorrhage",
    "rectal hemorrhage",
    "rectal haemorrhage",
    "hematochezia",
    "haematochezia",
    "melena",

    # Coagulation
    "inr increased",
    "increased inr",
    "inr fluctuation",
    "international normalized ratio",
    "international normalised ratio",
    "coagulation",

    # Cardiovascular
    "myocardial infarction",
    "heart attack",
    "cardiac event",
    "cardiac arrest",

    # Cerebrovascular
    "stroke",
    "cerebrovascular accident",
    "intracranial",
    "hematoma",
    "haematoma",

    # Thrombotic
    "thrombosis",
    "thromboembolism",
    "deep vein thrombosis",
    "dvt",
    "pulmonary embolism",
    "embolism",

    # General ADE
    "adverse event",
    "adverse drug reaction",
    "drug ineffective",
    "drug interaction",
    "drug-drug interaction",

    # Common adverse events
    "anemia",
    "anaemia",
    "nausea",
    "vomiting",
    "rash",
    "infection",
    "hypotension",
    "hypertension",
    "dizziness",
    "headache",
]


# ============================================================
# SAFE TEXT
# ============================================================

def safe_text(value):

    if value is None:
        return ""

    if isinstance(value, list):

        return " ".join(
            safe_text(x)
            for x in value
        )

    if isinstance(value, dict):

        return " ".join(
            safe_text(x)
            for x in value.values()
        )

    return str(value)


# ============================================================
# LOAD FAERS DATA
# ============================================================

def load_faers_data():

    global _FAERS_DATA

    if _FAERS_DATA is not None:
        return _FAERS_DATA

    if not os.path.exists(FAERS_FILE):

        raise FileNotFoundError(
            f"FAERS file not found:\n{FAERS_FILE}"
        )

    print()
    print("=" * 70)
    print("Loading local FAERS data...")
    print("=" * 70)

    print(
        f"File: {FAERS_FILE}"
    )

    df = pd.read_csv(
        FAERS_FILE,
        low_memory=False
    )

    df = df.fillna("")

    print(
        f"Loaded {len(df):,} FAERS records."
    )

    print(
        "FAERS columns:",
        list(df.columns)
    )

    _FAERS_DATA = df

    return df


# ============================================================
# DETECT QUERY DRUGS
# ============================================================

def detect_query_drugs(query):

    query_lower = safe_text(
        query
    ).lower()

    detected = []

    for drug, aliases in DRUG_ALIASES.items():

        for alias in aliases:

            if re.search(
                r"\b"
                + re.escape(alias.lower())
                + r"\b",
                query_lower
            ):

                detected.append(drug)

                break

    return detected


# ============================================================
# FIND TEXT COLUMN
# ============================================================

def find_text_column(df):

    preferred = [

        "text",
        "content",
        "chunk",
        "document",
        "body",
        "description",
        "summary",

    ]

    columns_lower = {

        str(col).lower(): col
        for col in df.columns

    }

    for name in preferred:

        if name in columns_lower:

            return columns_lower[name]

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    best_column = None
    best_length = 0

    for column in df.columns:

        try:

            avg_length = (

                df[column]
                .astype(str)
                .str.len()
                .mean()

            )

            if avg_length > best_length:

                best_length = avg_length
                best_column = column

        except Exception:

            continue

    return best_column


# ============================================================
# CREATE SEARCH TEXT
# ============================================================

def create_search_text(df):

    text_column = find_text_column(df)

    if text_column is None:

        return pd.Series(
            [""] * len(df),
            index=df.index
        )

    return (

        df[text_column]
        .astype(str)
        .str.lower()

    )


# ============================================================
# GET MATCHED DRUGS
# ============================================================

def matched_drugs(
    text,
    detected_drugs
):

    text = safe_text(
        text
    ).lower()

    matches = []

    for drug in detected_drugs:

        aliases = DRUG_ALIASES.get(
            drug,
            [drug]
        )

        for alias in aliases:

            if re.search(
                r"\b"
                + re.escape(alias.lower())
                + r"\b",
                text
            ):

                matches.append(
                    drug
                )

                break

    return matches


# ============================================================
# GET SAFETY TERMS
# ============================================================

def matched_safety_terms(text):

    text = safe_text(
        text
    ).lower()

    matches = []

    for term in SAFETY_TERMS:

        if term.lower() in text:

            matches.append(term)

    return matches


# ============================================================
# SCORE DOCUMENT
# ============================================================

def score_document(
    text,
    query,
    detected_drugs
):

    text = safe_text(
        text
    ).lower()

    query = safe_text(
        query
    ).lower()

    score = 0

    # ========================================================
    # DRUG MATCH SCORE
    # ========================================================

    drug_matches = matched_drugs(
        text,
        detected_drugs
    )

    # Strong reward when multiple requested drugs
    # appear in the SAME FAERS report.

    if detected_drugs:

        if len(drug_matches) == len(
            detected_drugs
        ):

            score += 40

        elif len(drug_matches) > 0:

            score += (
                15 * len(drug_matches)
            )

    # ========================================================
    # QUERY TERM MATCHING
    # ========================================================

    query_words = re.findall(
        r"\b[a-zA-Z]{4,}\b",
        query
    )

    unique_query_words = set(
        query_words
    )

    query_matches = 0

    for word in unique_query_words:

        if word in text:

            query_matches += 1

    score += min(
        query_matches * 2,
        20
    )

    # ========================================================
    # SAFETY TERMS
    # ========================================================

    safety_matches = matched_safety_terms(
        text
    )

    # Reward evidence containing safety signals.

    score += min(
        len(safety_matches) * 5,
        30
    )

    # ========================================================
    # COMBINATION / INTERACTION SIGNAL
    # ========================================================

    interaction_terms = [

        "concomitant",
        "concurrently",
        "coadministration",
        "co-administration",
        "combination",
        "combined",
        "concomitant use",
        "drug interaction",
        "drug-drug interaction",

    ]

    interaction_matches = sum(

        1
        for term in interaction_terms
        if term in text

    )

    score += min(
        interaction_matches * 4,
        12
    )

    # ========================================================
    # EVIDENCE LENGTH
    # ========================================================

    text_length = len(
        text.strip()
    )

    # Very short records are usually weak evidence.

    if text_length < 80:

        score -= 20

    elif text_length < 150:

        score -= 10

    elif text_length >= 300:

        score += 5

    elif text_length >= 600:

        score += 10

    # ========================================================
    # INFORMATION DENSITY
    # ========================================================

    # Reward reports containing several distinct
    # pharmacovigilance concepts.

    if len(safety_matches) >= 2:

        score += 5

    if len(safety_matches) >= 4:

        score += 5

    # ========================================================
    # RETURN
    # ========================================================

    return score


# ============================================================
# RETRIEVE FAERS
# ============================================================

def search_faers(
    query,
    limit=10
):

    df = load_faers_data()

    if df.empty:

        return []

    detected_drugs = detect_query_drugs(
        query
    )

    print(
        "Detected query drugs:",
        detected_drugs
    )

    # ========================================================
    # SEARCHABLE TEXT
    # ========================================================

    text_column = find_text_column(
        df
    )

    if text_column is None:

        print(
            "No usable FAERS text column found."
        )

        return []

    search_text = create_search_text(
        df
    )

    # ========================================================
    # CANDIDATE FILTERING
    # ========================================================

    candidate_mask = pd.Series(
        False,
        index=df.index
    )

    # ========================================================
    # CASE 1:
    # QUERY CONTAINS DRUGS
    # ========================================================

    if detected_drugs:

        # First try to find records containing
        # ALL requested drugs.

        all_drugs_mask = pd.Series(
            True,
            index=df.index
        )

        for drug in detected_drugs:

            aliases = DRUG_ALIASES.get(
                drug,
                [drug]
            )

            drug_mask = pd.Series(
                False,
                index=df.index
            )

            for alias in aliases:

                drug_mask |= (
                    search_text.str.contains(
                        re.escape(
                            alias.lower()
                        ),
                        regex=True,
                        na=False
                    )
                )

            all_drugs_mask &= drug_mask

        # Prefer reports containing all drugs.

        if all_drugs_mask.any():

            candidate_mask |= (
                all_drugs_mask
            )

            print(
                "Found records containing all "
                "detected query drugs."
            )

        # Add records containing at least
        # one requested drug as backup.

        for drug in detected_drugs:

            aliases = DRUG_ALIASES.get(
                drug,
                [drug]
            )

            for alias in aliases:

                candidate_mask |= (
                    search_text.str.contains(
                        re.escape(
                            alias.lower()
                        ),
                        regex=True,
                        na=False
                    )
                )

    # ========================================================
    # CASE 2:
    # NO DRUG DETECTED
    # ========================================================

    else:

        query_words = re.findall(
            r"\b[a-zA-Z]{4,}\b",
            safe_text(query).lower()
        )

        # Remove generic words.

        stop_words = {

            "what",
            "are",
            "the",
            "with",
            "from",
            "this",
            "that",
            "related",
            "safety",
            "concerns",
            "associated",
            "therapy",
            "risks",

        }

        query_words = [

            word
            for word in query_words
            if word not in stop_words

        ]

        for word in query_words:

            candidate_mask |= (

                search_text.str.contains(
                    re.escape(word),
                    regex=True,
                    na=False
                )

            )

        # Also explicitly search safety terms.

        for term in SAFETY_TERMS:

            if term in safe_text(
                query
            ).lower():

                candidate_mask |= (

                    search_text.str.contains(
                        re.escape(term),
                        regex=True,
                        na=False
                    )

                )

    # ========================================================
    # CANDIDATES
    # ========================================================

    candidates = df[
        candidate_mask
    ].copy()

    print(
        f"FAERS keyword candidates: "
        f"{len(candidates):,}"
    )

    if candidates.empty:

        print(
            "No FAERS candidates found."
        )

        return []

    # ========================================================
    # SCORE
    # ========================================================

    candidates["_score"] = (

        candidates[text_column]
        .astype(str)
        .apply(

            lambda text:
            score_document(
                text,
                query,
                detected_drugs
            )

        )

    )

    # ========================================================
    # REMOVE VERY WEAK DOCUMENTS
    # ========================================================

    candidates = candidates[
        candidates["_score"] >= 5
    ].copy()

    if candidates.empty:

        print(
            "All FAERS candidates were "
            "below the evidence-quality threshold."
        )

        return []

    # ========================================================
    # SORT
    # ========================================================

    candidates = candidates.sort_values(
        "_score",
        ascending=False
    )

    # ========================================================
    # LIMIT
    # ========================================================

    candidates = candidates.head(
        limit
    )

    # ========================================================
    # CONVERT TO APPLICATION FORMAT
    # ========================================================

    results = []

    for _, row in candidates.iterrows():

        properties = {}

        # ----------------------------------------------------
        # Copy original fields
        # ----------------------------------------------------

        for column in candidates.columns:

            if column == "_score":
                continue

            properties[column] = safe_text(
                row[column]
            )

        # ----------------------------------------------------
        # Standard source fields
        # ----------------------------------------------------

        properties["source"] = "FAERS"

        properties["database"] = "FAERS"

        if not properties.get(
            "journal"
        ):

            properties["journal"] = (
                "FDA FAERS"
            )

        if not properties.get(
            "title"
        ):

            properties["title"] = (
                "FDA FAERS Adverse Event Report"
            )

        # ----------------------------------------------------
        # ID
        # ----------------------------------------------------

        evidence_id = (

            properties.get("pmid")
            or properties.get("case_id")
            or properties.get("report_id")
            or properties.get("id")
            or properties.get("chunk_id")

        )

        if not evidence_id:

            evidence_id = (
                f"FAERS-{len(results) + 1}"
            )

        evidence_id = safe_text(
            evidence_id
        )

        properties["id"] = evidence_id

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        text = safe_text(
            row[text_column]
        )

        properties["text"] = text

        # ----------------------------------------------------
        # Evidence metadata
        # ----------------------------------------------------

        drug_matches = matched_drugs(
            text,
            detected_drugs
        )

        safety_matches = matched_safety_terms(
            text
        )

        properties[
            "matched_drugs"
        ] = ", ".join(
            drug_matches
        )

        properties[
            "matched_safety_terms"
        ] = ", ".join(
            safety_matches
        )

        properties[
            "evidence_quality_score"
        ] = str(
            row["_score"]
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        results.append(

            {

                "text": text,

                "content": text,

                "properties": properties,

                "source": "FAERS",

                "id": evidence_id,

                "score": float(
                    row["_score"]
                ),

            }

        )

    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    print(
        f"Retrieved {len(results)} "
        "FAERS candidates."
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"FAERS Result {index}: "
            f"{result['id']} | "
            f"score={result['score']} | "
            f"drugs={result['properties'].get('matched_drugs')} | "
            f"safety={result['properties'].get('matched_safety_terms')}"
        )

    return results


# ============================================================
# ALIAS
# ============================================================

def retrieve_faers(
    query,
    top_k=10
):

    return search_faers(
        query,
        limit=top_k
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    query = (
        "What are the risks of combining "
        "warfarin with aspirin?"
    )

    results = search_faers(
        query,
        limit=10
    )

    print()
    print("=" * 70)
    print("FAERS RETRIEVAL TEST")
    print("=" * 70)

    for index, result in enumerate(
        results,
        start=1
    ):

        print()
        print(
            f"Result {index}"
        )

        print(
            "ID:",
            result["id"]
        )

        print(
            "Score:",
            result["score"]
        )

        print(
            "Matched drugs:",
            result["properties"].get(
                "matched_drugs"
            )
        )

        print(
            "Safety terms:",
            result["properties"].get(
                "matched_safety_terms"
            )
        )

        print(
            "Text:",
            result["text"][:700]
        )