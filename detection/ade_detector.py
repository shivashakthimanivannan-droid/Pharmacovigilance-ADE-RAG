
"""PubMed and FAERS evidence detector for pharmacovigilance RAG.

Research prototype: extracted associations require source verification.
FAERS co-reporting does not establish causality or concurrent exposure.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence
from retrieval.retriever import DRUG_ALIASES as RETRIEVER_DRUG_ALIASES

DRUG_ALIASES = {
    "warfarin": ["warfarin", "warfarin sodium", "coumadin", "jantoven"],
    "aspirin": ["aspirin", "acetylsalicylic acid"],
    "apixaban": ["apixaban", "eliquis"],
    "rivaroxaban": ["rivaroxaban", "xarelto"],
    "dabigatran": ["dabigatran", "pradaxa"],
    "edoxaban": ["edoxaban", "savaysa"],
    "heparin": ["heparin", "unfractionated heparin"],
    "enoxaparin": ["enoxaparin", "lovenox"],
    "clopidogrel": ["clopidogrel", "plavix"],
    "ibuprofen": ["ibuprofen", "advil", "motrin"],
    "naproxen": ["naproxen", "aleve"],
    "diclofenac": ["diclofenac", "voltaren"],
    "amiodarone": ["amiodarone", "cordarone"],
}

# Synchronize drug recognition with the retrieval pipeline.
# Preserve detector-specific aliases such as Jantoven.
for drug_name, aliases in RETRIEVER_DRUG_ALIASES.items():
    existing = DRUG_ALIASES.get(drug_name, [])
    DRUG_ALIASES[drug_name] = list(
        dict.fromkeys(existing + aliases)
    )

EVENT_ALIASES = {
    "bleeding": [
        "bleeding", "bleed", "hemorrhage", "haemorrhage",
        "hematuria", "haematuria", "melena", "melaena",
        "hematochezia", "haematochezia", "blood loss",
    ],
    "coagulation abnormality": [
        "increased inr", "elevated inr", "high inr",
        "international normalized ratio increased",
        "international normalised ratio increased",
        "prothrombin time prolonged", "coagulation disorder",
        "anticoagulation drug level above therapeutic",
    ],
    "bruising": ["bruising", "bruises", "ecchymosis"],
    "thrombosis": [
        "thrombosis", "thromboembolism", "thromboembolic",
        "pulmonary embolism", "deep vein thrombosis", "blood clot",
    ],
    "stroke": ["stroke", "cerebrovascular accident", "cerebral infarction"],
    "anemia": ["anemia", "anaemia", "hemoglobin decreased", "haemoglobin decreased"],
    "kidney injury": ["acute kidney injury", "renal failure", "renal impairment", "kidney injury"],
    "liver injury": ["liver injury", "hepatic failure", "hepatotoxicity"],
    "cardiac event": [
        "myocardial infarction", "heart attack", "cardiac arrest",
        "arrhythmia", "heart failure",
    ],
    "hypotension": ["hypotension", "low blood pressure"],
    "hypertension": ["hypertension", "high blood pressure"],
    "nausea": ["nausea"],
    "vomiting": ["vomiting", "emesis"],
    "dizziness": ["dizziness", "vertigo", "lightheadedness"],
    "headache": ["headache", "migraine"],
    "rash": ["rash", "urticaria", "hives"],
    "hypoglycemia": ["hypoglycemia", "hypoglycaemia", "low blood glucose"],
    "infection": ["infection", "sepsis"],
    "ketoacidosis": ["ketoacidosis"],
    "drug ineffective": ["drug ineffective", "lack of efficacy", "treatment failure"],

    "lactic acidosis": [
    "lactic acidosis",
    "metformin-associated lactic acidosis",
    "hyperlactatemia",
],

"diarrhea": [
    "diarrhea",
    "diarrhoea",
],

"abdominal pain": [
    "abdominal pain",
    "stomach pain",
    "abdominal discomfort",
],

"vitamin b12 deficiency": [
    "vitamin b12 deficiency",
    "vitamin b12 decreased",
    "cobalamin deficiency",
],
}

FAERS_FIELDS = (
    "PRIMARYID", "CASEID", "DRUGNAME", "PROD_AI",
    "DRUGS", "PT", "REAC", "REACTIONS",
    "REPORTED REACTIONS", "OUTC_COD",
)

NEGATIVE_PATTERNS = [
    r"\bdid not (?:significantly |statistically )?increas\w*\b",
    r"\bdid not (?:significantly |statistically )?decreas\w*\b",
    r"\bdoes not (?:significantly )?increas\w*\b",
    r"\bnot (?:significantly |statistically )?associated\b",
    r"\bno (?:statistically )?significant (?:increase|decrease|difference)\b",
    r"\bno (?:increased|excess) risk\b",
    r"\bno association\b",
    r"\bno difference\b",
    r"\bno evidence of\b",
    r"\bno cases of\b",
    r"\bnot observed\b",
    r"\bwithout bleeding\b",
    r"\babsence of\b",
    r"\bsimilar (?:risk|rates)\b",
    r"\bneither group (?:presented|experienced|reported)\b",
]

UNCERTAIN_PATTERNS = [
    r"\bmay be associated\b",
    r"\bmight be associated\b",
    r"\bcould be associated\b",
    r"\bpossible association\b",
    r"\bpotential association\b",
    r"\bcannot exclude\b",
    r"\bunclear whether\b",
]

POSITIVE_PATTERNS = [
    r"\bassociated with\b",
    r"\bincreas(?:e|ed|es|ing)\b",
    r"\bdecreas(?:e|ed|es|ing)\b",
    r"\breduc(?:e|ed|es|ing|tion)\b",
    r"\bhigher\b",
    r"\blower\b",
    r"\bgreater\b",
    r"\bfewer\b",
    r"\bresult(?:ed|ing)? in\b",
    r"\bled to\b",
    r"\bleading to\b",
    r"\bcaused\b",
    r"\bcontributed to\b",
]

DIRECTION_PATTERNS = {
    "increased": [
        r"\bincreas(?:e|ed|es|ing)\b",
        r"\bhigher\b",
        r"\belevated\b",
        r"\bgreater\b",
        r"\bmore frequent\b",
    ],
    "decreased": [
        r"\bdecreas(?:e|ed|es|ing)\b",
        r"\breduc(?:e|ed|es|ing|tion)\b",
        r"\blower\b",
        r"\bfewer\b",
        r"\bless frequent\b",
    ],
}

COMBINATION_PATTERNS = [
    r"\bcombination of\b",
    r"\bcombined (?:use|therapy|treatment)\b",
    r"\bconcomitant (?:use|therapy|treatment)\b",
    r"\bco-?administration\b",
    r"\bdual therapy\b",
    r"\badding\b",
    r"\baddition of\b",
]

COMPARISON_PATTERNS = [
    r"\bwarfarin (?:versus|vs\.?|or) aspirin\b",
    r"\baspirin (?:versus|vs\.?|or) warfarin\b",
    r"\bwarfarin compared (?:with|to) aspirin\b",
    r"\baspirin compared (?:with|to) warfarin\b",
]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _has(text: str, phrase: str) -> bool:
    return bool(re.search(
        rf"(?<!\w){re.escape(phrase)}(?!\w)",
        text,
        re.IGNORECASE,
    ))


def _matches(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _aliases(text: str, aliases: dict) -> list[str]:
    return [
        key for key, terms in aliases.items()
        if any(_has(text, term) for term in terms)
    ]


def detect_drugs(text: str) -> list[str]:
    return _aliases(_clean(text), DRUG_ALIASES)


def detect_query_drugs(query: str) -> list[str]:
    return detect_drugs(query)


def detect_events(text: str) -> list[str]:
    return _aliases(_clean(text), EVENT_ALIASES)


def is_combination_query(
    query: str,
    query_drugs: Optional[Sequence[str]] = None,
) -> bool:
    drugs = query_drugs if query_drugs is not None else detect_query_drugs(query)
    return len(set(drugs)) >= 2


def _metadata(document: Any) -> dict:
    if isinstance(document, dict):
        result = dict(document.get("metadata") or {})
        result.update({k: v for k, v in document.items() if k != "metadata"})
        return result
    return dict(getattr(document, "metadata", {}) or {})


def get_document_text(document: Any) -> str:
    if isinstance(document, str):
        return document

    data = _metadata(document)
    for key in (
        "text", "content", "RAG_TEXT", "page_content",
        "chunk_text", "abstract", "narrative",
    ):
        if data.get(key):
            return str(data[key])

    return _clean(getattr(document, "page_content", ""))


def get_source(document: Any) -> str:
    data = _metadata(document)
    source = _clean(data.get("source") or data.get("database")).lower()

    if "faers" in source:
        return "FAERS"
    if "pubmed" in source:
        return "PubMed"
    if data.get("case_id") or data.get("CASEID"):
        return "FAERS"
    if data.get("pmid") or data.get("PMID"):
        return "PubMed"
    if re.search(r"\bCASEID\s*:", get_document_text(document), re.I):
        return "FAERS"
    return "Unknown"


def get_document_id(document: Any) -> str:
    data = _metadata(document)
    source = get_source(document)

    if source == "PubMed":
        value = _clean(data.get("pmid") or data.get("PMID"))
        if value:
            return value

    if source == "FAERS":
        value = _clean(data.get("case_id") or data.get("CASEID"))
        if not value:
            match = re.search(
                r"\bCASEID\s*:\s*(\d+)",
                get_document_text(document),
                re.I,
            )
            value = match.group(1) if match else ""

        if value:
            return value if value.upper().startswith("FAERS-CASE:") else f"FAERS-CASE:{value}"

    return _clean(
        data.get("document_id") or data.get("doc_id")
        or data.get("id") or "Unknown"
    )


def _sentences(text: str) -> list[str]:
    """Preserve scientific abbreviations and decimal comparisons."""
    if not text:
        return []

    protected = {
        "vs.": "vs<PERIOD>",
        "e.g.": "e<PERIOD>g<PERIOD>",
        "i.e.": "i<PERIOD>e<PERIOD>",
        "Dr.": "Dr<PERIOD>",
        "Prof.": "Prof<PERIOD>",
        "Fig.": "Fig<PERIOD>",
        "et al.": "et al<PERIOD>",
    }

    text = text.replace("\r\n", "\n")

    for original, replacement in protected.items():
        text = re.sub(
            re.escape(original),
            lambda _: replacement,
            text,
            flags=re.IGNORECASE,
        )

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+", text)
    return [
        part.replace("<PERIOD>", ".").strip()
        for part in parts
        if part.strip()
    ]


def _fields(text: str) -> dict[str, str]:
    names = sorted(FAERS_FIELDS, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(map(re.escape, names)) + r")\s*:\s*",
        re.I,
    )
    matches = list(pattern.finditer(text))
    result = {}

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.end():end].strip(" ;,|\r\n")
        if value:
            result[match.group(1).upper()] = value

    return result


def _event_mentions(sentence: str, event: str) -> list[tuple[int, int]]:
    positions = []
    for alias in EVENT_ALIASES.get(event, []):
        for match in re.finditer(
            rf"(?<!\w){re.escape(alias)}(?!\w)",
            sentence,
            re.I,
        ):
            positions.append((match.start(), match.end()))
    return sorted(set(positions))


def _local_clause(sentence: str, event: str) -> str:
    """Choose a comma/semicolon clause containing the event."""
    positions = _event_mentions(sentence, event)
    if not positions:
        return sentence

    boundaries = [0]
    boundaries.extend(m.end() for m in re.finditer(r"[;,]", sentence))
    boundaries.append(len(sentence))

    for start, _ in positions:
        for index in range(len(boundaries) - 1):
            left, right = boundaries[index:index + 2]
            if left <= start < right:
                return sentence[left:right].strip()

    return sentence


def _is_negated(sentence: str, event: str) -> bool:
    """
    Detect local negation.

    Whole-sentence fallback is used only when a sentence
    describes a single recognized event, to avoid negation
    of one outcome contaminating another.
    """
    clause = _local_clause(sentence, event)

    if _matches(clause, NEGATIVE_PATTERNS):
        return True

    if len(detect_events(sentence)) == 1:
        return _matches(sentence, NEGATIVE_PATTERNS)

    return False


def detect_outcome_direction(sentence: str, event: str) -> str:
    """Return increased, decreased, no_difference, or unknown."""
    if not _event_mentions(sentence, event):
        return "unknown"

    if _is_negated(sentence, event):
        return "no_difference"

    clause = _local_clause(sentence, event)
    events = _event_mentions(clause, event)

    def directions_in(text: str):
        found = []
        for direction, patterns in DIRECTION_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.I):
                    found.append((match.start(), match.end(), direction))
        return found

    directions = directions_in(clause)

    if not directions:
        clause = sentence
        events = _event_mentions(sentence, event)
        directions = directions_in(sentence)

    if not events or not directions:
        return "unknown"

    candidates = []

    for event_start, event_end in events:
        for direction_start, direction_end, direction in directions:
            if direction_end <= event_start:
                distance = event_start - direction_end
            else:
                distance = max(direction_start - event_end, 0) + 25

            candidates.append((distance, direction))

    distance, direction = min(candidates, key=lambda item: item[0])
    return direction if distance <= 110 else "unknown"


def _event_association_status(sentence: str, event: str) -> str:
    if not _event_mentions(sentence, event):
        return "none"

    if _is_negated(sentence, event):
        return "negated"

    clause = _local_clause(sentence, event)

    if _matches(clause, UNCERTAIN_PATTERNS):
        return "uncertain"

    if len(detect_events(sentence)) == 1 and _matches(
        sentence, UNCERTAIN_PATTERNS
    ):
        return "uncertain"

    direction = detect_outcome_direction(sentence, event)

    if direction in {"increased", "decreased"}:
        return "positive"

    for pattern in POSITIVE_PATTERNS:
        for match in re.finditer(pattern, sentence, re.I):
            for start, end in _event_mentions(sentence, event):
                distance = min(
                    abs(match.end() - start),
                    abs(end - match.start()),
                )
                if distance <= 100:
                    return "positive"

    return "none"


def _combination_supported(sentence: str, query_drugs: list[str]) -> bool:
    if len(query_drugs) < 2:
        return False

    if not set(query_drugs).issubset(set(detect_drugs(sentence))):
        return False

    # Explicit pair expressions, generated from the actual query drugs.
    aliases = [
        [re.escape(alias) for alias in DRUG_ALIASES.get(drug, [drug])]
        for drug in query_drugs
    ]

    if len(aliases) == 2:
        first = "(?:" + "|".join(aliases[0]) + ")"
        second = "(?:" + "|".join(aliases[1]) + ")"
        connector = r"\s*(?:plus|\+|and|with|combined with)\s*"

        if re.search(first + connector + second, sentence, re.I):
            return True
        if re.search(second + connector + first, sentence, re.I):
            return True

    if _matches(sentence, COMPARISON_PATTERNS):
        return False

    return _matches(sentence, COMBINATION_PATTERNS)


def _finding(
    event: str,
    source: str,
    doc_id: str,
    drugs: list[str],
    sentence: str,
    relevance: str,
    score: int,
    status: str,
    reason: str,
    outcome_direction: str = "unknown",
) -> dict:
    return {
        "event": event,
        "adverse_event": event,
        "source": source,
        "sources": [source],
        "document_id": doc_id,
        "supporting_document_ids": [doc_id],
        "document_count": 1,
        "drugs": drugs,
        "evidence_sentence": sentence[:1200],
        "sentence": sentence[:1200],
        "query_relevance": relevance,
        "query_relevance_score": score,
        "relevance": relevance,
        "association_status": status,
        "association_type": status,
        "outcome_direction": outcome_direction,
        "signal_strength": (
            "reported_cooccurrence_signal"
            if source == "FAERS"
            else "literature_association_signal"
            if relevance == "direct"
            else "limited_or_background_signal"
        ),
        "relevance_reasons": [reason],
        "finding_count": 1,
    }


def _analyze_pubmed(
    text: str,
    doc_id: str,
    query_drugs: list[str],
) -> list[dict]:
    findings = []

    for sentence in _sentences(text):
        events = detect_events(sentence)
        if not events:
            continue

        drugs = detect_drugs(sentence)
        overlap = set(drugs) & set(query_drugs)
        combination = _combination_supported(sentence, query_drugs)

        for event in events:
            status = _event_association_status(sentence, event)
            direction = detect_outcome_direction(sentence, event)

            if status == "negated":
                relevance, score = "background", 0
                reason = "Negated or null finding; not positive association evidence."
            elif not overlap:
                relevance, score = "background", 1
                reason = "No queried drug appears in this evidence sentence."
            elif combination and status == "positive":
                relevance, score = "direct", 9
                reason = (
                    "The sentence explicitly mentions the queried combination "
                    "and an affirmative association with this outcome. "
                    "The finding requires verification against the primary study."
                )
            else:
                relevance = "partial"
                score = 4 if status == "positive" else 3
                reason = (
                    "The sentence mentions a query-related outcome but does "
                    "not establish a direct affirmative combination association."
                )

            findings.append(_finding(
                event=event,
                source="PubMed",
                doc_id=doc_id,
                drugs=drugs,
                sentence=sentence,
                relevance=relevance,
                score=score,
                status=status,
                reason=reason,
                outcome_direction=direction,
            ))

    return findings


def _analyze_faers(
    text: str,
    doc_id: str,
    query_drugs: list[str],
) -> list[dict]:
    fields = _fields(text)

    drug_text = (
        fields.get("DRUGNAME")
        or fields.get("PROD_AI")
        or fields.get("DRUGS")
        or ""
    )

    event_text = (
        fields.get("PT")
        or fields.get("REAC")
        or fields.get("REPORTED REACTIONS")
        or fields.get("REACTIONS")
        or ""
    )

    if not drug_text or not event_text or not query_drugs:
        return []

    drugs = detect_drugs(drug_text)

    if not set(query_drugs).issubset(set(drugs)):
        return []

    reported_drugs = list(dict.fromkeys(
        part.strip() for part in drug_text.split(";") if part.strip()
    ))

    reactions = list(dict.fromkeys(
        part.strip() for part in event_text.split(";") if part.strip()
    ))

    findings = []

    for event in detect_events(event_text):
        matching = [
            reaction for reaction in reactions
            if event in detect_events(reaction)
        ]

        reaction = matching[0] if matching else event

        excerpt = (
            f"FAERS record {doc_id}: "
            f"reported drugs: {'; '.join(reported_drugs[:15])}; "
            f"reported reaction: {reaction}."
        )

        findings.append(_finding(
            event=event,
            source="FAERS",
            doc_id=doc_id,
            drugs=drugs,
            sentence=excerpt,
            relevance="partial",
            score=4,
            status="reported_cooccurrence",
            reason=(
                "Structured FAERS report contains the queried drugs and "
                "reaction. This does not establish causality, incidence, "
                "concurrent use, or a drug interaction."
            ),
            outcome_direction="unknown",
        ))

    return findings


def _deduplicate(findings: list[dict]) -> list[dict]:
    unique = {}

    for item in findings:
        key = (
            item["source"],
            item["document_id"],
            item["event"],
        )

        previous = unique.get(key)

        if previous is None or (
            item["query_relevance_score"]
            > previous["query_relevance_score"]
        ):
            unique[key] = item

    return sorted(
        unique.values(),
        key=lambda item: (
            -item["query_relevance_score"],
            item["event"],
            item["document_id"],
        ),
    )


def detect_adverse_events(
    query: str,
    documents: Optional[Sequence[Any]] = None,
    retrieved_documents: Optional[Sequence[Any]] = None,
) -> dict:
    docs = list(
        documents
        if documents is not None
        else (retrieved_documents or [])
    )

    query_drugs = detect_query_drugs(query)
    raw = []

    for document in docs:
        text = get_document_text(document)
        if not text:
            continue

        source = get_source(document)
        doc_id = get_document_id(document)

        if source == "FAERS":
            raw.extend(_analyze_faers(text, doc_id, query_drugs))
        elif source == "PubMed":
            raw.extend(_analyze_pubmed(text, doc_id, query_drugs))

    findings = _deduplicate(raw)

    direct = [f for f in findings if f["query_relevance"] == "direct"]
    partial = [f for f in findings if f["query_relevance"] == "partial"]
    background = [f for f in findings if f["query_relevance"] == "background"]

    return {
        "query": query,
        "query_drugs": query_drugs,
        "query_events": detect_events(query),
        "detected_drugs": sorted({
            drug for finding in findings for drug in finding["drugs"]
        }),
        "document_count": len(docs),
        "raw_findings": raw,
        "raw_finding_count": len(raw),
        "findings": findings,
        "aggregated_findings": findings,
        "direct_findings": direct,
        "partial_findings": partial,
        "background_findings": background,
        "direct_count": len(direct),
        "partial_count": len(partial),
        "background_count": len(background),
        "total_findings": len(findings),
        "is_combination_query": is_combination_query(query, query_drugs),
        "disclaimer": (
            "Research signals only. FAERS reports do not establish "
            "incidence, interaction, or causality."
        ),
    }


def summarize_ade_results(result: dict) -> dict:
    return {
        "total_findings": result.get("total_findings", 0),
        "raw_finding_count": result.get("raw_finding_count", 0),
        "direct_count": result.get("direct_count", 0),
        "partial_count": result.get("partial_count", 0),
        "background_count": result.get("background_count", 0),
        "document_count": result.get("document_count", 0),
    }


def run_tests():
    query = "What are the risks of combining warfarin with aspirin?"

    documents = [
        {
            "source": "PubMed",
            "pmid": "15242475",
            "text": (
                "Combination of warfarin plus aspirin significantly "
                "decreased thromboembolic events and increased major "
                "bleeding (RR, 1.58; 95% CI, 1.02 to 2.44) "
                "compared to warfarin alone."
            ),
        },
        {
            "source": "PubMed",
            "pmid": "16103468",
            "text": (
                "Compared with aspirin alone, warfarin plus aspirin "
                "was associated with a decrease in the annual rate "
                "of myocardial infarction (0.022 vs. 0.041; "
                "rate ratio, 0.56 [95% CI, 0.46 to 0.69]). "
                "Warfarin was associated with an increase in "
                "major bleeding (0.015 vs. 0.006; rate ratio, 2.5)."
            ),
        },
        {
            "source": "PubMed",
            "pmid": "33563422",
            "text": (
                "This study compared warfarin versus aspirin "
                "for prevention of thrombosis and bleeding."
            ),
        },
        {
            "source": "PubMed",
            "pmid": "99999999",
            "text": (
                "Warfarin plus aspirin did not significantly "
                "increase the risk of bleeding."
            ),
        },
        {
            "source": "FAERS",
            "case_id": "26777212",
            "text": (
                "CASEID: 26777212\n"
                "DRUGNAME: ASPIRIN; METOPROLOL; WARFARIN\n"
                "PROD_AI: ASPIRIN; METOPROLOL; WARFARIN\n"
                "PT: Haemorrhage urinary tract\n"
            ),
        },
    ]

    split = _sentences(
        "The annual rate was 0.022 vs. 0.041. "
        "Major bleeding increased."
    )

    assert len(split) == 2, f"Sentence splitting failed: {split}"
    assert "0.022 vs. 0.041" in split[0]

    result = detect_adverse_events(query, documents)

    print("\nADE DETECTOR REGRESSION TEST")
    print("=" * 65)

    for finding in result["findings"]:
        print(
            finding["source"],
            finding["document_id"],
            finding["event"],
            finding["query_relevance"],
            finding["association_status"],
            finding["outcome_direction"],
        )

    def find(doc_id: str, event: str) -> dict:
        return next(
            (
                f for f in result["findings"]
                if f["document_id"] == doc_id and f["event"] == event
            ),
            {},
        )

    bleeding = find("15242475", "bleeding")
    thrombosis = find("15242475", "thrombosis")
    cardiac = find("16103468", "cardiac event")
    negated = find("99999999", "bleeding")
    comparison = find("33563422", "bleeding")

    assert bleeding["query_relevance"] == "direct"
    assert bleeding["outcome_direction"] == "increased"

    assert thrombosis["query_relevance"] == "direct"
    assert thrombosis["outcome_direction"] == "decreased"

    assert cardiac["query_relevance"] == "direct"
    assert cardiac["outcome_direction"] == "decreased"
    assert "0.022 vs. 0.041" in cardiac["evidence_sentence"]

    assert negated["query_relevance"] == "background"
    assert negated["association_status"] == "negated"
    assert negated["outcome_direction"] == "no_difference"

    assert comparison["query_relevance"] != "direct"

    faers = [
        f for f in result["findings"]
        if f["source"] == "FAERS"
    ]

    assert faers
    assert all(f["query_relevance"] == "partial" for f in faers)
    assert all(f["outcome_direction"] == "unknown" for f in faers)
    assert faers[0]["evidence_sentence"].count("METOPROLOL") == 1

    print("\nAll ADE regression tests passed.")


if __name__ == "__main__":
    run_tests()
