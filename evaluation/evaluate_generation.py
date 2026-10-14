# ============================================================
# PHARMACOVIGILANCE RAG
# GENERATION & GROUNDING EVALUATION
# ============================================================

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from retrieval.retriever import hybrid_retrieve
from detection.ade_detector import detect_adverse_events
from generation.llm_generator import generate_safety_analysis


# ============================================================
# CONFIGURATION
# ============================================================

TEST_QUERIES = [
    "What are the risks of combining warfarin with aspirin?",
    "What adverse events are associated with anticoagulant therapy?",
    "What are the safety concerns related to gastrointestinal bleeding?",
]

TOP_K = 10

OUTPUT_FILE = (
    Path(__file__).resolve().parent
    / "generation_evaluation_results.json"
)


# ============================================================
# TEXT HELPERS
# ============================================================

def safe_text(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    try:
        return json.dumps(
            value,
            ensure_ascii=False
        )

    except Exception:
        return str(value)


def normalize(
    text: str
) -> str:

    text = safe_text(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize(
    text: str
) -> set[str]:

    text = normalize(text)

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9'-]{2,}\b",
        text
    )

    return set(words)


# ============================================================
# CITATION EXTRACTION
# ============================================================

CITATION_PATTERN = re.compile(
    r"\[\s*Evidence\s+(\d+)\s*\]",
    re.IGNORECASE
)


def extract_citations(
    text: str
) -> List[int]:

    citations = []

    for match in CITATION_PATTERN.finditer(
        safe_text(text)
    ):

        number = int(
            match.group(1)
        )

        if number not in citations:
            citations.append(number)

    return citations


# ============================================================
# SECTION HEADERS
# ============================================================

SECTION_HEADERS = {
    "safety summary",
    "relevant evidence",
    "potential adverse events",
    "explicitly reported events",
    "automated ade signals",
    "interpretation",
    "drug-drug interaction concerns",
    "monitoring considerations",
    "evidence limitations",
}


def is_section_header(
    line: str
) -> bool:

    clean = re.sub(
        r"[*#:_\-]",
        " ",
        line
    )

    clean = re.sub(
        r"\s+",
        " ",
        clean
    ).strip().lower()

    return clean in SECTION_HEADERS


# ============================================================
# METADATA FILTER
# ============================================================

def is_metadata_line(
    line: str
) -> bool:

    clean = line.strip()

    if not clean:
        return True

    # Remove common Markdown prefixes for checking.
    check = re.sub(
        r"^\s*(?:[-*•]|\d+[.)])\s*",
        "",
        clean
    )

    check = re.sub(
        r"\*+",
        "",
        check
    ).strip()

    if is_section_header(check):
        return True

    # Markdown separators.
    if re.fullmatch(
        r"[-=_*#\s]+",
        clean
    ):
        return True

    metadata_prefixes = [
        "source:",
        "title:",
        "journal:",
        "year:",
        "evidence id:",
        "evidence:",
        "document id:",
        "reranker score:",
        "cross-encoder",
        "source distribution:",
    ]

    lower = check.lower()

    for prefix in metadata_prefixes:

        if lower.startswith(prefix):
            return True

    # Pure evidence citation.
    if re.fullmatch(
        r"(?:\[\s*Evidence\s+\d+\s*\]\s*)+",
        check,
        re.IGNORECASE
    ):
        return True

    return False


# ============================================================
# CLAIM CLEANING
# ============================================================

def clean_claim_line(
    line: str
) -> str:

    line = line.strip()

    # Markdown bullets.
    line = re.sub(
        r"^\s*[-*•]\s+",
        "",
        line
    )

    # Numbered bullets.
    line = re.sub(
        r"^\s*\d+[.)]\s+",
        "",
        line
    )

    # Markdown headings.
    line = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        line
    )

    return line.strip()


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(
    text: str
) -> List[str]:

    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9*-])",
        text
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def extract_claims(
    analysis: str
) -> List[str]:

    claims = []

    for raw_line in safe_text(
        analysis
    ).splitlines():

        line = raw_line.strip()

        if not line:
            continue

        if is_section_header(line):
            continue

        if is_metadata_line(line):
            continue

        line = clean_claim_line(
            line
        )

        if not line:
            continue

        lower = line.lower()

        # Ignore explicit non-claims.
        ignored_prefixes = [
            "source:",
            "title:",
            "journal:",
            "year:",
            "evidence id:",
            "document id:",
            "reranker score:",
        ]

        if any(
            lower.startswith(prefix)
            for prefix in ignored_prefixes
        ):
            continue

        # Ignore generation errors.
        error_phrases = [
            "ai safety analysis failed",
            "model is currently experiencing high demand",
            "service unavailable",
            "resource exhausted",
            "rate limit",
            "too many requests",
        ]

        if any(
            phrase in lower
            for phrase in error_phrases
        ):
            continue

        # Split long paragraphs.
        sentences = split_sentences(
            line
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence) < 12:
                continue

            # Ignore simple labels such as "Evidence-supported findings".
            if re.fullmatch(
                r"[A-Za-z][A-Za-z0-9 /&()'-]{1,80}:?",
                sentence
            ) and "[" not in sentence:
                continue

            claims.append(
                sentence
            )

    # Deduplicate.
    unique = []

    seen = set()

    for claim in claims:

        key = normalize(
            claim
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(
            claim
        )

    return unique


# ============================================================
# SAFE SYSTEM CAVEATS
# ============================================================

# ============================================================
# PROTECTED METHODOLOGICAL / SAFETY STATEMENTS
# ============================================================

SAFE_SYSTEM_CAVEATS = [
    # FAERS limitations
    "does not establish causality",
    "do not establish causality",
    "cannot establish causality",
    "cannot prove causality",
    "does not prove causality",
    "do not prove causality",
    "does not confirm causality",
    "do not confirm causality",
    "not proof of causality",
    "not evidence of causality",
    "should not be interpreted as proof of causality",
    "should not be interpreted as establishing causality",

    # Automated ADE limitations
    "automated ade detection represents signals",
    "automated ade signals",
    "automated detection represents signals",
    "automated detection identifies matched terms",
    "automated signals do not confirm",
    "automated signals do not establish",
    "does not confirm adverse drug reactions",
    "do not confirm adverse drug reactions",
    "not confirmed adverse drug reactions",
    "not a confirmed adverse drug reaction",

    # Evidence limitations
    "retrieved evidence may be insufficient",
    "evidence may be insufficient",
    "retrieved evidence does not establish",
    "the evidence does not establish",
    "evidence does not establish",

    # Monitoring / recommendation limitation
    "does not establish specific monitoring recommendations",
    "does not provide specific monitoring recommendations",
    "retrieved evidence does not establish specific monitoring",
    "retrieved evidence does not provide specific monitoring",

    # General methodological statements
    "retrieved evidence",
    "evidence limitations",
    "faers limitations",
]


def is_safe_caveat(claim: str) -> bool:

    normalized = normalize(claim)

    return any(
        phrase in normalized
        for phrase in SAFE_SYSTEM_CAVEATS
    )

def is_automated_detection_statement(
    claim: str
) -> bool:

    normalized = normalize(claim)

    automated_markers = [
        "automated detection",
        "automated ade",
        "automated signal",
        "automated signals",
        "[automated detection]",
        "[automated ade detection]",
    ]

    return any(
        marker in normalized
        for marker in automated_markers
    )

# ============================================================
# EVIDENCE TEXT
# ============================================================

def evidence_text_for_ids(
    evidence: List[Dict[str, Any]],
    ids: List[int]
) -> str:

    chunks = []

    for evidence_id in ids:

        index = evidence_id - 1

        if index < 0:
            continue

        if index >= len(evidence):
            continue

        document = evidence[index]

        if not isinstance(
            document,
            dict
        ):
            continue

        chunks.append(
            safe_text(
                document.get(
                    "title",
                    ""
                )
            )
        )

        chunks.append(
            safe_text(
                document.get(
                    "text",
                    ""
                )
            )
        )

    return " ".join(
        chunks
    )


# ============================================================
# CLAIM SUPPORT
# ============================================================

def claim_supported(
    claim: str,
    evidence_text: str
) -> bool:

    if is_safe_caveat(claim):
        return True

    if is_automated_detection_statement(claim):
        return True

    claim_tokens = tokenize(
        claim
    )

    evidence_tokens = tokenize(
        evidence_text
    )

    if not claim_tokens:
        return False

    overlap = (
        len(
            claim_tokens
            & evidence_tokens
        )
        / len(claim_tokens)
    )

    # Strong lexical support.
    if overlap >= 0.55:
        return True

    # Important pharmacovigilance terms.
    critical_terms = {
        "bleeding",
        "hemorrhage",
        "haemorrhage",
        "intracranial",
        "gastrointestinal",
        "anemia",
        "anaemia",
        "stroke",
        "thrombosis",
        "interaction",
        "warfarin",
        "aspirin",
        "anticoagulant",
        "antiplatelet",
        "hematoma",
        "coagulopathy",
        "inr",
    }

    claim_critical = (
        claim_tokens
        & critical_terms
    )

    evidence_critical = (
        evidence_tokens
        & critical_terms
    )

    if (
        claim_critical
        and claim_critical.issubset(
            evidence_critical
        )
        and overlap >= 0.35
    ):
        return True

    return False


# ============================================================
# CLAIM EVALUATION
# ============================================================

def evaluate_claims(
    analysis: str,
    evidence: List[Dict[str, Any]]
) -> Dict[str, Any]:

    claims = extract_claims(
        analysis
    )

    supported = 0
    unsupported_claims = []

    cited_claims = 0

    evaluated_claims = []

    protected_claims = []

    for claim in claims:

        # ----------------------------------------------------
        # PROTECTED SAFETY / METHODOLOGICAL STATEMENTS
        # ----------------------------------------------------

        if (
            is_safe_caveat(claim)
            or is_automated_detection_statement(claim)
        ):

            protected_claims.append(
                claim
            )

            continue

        # ----------------------------------------------------
        # NORMAL EVIDENCE CLAIM
        # ----------------------------------------------------

        evaluated_claims.append(
            claim
        )

        citations = extract_citations(
            claim
        )

        valid_ids = [
            citation
            for citation in citations
            if 1 <= citation <= len(evidence)
        ]

        if valid_ids:
            cited_claims += 1

        support_text = evidence_text_for_ids(
            evidence,
            valid_ids
        )

        if claim_supported(
            claim,
            support_text
        ):

            supported += 1

        else:

            unsupported_claims.append(
                claim
            )

    # --------------------------------------------------------
    # ONLY REAL EVIDENCE CLAIMS COUNT
    # --------------------------------------------------------

    total_claims = len(
        evaluated_claims
    )

    claim_support = (
        supported / total_claims
        if total_claims
        else 1.0
    )

    citation_coverage = (
        cited_claims / total_claims
        if total_claims
        else 1.0
    )

    return {
        "claims": total_claims,

        "supported_claims":
            supported,

        "unsupported_claims":
            len(
                unsupported_claims
            ),

        "claim_support":
            claim_support,

        "citation_coverage":
            citation_coverage,

        "unsupported_claim_details":
            unsupported_claims,

        "protected_claims":
            len(
                protected_claims
            ),
    }


# ============================================================
# CITATION VALIDITY
# ============================================================

def evaluate_citations(
    analysis: str,
    evidence_count: int
) -> Dict[str, Any]:

    matches = list(
        CITATION_PATTERN.finditer(
            safe_text(analysis)
        )
    )

    total_citations = len(
        matches
    )

    valid_citations = 0

    invalid_citations = []

    for match in matches:

        number = int(
            match.group(1)
        )

        if 1 <= number <= evidence_count:

            valid_citations += 1

        else:

            invalid_citations.append(
                number
            )

    validity = (
        valid_citations / total_citations
        if total_citations
        else 0.0
    )

    return {
        "total_citations":
            total_citations,

        "valid_citations":
            valid_citations,

        "invalid_citations":
            invalid_citations,

        "citation_validity":
            max(
                0.0,
                min(
                    validity,
                    1.0
                )
            ),
    }


# ============================================================
# ADE GROUNDING
# ============================================================

def evaluate_ade_grounding(
    ade_results: Any,
    evidence: List[Dict[str, Any]]
) -> float:

    if not isinstance(
        ade_results,
        dict
    ):
        return 0.0

    findings = ade_results.get(
        "findings",
        []
    )

    if not isinstance(
        findings,
        list
    ):
        return 0.0

    if not findings:
        return 1.0

    evidence_text = " ".join(
        safe_text(
            document.get(
                "text",
                ""
            )
        )
        for document in evidence
        if isinstance(
            document,
            dict
        )
    )

    evidence_normalized = normalize(
        evidence_text
    )

    grounded = 0

    valid_findings = 0

    for finding in findings:

        if not isinstance(
            finding,
            dict
        ):
            continue

        valid_findings += 1

        event = normalize(
            finding.get(
                "adverse_event",
                ""
            )
        )

        matched_terms = finding.get(
            "matched_terms",
            []
        )

        terms = [
            normalize(term)
            for term in matched_terms
            if safe_text(term)
        ]

        event_grounded = (
            bool(event)
            and event in evidence_normalized
        )

        term_grounded = any(
            term in evidence_normalized
            for term in terms
        )

        if (
            event_grounded
            or term_grounded
        ):
            grounded += 1

    if valid_findings == 0:
        return 0.0

    return max(
        0.0,
        min(
            grounded / valid_findings,
            1.0
        )
    )


# ============================================================
# OVERALL SCORE
# ============================================================

def calculate_overall(
    claim_support: float,
    citation_coverage: float,
    citation_validity: float,
    ade_grounding: float
) -> float:

    score = (
        0.40 * claim_support
        + 0.25 * citation_coverage
        + 0.15 * citation_validity
        + 0.20 * ade_grounding
    )

    return max(
        0.0,
        min(
            score,
            1.0
        )
    )


# ============================================================
# FAERS CAUSALITY CHECK
# ============================================================

def faers_causality_check(
    analysis: str
) -> str:

    text = normalize(
        analysis
    )

    causality_warning = any(
        phrase in text
        for phrase in [
            "does not establish causality",
            "do not establish causality",
            "does not prove causality",
            "do not prove causality",
            "not establish causality",
            "not proof of causality",
            "reported safety information",
        ]
    )

    if causality_warning:
        return "PASS"

    return "REVIEW"


# ============================================================
# SOURCE DISTRIBUTION
# ============================================================

def source_distribution(
    evidence: List[Dict[str, Any]]
) -> Dict[str, int]:

    distribution = {
        "PubMed": 0,
        "FAERS": 0,
        "Other": 0,
    }

    for document in evidence:

        if not isinstance(
            document,
            dict
        ):
            continue

        source = normalize(
            document.get(
                "source",
                ""
            )
        )

        if "pubmed" in source:

            distribution["PubMed"] += 1

        elif "faers" in source:

            distribution["FAERS"] += 1

        else:

            distribution["Other"] += 1

    return distribution


# ============================================================
# SINGLE QUERY EVALUATION
# ============================================================

def evaluate_query(
    query: str
) -> Dict[str, Any]:

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Evaluating query:\n{query}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    print(
        "\nRunning hybrid retrieval..."
    )

    evidence = hybrid_retrieve(query)

    
    if not isinstance(
        evidence,
        list
    ):
        evidence = []

    print(
        f"\nRetrieved evidence: "
        f"{len(evidence)}"
    )

    distribution = source_distribution(
        evidence
    )

    print(
        "\nEvidence source distribution:"
    )

    print(
        f"  PubMed: {distribution['PubMed']}"
    )

    print(
        f"  FAERS: {distribution['FAERS']}"
    )

    print(
        f"  Other: {distribution['Other']}"
    )

    # --------------------------------------------------------
    # ADE
    # --------------------------------------------------------

    print(
        "\nRunning ADE signal detection..."
    )

    try:

        ade_results = detect_adverse_events(
            evidence
        )

    except TypeError:

        try:

            combined_text = "\n".join(
                safe_text(
                    document.get(
                        "text",
                        ""
                    )
                )
                for document in evidence
                if isinstance(
                    document,
                    dict
                )
            )

            ade_results = detect_adverse_events(
                combined_text
            )

        except Exception as exc:

            ade_results = {
                "status": "error",
                "error": str(exc),
                "findings": [],
            }

    except Exception as exc:

        ade_results = {
            "status": "error",
            "error": str(exc),
            "findings": [],
        }

    print(
        "ADE detection completed."
    )

    print(
        "\nADE findings:"
    )

    print(
        json.dumps(
            ade_results,
            indent=2,
            ensure_ascii=False
        )
    )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    print(
        "\nGenerating pharmacovigilance analysis..."
    )

    try:

        analysis = generate_safety_analysis(
            query=query,
            documents=evidence,
            ade_results=ade_results
        )

        analysis = safe_text(
            analysis
        )

    except Exception as exc:

        analysis = (
            "AI safety analysis failed.\n"
            f"{type(exc).__name__}: {exc}"
        )

    # --------------------------------------------------------
    # GENERATION FAILURE
    # --------------------------------------------------------

    failure_markers = [
        "ai safety analysis failed",
        "model is currently experiencing high demand",
        "service unavailable",
        "resource exhausted",
        "rate limit",
        "too many requests",
    ]

    analysis_normalized = normalize(
        analysis
    )

    generation_failed = (
        not analysis.strip()
        or any(
            marker in analysis_normalized
            for marker in failure_markers
        )
    )

    if generation_failed:

        print(
            "\nGeneration failed."
        )

        return {
            "query": query,
            "success": False,
            "generation_status": "failed",
            "evidence_count": len(evidence),
            "source_distribution": distribution,
            "analysis": analysis,
            "claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "claim_support": 0.0,
            "citation_coverage": 0.0,
            "citation_validity": 0.0,
            "ade_grounding": 0.0,
            "faers_causality": "REVIEW",
            "overall_grounding": 0.0,
            "unsupported_claim_details": [],
            "invalid_citations": [],
        }

    # --------------------------------------------------------
    # GROUNDING
    # --------------------------------------------------------

    claim_results = evaluate_claims(
        analysis,
        evidence
    )

    citation_results = evaluate_citations(
        analysis,
        len(evidence)
    )

    ade_grounding = evaluate_ade_grounding(
        ade_results,
        evidence
    )

    causality = faers_causality_check(
        analysis
    )

    overall = calculate_overall(
        claim_results["claim_support"],
        claim_results["citation_coverage"],
        citation_results["citation_validity"],
        ade_grounding
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        "\nGROUNDING RESULTS"
    )

    print(
        f"Claims: "
        f"{claim_results['claims']}"
    )

    print(
        f"Supported claims: "
        f"{claim_results['supported_claims']}"
    )

    print(
        f"Unsupported claims: "
        f"{claim_results['unsupported_claims']}"
    )

    print(
        f"Average claim support: "
        f"{claim_results['claim_support'] * 100:.2f}%"
    )

    print(
        f"Citation coverage: "
        f"{claim_results['citation_coverage'] * 100:.2f}%"
    )

    print(
        f"Citation validity: "
        f"{citation_results['citation_validity'] * 100:.2f}%"
    )

    print(
        f"ADE grounding: "
        f"{ade_grounding * 100:.2f}%"
    )

    print(
        f"FAERS causality check: "
        f"{causality}"
    )

    print(
        f"Overall grounding score: "
        f"{overall * 100:.2f}%"
    )

    if citation_results[
        "invalid_citations"
    ]:

        print(
            "\nInvalid citations:"
        )

        for citation in citation_results[
            "invalid_citations"
        ]:

            print(
                f"- Evidence {citation}"
            )

    if claim_results[
        "unsupported_claim_details"
    ]:

        print(
            "\nPotentially unsupported claims:"
        )

        for claim in claim_results[
            "unsupported_claim_details"
        ]:

            print(
                f"- {claim}"
            )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {
        "query": query,
        "success": True,
        "generation_status": "success",
        "evidence_count": len(evidence),
        "source_distribution": distribution,
        "analysis": analysis,
        "ade_results": ade_results,
        "claims":
            claim_results["claims"],
        "supported_claims":
            claim_results["supported_claims"],
        "unsupported_claims":
            claim_results["unsupported_claims"],
        "claim_support":
            claim_results["claim_support"],
        "citation_coverage":
            claim_results["citation_coverage"],
        "citation_validity":
            citation_results["citation_validity"],
        "ade_grounding":
            ade_grounding,
        "faers_causality":
            causality,
        "overall_grounding":
            overall,
        "unsupported_claim_details":
            claim_results[
                "unsupported_claim_details"
            ],
        "invalid_citations":
            citation_results[
                "invalid_citations"
            ],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "PHARMACOVIGILANCE RAG "
        "GENERATION & GROUNDING EVALUATION"
    )

    print(
        "=" * 70
    )

    results = []

    for query in TEST_QUERIES:

        try:

            result = evaluate_query(
                query
            )

        except Exception as exc:

            print(
                "\nQUERY EVALUATION ERROR:"
            )

            print(
                f"{type(exc).__name__}: {exc}"
            )

            result = {
                "query": query,
                "success": False,
                "generation_status": "error",
                "evidence_count": 0,
                "source_distribution": {
                    "PubMed": 0,
                    "FAERS": 0,
                    "Other": 0,
                },
                "analysis": "",
                "claims": 0,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "claim_support": 0.0,
                "citation_coverage": 0.0,
                "citation_validity": 0.0,
                "ade_grounding": 0.0,
                "faers_causality": "REVIEW",
                "overall_grounding": 0.0,
                "unsupported_claim_details": [
                    f"{type(exc).__name__}: {exc}"
                ],
                "invalid_citations": [],
            }

        results.append(
            result
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    successful = [
        result
        for result in results
        if result.get("success")
    ]

    def average(
        key: str
    ) -> float:

        if not successful:
            return 0.0

        return sum(
            float(
                result.get(
                    key,
                    0.0
                )
            )
            for result in successful
        ) / len(successful)

    source_totals = {
        "PubMed": 0,
        "FAERS": 0,
        "Other": 0,
    }

    for result in results:

        distribution = result.get(
            "source_distribution",
            {}
        )

        for source in source_totals:

            source_totals[source] += int(
                distribution.get(
                    source,
                    0
                )
            )

    summary = {
        "test_queries": len(
            TEST_QUERIES
        ),
        "successful_queries": len(
            successful
        ),
        "failed_queries":
            len(TEST_QUERIES)
            - len(successful),
        "success_rate": (
            len(successful)
            / len(TEST_QUERIES)
            if TEST_QUERIES
            else 0.0
        ),
        "average_claim_support":
            average("claim_support"),
        "average_citation_coverage":
            average("citation_coverage"),
        "average_citation_validity":
            average("citation_validity"),
        "average_ade_grounding":
            average("ade_grounding"),
        "average_grounding_score":
            average("overall_grounding"),
        "source_distribution":
            source_totals,
    }

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GENERATION EVALUATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Test queries: "
        f"{summary['test_queries']}"
    )

    print(
        f"Successful queries: "
        f"{summary['successful_queries']}"
    )

    print(
        f"Failed queries: "
        f"{summary['failed_queries']}"
    )

    print(
        f"Success rate: "
        f"{summary['success_rate'] * 100:.2f}%"
    )

    print(
        f"Average claim support: "
        f"{summary['average_claim_support'] * 100:.2f}%"
    )

    print(
        f"Average citation coverage: "
        f"{summary['average_citation_coverage'] * 100:.2f}%"
    )

    print(
        f"Average citation validity: "
        f"{summary['average_citation_validity'] * 100:.2f}%"
    )

    print(
        f"Average ADE grounding: "
        f"{summary['average_ade_grounding'] * 100:.2f}%"
    )

    print(
        f"Average grounding score: "
        f"{summary['average_grounding_score'] * 100:.2f}%"
    )

    print(
        "\nTOTAL RETRIEVED SOURCE DISTRIBUTION"
    )

    print(
        f"PubMed: "
        f"{source_totals['PubMed']}"
    )

    print(
        f"FAERS: "
        f"{source_totals['FAERS']}"
    )

    print(
        f"Other: "
        f"{source_totals['Other']}"
    )

    output = {
        "summary": summary,
        "queries": results,
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()