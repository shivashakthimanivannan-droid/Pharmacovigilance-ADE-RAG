
"""Pharmacovigilance RAG - final Streamlit dashboard."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.retriever import hybrid_retrieve
from detection.ade_detector import (
    detect_adverse_events,
    summarize_ade_results,
    get_document_text,
    get_document_id,
    get_source,
)
from generation.llm_generator import generate_from_retrieval
from evaluation.grounding_evaluator import evaluate_answer


st.set_page_config(
    page_title="Pharmacovigilance RAG",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1220;
        color: #e5edf7;
    }
    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
    }
    section[data-testid="stSidebar"] {
        background: #101827;
    }
    div[data-testid="stMetric"] {
        background: #172235;
        border: 1px solid #344258;
        border-radius: 12px;
        padding: 12px;
    }
    div[data-testid="stMetricLabel"] {
        color: #b7c8df;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def excerpt(value: Any, limit: int = 1400) -> str:
    value = clean(value)
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0] + "..."


def source_name(doc: Any) -> str:
    try:
        return clean(get_source(doc))
    except Exception:
        return "Unknown"


def document_id(doc: Any) -> str:
    try:
        return clean(get_document_id(doc))
    except Exception:
        return "Unknown"


def document_text(doc: Any) -> str:
    try:
        return clean(get_document_text(doc))
    except Exception:
        return ""


def pubmed_url(identifier: str) -> str | None:
    value = clean(identifier)
    if value.upper().startswith("PMID:"):
        value = value.split(":", 1)[1].strip()

    if value.isdigit():
        return f"https://pubmed.ncbi.nlm.nih.gov/{value}/"

    return None


def to_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        indent=2,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def count_by_source(documents: list[Any], name: str) -> int:
    return sum(
        source_name(doc).lower() == name.lower()
        for doc in documents
    )


def render_documents(
    documents: list[Any],
    source: str,
) -> None:
    filtered = [
        doc for doc in documents
        if source_name(doc).lower() == source.lower()
    ]

    if not filtered:
        st.info(f"No {source} documents retrieved.")
        return

    for index, doc in enumerate(filtered, 1):
        identifier = document_id(doc)

        with st.expander(
            f"{index}. {identifier}",
            expanded=False,
        ):
            st.write(
                excerpt(document_text(doc), 4000)
                or "No text available."
            )

            if source.lower() == "pubmed":
                url = pubmed_url(identifier)
                if url:
                    st.link_button(
                        "Open PubMed article",
                        url,
                    )
            else:
                st.caption(
                    "FAERS records are spontaneous reports. "
                    "They do not establish simultaneous exposure, "
                    "causality, incidence, or comparative risk."
                )


def render_findings(result: dict) -> None:
    findings = result.get("findings") or []

    if not findings:
        st.info("No ADE findings identified.")
        return

    source_filter = st.selectbox(
        "Evidence source",
        ["All", "PubMed", "FAERS"],
        key="findings_source",
    )

    relevance_filter = st.selectbox(
        "Detector relevance",
        ["All", "direct", "partial", "background"],
        key="findings_relevance",
    )

    filtered = []

    for finding in findings:
        source = clean(finding.get("source"))
        relevance = clean(
            finding.get("query_relevance", "background")
        ).lower()

        if (
            source_filter != "All"
            and source.lower() != source_filter.lower()
        ):
            continue

        if (
            relevance_filter != "All"
            and relevance != relevance_filter
        ):
            continue

        filtered.append(finding)

    st.caption(
        f"Showing {len(filtered)} of {len(findings)} "
        "event-document findings."
    )

    max_cards = st.slider(
        "Findings to display",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
    )

    for finding in filtered[:max_cards]:
        event = clean(
            finding.get("event")
            or finding.get("adverse_event")
            or "Unspecified event"
        )

        source = clean(finding.get("source", "Unknown"))
        identifier = clean(
            finding.get("document_id", "Unknown")
        )

        relevance = clean(
            finding.get("query_relevance", "background")
        )

        direction = clean(
            finding.get("outcome_direction", "unknown")
        )

        association = clean(
            finding.get("association_status", "none")
        )

        evidence = (
            finding.get("evidence_sentence")
            or finding.get("sentence")
            or ""
        )

        with st.container(border=True):
            st.markdown(f"**{event.title()}**")
            st.caption(
                f"{source} | {identifier} | "
                f"Relevance: {relevance}"
            )

            st.write(
                excerpt(evidence, 1400)
                or "No evidence excerpt available."
            )

            c1, c2 = st.columns(2)
            c1.caption(f"Direction: {direction}")
            c2.caption(f"Association: {association}")

            if source.lower() == "faers":
                st.caption(
                    "Co-report only; not proof of "
                    "a drug interaction or causal effect."
                )
            else:
                url = pubmed_url(identifier)
                if url:
                    st.link_button("View PubMed", url)


def render_grounding(report: dict | None) -> None:
    st.subheader("🧪 Claim-level grounding audit")

    if not report:
        st.info(
            "No grounding report is available. "
            "The generated answer may have failed validation."
        )
        return

    counts = report.get("counts") or {}
    claims = report.get("claims") or []

    st.caption(
        "Method: "
        + clean(report.get("method", "unknown"))
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Claims",
        report.get("total_claims", len(claims)),
    )
    b.metric(
        "Supported",
        counts.get("supported", 0),
    )
    c.metric(
        "Uncited",
        counts.get("uncited", 0),
    )
    d.metric(
        "Needs review",
        counts.get("needs_review", 0),
    )

    flags = (
        counts.get("contradicted", 0)
        + counts.get("insufficient_evidence", 0)
        + counts.get("invalid_citation", 0)
        + counts.get("unsupported", 0)
    )

    if flags:
        st.warning(
            f"{flags} claim(s) were flagged for "
            "potential grounding problems."
        )
    elif report.get("method") == "gemini_assisted_audit":
        st.success(
            "No unsupported, contradicted, or invalidly "
            "cited claims were flagged by this audit."
        )
    else:
        st.info(
            "Deterministic screening completed. "
            "Cited claims marked 'needs_review' have "
            "not been semantically verified."
        )

    if counts.get("methodological_disclaimer", 0):
        st.caption(
            "Recognized FAERS methodological disclaimers: "
            + str(
                counts.get("methodological_disclaimer", 0)
            )
        )

    st.caption(
        "Automated grounding classifications are provisional. "
        "A supported label is not independent clinical verification."
    )

    for index, item in enumerate(claims, 1):
        label = clean(
            item.get("status", "needs_review")
        )

        claim = clean(item.get("claim"))
        reason = clean(item.get("reason"))
        citations = item.get("citations") or []

        with st.expander(
            f"{index}. {label.upper()} — "
            f"{excerpt(claim, 85)}",
            expanded=label in {
                "contradicted",
                "insufficient_evidence",
                "invalid_citation",
                "unsupported",
            },
        ):
            st.markdown("**Claim**")
            st.write(claim)

            st.markdown("**Audit explanation**")
            st.write(reason)

            st.caption(
                "Citations: "
                + (
                    ", ".join(map(str, citations))
                    if citations
                    else "None"
                )
            )

            for evidence in item.get("evidence") or []:
                with st.expander(
                    "Evidence: "
                    + clean(evidence.get("citation"))
                ):
                    st.write(
                        excerpt(
                            evidence.get("excerpt"),
                            3500,
                        )
                    )


def render_generated(generated: Any) -> None:
    st.subheader("🤖 Grounded Gemini analysis")

    if not isinstance(generated, dict):
        st.warning("No structured generation result.")
        if generated:
            st.write(str(generated))
        return

    answer = str(
        generated.get("answer")
        or generated.get("response")
        or generated.get("text")
        or ""
    ).strip()

    answer = re.sub(
    r"(?<!\n)(Summary:|PubMed Evidence:|FAERS Signals:|Limitations:)",
    r"\n\n### \1\n",
    answer,
)

    answer = re.sub(
    r"\s+- (?=\S)",
    "\n- ",
    answer,
)

    validation = (
        generated.get("citation_validation") or {}
    )

    valid = (
        generated.get("status") == "validated"
        and validation.get("is_valid") is True
        and validation.get("has_citations") is True
        and not validation.get("invalid_citations")
    )

    if valid:
        st.success(
            "Citation identifiers validated "
            "against retrieved evidence."
        )
    else:
        st.warning(
            "Citation validation is incomplete or failed."
        )

    if answer:
        st.markdown(answer)
    else:
        st.info("No generated answer available.")

    with st.expander("Citation validation details"):
        st.json(validation)
        st.write(
            "Generation status:",
            generated.get("status", "unknown"),
        )
        st.write(
            "Generation attempts:",
            generated.get("generation_attempts", "unknown"),
        )

        if generated.get("error"):
            st.error(str(generated["error"]))

        if generated.get("grounding_error"):
            st.warning(
                str(generated["grounding_error"])
            )


if "analysis_payload" not in st.session_state:
    st.session_state.analysis_payload = None


with st.sidebar:
    st.header("⚙️ Pipeline settings")

    st.write(
        "PubMed + FDA FAERS → "
        "Reranking → ADE detection → "
        "Gemini → Grounding audit"
    )

    semantic_audit = st.checkbox(
        "Enable Gemini semantic audit",
        value=False,
        help=(
            "Uses additional Gemini API calls to check "
            "whether cited evidence supports each claim."
        ),
    )

    st.caption(
        "Disabled: fast deterministic screening. "
        "Enabled: slower Gemini-assisted verification."
    )

    st.divider()

    if st.button(
        "Clear previous analysis",
        use_container_width=True,
    ):
        st.session_state.analysis_payload = None
        st.rerun()

    st.divider()

    st.warning(
        "Research prototype only. "
        "Not validated for clinical decision-making. "
        "FAERS reports do not establish causality "
        "or incidence."
    )


st.title("💊 Pharmacovigilance & ADE Evidence Explorer")

st.caption(
    "Query-aware PubMed and FDA FAERS retrieval, "
    "adverse event detection, citation-validated "
    "Gemini synthesis, and claim-level grounding audit."
)

query = st.text_area(
    "Drug safety question",
    value=(
        "What are the risks of combining "
        "warfarin with aspirin?"
    ),
    height=100,
)

run = st.button(
    "🚀 Analyze safety evidence",
    type="primary",
    use_container_width=True,
)


if run:
    if not query.strip():
        st.error("Enter a drug safety question.")
        st.stop()

    st.session_state.analysis_payload = None

    documents = []
    result = {}
    summary = {}
    generated = None
    grounding_report = None

    with st.status(
        "Running pharmacovigilance pipeline...",
        expanded=True,
    ) as status:

        try:
            st.write(
                "Step 1: Retrieving and reranking evidence..."
            )

            documents = list(
                hybrid_retrieve(query.strip())
            )

            st.write(
                f"Retrieved {len(documents)} documents."
            )

        except Exception as exc:
            status.update(
                label="Retrieval failed",
                state="error",
            )
            st.error(f"Retrieval error: {exc}")
            st.stop()

        try:
            st.write(
                "Step 2: Detecting adverse event findings..."
            )

            result = detect_adverse_events(
                query=query.strip(),
                documents=documents,
            )

            summary = summarize_ade_results(result)

            st.write(
                "Detected "
                + str(summary.get("total_findings", 0))
                + " event-document findings."
            )

        except Exception as exc:
            status.update(
                label="ADE detection failed",
                state="error",
            )
            st.error(f"ADE detection error: {exc}")
            st.stop()

        try:
            st.write(
                "Step 3: Generating citation-validated answer..."
            )

            # The generator already performs deterministic
            # grounding screening when generation succeeds.
            generated = generate_from_retrieval(
                query=query.strip(),
                retrieved_documents=documents,
                grounding_use_llm=semantic_audit,
            )

            if isinstance(generated, dict):
                grounding_report = generated.get(
                    "grounding_report"
                )

                # Fallback for generator versions without
                # an integrated grounding report.
                if (
                    not grounding_report
                    and generated.get("status") == "validated"
                    and generated.get("answer")
                ):
                    st.write(
                        "Step 4: Evaluating answer grounding..."
                    )

                    grounding_report = evaluate_answer(
                        answer=generated["answer"],
                        documents=documents,
                        use_llm=semantic_audit,
                    )

        except Exception as exc:
            st.warning(
                "Generation or grounding audit failed. "
                "Retrieved evidence remains available."
            )
            st.caption(str(exc))

        status.update(
            label="Pipeline execution complete",
            state="complete",
            expanded=False,
        )

    st.session_state.analysis_payload = {
        "query": query.strip(),
        "documents": documents,
        "result": result,
        "summary": summary,
        "generated": generated,
        "grounding_report": grounding_report,
        "semantic_audit": semantic_audit,
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
    }


payload = st.session_state.analysis_payload

if payload is None:
    st.info(
        "Enter a drug safety question and click "
        "'Analyze safety evidence' to begin."
    )

else:
    documents = payload["documents"]
    result = payload["result"]
    summary = payload["summary"]
    generated = payload["generated"]
    grounding_report = payload["grounding_report"]

    st.subheader("📊 Evidence overview")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Retrieved", len(documents))
    m2.metric(
        "PubMed",
        count_by_source(documents, "PubMed"),
    )
    m3.metric(
        "FAERS",
        count_by_source(documents, "FAERS"),
    )
    m4.metric(
        "ADE findings",
        summary.get("total_findings", 0),
    )

    st.caption(
        "ADE findings are event-document associations, "
        "not patient counts or incidence estimates."
    )

    st.subheader("🧠 Query context")

    c1, c2 = st.columns(2)

    c1.write(
        "**Recognized drugs:** "
        + (
            ", ".join(result.get("query_drugs", []))
            or "None identified"
        )
    )

    c2.write(
        "**Explicit adverse events:** "
        + (
            ", ".join(result.get("query_events", []))
            or "None specified"
        )
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🤖 AI Analysis",
            "🧪 Grounding Audit",
            "🛡️ ADE Findings",
            "📚 Sources",
        ]
    )

    with tab1:
        render_generated(generated)

    with tab2:
        render_grounding(grounding_report)

    with tab3:
        a, b, c = st.columns(3)

        a.metric(
            "Direct",
            summary.get("direct_count", 0),
        )
        b.metric(
            "Partial / co-reports",
            summary.get("partial_count", 0),
        )
        c.metric(
            "Background",
            summary.get("background_count", 0),
        )

        st.caption(
            "These are detector relevance categories, "
            "not validated clinical risk grades."
        )

        render_findings(result)

    with tab4:
        pubmed_tab, faers_tab = st.tabs(
            ["🔬 PubMed", "🏥 FDA FAERS"]
        )

        with pubmed_tab:
            render_documents(documents, "PubMed")

        with faers_tab:
            render_documents(documents, "FAERS")

    st.divider()

    st.subheader("📥 Export research report")

    export = {
        "query": payload["query"],
        "timestamp": payload["timestamp"],
        "semantic_audit_enabled": payload["semantic_audit"],
        "retrieved_document_count": len(documents),
        "pubmed_document_count": count_by_source(
            documents, "PubMed"
        ),
        "faers_document_count": count_by_source(
            documents, "FAERS"
        ),
        "ade_summary": summary,
        "ade_findings": result.get("findings", []),
        "generated_answer": (
            generated.get("answer")
            if isinstance(generated, dict)
            else None
        ),
        "citation_validation": (
            generated.get("citation_validation")
            if isinstance(generated, dict)
            else None
        ),
        "grounding_report": grounding_report,
        "warning": (
            "Research prototype only. "
            "Not validated for clinical decision-making."
        ),
    }

    st.download_button(
        "Download JSON analysis report",
        data=to_json_bytes(export),
        file_name="pharmacovigilance_rag_report.json",
        mime="application/json",
        use_container_width=True,
    )

    st.warning(
        "Research use only. AI-generated findings and "
        "grounding classifications require independent "
        "verification. FAERS co-reports cannot establish "
        "causality, incidence, or drug interaction risk."
    )
