
# Pharmacovigilance & Adverse Drug Event (ADE) RAG System

An AI-powered pharmacovigilance research prototype that retrieves drug-safety evidence from PubMed and FDA FAERS, detects adverse event associations, and generates citation-grounded responses using Google Gemini.

The system combines query-aware retrieval, semantic reranking, rule-based adverse event detection, citation validation, and claim-level grounding evaluation in an interactive Streamlit application.

> **Disclaimer:** This is a research and portfolio prototype, not a clinically validated decision-support system. FAERS reports do not establish causality, incidence, or confirmed drug interactions.

## Key Features

- **Hybrid evidence retrieval:** Combines PubMed research literature with locally processed FDA FAERS reports.
- **Query-aware drug recognition:** Detects supported medications, aliases, and drug combinations.
- **Semantic reranking:** Uses a Cross-Encoder to prioritize retrieved evidence.
- **ADE detection:** Extracts recognized adverse-event associations and classifies evidence relevance.
- **Gemini-powered analysis:** Generates natural-language research summaries using retrieved evidence.
- **Citation validation:** Checks generated citations against retrieved document identifiers.
- **Grounding audit:** Evaluates individual claims using deterministic screening or optional Gemini-assisted verification.
- **Interactive dashboard:** Displays AI analysis, grounding audit results, ADE findings, and evidence sources.
- **Research report export:** Supports exporting analysis results as JSON.

## System Architecture

```text
                    User Drug-Safety Query
                              |
                              v
                     Query Understanding
                    (Drugs, Aliases, Events)
                              |
                 +------------+------------+
                 |                         |
                 v                         v
           PubMed API                Local FDA FAERS
          Literature Search          Document Search
                 |                         |
                 +------------+------------+
                              |
                              v
                      Evidence Filtering
                              |
                              v
                    Cross-Encoder Reranking
                              |
                              v
                    Top Retrieved Evidence
                              |
                 +------------+------------+
                 |                         |
                 v                         v
           ADE Detection             Gemini Generation
                 |                         |
                 |                         v
                 |                  Citation Validation
                 |                         |
                 +------------+------------+
                              |
                              v
                       Grounding Audit
                              |
                              v
                       Streamlit Dashboard
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| User interface | Streamlit |
| Language model | Google Gemini |
| Literature source | PubMed / NCBI E-utilities |
| Safety report source | FDA FAERS |
| Semantic retrieval | Sentence Transformers |
| Evidence reranking | Cross-Encoder |
| Data processing | Pandas |
| Grounding evaluation | Python and Gemini-assisted auditing |
| Structured data validation | Pydantic |
| Additional vector database integration | Weaviate |

## Project Structure

```text
Pharmacovigilence_Rag/
|
|-- app/
|   |-- app.py
|
|-- ingestion/
|   |-- pubmed_loader.py
|   |-- faers_loader.py
|
|-- preprocessing/
|   |-- text_cleaner.py
|   |-- chunker.py
|
|-- embeddings/
|   |-- embedder.py
|
|-- retrieval/
|   |-- retriever.py
|   |-- faers_retriever.py
|   |-- reranker.py
|   |-- test_reranker.py
|
|-- detection/
|   |-- ade_detector.py
|   |-- test_ade_detector.py
|   |-- test_pipeline.py
|
|-- generation/
|   |-- llm_generator.py
|   |-- test_generator.py
|   |-- test_queries.json
|
|-- evaluation/
|   |-- grounding_evaluator.py
|   |-- benchmark.py
|   |-- benchmark_cases.json
|   |-- evaluate_generation.py
|   |-- evaluate_retrieval.py
|   |-- final_evaluation.py
|
|-- vector_db/
|   |-- weaviate_client.py
|   |-- faers_ingest.py
|   |-- test_weaviate.py
|
|-- data/
|   |-- raw/
|   |-- processed/
|
|-- requirements.txt
|-- test_environment.py
|-- .gitignore
|-- README.md
```

**Note:** Raw and processed datasets are excluded from the Git repository. The architecture above represents the main application workflow; additional experimental modules are included in the repository.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the placeholder URL with the actual repository URL after publishing.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

Additional environment variables may be required for optional integrations.

Never commit real API keys or credentials to GitHub.

### 5. Prepare datasets

The retrieval pipeline uses PubMed literature and locally processed FDA FAERS reports.

- PubMed literature is retrieved through NCBI E-utilities.
- FAERS reports must be obtained from the FDA's public data resources and processed into the document format expected by the local retrieval pipeline.
- The local application expects the processed FAERS dataset at `data/processed/faers_rag_documents.csv`.

The processed FAERS dataset is not included in this repository. A fresh clone will require local data preparation before the full hybrid retrieval pipeline can run.

Data sources:

- [PubMed](https://pubmed.ncbi.nlm.nih.gov/)
- [FDA FAERS](https://www.fda.gov/drugs/surveillance/fda-adverse-event-reporting-system-faers)

## Running the Application

From the project root:

```bash
streamlit run app/app.py
```

Enter a drug-safety question to retrieve and analyze evidence.

Example queries:

```text
What are the risks of combining warfarin with aspirin?

What are the adverse effects of metformin?
```

## Testing

Test the retrieval pipeline:

```bash
python -m retrieval.retriever
```

Test the ADE detector:

```bash
python -m detection.ade_detector
```

Run the grounding benchmark:

```bash
python -m evaluation.benchmark
```

The benchmark uses a small synthetic evaluation set. Results should not be interpreted as clinical performance measurements.

## Evaluation

During local development, the grounding benchmark passed all nine synthetic test cases.

| Metric | Local test result |
|---|---|
| Synthetic benchmark cases | 9 |
| Cases passed | 9 |
| Macro F1 | 1.00 |

These figures describe performance on a limited synthetic benchmark, not real-world clinical accuracy, retrieval completeness, or safety.

A metformin dashboard test retrieved five PubMed documents and five FAERS reports. Its Gemini-assisted grounding audit classified 14 claims:

| Classification | Count |
|---|---:|
| Supported | 10 |
| Uncited | 3 |
| Methodological disclaimer | 1 |

Automated claim classifications are provisional and require independent verification.

## Limitations

1. Drug recognition relies on a configured vocabulary of medication names and aliases.
2. Retrieved literature may contain background information rather than direct evidence for a specific question.
3. FAERS reports are spontaneous reports and cannot establish causality, incidence, or concurrent drug exposure.
4. ADE findings represent extracted event-document associations, not patient counts.
5. Gemini-generated answers may contain uncited or insufficiently supported statements.
6. Grounding audit classifications do not replace expert clinical review.
7. The project has not undergone clinical validation.

## Future Improvements

Potential extensions include broader drug-name normalization, expanded adverse-event terminology, more systematic retrieval evaluation, and expert-reviewed clinical benchmark datasets.

## Intended Use

This project demonstrates applied skills in:

- Retrieval-Augmented Generation (RAG)
- Natural Language Processing (NLP)
- LLM application development
- Semantic search and reranking
- Evidence-based answer generation
- AI evaluation and guardrails
- Data engineering
- Streamlit application development

**For educational and research demonstration purposes only.**
