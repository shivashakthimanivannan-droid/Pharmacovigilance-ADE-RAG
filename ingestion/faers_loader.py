import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FAERS_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "faers"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)


# ============================================================
# FILES
# ============================================================

FILES = {
    "demo": os.path.join(FAERS_DIR, "DEMO26Q2.txt"),
    "drug": os.path.join(FAERS_DIR, "DRUG26Q2.txt"),
    "reac": os.path.join(FAERS_DIR, "REAC26Q2.txt"),
    "outc": os.path.join(FAERS_DIR, "OUTC26Q2.txt"),
}


# ============================================================
# LOAD
# ============================================================

def load_file(path):

    print(f"Loading {os.path.basename(path)}...")

    df = pd.read_csv(
        path,
        sep="$",
        dtype=str,
        encoding="latin1",
        low_memory=False
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
    )

    print(f"Records: {len(df):,}")

    return df


# ============================================================
# FIND CASE COLUMN
# ============================================================

def find_case_column(df):

    possible = [
        "CASEID",
        "CASE_ID",
        "CASE ID"
    ]

    for column in possible:
        if column in df.columns:
            return column

    raise ValueError(
        f"Case identifier not found. "
        f"Available columns: {df.columns.tolist()}"
    )


# ============================================================
# CREATE SAFETY DOCUMENTS
# ============================================================

def create_documents(
    demo,
    drug,
    reac,
    outc
):

    demo_case = find_case_column(demo)
    drug_case = find_case_column(drug)
    reac_case = find_case_column(reac)
    outc_case = find_case_column(outc)

    # --------------------------------------------------------
    # DRUG INFORMATION
    # --------------------------------------------------------

    drug_columns = [
        column
        for column in [
            "DRUGNAME",
            "PROD_AI",
            "ROLE_COD",
            "DRUG_SEQ"
        ]
        if column in drug.columns
    ]

    drug_data = (
        drug[
            [drug_case] + drug_columns
        ]
        .fillna("")
        .groupby(drug_case)[drug_columns]
        .agg(
            lambda x: "; ".join(
                sorted(
                    set(
                        str(value)
                        for value in x
                        if str(value).strip()
                    )
                )
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # REACTION INFORMATION
    # --------------------------------------------------------

    reaction_columns = [
        column
        for column in [
            "PT",
            "DRT"
        ]
        if column in reac.columns
    ]

    if reaction_columns:

        reaction_data = (
            reac[
                [reac_case] + reaction_columns
            ]
            .fillna("")
            .groupby(reac_case)[reaction_columns]
            .agg(
                lambda x: "; ".join(
                    sorted(
                        set(
                            str(value)
                            for value in x
                            if str(value).strip()
                        )
                    )
                )
            )
            .reset_index()
        )

    else:

        reaction_data = pd.DataFrame(
            columns=[reac_case]
        )

    # --------------------------------------------------------
    # OUTCOME INFORMATION
    # --------------------------------------------------------

    outcome_columns = [
        column
        for column in [
            "OUTC_COD"
        ]
        if column in outc.columns
    ]

    if outcome_columns:

        outcome_data = (
            outc[
                [outc_case] + outcome_columns
            ]
            .fillna("")
            .groupby(outc_case)[outcome_columns]
            .agg(
                lambda x: "; ".join(
                    sorted(
                        set(
                            str(value)
                            for value in x
                            if str(value).strip()
                        )
                    )
                )
            )
            .reset_index()
        )

    else:

        outcome_data = pd.DataFrame(
            columns=[outc_case]
        )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    documents = demo.copy()

    documents = documents.merge(
        drug_data,
        left_on=demo_case,
        right_on=drug_case,
        how="left",
        suffixes=("", "_DRUG")
    )

    if not reaction_data.empty:

        documents = documents.merge(
            reaction_data,
            left_on=demo_case,
            right_on=reac_case,
            how="left",
            suffixes=("", "_REAC")
        )

    if not outcome_data.empty:

        documents = documents.merge(
            outcome_data,
            left_on=demo_case,
            right_on=outc_case,
            how="left",
            suffixes=("", "_OUTC")
        )

    return documents


# ============================================================
# CREATE TEXT FOR RAG
# ============================================================

def create_rag_text(row):

    parts = []

    for column in row.index:

        if column in [
            "RAG_TEXT",
            "REPORT_DATE"
        ]:
            continue

        value = row[column]

        if pd.notna(value):

            value = str(value).strip()

            if value:

                parts.append(
                    f"{column}: {value}"
                )

    return "\n".join(parts)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FAERS → RAG DOCUMENT PIPELINE")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # Load data
    demo = load_file(FILES["demo"])
    drug = load_file(FILES["drug"])
    reac = load_file(FILES["reac"])
    outc = load_file(FILES["outc"])

    print("\nCreating unified safety records...")

    documents = create_documents(
        demo,
        drug,
        reac,
        outc
    )

    print(
        f"Unified records: {len(documents):,}"
    )

    # Create RAG text
    print("\nCreating RAG text...")

    documents["RAG_TEXT"] = documents.apply(
        create_rag_text,
        axis=1
    )

    # Keep useful fields
    output = documents[
        [
            column
            for column in [
                find_case_column(documents),
                "RAG_TEXT"
            ]
            if column in documents.columns
        ]
    ].copy()

    output_file = os.path.join(
        OUTPUT_DIR,
        "faers_rag_documents.csv"
    )

    output.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("FAERS RAG DATASET CREATED")
    print("=" * 70)

    print(
        f"\nSaved to:\n{output_file}"
    )

    print(
        f"\nDocuments created: "
        f"{len(output):,}"
    )

    print("\nSample RAG document:")
    print("-" * 70)

    if len(output) > 0:
        print(
            output.iloc[0]["RAG_TEXT"][:3000]
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()