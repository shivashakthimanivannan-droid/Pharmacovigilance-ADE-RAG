from generation.llm_generator import generate_safety_analysis


def main():

    query = (
        "What are the risks of combining "
        "warfarin with aspirin?"
    )

    findings = [
        {
            "pmid": "TEST123",
            "title": "Example Drug Interaction Study",
            "year": "2025",
            "drugs": [
                "warfarin",
                "aspirin"
            ],
            "interaction_terms": [
                "combination",
                "increased risk"
            ],
            "adverse_events": [
                "bleeding"
            ],
            "evidence": (
                "Example retrieved evidence "
                "describing the interaction."
            )
        }
    ]

    result = generate_safety_analysis(
        query,
        findings
    )

    print("\n")
    print("=" * 70)
    print("LLM PHARMACOVIGILANCE ANALYSIS")
    print("=" * 70)

    print(result)


if __name__ == "__main__":
    main()