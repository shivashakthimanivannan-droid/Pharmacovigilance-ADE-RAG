from ade_detector import detect_adverse_events


def main():

    text = """
    The patient received warfarin therapy.
    The combination was associated with an
    increased risk of bleeding and anemia.
    """

    events = detect_adverse_events(text)

    print("\nDetected Adverse Events:")
    
    for event in events:
        print(f"- {event}")


if __name__ == "__main__":
    main()