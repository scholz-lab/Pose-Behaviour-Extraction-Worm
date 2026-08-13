import json
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD JSON
# ============================================================

def load_label_studio_json(file_path):
    """Load a Label Studio JSON export."""
    with open(file_path, "r") as f:
        return json.load(f)


# ============================================================
# 2. GET ANNOTATION RESULTS
# ============================================================

def get_results(data):
    """Return the annotation result list."""
    return data[0]["annotations"][0]["result"]


# ============================================================
# 3. EXTRACT RANGES FOR ONE LABEL
# ============================================================

def extract_ranges(results, target_label):
    """
    Extract all (start, end) ranges belonging to target_label.
    """

    ranges_found = []

    for item in results:

        value = item.get("value", {})

        labels = value.get("timelinelabels", [])
        ranges = value.get("ranges", [])

        if target_label not in labels:
            continue

        for r in ranges:
            ranges_found.append(
                (r["start"], r["end"])
            )

    return ranges_found


# ============================================================
# 4. PRINT EVENTS
# ============================================================

def print_events(name, ranges):
    """Print start, end and duration for each event."""

    print(f"\n{name}")
    print("-" * 40)

    for i, (start, end) in enumerate(ranges, start=1):

        duration = end - start

        print(
            f"Event {i:02d}: "
            f"start={start}, "
            f"end={end}, "
            f"duration={duration}"
        )


# ============================================================
# 5. CONVERT TO PLOT FORMAT
# ============================================================

def to_broken_bar_format(ranges):
    """
    broken_barh expects:
    (start, duration)

    instead of:
    (start, end)
    """

    return [
        (start, end - start)
        for start, end in ranges
    ]


# ============================================================
# 6. PLOT LEFT + RIGHT CONTACT
# ============================================================

def plot_contact_timeline(left_ranges, right_ranges):

    fig, ax = plt.subplots(figsize=(14, 4))

    left_plot = to_broken_bar_format(left_ranges)
    right_plot = to_broken_bar_format(right_ranges)

    ax.broken_barh(
        left_plot,
        (20, 8),
        label="Cont L Larva"
    )

    ax.broken_barh(
        right_plot,
        (5, 8),
        label="Cont R Larva"
    )

    ax.set_yticks([24, 9])

    ax.set_yticklabels([
        "Left larva contact",
        "Right larva contact"
    ])

    ax.set_xlabel("Recording position (frames)")
    ax.set_title("Larva Contact Timeline")

    ax.set_ylim(0, 35)

    ax.grid(
        axis="x",
        alpha=0.25
    )

    ax.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    file_path = "C:/Users/pyaasa/Downloads/spool_20260723_2.json"

    # Load
    data = load_label_studio_json(file_path)

    # Annotation results
    results = get_results(data)

    # Extract only the two labels we want
    left_contacts = extract_ranges(
        results,
        "Cont L Larva"
    )

    right_contacts = extract_ranges(
        results,
        "Cont R Larva"
    )

    # Print ranges
    print_events(
        "LEFT LARVA CONTACT",
        left_contacts
    )

    print_events(
        "RIGHT LARVA CONTACT",
        right_contacts
    )

    # Plot
    plot_contact_timeline(
        left_contacts,
        right_contacts
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()