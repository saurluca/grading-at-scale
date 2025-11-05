from pathlib import Path
import pandas as pd
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def process_grading_results(json_path):
    """Process JSON grading results and add to CSV"""
    csv_path = PROJECT_ROOT / "data" / "logic" / "quiz_1_subtasks.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not Path(json_path).exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Load CSV
    df = pd.read_csv(csv_path, sep=";")

    # Load JSON grades
    with open(json_path, "r", encoding="utf-8") as f:
        grades = json.load(f)

    # Add new columns for subtask points (prefilled with 1.0)
    sentences_letters = ["a", "b", "c", "d", "e", "f"]
    validity_letters = ["a", "b", "c", "d", "e", "f", "g", "h"]

    for letter in sentences_letters:
        col_name = f"sentences_{letter}_points"
        df[col_name] = 1.0

    for letter in validity_letters:
        col_name = f"validity_and_soundness_{letter}_points"
        df[col_name] = 1.0

    # Populate grades (JSON uses Teilnehmer name as key)
    warnings = []
    for idx, row in df.iterrows():
        teilnehmer = str(row["Teilnehmer"])

        if teilnehmer in grades:
            group_grades = grades[teilnehmer]

            # Process sentences grades
            for letter in sentences_letters:
                col_name = f"sentences_{letter}_points"
                key = f"sentences_{letter}"
                if key in group_grades:
                    df.at[idx, col_name] = float(group_grades[key])

            # Process validity grades
            for letter in validity_letters:
                col_name = f"validity_and_soundness_{letter}_points"
                key = f"validity_and_soundness_{letter}"
                if key in group_grades:
                    df.at[idx, col_name] = float(group_grades[key])

        # Validate sums
        sentences_sum = sum(
            [df.at[idx, f"sentences_{letter}_points"] for letter in sentences_letters]
        )
        validity_sum = sum(
            [
                df.at[idx, f"validity_and_soundness_{letter}_points"]
                for letter in validity_letters
            ]
        )

        expected_sentences = (
            float(row["sentences_points"]) if pd.notna(row["sentences_points"]) else 0.0
        )
        expected_validity = (
            float(row["validity_and_soundness_points"])
            if pd.notna(row["validity_and_soundness_points"])
            else 0.0
        )

        if abs(sentences_sum - expected_sentences) > 0.01:
            warnings.append(
                f"Group '{teilnehmer}': Sentences sum mismatch. "
                f"Calculated: {sentences_sum:.1f}, Expected: {expected_sentences:.1f}"
            )

        if abs(validity_sum - expected_validity) > 0.01:
            warnings.append(
                f"Group '{teilnehmer}': Validity sum mismatch. "
                f"Calculated: {validity_sum:.1f}, Expected: {expected_validity:.1f}"
            )

    # Print warnings
    if warnings:
        print("\n⚠️  Warnings (sum mismatches):")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    else:
        print("\n✓ All sums match expected totals!\n")

    # Save to new CSV
    output_path = PROJECT_ROOT / "data" / "logic" / "quiz_1_subtasks_graded.csv"
    df.to_csv(output_path, index=False, sep=";")
    print(f"Graded CSV saved to: {output_path}")
    print(f"Total groups processed: {len(df)}")

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_grading_results.py <path_to_json_file>")
        print("Example: python process_grading_results.py grading_results.json")
        sys.exit(1)

    json_path = sys.argv[1]
    process_grading_results(json_path)
