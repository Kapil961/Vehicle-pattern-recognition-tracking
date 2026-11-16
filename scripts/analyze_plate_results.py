import pandas as pd
from collections import Counter

FILE = "outputs/reports/final_vehicle_summary_improved.csv"

df = pd.read_csv(FILE)

print("\nPLATE OCR VALIDATION REPORT")
print("=" * 65)

candidates = df[df["plate_confirmed"] == True].copy()

print(f"\nTotal tracked vehicles: {len(df)}")
print(f"Vehicles with OCR consensus: {len(candidates)}")

for _, row in candidates.iterrows():
    print("\n" + "-" * 50)
    print(f"Track ID: {row['track_id']}")
    print(f"Vehicle type: {row['vehicle_type']}")
    print(f"Plate candidate: {row['plate_text']}")
    print(f"Votes: {row['plate_votes']}")
    print(f"Readings: {row['plate_readings']}")
    print(f"Frames: {row['first_frame']} - {row['last_frame']}")

    if str(row["plate_text"]).isalpha():
        print("WARNING: Alphabet-only candidate; inspect for false positive.")

print("\n" + "=" * 65)
print("COLOR DISTRIBUTION")
print(df["dominant_color"].value_counts(dropna=False).to_string())
