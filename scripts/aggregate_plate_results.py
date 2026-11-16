import re
import pandas as pd

INPUT = "outputs/reports/plate_detection_test.csv"
OUTPUT = "outputs/reports/plate_consensus.csv"

df = pd.read_csv(INPUT)

df["plate_text"] = (
    df["plate_text"]
    .fillna("")
    .str.upper()
    .str.replace(r"[^A-Z0-9]", "", regex=True)
)

# Keep plausible-length OCR strings and reasonably confident detections
df = df[
    df["vehicle_track_id"].notna()
    & df["plate_text"].str.len().between(5, 12)
    & (df["plate_confidence"] >= 0.35)
].copy()

if df.empty:
    print("No usable plate readings found.")
    raise SystemExit(0)

# Count each reading only once per frame
df = df.drop_duplicates(
    subset=["vehicle_track_id", "frame", "plate_text"]
)

summary = (
    df.groupby(["vehicle_track_id", "plate_text"])
    .agg(
        distinct_frames=("frame", "nunique"),
        mean_detector_confidence=("plate_confidence", "mean"),
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        vehicle_type=("vehicle_type", "first")
    )
    .reset_index()
)

summary["mean_detector_confidence"] = (
    summary["mean_detector_confidence"].round(3)
)

summary = summary.sort_values(
    ["vehicle_track_id", "distinct_frames",
     "mean_detector_confidence"],
    ascending=[True, False, False]
)

# Select the most frequently repeated OCR result per vehicle
best = summary.drop_duplicates(
    subset=["vehicle_track_id"],
    keep="first"
).copy()

best["reading_status"] = best["distinct_frames"].apply(
    lambda n: "Repeated reading" if n >= 2
    else "Single reading - unverified"
)

best.to_csv(OUTPUT, index=False)

print("\nPLATE OCR CONSENSUS")
print("=" * 65)
print("Vehicles with usable OCR:", len(best))
print("Repeated readings:", (best["distinct_frames"] >= 2).sum())

print("\nSelected readings:")
print(best.to_string(index=False))

print("\nSaved:", OUTPUT)
