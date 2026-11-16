import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIDEO_PATH = "data/raw/highway_videos/highway_traffic_h264.mp4"
SUMMARY_PATH = "outputs/reports/final_vehicle_summary_improved.csv"
PLATE_PATH = "outputs/reports/license_plate_detections_improved.csv"

OUTPUT_DIR = "outputs/plate_review"
os.makedirs(OUTPUT_DIR, exist_ok=True)

summary = pd.read_csv(SUMMARY_PATH)
plates = pd.read_csv(PLATE_PATH)

candidates = summary[summary["plate_confirmed"] == True].copy()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open input video.")

font = ImageFont.load_default()

tiles = []

for _, vehicle in candidates.iterrows():
    track_id = int(vehicle["track_id"])
    candidate_text = str(vehicle["plate_text"])

    readings = plates[
        (plates["track_id"] == track_id) &
        (plates["plate_text"] == candidate_text)
    ].copy()

    readings["quality"] = (
        readings["ocr_confidence"] * 0.5 +
        readings["plate_confidence"] * 0.5
    )

    readings = readings.sort_values("quality", ascending=False)
    readings = readings.drop_duplicates(subset=["frame"]).head(3)

    tile = Image.new("RGB", (600, 180), "white")
    draw = ImageDraw.Draw(tile)

    draw.text(
        (10, 8),
        f"ID {track_id} | Candidate: {candidate_text} | "
        f"Votes: {int(vehicle['plate_votes'])}",
        fill="black",
        font=font
    )

    for index, (_, reading) in enumerate(readings.iterrows()):
        frame_number = int(reading["frame"])

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        success, frame = cap.read()

        if not success:
            continue

        x1 = max(0, int(reading["x1"]) - 15)
        y1 = max(0, int(reading["y1"]) - 15)
        x2 = min(frame.shape[1], int(reading["x2"]) + 15)
        y2 = min(frame.shape[0], int(reading["y2"]) + 15)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_image = Image.fromarray(crop)
        crop_image.thumbnail((185, 115))

        x_position = 10 + index * 195
        y_position = 35

        tile.paste(crop_image, (x_position, y_position))

        draw.text(
            (x_position, 155),
            f"Frame {frame_number}",
            fill="black",
            font=font
        )

    tiles.append(tile)

    tile.save(
        os.path.join(OUTPUT_DIR, f"vehicle_{track_id}.jpg")
    )

cap.release()

columns = 3
rows = (len(tiles) + columns - 1) // columns

contact_sheet = Image.new(
    "RGB",
    (columns * 600, rows * 180),
    (220, 220, 220)
)

for index, tile in enumerate(tiles):
    x = (index % columns) * 600
    y = (index // columns) * 180
    contact_sheet.paste(tile, (x, y))

output_path = os.path.join(OUTPUT_DIR, "plate_candidates_contact_sheet.jpg")
contact_sheet.save(output_path)

print("\nPLATE REVIEW IMAGES CREATED")
print("Candidates:", len(tiles))
print("Individual crops:", OUTPUT_DIR)
print("Contact sheet:", output_path)
