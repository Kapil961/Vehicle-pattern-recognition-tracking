import os
import cv2
import pandas as pd
import easyocr

from ultralytics import YOLO

VIDEO = "data/raw/highway_videos/highway_traffic_h264.mp4"
PLATE_MODEL = "models/license_plate/license_plate_detector.pt"
TRACKING_CSV = "outputs/reports/vehicle_tracking.csv"

OUTPUT_DIR = "outputs/plate_test"
CSV_PATH = "outputs/reports/plate_detection_test.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading plate detector...")
model = YOLO(PLATE_MODEL)

print("Loading EasyOCR...")
reader = easyocr.Reader(["en"], gpu=False)

tracking = pd.read_csv(TRACKING_CSV)

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Cannot open highway video")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

records = []
sample_interval = int(fps)  # One frame per second

print("\nTesting plate detection on sampled frames...")

for frame_index in range(0, total_frames, sample_interval):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()

    if not ret:
        continue

    frame_number = frame_index + 1

    results = model.predict(
        frame,
        imgsz=960,
        conf=0.15,
        device="cpu",
        verbose=False
    )

    result = results[0]
    annotated = frame.copy()

    if result.boxes is None:
        continue

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    for plate_index, (box, confidence) in enumerate(
        zip(boxes, confidences)
    ):

        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            continue

        plate_crop = frame[y1:y2, x1:x2]

        if plate_crop.size == 0:
            continue

        # Improve OCR input size
        scale = max(2, 160 // max(1, plate_crop.shape[1]))
        enlarged = cv2.resize(
            plate_crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        ocr_results = reader.readtext(
            enlarged,
            detail=0,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        plate_text = "".join(ocr_results).upper().replace(" ", "")

        # Match plate center to an existing vehicle bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        candidates = tracking[
            tracking["frame"] == frame_number
        ]

        matched_track = None
        matched_type = None

        for _, vehicle in candidates.iterrows():

            if (
                vehicle["x1"] <= center_x <= vehicle["x2"]
                and vehicle["y1"] <= center_y <= vehicle["y2"]
            ):
                matched_track = int(vehicle["track_id"])
                matched_type = vehicle["vehicle_type"]
                break

        records.append({
            "frame": frame_number,
            "timestamp_seconds": round(frame_index / fps, 2),
            "plate_index": plate_index,
            "plate_confidence": round(float(confidence), 4),
            "plate_text": plate_text,
            "vehicle_track_id": matched_track,
            "vehicle_type": matched_type,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

        cv2.rectangle(
            annotated, (x1, y1), (x2, y2), (0, 0, 255), 3
        )

        label = plate_text if plate_text else "Plate detected"
        cv2.putText(
            annotated,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    if result.boxes is not None and len(result.boxes) > 0:
        output_path = os.path.join(
            OUTPUT_DIR, f"frame_{frame_number:04d}.jpg"
        )
        cv2.imwrite(output_path, annotated)

    if frame_number % 600 == 1:
        print(f"Checked frame {frame_number}/{total_frames}")

cap.release()

df = pd.DataFrame(records)
df.to_csv(CSV_PATH, index=False)

print("\n" + "=" * 45)
print("PLATE DETECTION TEST COMPLETED")
print("=" * 45)

print("Sampled frames:", len(range(0, total_frames, sample_interval)))
print("Plate detections:", len(df))

if not df.empty:
    print("Unique OCR texts:", df["plate_text"].replace("", pd.NA).nunique())
    print("Non-empty OCR results:", (df["plate_text"] != "").sum())
    print("Matched vehicle IDs:", df["vehicle_track_id"].nunique())

    print("\nDETECTION RESULTS:")
    print(df.head(20).to_string(index=False))

print("\nCSV:", CSV_PATH)
print("Annotated frames:", OUTPUT_DIR)
