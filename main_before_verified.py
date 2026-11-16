
import os
import re
import json
import time
import cv2
import torch
import easyocr
import difflib
import numpy as np
import pandas as pd
import imageio_ffmpeg

from collections import defaultdict, Counter, deque
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_VIDEO = "data/raw/highway_videos/highway_traffic_h264.mp4"

VEHICLE_MODEL_PATH = "models/vehicle/yolov8n.pt"
PLATE_MODEL_PATH = "models/license_plate/license_plate_detector.pt"
SPEED_CALIBRATION_PATH = "configs/speed_calibration.json"

OUTPUT_VIDEO = "outputs/videos/final_vehicle_recognition_improved.mp4"

TRACKING_CSV = "outputs/reports/vehicle_tracking_improved.csv"
PLATE_CSV = "outputs/reports/license_plate_detections_improved.csv"
SUMMARY_CSV = "outputs/reports/final_vehicle_summary_improved.csv"
SPEED_CSV = "outputs/reports/vehicle_speed_tracking_improved.csv"

GROUND_TRUTH_CSV = "outputs/reports/ground_truth_validation.csv"
VALIDATION_CSV = "outputs/reports/model_accuracy_validation.csv"

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

VEHICLE_CONFIDENCE = 0.25
PLATE_CONFIDENCE = 0.25

PLATE_DETECTION_INTERVAL = 10
PLATE_IMAGE_SIZE = 1280
PLATE_PERSISTENCE_FRAMES = 90

MIN_PLATE_VOTES = 3
MIN_COLOR_OBSERVATIONS = 3
MIN_OCR_CONFIDENCE = 0.25

COLOR_HISTORY_SIZE = 30
PLATE_HISTORY_SIZE = 100

DEVICE = "cpu"
MAX_REASONABLE_SPEED = 180.0


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for folder in [
    "outputs/videos",
    "outputs/reports",
    "outputs/plots"
]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# VEHICLE COLOR CLASSIFICATION
# ============================================================

def classify_color(vehicle_crop):

    if vehicle_crop is None or vehicle_crop.size == 0:
        return "Unknown", 0.0

    height, width = vehicle_crop.shape[:2]

    if height < 15 or width < 15:
        return "Unknown", 0.0

    # Focus on the central/lower body.
    x1 = int(width * 0.20)
    x2 = int(width * 0.80)
    y1 = int(height * 0.35)
    y2 = int(height * 0.78)

    crop = vehicle_crop[y1:y2, x1:x2]

    if crop.size == 0:
        return "Unknown", 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # Ignore extreme shadows and overexposed pixels.
    valid = (v > 35) & (v < 245)

    if np.count_nonzero(valid) < 20:
        return "Unknown", 0.0

    hue = h[valid]
    saturation = s[valid]
    value = v[valid]

    # Neutral color classification.
    neutral_mask = saturation < 45
    neutral_ratio = float(np.mean(neutral_mask))

    if neutral_ratio >= 0.55:

        neutral_values = value[neutral_mask]

        if len(neutral_values) == 0:
            return "Unknown", 0.0

        median_value = float(np.median(neutral_values))

        if median_value > 185:
            color = "White"
        elif median_value > 85:
            color = "Gray"
        else:
            color = "Black"

        return color, neutral_ratio * 100.0

    # Chromatic color classification.
    chromatic_mask = saturation >= 45

    hue = hue[chromatic_mask]
    saturation = saturation[chromatic_mask]

    if len(hue) < 15:
        return "Unknown", 0.0

    color_ranges = {
        "Red": (hue <= 10) | (hue >= 170),
        "Orange": (hue > 10) & (hue <= 22),
        "Yellow": (hue > 22) & (hue <= 35),
        "Green": (hue > 35) & (hue <= 85),
        "Blue": (hue > 85) & (hue <= 130),
        "Purple": (hue > 130) & (hue < 170)
    }

    scores = {}

    for color, mask in color_ranges.items():
        scores[color] = float(
            np.sum(saturation[mask])
        )

    total_score = sum(scores.values())

    if total_score <= 0:
        return "Unknown", 0.0

    best_color = max(scores, key=scores.get)

    confidence = (
        scores[best_color] / total_score
    ) * 100.0

    if confidence < 35.0:
        return "Unknown", confidence

    return best_color, confidence


def get_stable_color(color_history, confidence_history):

    recent_colors = color_history[-COLOR_HISTORY_SIZE:]
    recent_confidences = confidence_history[-COLOR_HISTORY_SIZE:]

    valid_records = [
        (color, confidence)
        for color, confidence in zip(
            recent_colors,
            recent_confidences
        )
        if color != "Unknown"
    ]

    if len(valid_records) < MIN_COLOR_OBSERVATIONS:
        return "Unknown", 0.0

    counts = Counter(
        color for color, _ in valid_records
    )

    best_color, count = counts.most_common(1)[0]

    # Require a minimum majority of valid observations.
    consensus = count / len(valid_records)

    if consensus < 0.40:
        return "Unknown", 0.0

    winning_confidences = [
        confidence
        for color, confidence in valid_records
        if color == best_color
    ]

    mean_confidence = (
        float(np.mean(winning_confidences))
        if winning_confidences
        else 0.0
    )

    # Combine visual confidence with temporal agreement.
    stable_confidence = mean_confidence * consensus

    return best_color, float(stable_confidence)


# ============================================================
# LICENSE PLATE TEXT CLEANING
# ============================================================

def clean_plate_text(text):

    text = str(text).upper()

    text = re.sub(r"[^A-Z0-9]", "", text)

    if len(text) < 5 or len(text) > 10:
        return ""

    if not any(char.isdigit() for char in text):
        return ""

    if not any(char.isalpha() for char in text):
        return ""

    return text


# ============================================================
# LICENSE PLATE OCR
# ============================================================

def recognize_plate(plate_crop, reader):

    if plate_crop is None or plate_crop.size == 0:
        return "", 0.0

    height, width = plate_crop.shape[:2]

    if height < 8 or width < 15:
        return "", 0.0

    gray = cv2.cvtColor(
        plate_crop,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(
        enhanced,
        -1,
        sharpen_kernel
    )

    _, threshold = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants = [
        enhanced,
        sharpened,
        threshold
    ]

    candidates = []

    for image in variants:

        try:

            results = reader.readtext(
                image,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            for result in results:

                text = clean_plate_text(result[1])
                confidence = float(result[2])

                if text and confidence >= MIN_OCR_CONFIDENCE:
                    candidates.append((text, confidence))

        except Exception:
            continue

    if not candidates:
        return "", 0.0

    # Combine repeated readings of the same text.
    grouped = defaultdict(list)

    for text, confidence in candidates:
        grouped[text].append(confidence)

    ranked = []

    for text, values in grouped.items():

        ranked.append((
            text,
            float(np.mean(values)),
            len(values)
        ))

    ranked.sort(
        key=lambda item: (
            item[1],
            item[2]
        ),
        reverse=True
    )

    return ranked[0][0], ranked[0][1]


# ============================================================
# MATCH PLATE WITH VEHICLE
# ============================================================

def match_plate_to_vehicle(plate_box, vehicle_boxes):

    px1, py1, px2, py2 = plate_box

    plate_center_x = (px1 + px2) / 2
    plate_center_y = (py1 + py2) / 2

    best_track_id = None
    best_score = float("inf")

    for vehicle in vehicle_boxes:

        vx1, vy1, vx2, vy2 = vehicle["box"]

        if not (
            vx1 <= plate_center_x <= vx2
            and vy1 <= plate_center_y <= vy2
        ):
            continue

        vehicle_width = vx2 - vx1
        vehicle_height = vy2 - vy1

        if vehicle_width <= 0 or vehicle_height <= 0:
            continue

        relative_y = (
            (plate_center_y - vy1) / vehicle_height
        )

        width_ratio = (
            (px2 - px1) / vehicle_width
        )

        if relative_y < 0.35 or relative_y > 0.98:
            continue

        if width_ratio < 0.08 or width_ratio > 0.90:
            continue

        score = abs(relative_y - 0.75)

        if score < best_score:
            best_score = score
            best_track_id = vehicle["track_id"]

    return best_track_id


# ============================================================
# GET STABLE PLATE READING
# ============================================================

def get_stable_plate(plate_history):

    if not plate_history:
        return "", 0, 0.0

    # Keep only recent OCR observations.
    recent_history = plate_history[-PLATE_HISTORY_SIZE:]

    valid_records = [
        record
        for record in recent_history
        if (
            record["text"]
            and record["confidence"] >= MIN_OCR_CONFIDENCE
        )
    ]

    if not valid_records:
        return "", 0, 0.0

    # Group exact readings first.
    groups = []

    for record in valid_records:

        text = record["text"]

        matched_group = None

        for group in groups:

            representative = group[0]["text"]

            similarity = difflib.SequenceMatcher(
                None,
                text,
                representative
            ).ratio()

            if similarity >= 0.85:
                matched_group = group
                break

        if matched_group is None:
            groups.append([record])
        else:
            matched_group.append(record)

    ranked_groups = []

    for group in groups:

        # Count distinct frames, not repeated OCR variants
        # from the same frame.
        frame_records = {}

        for record in group:

            frame = record["frame"]

            if (
                frame not in frame_records
                or record["confidence"] >
                frame_records[frame]["confidence"]
            ):
                frame_records[frame] = record

        distinct_records = list(
            frame_records.values()
        )

        if not distinct_records:
            continue

        confidence = float(np.mean([
            record["confidence"]
            for record in distinct_records
        ]))

        # Select the best-supported text.
        text_scores = defaultdict(float)

        for record in distinct_records:
            text_scores[record["text"]] += record["confidence"]

        best_text = max(
            text_scores,
            key=text_scores.get
        )

        ranked_groups.append((
            best_text,
            len(distinct_records),
            confidence
        ))

    if not ranked_groups:
        return "", 0, 0.0

    ranked_groups.sort(
        key=lambda item: (
            item[1],
            item[2]
        ),
        reverse=True
    )

    best_text, votes, mean_confidence = ranked_groups[0]

    return best_text, votes, mean_confidence


# ============================================================
# LOAD SPEED CALIBRATION
# ============================================================

def load_speed_calibration():

    if not os.path.exists(SPEED_CALIBRATION_PATH):

        print("Speed calibration file not found.")

        return None, "unavailable"

    try:

        with open(
            SPEED_CALIBRATION_PATH,
            "r"
        ) as file:

            calibration = json.load(file)

        image_points = np.array(
            calibration["image_points"],
            dtype=np.float32
        )

        world_points = np.array(
            calibration["world_points_metres"],
            dtype=np.float32
        )

        homography, _ = cv2.findHomography(
            image_points,
            world_points
        )

        if homography is None:
            return None, "unavailable"

        status = calibration.get(
            "calibration_status",
            "approximate_unverified"
        )

        return homography, status

    except Exception as error:

        print("Calibration error:", error)

        return None, "unavailable"


# ============================================================
# SPEED ESTIMATION
# ============================================================

def estimate_speed(
    track_id,
    box,
    frame_number,
    fps,
    homography,
    position_history,
    speed_history
):

    if homography is None:
        return None

    x1, y1, x2, y2 = box

    bottom_center = np.array(
        [[[
            (x1 + x2) / 2,
            y2
        ]]],
        dtype=np.float32
    )

    transformed = cv2.perspectiveTransform(
        bottom_center,
        homography
    )

    world_x, world_y = transformed[0][0]

    history = position_history[track_id]

    history.append((
        frame_number,
        float(world_x),
        float(world_y)
    ))

    if len(history) < 2:
        return None

    old_frame, old_x, old_y = history[0]
    new_frame, new_x, new_y = history[-1]

    frame_difference = new_frame - old_frame

    if frame_difference <= 0:
        return None

    time_difference = frame_difference / fps

    distance = np.sqrt(
        (new_x - old_x) ** 2
        +
        (new_y - old_y) ** 2
    )

    speed_kmh = (
        distance / time_difference
    ) * 3.6

    if not np.isfinite(speed_kmh):
        return None

    if speed_kmh > MAX_REASONABLE_SPEED:
        return None

    speed_history[track_id].append(
        float(speed_kmh)
    )

    return float(
        np.median(speed_history[track_id])
    )


# ============================================================
# GROUND-TRUTH ACCURACY EVALUATION
# ============================================================

def evaluate_ground_truth(summary_df):

    if not os.path.exists(GROUND_TRUTH_CSV):

        print("\nGround-truth validation skipped.")
        print("No manually verified labels were found.")

        print(
            "Create this file when verified labels are available:"
        )
        print(GROUND_TRUTH_CSV)

        print(
            "Required columns: track_id,true_color,true_plate"
        )

        return

    try:

        truth_df = pd.read_csv(GROUND_TRUTH_CSV)

        required_columns = {
            "track_id",
            "true_color",
            "true_plate"
        }

        if not required_columns.issubset(truth_df.columns):

            print("\nGround-truth CSV has missing columns.")
            print("Required:", required_columns)

            return

        truth_df["track_id"] = pd.to_numeric(
            truth_df["track_id"],
            errors="coerce"
        )

        truth_df = truth_df.dropna(
            subset=["track_id"]
        )

        truth_df["track_id"] = (
            truth_df["track_id"].astype(int)
        )

        merged = summary_df.merge(
            truth_df,
            on="track_id",
            how="inner"
        )

        if merged.empty:

            print("\nNo matching track IDs for validation.")

            return

        # Color accuracy.
        color_rows = merged[
            merged["true_color"].notna()
            & (merged["true_color"].astype(str).str.strip() != "")
            & (merged["dominant_color"] != "Unknown")
        ].copy()

        if len(color_rows) > 0:

            color_correct = (
                color_rows["dominant_color"].str.upper()
                ==
                color_rows["true_color"].astype(str).str.strip().str.upper()
            )

            color_accuracy = (
                float(color_correct.mean()) * 100.0
            )

        else:
            color_accuracy = None

        # Exact plate accuracy.
        plate_rows = merged[
            merged["true_plate"].notna()
            & (merged["true_plate"].astype(str).str.strip() != "")
            & (merged["plate_text"].astype(str).str.strip() != "")
        ].copy()

        if len(plate_rows) > 0:

            true_plates = plate_rows["true_plate"].apply(
                clean_plate_text
            )

            predicted_plates = plate_rows["plate_text"].apply(
                clean_plate_text
            )

            plate_correct = (
                true_plates == predicted_plates
            )

            plate_accuracy = (
                float(plate_correct.mean()) * 100.0
            )

        else:
            plate_accuracy = None

        validation_records = [
            {
                "metric": "Color exact accuracy",
                "accuracy_percent": color_accuracy,
                "samples": len(color_rows)
            },
            {
                "metric": "License plate exact accuracy",
                "accuracy_percent": plate_accuracy,
                "samples": len(plate_rows)
            }
        ]

        validation_df = pd.DataFrame(
            validation_records
        )

        validation_df.to_csv(
            VALIDATION_CSV,
            index=False
        )

        print("\nGROUND-TRUTH VALIDATION")
        print("-----------------------")

        if color_accuracy is not None:

            print(
                f"Color accuracy: {color_accuracy:.2f}% "
                f"({len(color_rows)} labeled vehicles)"
            )

        else:
            print("Color accuracy: Not available")

        if plate_accuracy is not None:

            print(
                f"Plate exact accuracy: {plate_accuracy:.2f}% "
                f"({len(plate_rows)} labeled vehicles)"
            )

        else:
            print("Plate exact accuracy: Not available")

        print("Validation report:", VALIDATION_CSV)

    except Exception as error:

        print("Ground-truth evaluation error:", error)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    start_time = time.time()

    print("\nLoading vehicle model...\n")

    vehicle_model = YOLO(
        VEHICLE_MODEL_PATH
    )

    print("Loading license plate model...\n")

    plate_model = YOLO(
        PLATE_MODEL_PATH
    )

    print("Initializing EasyOCR...\n")

    reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    print("Loading speed calibration...\n")

    homography, calibration_status = (
        load_speed_calibration()
    )

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():

        raise RuntimeError(
            "Unable to open input video: " + INPUT_VIDEO
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        fps = 30.0

    print("\nVIDEO INFORMATION")
    print("-------------------------")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps}")
    print(f"Frames: {total_frames}")
    print(f"Duration: {total_frames / fps:.1f} seconds")
    print(f"Calibration: {calibration_status}")

    writer = imageio_ffmpeg.write_frames(
        OUTPUT_VIDEO,
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1
    )

    writer.send(None)

    tracking_records = []
    plate_records = []
    speed_records = []

    vehicle_data = defaultdict(
        lambda: {
            "vehicle_type": "Unknown",
            "colors": [],
            "color_confidences": [],
            "plates": [],
            "plate_relative_box": None,
            "last_plate_frame": None,
            "confidences": [],
            "first_frame": None,
            "last_frame": None,
            "observations": 0,
            "speeds": []
        }
    )

    position_history = defaultdict(
        lambda: deque(maxlen=15)
    )

    speed_history = defaultdict(
        lambda: deque(maxlen=5)
    )

    frame_number = 0
    total_vehicle_observations = 0

    print("\nStarting improved processing pipeline...\n")

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1
            timestamp = frame_number / fps

            # ------------------------------------------------
            # VEHICLE DETECTION AND TRACKING
            # ------------------------------------------------

            results = vehicle_model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=list(VEHICLE_CLASSES.keys()),
                conf=VEHICLE_CONFIDENCE,
                imgsz=640,
                device=DEVICE,
                verbose=False
            )

            vehicle_boxes = []

            if results and results[0].boxes is not None:

                boxes = results[0].boxes

                if boxes.id is not None:

                    xyxy = boxes.xyxy.cpu().numpy()

                    class_ids = (
                        boxes.cls.cpu().numpy().astype(int)
                    )

                    confidences = (
                        boxes.conf.cpu().numpy()
                    )

                    track_ids = (
                        boxes.id.cpu().numpy().astype(int)
                    )

                    for box, class_id, confidence, track_id in zip(
                        xyxy,
                        class_ids,
                        confidences,
                        track_ids
                    ):

                        x1, y1, x2, y2 = map(int, box)

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)

                        if x2 <= x1 or y2 <= y1:
                            continue

                        vehicle_type = VEHICLE_CLASSES.get(
                            class_id,
                            "Unknown"
                        )

                        vehicle_crop = frame[
                            y1:y2,
                            x1:x2
                        ]

                        raw_color, raw_color_confidence = (
                            classify_color(vehicle_crop)
                        )

                        data = vehicle_data[int(track_id)]

                        data["vehicle_type"] = vehicle_type

                        data["colors"].append(raw_color)

                        data["color_confidences"].append(
                            raw_color_confidence
                        )

                        data["confidences"].append(
                            float(confidence)
                        )

                        if data["first_frame"] is None:
                            data["first_frame"] = frame_number

                        data["last_frame"] = frame_number
                        data["observations"] += 1

                        stable_color, stable_color_confidence = (
                            get_stable_color(
                                data["colors"],
                                data["color_confidences"]
                            )
                        )

                        vehicle_boxes.append({
                            "box": (x1, y1, x2, y2),
                            "track_id": int(track_id),
                            "vehicle_type": vehicle_type,
                            "confidence": float(confidence),
                            "raw_color": raw_color,
                            "raw_color_confidence": raw_color_confidence,
                            "color": stable_color,
                            "color_confidence": stable_color_confidence
                        })

                        total_vehicle_observations += 1

                        tracking_records.append({
                            "frame": frame_number,
                            "timestamp_seconds": timestamp,
                            "track_id": int(track_id),
                            "vehicle_type": vehicle_type,
                            "raw_color": raw_color,
                            "raw_color_confidence": raw_color_confidence,
                            "color": stable_color,
                            "color_confidence": stable_color_confidence,
                            "confidence": float(confidence),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        })

            # ------------------------------------------------
            # LICENSE PLATE DETECTION AND OCR
            # ------------------------------------------------

            if frame_number % PLATE_DETECTION_INTERVAL == 0:

                plate_results = plate_model.predict(
                    frame,
                    conf=PLATE_CONFIDENCE,
                    imgsz=PLATE_IMAGE_SIZE,
                    device=DEVICE,
                    verbose=False
                )

                if (
                    plate_results
                    and plate_results[0].boxes is not None
                ):

                    plate_boxes = (
                        plate_results[0].boxes.xyxy.cpu().numpy()
                    )

                    plate_confidences = (
                        plate_results[0].boxes.conf.cpu().numpy()
                    )

                    for plate_box, plate_confidence in zip(
                        plate_boxes,
                        plate_confidences
                    ):

                        px1, py1, px2, py2 = map(
                            int,
                            plate_box
                        )

                        px1 = max(0, px1)
                        py1 = max(0, py1)
                        px2 = min(width, px2)
                        py2 = min(height, py2)

                        if px2 <= px1 or py2 <= py1:
                            continue

                        track_id = match_plate_to_vehicle(
                            (px1, py1, px2, py2),
                            vehicle_boxes
                        )

                        if track_id is None:
                            continue

                        matched_vehicle = next(
                            (
                                vehicle
                                for vehicle in vehicle_boxes
                                if vehicle["track_id"] == track_id
                            ),
                            None
                        )

                        if matched_vehicle is None:
                            continue

                        # Save the plate box relative to its vehicle.
                        vx1, vy1, vx2, vy2 = matched_vehicle["box"]

                        vehicle_width = max(1, vx2 - vx1)
                        vehicle_height = max(1, vy2 - vy1)

                        vehicle_info = vehicle_data[track_id]

                        vehicle_info["plate_relative_box"] = (
                            (px1 - vx1) / vehicle_width,
                            (py1 - vy1) / vehicle_height,
                            (px2 - vx1) / vehicle_width,
                            (py2 - vy1) / vehicle_height
                        )

                        vehicle_info["last_plate_frame"] = frame_number

                        plate_crop = frame[
                            py1:py2,
                            px1:px2
                        ]

                        plate_text, ocr_confidence = recognize_plate(
                            plate_crop,
                            reader
                        )

                        if (
                            not plate_text
                            or ocr_confidence < MIN_OCR_CONFIDENCE
                        ):
                            continue

                        plate_record = {
                            "frame": frame_number,
                            "text": plate_text,
                            "confidence": float(ocr_confidence)
                        }

                        vehicle_info["plates"].append(
                            plate_record
                        )

                        vehicle_type = matched_vehicle["vehicle_type"]

                        plate_records.append({
                            "frame": frame_number,
                            "timestamp_seconds": timestamp,
                            "track_id": track_id,
                            "vehicle_type": vehicle_type,
                            "plate_text": plate_text,
                            "ocr_confidence": float(ocr_confidence),
                            "plate_confidence": float(plate_confidence),
                            "x1": px1,
                            "y1": py1,
                            "x2": px2,
                            "y2": py2
                        })

            # ------------------------------------------------
            # VEHICLE SPEED ESTIMATION AND ANNOTATION
            # ------------------------------------------------

            for vehicle in vehicle_boxes:

                x1, y1, x2, y2 = vehicle["box"]

                track_id = vehicle["track_id"]
                vehicle_type = vehicle["vehicle_type"]
                color = vehicle["color"]
                color_confidence = vehicle["color_confidence"]

                speed = estimate_speed(
                    track_id,
                    vehicle["box"],
                    frame_number,
                    fps,
                    homography,
                    position_history,
                    speed_history
                )

                if speed is not None:

                    speed_records.append({
                        "frame": frame_number,
                        "timestamp_seconds": timestamp,
                        "track_id": track_id,
                        "vehicle_type": vehicle_type,
                        "speed_kmh": speed,
                        "speed_status": calibration_status
                    })

                    vehicle_data[track_id]["speeds"].append(speed)

                stable_plate, plate_votes, plate_ocr_confidence = (
                    get_stable_plate(
                        vehicle_data[track_id]["plates"]
                    )
                )

                # Vehicle bounding box.
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Vehicle label.
                if color == "Unknown":
                    color_label = "Color: Unknown"
                else:
                    color_label = (
                        f"{color} {color_confidence:.0f}%"
                    )

                label = (
                    f"ID:{track_id} "
                    f"{vehicle_type} "
                    f"{color_label}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(25, y1 - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                # Persistent plate display.
                vehicle_info = vehicle_data[track_id]

                plate_box = vehicle_info["plate_relative_box"]
                last_plate_frame = vehicle_info["last_plate_frame"]

                plate_is_recent = (
                    plate_box is not None
                    and last_plate_frame is not None
                    and frame_number - last_plate_frame
                    <= PLATE_PERSISTENCE_FRAMES
                )

                if plate_is_recent:

                    rx1, ry1, rx2, ry2 = plate_box

                    p1 = (
                        max(
                            0,
                            min(
                                width - 1,
                                int(x1 + rx1 * (x2 - x1))
                            )
                        ),
                        max(
                            0,
                            min(
                                height - 1,
                                int(y1 + ry1 * (y2 - y1))
                            )
                        )
                    )

                    p2 = (
                        max(
                            0,
                            min(
                                width - 1,
                                int(x1 + rx2 * (x2 - x1))
                            )
                        ),
                        max(
                            0,
                            min(
                                height - 1,
                                int(y1 + ry2 * (y2 - y1))
                            )
                        )
                    )

                    if p2[0] > p1[0] and p2[1] > p1[1]:

                        cv2.rectangle(
                            frame,
                            p1,
                            p2,
                            (0, 255, 255),
                            2
                        )

                    if stable_plate and plate_votes >= MIN_PLATE_VOTES:

                        plate_label = (
                            f"Plate: {stable_plate} "
                            f"OCR:{plate_ocr_confidence * 100:.0f}%"
                        )

                    else:

                        plate_label = (
                            f"Plate: Reading "
                            f"({plate_votes}/{MIN_PLATE_VOTES})"
                        )

                    cv2.putText(
                        frame,
                        plate_label,
                        (x1, min(height - 10, y1 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2
                    )

                # Approximate speed information.
                if speed is not None:

                    speed_label = (
                        f"Speed: {speed:.1f} km/h *"
                    )

                    cv2.putText(
                        frame,
                        speed_label,
                        (x1, min(height - 10, y1 + 50)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2
                    )

            # ------------------------------------------------
            # FRAME INFORMATION
            # ------------------------------------------------

            vehicle_count = len(vehicle_boxes)

            cv2.putText(
                frame,
                f"Frame: {frame_number}/{total_frames} "
                f"Vehicles: {vehicle_count}",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Speed estimates are approximate and unverified (*)",
                (25, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2
            )

            writer.send(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )
            )

            if frame_number % 100 == 0:

                print(
                    f"Processed {frame_number}/{total_frames} "
                    f"| Vehicles: {total_vehicle_observations} "
                    f"| Plate readings: {len(plate_records)}"
                )

    finally:

        cap.release()
        writer.close()

    # ========================================================
    # SAVE VEHICLE TRACKING CSV
    # ========================================================

    tracking_df = pd.DataFrame(
        tracking_records
    )

    tracking_df.to_csv(
        TRACKING_CSV,
        index=False
    )

    # ========================================================
    # SAVE LICENSE PLATE CSV
    # ========================================================

    plate_columns = [
        "frame",
        "timestamp_seconds",
        "track_id",
        "vehicle_type",
        "plate_text",
        "ocr_confidence",
        "plate_confidence",
        "x1",
        "y1",
        "x2",
        "y2"
    ]

    plate_df = pd.DataFrame(
        plate_records,
        columns=plate_columns
    )

    plate_df.to_csv(
        PLATE_CSV,
        index=False
    )

    # ========================================================
    # SAVE SPEED CSV
    # ========================================================

    speed_columns = [
        "frame",
        "timestamp_seconds",
        "track_id",
        "vehicle_type",
        "speed_kmh",
        "speed_status"
    ]

    speed_df = pd.DataFrame(
        speed_records,
        columns=speed_columns
    )

    speed_df.to_csv(
        SPEED_CSV,
        index=False
    )

    # ========================================================
    # GENERATE FINAL VEHICLE SUMMARY
    # ========================================================

    summary_records = []

    for track_id, data in vehicle_data.items():

        stable_color, stable_color_confidence = get_stable_color(
            data["colors"],
            data["color_confidences"]
        )

        valid_colors = [
            color
            for color in data["colors"]
            if color != "Unknown"
        ]

        stable_plate, plate_votes, plate_ocr_confidence = (
            get_stable_plate(
                data["plates"]
            )
        )

        plate_readings = len(data["plates"])

        plate_consensus_reached = (
            plate_votes >= MIN_PLATE_VOTES
        )

        speeds = data["speeds"]

        average_speed = (
            float(np.mean(speeds))
            if speeds
            else None
        )

        summary_records.append({

            "track_id": track_id,

            "vehicle_type": data["vehicle_type"],

            "dominant_color": stable_color,

            "color_confidence_percent": stable_color_confidence,

            "color_observations": len(valid_colors),

            "plate_text": (
                stable_plate
                if plate_consensus_reached
                else ""
            ),

            "plate_votes": plate_votes,

            "plate_readings": plate_readings,

            "plate_ocr_confidence_percent": (
                plate_ocr_confidence * 100.0
            ),

            "plate_consensus_reached": plate_consensus_reached,

            "first_frame": data["first_frame"],

            "last_frame": data["last_frame"],

            "observations": data["observations"],

            "average_confidence": (
                float(np.mean(data["confidences"]))
                if data["confidences"]
                else 0.0
            ),

            "average_speed_kmh": average_speed,

            "speed_status": calibration_status

        })

    summary_df = pd.DataFrame(
        summary_records
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False
    )

    # ========================================================
    # OPTIONAL GROUND-TRUTH ACCURACY EVALUATION
    # ========================================================

    evaluate_ground_truth(summary_df)

    # ========================================================
    # FINAL PROJECT SUMMARY
    # ========================================================

    processing_time = time.time() - start_time

    unique_vehicles = len(vehicle_data)

    vehicle_type_counts = Counter(
        data["vehicle_type"]
        for data in vehicle_data.values()
    )

    color_counts = Counter(
        record["dominant_color"]
        for record in summary_records
    )

    confirmed_candidates = sum(
        record["plate_consensus_reached"]
        for record in summary_records
    )

    vehicles_with_speed = sum(
        bool(data["speeds"])
        for data in vehicle_data.values()
    )

    all_speeds = [
        speed
        for data in vehicle_data.values()
        for speed in data["speeds"]
    ]

    print("\n")
    print("=" * 55)
    print("VEHICLE RECOGNITION AND TRACKING COMPLETED")
    print("=" * 55)

    print(f"\nTotal frames processed: {frame_number}")

    print(
        f"Vehicle observations: {total_vehicle_observations}"
    )

    print(
        f"Unique tracked vehicles: {unique_vehicles}"
    )

    print("\nVehicle type distribution:")

    for vehicle_type, count in vehicle_type_counts.items():
        print(f"{vehicle_type}: {count}")

    print("\nVehicle color distribution:")

    for color, count in color_counts.items():
        print(f"{color}: {count}")

    print("\nLicense plate recognition:")

    print(
        f"Usable OCR readings: {len(plate_records)}"
    )

    print(
        f"Vehicles with plate consensus: {confirmed_candidates}"
    )

    print(
        "OCR confidence and consensus are not accuracy."
    )

    print("\nSpeed estimation:")

    print(
        f"Speed observations: {len(speed_records)}"
    )

    print(
        f"Vehicles with speed estimates: {vehicles_with_speed}"
    )

    if all_speeds:

        print(
            f"Mean estimated speed: {np.mean(all_speeds):.2f} km/h"
        )

    print(
        f"Calibration status: {calibration_status}"
    )

    print(
        "Speed values are approximate and unverified."
    )

    print("\nOutput files:")

    print(OUTPUT_VIDEO)
    print(TRACKING_CSV)
    print(PLATE_CSV)
    print(SUMMARY_CSV)
    print(SPEED_CSV)

    print(
        f"\nProcessing time: {processing_time / 60:.2f} minutes"
    )

    print("\nPROJECT PROCESSING COMPLETED")
    print("=" * 55)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
