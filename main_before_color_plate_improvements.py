
import os
import re
import cv2
import json
import time
import torch
import easyocr
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

MIN_PLATE_VOTES = 3
MIN_COLOR_OBSERVATIONS = 3

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
        return "Unknown"

    height, width = vehicle_crop.shape[:2]

    if height < 10 or width < 10:
        return "Unknown"

    x1 = int(width * 0.18)
    x2 = int(width * 0.82)

    y1 = int(height * 0.38)
    y2 = int(height * 0.88)

    crop = vehicle_crop[y1:y2, x1:x2]

    if crop.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    mean_saturation = np.mean(s)
    mean_value = np.mean(v)
    mean_hue = np.mean(h)

    if mean_value < 45:
        return "Black"

    if mean_saturation < 35:

        if mean_value > 190:
            return "White"

        if mean_value > 90:
            return "Gray"

        return "Black"

    if mean_hue < 10 or mean_hue > 170:
        return "Red"

    if mean_hue < 22:
        return "Orange"

    if mean_hue < 35:
        return "Yellow"

    if mean_hue < 85:
        return "Green"

    if mean_hue < 130:
        return "Blue"

    if mean_hue < 160:
        return "Purple"

    return "Unknown"


# ============================================================
# LICENSE PLATE TEXT CLEANING
# ============================================================

def clean_plate_text(text):

    text = text.upper()

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

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

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

                if text:
                    candidates.append((text, confidence))

        except Exception:
            continue

    if not candidates:
        return "", 0.0

    candidates.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return candidates[0]


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

    grouped = defaultdict(list)

    for record in plate_history:

        text = record["text"]

        if text:
            grouped[text].append(record)

    if not grouped:
        return "", 0

    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            len(item[1]),
            np.mean([
                record["confidence"]
                for record in item[1]
            ])
        ),
        reverse=True
    )

    best_text, records = ranked[0]

    distinct_frames = {
        record["frame"]
        for record in records
    }

    return best_text, len(distinct_frames)


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

    print("VIDEO INFORMATION")
    print("-------------------------")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps}")
    print(f"Frames: {total_frames}")
    print(f"Duration: {total_frames / fps:.1f} seconds")
    print(f"Calibration: {calibration_status}")

    # Initialize video writer using bundled FFmpeg.

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
            "plates": [],
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

                        x1, y1, x2, y2 = map(
                            int,
                            box
                        )

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

                        color = classify_color(
                            vehicle_crop
                        )

                        vehicle_boxes.append({
                            "box": (x1, y1, x2, y2),
                            "track_id": int(track_id),
                            "vehicle_type": vehicle_type,
                            "confidence": float(confidence),
                            "color": color
                        })

                        total_vehicle_observations += 1

                        tracking_records.append({
                            "frame": frame_number,
                            "timestamp_seconds": timestamp,
                            "track_id": int(track_id),
                            "vehicle_type": vehicle_type,
                            "color": color,
                            "confidence": float(confidence),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        })

                        data = vehicle_data[track_id]

                        data["vehicle_type"] = vehicle_type
                        data["colors"].append(color)
                        data["confidences"].append(
                            float(confidence)
                        )

                        if data["first_frame"] is None:
                            data["first_frame"] = frame_number

                        data["last_frame"] = frame_number

                        data["observations"] += 1

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

                if plate_results and plate_results[0].boxes is not None:

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

                        plate_crop = frame[
                            py1:py2,
                            px1:px2
                        ]

                        plate_text, ocr_confidence = (
                            recognize_plate(
                                plate_crop,
                                reader
                            )
                        )

                        track_id = match_plate_to_vehicle(
                            (px1, py1, px2, py2),
                            vehicle_boxes
                        )

                        if not plate_text or track_id is None:
                            continue

                        vehicle_type = next(
                            (
                                vehicle["vehicle_type"]
                                for vehicle in vehicle_boxes
                                if vehicle["track_id"] == track_id
                            ),
                            "Unknown"
                        )

                        plate_records.append({
                            "frame": frame_number,
                            "timestamp_seconds": timestamp,
                            "track_id": track_id,
                            "vehicle_type": vehicle_type,
                            "plate_text": plate_text,
                            "ocr_confidence": ocr_confidence,
                            "plate_confidence": float(plate_confidence),
                            "x1": px1,
                            "y1": py1,
                            "x2": px2,
                            "y2": py2
                        })

                        vehicle_data[track_id]["plates"].append({
                            "frame": frame_number,
                            "text": plate_text,
                            "confidence": ocr_confidence
                        })

                        cv2.rectangle(
                            frame,
                            (px1, py1),
                            (px2, py2),
                            (0, 255, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            plate_text,
                            (px1, max(20, py1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )

            # ------------------------------------------------
            # VEHICLE SPEED ESTIMATION AND ANNOTATION
            # ------------------------------------------------

            for vehicle in vehicle_boxes:

                x1, y1, x2, y2 = vehicle["box"]

                track_id = vehicle["track_id"]

                vehicle_type = vehicle["vehicle_type"]

                color = vehicle["color"]

                confidence = vehicle["confidence"]

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

                    vehicle_data[track_id]["speeds"].append(
                        speed
                    )

                stable_plate, plate_votes = get_stable_plate(
                    vehicle_data[track_id]["plates"]
                )

                # Vehicle bounding box

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Vehicle information

                label = (
                    f"ID:{track_id} "
                    f"{vehicle_type} "
                    f"{color}"
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

                # Plate information

                if stable_plate and plate_votes >= MIN_PLATE_VOTES:

                    plate_label = (
                        f"Plate: {stable_plate}"
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

                # Approximate speed information

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
                f"Frame: {frame_number}/{total_frames}  "
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

            # Write frame through FFmpeg.

            writer.send(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )
            )

            # Progress information.

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

    plate_df = pd.DataFrame(
        plate_records,
        columns=[
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
    )

    plate_df.to_csv(
        PLATE_CSV,
        index=False
    )

    # ========================================================
    # SAVE SPEED CSV
    # ========================================================

    speed_df = pd.DataFrame(
        speed_records,
        columns=[
            "frame",
            "timestamp_seconds",
            "track_id",
            "vehicle_type",
            "speed_kmh",
            "speed_status"
        ]
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

        colors = [
            color
            for color in data["colors"]
            if color != "Unknown"
        ]

        if colors:

            color_counts = Counter(colors)

            dominant_color, color_count = (
                color_counts.most_common(1)[0]
            )

        else:

            dominant_color = "Unknown"
            color_count = 0

        stable_plate, plate_votes = get_stable_plate(
            data["plates"]
        )

        plate_readings = len(
            data["plates"]
        )

        plate_consensus_reached = (
            plate_votes >= MIN_PLATE_VOTES
        )

        speeds = data["speeds"]

        average_speed = (
            float(np.mean(speeds))
            if speeds else None
        )

        summary_records.append({

            "track_id": track_id,

            "vehicle_type": data["vehicle_type"],

            "dominant_color": dominant_color,

            "color_observations": color_count,

            "plate_text": (
                stable_plate
                if plate_consensus_reached
                else ""
            ),

            "plate_votes": plate_votes,

            "plate_readings": plate_readings,

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
    # FINAL PROJECT SUMMARY
    # ========================================================

    processing_time = (
        time.time() - start_time
    )

    unique_vehicles = len(
        vehicle_data
    )

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

        print(
            f"{vehicle_type}: {count}"
        )

    print("\nVehicle color distribution:")

    for color, count in color_counts.items():

        print(
            f"{color}: {count}"
        )

    print("\nLicense plate recognition:")

    print(
        f"Usable OCR readings: {len(plate_records)}"
    )

    print(
        f"Vehicles with plate consensus: {confirmed_candidates}"
    )

    print(
        "Note: OCR consensus is not ground-truth verification."
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
