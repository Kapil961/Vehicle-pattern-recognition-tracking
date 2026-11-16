import os
import json
import pandas as pd

REPORT_DIR = "outputs/reports"

# Latest improved pipeline outputs
VEHICLE_CSV = f"{REPORT_DIR}/vehicle_tracking_improved.csv"
SUMMARY_CSV = f"{REPORT_DIR}/final_vehicle_summary_improved.csv"
PLATE_CSV = f"{REPORT_DIR}/license_plate_detections_improved.csv"
SPEED_CSV = f"{REPORT_DIR}/vehicle_speed_tracking_improved.csv"

OUTPUT_FILE = f"{REPORT_DIR}/project_evaluation_report.txt"
JSON_OUTPUT = f"{REPORT_DIR}/project_evaluation.json"


def main():
    print("Loading latest project results...")

    required_files = [
        VEHICLE_CSV,
        SUMMARY_CSV,
        PLATE_CSV,
        SPEED_CSV
    ]

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    vehicles = pd.read_csv(VEHICLE_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    plates = pd.read_csv(PLATE_CSV)
    speeds = pd.read_csv(SPEED_CSV)

    # 1. Video processing statistics
    total_frames = int(vehicles["frame"].nunique())
    total_observations = int(len(vehicles))
    total_vehicles = int(summary["track_id"].nunique())

    # 2. Vehicle type distribution
    vehicle_distribution = (
        summary.groupby("vehicle_type")["track_id"]
        .nunique()
        .to_dict()
    )

    # 3. Color distribution
    color_distribution = (
        summary["dominant_color"]
        .fillna("Unknown")
        .value_counts()
        .to_dict()
    )

    # 4. License plate statistics
    total_plate_readings = int(len(plates))

    vehicles_with_plate_readings = int(
        plates["track_id"].nunique()
    )

    vehicles_with_plate_consensus = int(
        summary["plate_consensus_reached"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    # 5. Speed statistics
    valid_speeds = speeds["speed_kmh"].dropna()

    vehicles_with_speed = int(
        speeds.loc[
            speeds["speed_kmh"].notna(),
            "track_id"
        ].nunique()
    )

    speed_statuses = (
        speeds["speed_status"]
        .dropna()
        .value_counts()
        .to_dict()
    )

    speed_results = {
        "speed_observations": int(len(valid_speeds)),
        "vehicles_with_speed_estimates": vehicles_with_speed,
        "mean_speed_kmh": (
            round(float(valid_speeds.mean()), 2)
            if len(valid_speeds) else None
        ),
        "median_speed_kmh": (
            round(float(valid_speeds.median()), 2)
            if len(valid_speeds) else None
        ),
        "maximum_speed_kmh": (
            round(float(valid_speeds.max()), 2)
            if len(valid_speeds) else None
        ),
        "calibration_status": (
            ", ".join(map(str, speed_statuses.keys()))
            if speed_statuses else "not_available"
        ),
        "reliability": "approximate_unverified"
    }

    # 6. Machine-readable evaluation
    results = {
        "project": "Vehicle Pattern Recognition and Tracking",
        "video_processing": {
            "total_frames_processed": total_frames,
            "vehicle_observations": total_observations
        },
        "vehicle_detection_and_tracking": {
            "unique_tracked_vehicles": total_vehicles,
            "vehicle_type_distribution": vehicle_distribution
        },
        "vehicle_color_classification": {
            "color_distribution": color_distribution,
            "method": "OpenCV-based heuristic",
            "ground_truth_validated": False
        },
        "license_plate_recognition": {
            "usable_ocr_readings": total_plate_readings,
            "vehicles_with_plate_readings": vehicles_with_plate_readings,
            "vehicles_with_plate_consensus": vehicles_with_plate_consensus,
            "ground_truth_validated": False
        },
        "speed_estimation": speed_results,
        "model_information": {
            "vehicle_detector": "Pretrained YOLOv8n (COCO weights)",
            "tracker": "ByteTrack",
            "license_plate_detector": "Pretrained YOLO license plate model",
            "ocr": "EasyOCR",
            "vehicle_color": "OpenCV heuristic",
            "vehicle_specific_fine_tuning": False,
            "ua_detrac_used_for_training": False
        },
        "evaluation_limitations": [
            "No ground-truth vehicle annotations for this test video.",
            "Detection precision, recall and mAP were not measured.",
            "Tracking ID accuracy was not ground-truth validated.",
            "License plate OCR accuracy was not ground-truth validated.",
            "Color predictions require further validation.",
            "Speed estimates require measured road calibration."
        ]
    }

    with open(JSON_OUTPUT, "w") as file:
        json.dump(results, file, indent=4)

    # 7. Human-readable report
    lines = [
        "VEHICLE PATTERN RECOGNITION AND TRACKING",
        "PROJECT EVALUATION REPORT",
        "=" * 60,
        "",
        "1. VIDEO PROCESSING",
        f"Total frames processed: {total_frames}",
        f"Vehicle observations: {total_observations}",
        "",
        "2. VEHICLE DETECTION AND TRACKING",
        f"Unique tracked vehicles: {total_vehicles}",
        "Vehicle type distribution:"
    ]

    for vehicle_type, count in vehicle_distribution.items():
        lines.append(f"  {vehicle_type}: {count}")

    lines += [
        "",
        "3. VEHICLE COLOR CLASSIFICATION",
        "Color distribution:"
    ]

    for color, count in color_distribution.items():
        lines.append(f"  {color}: {count}")

    lines += [
        "",
        "Method: OpenCV-based color heuristic.",
        "Color classification has not been ground-truth validated.",
        "",
        "4. LICENSE PLATE DETECTION AND OCR",
        f"Usable OCR readings: {total_plate_readings}",
        f"Vehicles with OCR readings: {vehicles_with_plate_readings}",
        f"Vehicles with plate consensus: {vehicles_with_plate_consensus}",
        "",
        "Note: OCR consensus does not establish that a plate is correct.",
        "Plate recognition accuracy has not been ground-truth validated.",
        "",
        "5. SPEED ESTIMATION",
        f"Speed observations: {speed_results['speed_observations']}",
        f"Vehicles with speed estimates: {vehicles_with_speed}",
        f"Mean estimated speed: {speed_results['mean_speed_kmh']} km/h",
        f"Median estimated speed: {speed_results['median_speed_kmh']} km/h",
        f"Maximum estimated speed: {speed_results['maximum_speed_kmh']} km/h",
        f"Calibration status: {speed_results['calibration_status']}",
        "Reliability: Approximate and unverified.",
        "Calibration uses assumed road dimensions, not measured ground distances.",
        "",
        "6. MODEL AND DATASET INFORMATION",
        "Vehicle detector: Pretrained YOLOv8n (COCO weights).",
        "Multi-object tracker: ByteTrack.",
        "License plate detector: Pretrained YOLO license plate model.",
        "OCR engine: EasyOCR.",
        "Color recognition: OpenCV-based heuristic.",
        "Vehicle-specific model fine-tuning: Not performed.",
        "UA-DETRAC: Not used to train this pipeline.",
        "",
        "7. EVALUATION LIMITATIONS",
        "- No ground-truth vehicle annotations for this test video.",
        "- Detection precision, recall and mAP were not measured.",
        "- Tracking ID accuracy was not ground-truth validated.",
        "- License plate OCR accuracy was not ground-truth validated.",
        "- Color predictions require further validation.",
        "- Speed estimates require measured road calibration.",
        "",
        "8. OVERALL EVALUATION",
        "The pipeline successfully processed the complete test video.",
        "Vehicle detection, tracking, plate OCR, color estimation and",
        "approximate speed estimation produced output results.",
        "These results demonstrate pipeline execution, not verified",
        "real-world accuracy.",
        "",
        "END OF REPORT"
    ]

    with open(OUTPUT_FILE, "w") as file:
        file.write("\n".join(lines))

    print("\nPROJECT EVALUATION COMPLETED")
    print("=" * 60)
    print("\n".join(lines))

    print("\nSaved reports:")
    print(OUTPUT_FILE)
    print(JSON_OUTPUT)


if __name__ == "__main__":
    main()
