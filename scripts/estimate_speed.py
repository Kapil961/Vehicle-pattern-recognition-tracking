import cv2
import numpy as np
import pandas as pd
import json
import os

INPUT_CSV = "outputs/reports/vehicle_tracking.csv"
CALIBRATION_FILE = "configs/speed_calibration.json"
VIDEO_PATH = "data/raw/highway_videos/highway_traffic_h264.mp4"

FRAME_OUTPUT = "outputs/reports/vehicle_speed_tracking.csv"
SUMMARY_OUTPUT = "outputs/reports/vehicle_speed_summary.csv"

MAX_REASONABLE_SPEED = 180.0
MAX_FRAME_GAP = 5


def main():
    print("Loading vehicle tracking data...")

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(INPUT_CSV)

    if not os.path.exists(CALIBRATION_FILE):
        raise FileNotFoundError(CALIBRATION_FILE)

    df = pd.read_csv(INPUT_CSV)

    if df.empty:
        raise RuntimeError("Vehicle tracking CSV is empty.")

    print("Loading video information...")

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        raise RuntimeError("Invalid video FPS.")

    print("Video FPS:", fps)

    print("Loading calibration...")

    with open(CALIBRATION_FILE) as f:
        calibration = json.load(f)

    image_points = np.array(
        calibration["image_points"], dtype=np.float32
    )

    world_points = np.array(
        calibration["world_points_metres"], dtype=np.float32
    )

    if image_points.shape != (4, 2):
        raise ValueError("Calibration must contain exactly four image points.")

    if world_points.shape != (4, 2):
        raise ValueError("Calibration must contain exactly four world points.")

    homography = cv2.getPerspectiveTransform(
        image_points, world_points
    )

    # Calculate the bottom-centre of each vehicle bounding box.
    df["center_x"] = (df["x1"] + df["x2"]) / 2.0
    df["bottom_y"] = df["y2"]

    # Keep observations inside the calibrated road region.
    polygon = image_points.reshape((-1, 1, 2))

    inside = []

    for x, y in zip(df["center_x"], df["bottom_y"]):
        inside.append(
            cv2.pointPolygonTest(
                polygon,
                (float(x), float(y)),
                False
            ) >= 0
        )

    df = df[np.array(inside)].copy()

    if df.empty:
        raise RuntimeError(
            "No vehicle observations inside calibration region. "
            "Check the selected calibration points."
        )

    print("Observations inside calibration region:", len(df))

    # Convert image positions into approximate road-plane coordinates.
    image_positions = df[
        ["center_x", "bottom_y"]
    ].to_numpy(dtype=np.float32).reshape(-1, 1, 2)

    world_positions = cv2.perspectiveTransform(
        image_positions, homography
    ).reshape(-1, 2)

    df["world_x"] = world_positions[:, 0]
    df["world_y"] = world_positions[:, 1]

    df = df.sort_values(["track_id", "frame"]).copy()

    # Smooth trajectories to reduce bounding-box jitter.
    df["smooth_x"] = (
        df.groupby("track_id")["world_x"]
        .transform(
            lambda s: s.rolling(
                window=5,
                min_periods=3,
                center=True
            ).mean()
        )
    )

    df["smooth_y"] = (
        df.groupby("track_id")["world_y"]
        .transform(
            lambda s: s.rolling(
                window=5,
                min_periods=3,
                center=True
            ).mean()
        )
    )

    # Previous observation for each vehicle.
    grouped = df.groupby("track_id")

    df["previous_frame"] = grouped["frame"].shift(1)
    df["previous_x"] = grouped["smooth_x"].shift(1)
    df["previous_y"] = grouped["smooth_y"].shift(1)

    df["frame_gap"] = df["frame"] - df["previous_frame"]
    df["time_difference"] = df["frame_gap"] / fps

    dx = df["smooth_x"] - df["previous_x"]
    dy = df["smooth_y"] - df["previous_y"]

    distance_metres = np.sqrt(dx ** 2 + dy ** 2)

    df["speed_kmh"] = (
        distance_metres / df["time_difference"] * 3.6
    )

    # Reject invalid intervals and large tracking gaps.
    valid = (
        (df["time_difference"] > 0)
        & (df["frame_gap"] <= MAX_FRAME_GAP)
        & (df["speed_kmh"] >= 0)
        & (df["speed_kmh"] <= MAX_REASONABLE_SPEED)
        & df["speed_kmh"].notna()
    )

    df.loc[~valid, "speed_kmh"] = np.nan

    df["speed_status"] = "approximate_unverified"

    # Create output directory.
    os.makedirs("outputs/reports", exist_ok=True)

    # Save frame-level speed results.
    output_columns = [
        "frame",
        "timestamp_seconds",
        "track_id",
        "vehicle_type",
        "speed_kmh",
        "speed_status"
    ]

    speed_df = df[output_columns].copy()
    speed_df.to_csv(FRAME_OUTPUT, index=False)

    # Generate per-vehicle speed summary.
    summary = (
        df.groupby(["track_id", "vehicle_type"])
        .agg(
            valid_speed_observations=("speed_kmh", "count"),
            average_speed_kmh=("speed_kmh", "mean"),
            median_speed_kmh=("speed_kmh", "median"),
            maximum_speed_kmh=("speed_kmh", "max")
        )
        .reset_index()
    )

    summary["calibration_status"] = (
        calibration.get(
            "calibration_status",
            "unknown"
        )
    )

    summary["speed_status"] = "approximate_unverified"

    summary = summary.round(2)

    summary.to_csv(SUMMARY_OUTPUT, index=False)

    # Display results.
    valid_count = speed_df["speed_kmh"].count()

    print("\n" + "=" * 55)
    print("SPEED ESTIMATION COMPLETED")
    print("=" * 55)

    print("Video FPS:", fps)
    print("Observations inside calibration region:", len(df))
    print("Valid speed observations:", valid_count)
    print("Vehicles with speed records:", len(summary))

    print("\nVEHICLE SPEED SUMMARY")

    print(
        summary.head(20).to_string(index=False)
    )

    print("\nOUTPUT FILES")
    print("Frame-level:", FRAME_OUTPUT)
    print("Vehicle summary:", SUMMARY_OUTPUT)

    print("\nCALIBRATION WARNING")
    print("Lane width and longitudinal distance are assumed.")
    print("Speed values are approximate and unverified.")
    print("Do not treat them as ground-truth traffic speeds.")


if __name__ == "__main__":
    main()
