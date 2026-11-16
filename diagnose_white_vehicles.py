
import os
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "data/raw/highway_videos/highway_traffic_h264.mp4"
MODEL_PATH = "yolov8n-seg.pt"
FRAME_NUMBER = 150

SEGMENTS = [2, 3, 7, 11, 12, 18, 20, 23]

OUTPUT_DIR = "outputs/reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# READ VIDEO FRAME
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Cannot read frame")

h, w = frame.shape[:2]
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# --------------------------------------------------
# LOAD SEGMENTATION MODEL
# --------------------------------------------------

model = YOLO(MODEL_PATH)

result = model(
    frame,
    conf=0.25,
    imgsz=640,
    device="cpu",
    classes=[2, 3, 5, 7],
    verbose=False
)[0]

if result.masks is None:
    raise RuntimeError("No segmentation masks found")

# --------------------------------------------------
# OUTPUT COLLECTIONS
# --------------------------------------------------

tiles = []
region_tiles = []

regions = [
    ("Top 10-30%", 0.10, 0.30),
    ("Upper 20-45%", 0.20, 0.45),
    ("Middle 35-60%", 0.35, 0.60),
    ("Lower 55-80%", 0.55, 0.80),
    ("Bottom 70-90%", 0.70, 0.90),
]

# --------------------------------------------------
# PROCESS SELECTED SEGMENTS
# --------------------------------------------------

for segment_no in SEGMENTS:

    i = segment_no - 1

    if i >= len(result.masks.xy):
        print(f"\nSegment {segment_no}: Index not available")
        continue

    x1, y1, x2, y2 = map(
        int, result.boxes.xyxy[i].cpu().numpy()
    )

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        continue

    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(
        mask,
        [np.round(result.masks.xy[i]).astype(np.int32)],
        255
    )

    local_mask = mask[y1:y2, x1:x2] > 0
    crop = frame[y1:y2, x1:x2].copy()
    crop_hsv = hsv[y1:y2, x1:x2]

    pixels = crop_hsv[local_mask]

    if len(pixels) == 0:
        continue

    print(
        f"\nHORIZONTAL BODY REGION ANALYSIS "
        f"- SEGMENT {segment_no}"
    )

    crop_height, crop_width = local_mask.shape

    # --------------------------------------------------
    # ANALYSE AND VISUALISE FIVE HORIZONTAL REGIONS
    # --------------------------------------------------

    for region_name, start, end in regions:

        ry1 = int(crop_height * start)
        ry2 = int(crop_height * end)

        rx1 = int(crop_width * 0.20)
        rx2 = int(crop_width * 0.80)

        region_mask = local_mask[ry1:ry2, rx1:rx2]
        region_hsv = crop_hsv[ry1:ry2, rx1:rx2]

        region_pixels = region_hsv[region_mask]

        if len(region_pixels) < 20:
            continue

        rh = region_pixels[:, 0]
        rs = region_pixels[:, 1]
        rv = region_pixels[:, 2]

        bright_neutral = (rs < 80) & (rv >= 170)

        print(f"\n{region_name}")
        print("Pixels:", len(region_pixels))

        print(
            "Median HSV:",
            np.median(region_pixels, axis=0).round(1)
        )

        print(
            "Bright, low-saturation pixels:",
            f"{bright_neutral.mean() * 100:.2f}%"
        )

        print(
            "Saturation percentiles:",
            np.percentile(rs, [25, 50, 75]).round(1)
        )

        # ----------------------------------------------
        # CREATE MASKED REGION IMAGE
        # ----------------------------------------------

        region_crop = crop[ry1:ry2, rx1:rx2].copy()
        region_crop[~region_mask] = 0

        tile_width = 360
        image_height = 100
        label_height = 30

        region_height, region_width = region_crop.shape[:2]

        if region_height == 0 or region_width == 0:
            continue

        scale = min(
            tile_width / region_width,
            image_height / region_height
        )

        resized = cv2.resize(
            region_crop,
            (
                max(1, int(region_width * scale)),
                max(1, int(region_height * scale))
            )
        )

        tile = np.zeros(
            (image_height + label_height, tile_width, 3),
            dtype=np.uint8
        )

        x_offset = (tile_width - resized.shape[1]) // 2

        y_offset = label_height + (
            image_height - resized.shape[0]
        ) // 2

        tile[
            y_offset:y_offset + resized.shape[0],
            x_offset:x_offset + resized.shape[1]
        ] = resized

        cv2.putText(
            tile,
            f"SEG {segment_no} | {region_name}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        region_tiles.append(tile)

    # --------------------------------------------------
    # OVERALL SEGMENT HSV DIAGNOSTICS
    # --------------------------------------------------

    saturation = pixels[:, 1]
    brightness = pixels[:, 2]

    print("\n" + "=" * 55)
    print(f"SEGMENT {segment_no}")
    print("Class ID:", int(result.boxes.cls[i].item()))

    print(
        "Median BGR:",
        np.median(crop[local_mask], axis=0).astype(int)
    )

    for threshold in [40, 50, 60, 70]:

        neutral = saturation < threshold
        bright_neutral = neutral & (brightness >= 170)

        print(
            f"S < {threshold}: "
            f"{neutral.mean() * 100:.2f}% neutral, "
            f"{bright_neutral.mean() * 100:.2f}% bright-neutral"
        )

    print("Saturation percentiles:")
    print(
        np.percentile(
            saturation, [10, 25, 50, 75, 90]
        ).round(1)
    )

    print("Brightness percentiles:")
    print(
        np.percentile(
            brightness, [10, 25, 50, 75, 90]
        ).round(1)
    )

    # --------------------------------------------------
    # ORIGINAL FULL VEHICLE CONTACT SHEET
    # --------------------------------------------------

    masked_crop = crop.copy()
    masked_crop[~local_mask] = 0

    masked_crop = cv2.resize(masked_crop, (360, 240))

    label = np.zeros((35, 360, 3), dtype=np.uint8)

    cv2.putText(
        label,
        f"SEGMENT {segment_no}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    tiles.append(np.vstack([label, masked_crop]))

# --------------------------------------------------
# SAVE ORIGINAL VEHICLE CONTACT SHEET
# --------------------------------------------------

if not tiles:
    raise RuntimeError("No valid vehicle crops generated")

columns = 2
rows = (len(tiles) + columns - 1) // columns

sheet = np.zeros(
    (rows * 275, columns * 360, 3),
    dtype=np.uint8
)

for index, tile in enumerate(tiles):

    row = index // columns
    col = index % columns

    sheet[
        row * 275:(row + 1) * 275,
        col * 360:(col + 1) * 360
    ] = tile

output_path = os.path.join(
    OUTPUT_DIR,
    "white_vehicle_pixel_check.jpg"
)

cv2.imwrite(output_path, sheet)

# --------------------------------------------------
# SAVE FIVE-REGION COMPARISON SHEET
# --------------------------------------------------

if not region_tiles:
    raise RuntimeError("No valid region images generated")

region_columns = 4
region_tile_width = 360
region_tile_height = 130

region_rows = (
    len(region_tiles) + region_columns - 1
) // region_columns

region_sheet = np.zeros(
    (
        region_rows * region_tile_height,
        region_columns * region_tile_width,
        3
    ),
    dtype=np.uint8
)

for index, tile in enumerate(region_tiles):

    row = index // region_columns
    col = index % region_columns

    y = row * region_tile_height
    x = col * region_tile_width

    region_sheet[
        y:y + region_tile_height,
        x:x + region_tile_width
    ] = tile

region_output_path = os.path.join(
    OUTPUT_DIR,
    "white_vehicle_region_comparison.jpg"
)

cv2.imwrite(region_output_path, region_sheet)

# --------------------------------------------------
# FINAL OUTPUT
# --------------------------------------------------

print("\n" + "=" * 55)
print("DIAGNOSTIC COMPLETED")
print("Vehicle contact sheet:", output_path)
print("Region comparison sheet:", region_output_path)
print("Total vehicle tiles:", len(tiles))
print("Total region tiles:", len(region_tiles))
