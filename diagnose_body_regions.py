import cv2
import numpy as np
from ultralytics import YOLO

VIDEO = "data/raw/highway_videos/highway_traffic_h264.mp4"
SEGMENTS = [2, 3, 7, 11, 12, 18, 20, 23]

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Cannot read frame")

h, w = frame.shape[:2]
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

model = YOLO("yolov8n-seg.pt")

result = model(
    frame,
    conf=0.25,
    imgsz=640,
    device="cpu",
    classes=[2, 3, 5, 7],
    verbose=False
)[0]

regions = {
    "Candidate top body": (0.10, 0.30),
    "Upper body": (0.20, 0.45),
    "Current classifier": (0.35, 0.78),
    "Middle body": (0.45, 0.65),
}

for segment_no in SEGMENTS:
    i = segment_no - 1

    if i >= len(result.masks.xy):
        continue

    x1, y1, x2, y2 = map(
        int, result.boxes.xyxy[i].cpu().numpy()
    )

    mask = np.zeros((h, w), dtype=np.uint8)

    polygon = np.round(
        result.masks.xy[i]
    ).astype(np.int32)

    cv2.fillPoly(mask, [polygon], 255)

    local_mask = mask[y1:y2, x1:x2] > 0
    local_hsv = hsv[y1:y2, x1:x2]

    ch, cw = local_mask.shape

    print("\n" + "=" * 55)
    print("SEGMENT:", segment_no)
    print("CLASS:", int(result.boxes.cls[i].item()))

    for name, (start, end) in regions.items():

        xa = int(cw * 0.20)
        xb = int(cw * 0.80)
        ya = int(ch * start)
        yb = int(ch * end)

        selected_mask = local_mask[ya:yb, xa:xb]
        selected_hsv = local_hsv[ya:yb, xa:xb]

        pixels = selected_hsv[selected_mask]

        if len(pixels) == 0:
            continue

        saturation = pixels[:, 1]
        brightness = pixels[:, 2]

        bright_low_saturation = (
            (saturation < 80) &
            (brightness >= 170)
        )

        print(f"\n{name}")
        print("Pixels:", len(pixels))
        print(
            "Median HSV:",
            np.median(pixels, axis=0).round().astype(int)
        )
        print(
            "Saturation percentiles:",
            np.percentile(
                saturation, [25, 50, 75]
            ).round(1)
        )
        print(
            "Brightness percentiles:",
            np.percentile(
                brightness, [25, 50, 75]
            ).round(1)
        )
        print(
            "Bright low-saturation pixels:",
            f"{bright_low_saturation.mean()*100:.2f}%"
        )

print("\nDiagnostic completed.")
