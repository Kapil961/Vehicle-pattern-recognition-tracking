import cv2
import json
import os

IMAGE_PATH = "outputs/plots/speed_calibration_frame.jpg"
OUTPUT_PATH = "configs/speed_calibration.json"

points = []
display_scale = 1.0
original_image = None

def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        original_x = int(x / display_scale)
        original_y = int(y / display_scale)
        points.append([original_x, original_y])
        print(f"Point {len(points)} selected: ({original_x}, {original_y})")

def main():
    global display_scale, original_image

    original_image = cv2.imread(IMAGE_PATH)

    if original_image is None:
        raise FileNotFoundError(IMAGE_PATH)

    height, width = original_image.shape[:2]

    display_scale = min(1280 / width, 720 / height, 1.0)

    display_width = int(width * display_scale)
    display_height = int(height * display_scale)

    print("\nSELECT FOUR POINTS IN THIS ORDER:")
    print("1. Near-left corner of one lane")
    print("2. Near-right corner of the same lane")
    print("3. Far-right corner of that lane")
    print("4. Far-left corner of that lane")
    print("\nChoose points on the road surface, not on vehicles.")
    print("Press Q when all four points are selected.")
    print("Press R to reset your selections.\n")

    cv2.namedWindow("Speed Calibration")
    cv2.setMouseCallback("Speed Calibration", mouse_callback)

    while True:
        display = cv2.resize(
            original_image,
            (display_width, display_height)
        )

        for i, point in enumerate(points):
            x = int(point[0] * display_scale)
            y = int(point[1] * display_scale)

            cv2.circle(display, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(
                display,
                str(i + 1),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("Speed Calibration", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):
            points.clear()
            print("Selection reset.")

    cv2.destroyAllWindows()

    if len(points) != 4:
        print("Calibration cancelled: exactly four points are required.")
        return

    print("\nSelected points:", points)

    lane_width = input("Enter lane width in metres [3.5]: ").strip()
    road_length = input("Enter longitudinal distance in metres [9.0]: ").strip()

    lane_width = float(lane_width) if lane_width else 3.5
    road_length = float(road_length) if road_length else 9.0

    if lane_width <= 0 or road_length <= 0:
        raise ValueError("Distances must be positive.")

    # World-coordinate order matches the selected image-point order.
    world_points = [
        [0.0, 0.0],
        [lane_width, 0.0],
        [lane_width, road_length],
        [0.0, road_length]
    ]

    calibration = {
        "image_points": points,
        "world_points_metres": world_points,
        "lane_width_metres": lane_width,
        "longitudinal_distance_metres": road_length,
        "calibration_status": "approximate_assumed_dimensions"
    }

    os.makedirs("configs", exist_ok=True)

    with open(OUTPUT_PATH, "w") as file:
        json.dump(calibration, file, indent=4)

    print("\nCalibration saved:", OUTPUT_PATH)
    print("Lane width:", lane_width, "metres")
    print("Longitudinal distance:", road_length, "metres")
    print("Status: APPROXIMATE — based on assumed dimensions.")

if __name__ == "__main__":
    main()
