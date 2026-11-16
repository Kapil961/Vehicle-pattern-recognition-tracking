from pathlib import Path

path = Path("main.py")
code = path.read_text()


def replace_section(start_marker, end_marker, replacement):
    global code

    start = code.index(start_marker)
    end = code.index(end_marker, start)

    code = code[:start] + replacement + "\n\n" + code[end:]


# ============================================================
# 1. IMPROVED VEHICLE COLOR CLASSIFICATION
# ============================================================

replace_section(
    "def classify_color(vehicle_crop):",
    "# ============================================================\n# LICENSE PLATE TEXT CLEANING",
    '''def classify_color(vehicle_crop):

    if vehicle_crop is None or vehicle_crop.size == 0:
        return "Unknown"

    height, width = vehicle_crop.shape[:2]

    if height < 15 or width < 15:
        return "Unknown"

    # Focus on the central vehicle body.
    x1 = int(width * 0.20)
    x2 = int(width * 0.80)
    y1 = int(height * 0.35)
    y2 = int(height * 0.78)

    crop = vehicle_crop[y1:y2, x1:x2]

    if crop.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # Remove very dark, very bright and weakly saturated pixels
    # from the chromatic color calculation.
    valid = (
        (v > 35) &
        (v < 245)
    )

    if np.count_nonzero(valid) < 20:
        return "Unknown"

    saturation = s[valid]
    hue = h[valid]
    value = v[valid]

    # Determine whether the vehicle is mostly achromatic.
    achromatic_ratio = np.mean(saturation < 45)

    if achromatic_ratio >= 0.55:

        neutral_values = value[saturation < 45]

        if len(neutral_values) == 0:
            return "Unknown"

        median_value = np.median(neutral_values)

        if median_value > 185:
            return "White"

        if median_value > 85:
            return "Gray"

        return "Black"

    # Use the distribution of hue pixels rather than mean hue.
    chromatic = saturation >= 45

    hue = hue[chromatic]
    saturation = saturation[chromatic]

    if len(hue) < 15:
        return "Unknown"

    color_ranges = {
        "Red": ((hue <= 10) | (hue >= 170)),
        "Orange": ((hue > 10) & (hue <= 22)),
        "Yellow": ((hue > 22) & (hue <= 35)),
        "Green": ((hue > 35) & (hue <= 85)),
        "Blue": ((hue > 85) & (hue <= 130)),
        "Purple": ((hue > 130) & (hue < 170))
    }

    scores = {}

    for color, mask in color_ranges.items():
        scores[color] = float(
            np.sum(saturation[mask])
        )

    best_color = max(scores, key=scores.get)

    total_score = sum(scores.values())

    if total_score <= 0:
        return "Unknown"

    dominance = scores[best_color] / total_score

    # Avoid reporting a color when the evidence is ambiguous.
    if dominance < 0.35:
        return "Unknown"

    return best_color


def get_stable_color(color_history):

    recent = [
        color
        for color in color_history[-30:]
        if color != "Unknown"
    ]

    if not recent:
        return "Unknown"

    counts = Counter(recent)

    best_color, count = counts.most_common(1)[0]

    if count < 2:
        return "Unknown"

    return best_color
'''
)


# ============================================================
# 2. IMPROVED CONFIDENCE-WEIGHTED PLATE VOTING
# ============================================================

replace_section(
    "def get_stable_plate(plate_history):",
    "# ============================================================\n# LOAD SPEED CALIBRATION",
    '''def get_stable_plate(plate_history):

    grouped = defaultdict(list)

    for record in plate_history:

        text = record["text"]

        confidence = float(record["confidence"])

        if text and confidence >= 0.25:
            grouped[text].append(record)

    if not grouped:
        return "", 0

    ranked = []

    for text, records in grouped.items():

        distinct_frames = len({
            record["frame"]
            for record in records
        })

        total_confidence = sum(
            float(record["confidence"])
            for record in records
        )

        mean_confidence = (
            total_confidence / len(records)
        )

        ranked.append((
            text,
            distinct_frames,
            total_confidence,
            mean_confidence
        ))

    ranked.sort(
        key=lambda item: (
            item[1],
            item[2],
            item[3]
        ),
        reverse=True
    )

    best_text = ranked[0][0]
    best_votes = ranked[0][1]

    return best_text, best_votes
'''
)


# ============================================================
# 3. ADD OCR CONFIDENCE THRESHOLD
# ============================================================

code = code.replace(
    "MIN_PLATE_VOTES = 3\nMIN_COLOR_OBSERVATIONS = 3",
    "MIN_PLATE_VOTES = 3\nMIN_COLOR_OBSERVATIONS = 3\nMIN_OCR_CONFIDENCE = 0.25"
)

# ============================================================
# 4. ADD PERSISTENT PLATE LOCATION TO VEHICLE HISTORY
# ============================================================

old = '''            "colors": [],
            "plates": [],
            "confidences": [],'''

new = '''            "colors": [],
            "plates": [],
            "plate_relative_box": None,
            "last_plate_frame": None,
            "confidences": [],'''

assert old in code, "Vehicle history section not found"
code = code.replace(old, new, 1)


# ============================================================
# 5. USE TEMPORAL COLOR VOTING
# ============================================================

old = '''                        data["observations"] += 1

            # ------------------------------------------------
            # LICENSE PLATE DETECTION AND OCR'''

new = '''                        data["observations"] += 1

                        # Use recent observations for a stable color.
                        vehicle_boxes[-1]["color"] = get_stable_color(
                            data["colors"]
                        )

            # ------------------------------------------------
            # LICENSE PLATE DETECTION AND OCR'''

assert old in code, "Color update location not found"
code = code.replace(old, new, 1)


# ============================================================
# 6. UPDATE PLATE ASSOCIATION AND SAVE RELATIVE COORDINATES
# ============================================================

old = '''                        track_id = match_plate_to_vehicle(
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
                        )'''

new = '''                        track_id = match_plate_to_vehicle(
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

                        # Store plate position relative to the vehicle.
                        # This lets us redraw it when the vehicle moves.
                        vx1, vy1, vx2, vy2 = matched_vehicle["box"]

                        vehicle_width = max(1, vx2 - vx1)
                        vehicle_height = max(1, vy2 - vy1)

                        vehicle_data[track_id]["plate_relative_box"] = (
                            (px1 - vx1) / vehicle_width,
                            (py1 - vy1) / vehicle_height,
                            (px2 - vx1) / vehicle_width,
                            (py2 - vy1) / vehicle_height
                        )

                        vehicle_data[track_id]["last_plate_frame"] = (
                            frame_number
                        )

                        vehicle_type = matched_vehicle["vehicle_type"]

                        if (
                            not plate_text
                            or ocr_confidence < MIN_OCR_CONFIDENCE
                        ):
                            continue'''

assert old in code, "Plate association section not found"
code = code.replace(old, new, 1)


# ============================================================
# 7. REMOVE THE OLD BLINKING PLATE DRAWING
# ============================================================

old = '''                        cv2.rectangle(
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

'''

assert old in code, "Old plate drawing section not found"
code = code.replace(old, "", 1)


# ============================================================
# 8. REPLACE PLATE ANNOTATION WITH PERSISTENT OVERLAY
# ============================================================

old = '''                # Plate information

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

                # Approximate speed information'''

new = '''                # Persistent plate information.
                # Expire an old plate location if it has not been
                # detected again for 45 frames.

                vehicle_info = vehicle_data[track_id]

                plate_box = vehicle_info["plate_relative_box"]

                last_plate_frame = vehicle_info["last_plate_frame"]

                if (
                    plate_box is not None
                    and last_plate_frame is not None
                    and frame_number - last_plate_frame <= 45
                ):

                    rx1, ry1, rx2, ry2 = plate_box

                    p1 = (
                        max(0, min(width - 1, int(x1 + rx1 * (x2 - x1)))),
                        max(0, min(height - 1, int(y1 + ry1 * (y2 - y1))))
                    )

                    p2 = (
                        max(0, min(width - 1, int(x1 + rx2 * (x2 - x1)))),
                        max(0, min(height - 1, int(y1 + ry2 * (y2 - y1))))
                    )

                    cv2.rectangle(
                        frame,
                        p1,
                        p2,
                        (0, 255, 255),
                        2
                    )

                    if (
                        stable_plate
                        and plate_votes >= MIN_PLATE_VOTES
                    ):
                        plate_label = f"Plate: {stable_plate}"
                    else:
                        plate_label = "Plate: Reading..."

                    cv2.putText(
                        frame,
                        plate_label,
                        (x1, min(height - 10, y1 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2
                    )

                # Approximate speed information'''

assert old in code, "Old plate annotation section not found"
code = code.replace(old, new, 1)


# ============================================================
# SAVE CHANGES
# ============================================================

path.write_text(code)

print("DETECTION IMPROVEMENTS APPLIED")
print("=" * 45)
print("Improved pixel-based vehicle color classification")
print("Temporal color voting")
print("Confidence-weighted plate voting")
print("Persistent plate location across frames")
print("Persistent plate label after OCR consensus")
print("OCR confidence filtering")
print()
print("Original backup: main_before_color_plate_improvements.py")
print("Updated file: main.py")
