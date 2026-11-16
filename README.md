# Vehicle Pattern Recognition and Tracking Using Deep Learning

## Overview
A computer vision project that detects, tracks, and analyzes vehicles in highway surveillance videos using deep learning.

## Features
- Vehicle detection using YOLOv8
- Vehicle segmentation
- Multi-object tracking using ByteTrack
- Vehicle type classification
- Vehicle color recognition
- License plate detection and OCR using EasyOCR
- Track-wise license plate consensus
- Approximate vehicle speed estimation
- Annotated video generation
- CSV-based reports and analytics

## Technologies
- Python 3.11
- PyTorch
- YOLOv8 (Ultralytics)
- ByteTrack
- OpenCV
- EasyOCR
- NumPy
- Pandas
- Matplotlib

## Installation

Clone the repository:

```bash
git clone https://github.com/Kapil961/Vehicle-pattern-recognition-tracking.git
cd Vehicle-pattern-recognition-tracking
```

Create and activate a virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Input Video

The original input video is not included in the repository. Place it at:

`data/raw/highway_videos/highway_traffic_h264.mp4`

## Run

```bash
python main.py
```

## Experimental Results

Results from the 1,800-frame test:

| Metric | Result |
|---|---:|
| Frames processed | 1,800 |
| Vehicle observations | 24,529 |
| Unique tracked vehicles | 67 |
| Cars | 63 |
| Trucks | 3 |
| Buses | 1 |
| OCR readings | 161 |
| Accepted plate candidates | 8 |
| Processing time | 21.8 minutes |

## Generated Outputs

- `outputs/videos/final_vehicle_recognition_verified.mp4`
- `outputs/reports/vehicle_tracking_verified.csv`
- `outputs/reports/final_vehicle_summary_verified.csv`
- `outputs/reports/license_plate_detections_verified.csv`
- `outputs/reports/vehicle_speed_tracking_verified.csv`

## Limitations

- Speed estimates are approximate and require real-world calibration.
- OCR results have not been independently verified against ground-truth plates.
- Detection performance depends on lighting, resolution, and vehicle visibility.

## Author

**Kapil Kumawat**

B.Tech – Computer Science and Artificial Intelligence  
JK Lakshmipat University, Jaipur

GitHub: https://github.com/Kapil961

---

Academic and experimental project.