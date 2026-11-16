import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT = "outputs/reports/vehicle_tracking.csv"
SUMMARY = "outputs/reports/final_vehicle_summary.csv"

REPORT_DIR = "outputs/reports"
PLOT_DIR = "outputs/plots"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(INPUT)
    summary = pd.read_csv(SUMMARY)

    print("Generating vehicle analytics...")

    # Unique tracked vehicles
    total_vehicles = summary["track_id"].nunique()

    # Vehicle type distribution
    type_counts = (
        summary.groupby("vehicle_type")["track_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    # Frame-level traffic volume
    frame_counts = (
        df.groupby("frame")["track_id"]
        .nunique()
        .reset_index(name="vehicle_count")
    )

    # Tracking duration
    duration = (
        df.groupby("track_id")
        .agg(
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
            observations=("frame", "count")
        )
        .reset_index()
    )

    duration["tracking_duration_seconds"] = (
        duration["last_frame"] - duration["first_frame"]
    ) / 60.0

    # Save reports
    type_counts.to_csv(
        f"{REPORT_DIR}/vehicle_type_distribution.csv"
    )

    frame_counts.to_csv(
        f"{REPORT_DIR}/frame_vehicle_counts.csv",
        index=False
    )

    duration.to_csv(
        f"{REPORT_DIR}/vehicle_tracking_duration.csv",
        index=False
    )

    # Vehicle type chart
    plt.figure(figsize=(9, 5))
    type_counts.plot(kind="bar")
    plt.title("Vehicle Type Distribution")
    plt.xlabel("Vehicle Type")
    plt.ylabel("Number of Tracked Vehicles")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/vehicle_type_distribution.png")
    plt.close()

    # Traffic volume chart
    plt.figure(figsize=(12, 5))
    plt.plot(
        frame_counts["frame"] / 60.0,
        frame_counts["vehicle_count"]
    )
    plt.title("Vehicle Count Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Vehicles Visible")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/traffic_volume.png")
    plt.close()

    # Tracking duration chart
    plt.figure(figsize=(9, 5))
    plt.hist(
        duration["tracking_duration_seconds"],
        bins=20
    )
    plt.title("Vehicle Tracking Duration")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of Vehicles")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/tracking_duration.png")
    plt.close()

    # Overall report
    with open(f"{REPORT_DIR}/analytics_summary.txt", "w") as f:
        f.write("VEHICLE TRAFFIC ANALYTICS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total unique vehicles: {total_vehicles}\n")
        f.write(f"Total observations: {len(df)}\n")
        f.write(f"Video frames analyzed: {df['frame'].nunique()}\n")
        f.write("\nVehicle type distribution:\n")
        f.write(type_counts.to_string())

    print("\nANALYTICS COMPLETED")
    print("Unique vehicles:", total_vehicles)
    print("Total observations:", len(df))
    print("\nVehicle type distribution:")
    print(type_counts)

    print("\nGenerated charts:")
    print("outputs/plots/vehicle_type_distribution.png")
    print("outputs/plots/traffic_volume.png")
    print("outputs/plots/tracking_duration.png")

    print("\nGenerated reports:")
    print("outputs/reports/analytics_summary.txt")
    print("outputs/reports/frame_vehicle_counts.csv")
    print("outputs/reports/vehicle_tracking_duration.csv")


if __name__ == "__main__":
    main()
