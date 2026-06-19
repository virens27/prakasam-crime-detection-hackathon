"""
main.py

Main pipeline: runs all three detection modules on a video source and
logs/aggregates alerts.

TODO (Team):
1. Once each module in src/modules/ is working individually, wire them
   together here.
2. Decide on FRAME_SAMPLE_INTERVAL based on testing -- lower = more accurate
   but slower on CPU; higher = faster but may miss brief events.
3. Replace the print-based alert log with writes to a file/database that
   the dashboard (src/dashboard/app.py) can read from.
4. Add snapshot saving for flagged frames (cv2.imwrite) so alerts have
   visual evidence attached.
"""

import cv2
import time
from modules.weapon_detection import WeaponDetector
from modules.fight_detection import FightDetector
from modules.abandoned_object import AbandonedObjectDetector

VIDEO_SOURCE = "data/sample_clips/test_clip.mp4"  # TODO: point to your test clip(s)
FRAME_SAMPLE_INTERVAL = 0.5  # seconds between analyzed frames (CPU-friendly sampling)


def log_alert(alert_type: str, details: dict):
    """
    TODO: replace this with proper logging -- e.g. append to a JSON/CSV file
    or a lightweight database (SQLite) that the dashboard reads from.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ALERT] {timestamp} | {alert_type} | {details}")


def run_pipeline(video_source: str = VIDEO_SOURCE):
    weapon_detector = WeaponDetector()
    fight_detector = FightDetector()
    object_detector = AbandonedObjectDetector()

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Could not open video source: {video_source}")
        print("Add a test clip to data/sample_clips/ and update VIDEO_SOURCE.")
        return

    last_processed_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if current_time - last_processed_time < FRAME_SAMPLE_INTERVAL:
            continue  # skip frame for CPU efficiency
        last_processed_time = current_time

        # --- Weapon detection ---
        weapon_results = weapon_detector.detect_weapons(frame)
        if weapon_results:
            log_alert("WEAPON_DETECTED", weapon_results)

        # --- Fight detection ---
        fight_result = fight_detector.detect_fight(frame)
        if fight_result["flagged"]:
            log_alert("FIGHT_DETECTED", fight_result)

        # --- Abandoned object detection ---
        object_result = object_detector.detect_abandoned_objects(frame)
        if object_result["flagged_ids"]:
            log_alert("ABANDONED_OBJECT", object_result)

    cap.release()
    print("Pipeline finished processing video.")


if __name__ == "__main__":
    run_pipeline()
