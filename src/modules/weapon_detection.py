"""
weapon_detection.py

Module: Weapon Detection
Detects visible weapons in video frames using YOLOv8n.

Note on model choice:
    The ideal setup is a YOLOv8n model fine-tuned specifically on weapon
    images (e.g. trained on the Roboflow buildx/weapon-detection-7kro8
    dataset which is already downloaded in Weapon-Detection--2/).
    
    For this MVP we use the base yolov8n.pt model and filter its detections
    to only flag COCO classes that overlap with weapons (person is excluded,
    but items like "knife" are in COCO). The weapon-specific class list from
    the Roboflow dataset is also included so the module is ready to swap in
    fine-tuned weights when available.

    To use fine-tuned weights (future upgrade):
    1. Train: yolo train model=yolov8n.pt data=Weapon-Detection--2/data.yaml epochs=20
    2. Replace MODEL_PATH below with the output weights path
    3. Set USE_WEAPON_DATASET_CLASSES = True

Future scope:
    Train yolov8n on the downloaded Weapon-Detection--2 dataset for
    significantly higher accuracy on real-world weapon detection scenarios.
"""

import cv2
import time
import sys
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = "yolov8n.pt"          # base YOLOv8n -- swap with fine-tuned weights when available
CONFIDENCE_THRESHOLD = 0.45        # minimum confidence to flag a detection

# COCO classes (in base yolov8n) that are weapon-related
# These are the only detections we care about from the base model
COCO_WEAPON_CLASSES = [
    "knife"
]

# Full weapon class list from the Roboflow buildx dataset
# Used when fine-tuned weights are loaded
WEAPON_DATASET_CLASSES = [
    "Knife", "ak", "ax", "cleaver", "cutter", "eto",
    "long sword", "m16", "revolver", "rifle",
    "semi automatic", "short sword", "shotgun", "spear"
]

# Set to True when using fine-tuned weapon-detection weights
USE_WEAPON_DATASET_CLASSES = False


class WeaponDetector:
    def __init__(self, model_path: str = MODEL_PATH, conf_threshold: float = CONFIDENCE_THRESHOLD):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        if USE_WEAPON_DATASET_CLASSES:
            self.target_classes = [c.lower() for c in WEAPON_DATASET_CLASSES]
        else:
            self.target_classes = [c.lower() for c in COCO_WEAPON_CLASSES]

    def detect_weapons(self, frame):
        """
        Run weapon detection on a single video frame.

        Args:
            frame: numpy array (BGR format from OpenCV)

        Returns:
            List of detections, each as a dict:
            {
                "class_name": str,
                "confidence": float,
                "bbox": (x1, y1, x2, y2)
            }
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]

            if conf >= self.conf_threshold and class_name.lower() in self.target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "class_name": class_name,
                    "confidence": round(conf, 3),
                    "bbox": (x1, y1, x2, y2)
                })

        return detections


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_clips/test_clip.mp4"

    detector = WeaponDetector()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
    else:
        print(f"Testing weapon detection on: {video_path}")
        print(f"Watching for: {detector.target_classes}\n")

        frame_count = 0
        alert_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 3 != 0:
                continue

            detections = detector.detect_weapons(frame)
            if detections:
                alert_count += 1
                print(f"[Frame {frame_count}] WEAPON DETECTED: {detections}")

        elapsed = time.time() - start_time
        print(f"\nFinished: {frame_count} frames in {elapsed:.1f}s")
        print(f"Total weapon alerts: {alert_count}")
        cap.release()