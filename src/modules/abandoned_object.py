"""
abandoned_object.py

Module: Abandoned Object Detection
Uses OpenCV background subtraction to detect objects that remain stationary
in a scene for longer than a defined time threshold without a person nearby.

Key improvements over v1:
- Increased MIN_CONTOUR_AREA to filter out small noise and body parts
- Increased STATIONARY_TIME_THRESHOLD to 20 seconds (from 10)
- Added person-proximity check using YOLOv8n -- if a detected blob is close
  to an active person, it's NOT flagged (avoids flagging people standing still)
- Cleaned up tracker to properly prune stale tracked objects
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO

MIN_CONTOUR_AREA = 5000          # increased from 1500 -- filters out body parts, shadows, small noise
STATIONARY_TIME_THRESHOLD = 20   # seconds -- increased from 10 for stricter detection
POSITION_TOLERANCE = 30          # pixels of movement still counted as "stationary"
PERSON_PROXIMITY_THRESHOLD = 120 # pixels -- if a person is this close to a blob, don't flag it
PERSON_CONF_THRESHOLD = 0.4      # YOLO confidence to count as a detected person

PERSON_DETECTOR_MODEL = "yolov8n.pt"


class AbandonedObjectDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=700,         # longer history = more stable background model
            varThreshold=60,     # higher = less sensitive to minor changes
            detectShadows=True
        )
        self.person_detector = YOLO(PERSON_DETECTOR_MODEL)
        # Tracks: {object_id: {"position": (cx, cy), "first_seen": timestamp, "bbox": (x,y,w,h)}}
        self.tracked_objects = {}
        self.next_object_id = 0

    def detect_blobs(self, frame):
        """
        Run background subtraction and return candidate foreground blobs
        large enough to be objects (not noise or body parts).
        """
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows (gray pixels, value=127) -- keep only definite foreground (white, value=255)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological closing to fill gaps within objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
                boxes.append(cv2.boundingRect(cnt))

        return boxes

    def detect_people(self, frame):
        """Detect person bounding boxes using YOLOv8n."""
        results = self.person_detector(frame, verbose=False, classes=[0])[0]
        person_centers = []

        for box in results.boxes:
            if float(box.conf[0]) >= PERSON_CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                person_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

        return person_centers

    def is_near_person(self, blob_center, person_centers):
        """Return True if blob is close to any detected person (so we don't flag people)."""
        for pc in person_centers:
            dist = np.linalg.norm(np.array(blob_center) - np.array(pc))
            if dist < PERSON_PROXIMITY_THRESHOLD:
                return True
        return False

    def update_tracking(self, boxes, person_centers):
        """
        Match detected blobs to tracked objects (or create new ones).
        Prune objects that are now near a person or have disappeared.

        Returns:
            List of object_ids that have been stationary past the time threshold.
        """
        current_time = time.time()
        matched_ids = set()
        flagged_ids = []

        for (x, y, w, h) in boxes:
            center = (x + w // 2, y + h // 2)

            # Skip blobs that are right next to a person
            if self.is_near_person(center, person_centers):
                continue

            matched_id = None
            for obj_id, data in self.tracked_objects.items():
                dist = np.linalg.norm(np.array(center) - np.array(data["position"]))
                if dist <= POSITION_TOLERANCE:
                    matched_id = obj_id
                    break

            if matched_id is not None:
                elapsed = current_time - self.tracked_objects[matched_id]["first_seen"]
                if elapsed >= STATIONARY_TIME_THRESHOLD:
                    flagged_ids.append(matched_id)
                matched_ids.add(matched_id)
            else:
                self.tracked_objects[self.next_object_id] = {
                    "position": center,
                    "first_seen": current_time,
                    "bbox": (x, y, w, h)
                }
                matched_ids.add(self.next_object_id)
                self.next_object_id += 1

        # Remove tracked objects that are no longer detected
        for obj_id in list(self.tracked_objects.keys()):
            if obj_id not in matched_ids:
                del self.tracked_objects[obj_id]

        return flagged_ids

    def detect_abandoned_objects(self, frame):
        """
        Main entry point: detect blobs, filter by person proximity,
        and flag any that have been stationary past the time threshold.

        Returns:
            dict with "boxes" (all detected blobs) and "flagged_ids"
        """
        boxes = self.detect_blobs(frame)
        person_centers = self.detect_people(frame)
        flagged_ids = self.update_tracking(boxes, person_centers)

        return {
            "boxes": boxes,
            "flagged_ids": flagged_ids
        }


if __name__ == "__main__":
    import sys

    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_clips/test_clip.mp4"

    detector = AbandonedObjectDetector()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
    else:
        print(f"Testing on: {video_path}")
        print(f"Stationary time threshold: {STATIONARY_TIME_THRESHOLD}s\n")
        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Process every 5th frame -- abandoned objects are slow events,
            # so we can sample even less frequently than fight detection
            if frame_count % 5 != 0:
                continue

            result = detector.detect_abandoned_objects(frame)
            if result["flagged_ids"]:
                print(f"[Frame {frame_count}] Abandoned object(s) flagged: {result['flagged_ids']}")

        elapsed = time.time() - start_time
        print(f"\nFinished processing {frame_count} frames in {elapsed:.1f}s")
        cap.release()