"""
abandoned_object.py

Module: Abandoned Object Detection
Uses OpenCV background subtraction to detect objects that remain stationary
in a scene for longer than a defined time threshold without a person nearby
(suggesting it was abandoned, e.g., an unattended bag).

Approach:
1. Use background subtraction (MOG2) to detect foreground "blobs" (moving or
   newly introduced objects) in each frame.
2. Track blobs across frames -- if a blob stays roughly in the same position
   without moving for STATIONARY_TIME_THRESHOLD seconds, flag it.
3. (Optional refinement) Check if a person was near the object recently and
   has since moved away, to better distinguish "abandoned" from "just parked".

TODO (Team):
1. Tune STATIONARY_TIME_THRESHOLD and MIN_CONTOUR_AREA based on test footage
2. Test on sample clips with a bag/object being placed and left
3. (Stretch goal) Add person-proximity check using a lightweight person
   detector (e.g. reuse YOLOv8n with the 'person' class) to reduce false
   positives from static background elements
"""

import cv2
import time

MIN_CONTOUR_AREA = 1500       # ignore tiny noise blobs
STATIONARY_TIME_THRESHOLD = 10  # seconds an object must stay still to be "abandoned"
POSITION_TOLERANCE = 20       # pixels of allowed movement to still count as "stationary"


class AbandonedObjectDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        # Tracks: {object_id: {"position": (x, y), "first_seen": timestamp}}
        self.tracked_objects = {}
        self.next_object_id = 0

    def detect_objects(self, frame):
        """
        Run background subtraction and find candidate foreground objects.

        Returns:
            List of (x, y, w, h) bounding boxes for detected blobs.
        """
        fg_mask = self.bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
                boxes.append(cv2.boundingRect(cnt))

        return boxes

    def update_tracking(self, boxes):
        """
        Match current detections to tracked objects and check how long each
        has remained roughly stationary.

        TODO: This is a simplified nearest-position matcher. For a more robust
        MVP, consider using a proper tracker (e.g. OpenCV's built-in trackers,
        or a simple centroid-tracking algorithm).

        Returns:
            List of object_ids that have exceeded STATIONARY_TIME_THRESHOLD.
        """
        current_time = time.time()
        flagged_ids = []

        for (x, y, w, h) in boxes:
            center = (x + w // 2, y + h // 2)
            matched_id = None

            for obj_id, data in self.tracked_objects.items():
                prev_center = data["position"]
                dist = ((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2) ** 0.5
                if dist <= POSITION_TOLERANCE:
                    matched_id = obj_id
                    break

            if matched_id is not None:
                elapsed = current_time - self.tracked_objects[matched_id]["first_seen"]
                if elapsed >= STATIONARY_TIME_THRESHOLD:
                    flagged_ids.append(matched_id)
            else:
                self.tracked_objects[self.next_object_id] = {
                    "position": center,
                    "first_seen": current_time
                }
                self.next_object_id += 1

        return flagged_ids

    def detect_abandoned_objects(self, frame):
        """
        Main entry point: detect objects and flag any that are abandoned.

        Returns:
            dict with "boxes" (all detected blobs) and "flagged_ids"
            (objects stationary past threshold)
        """
        boxes = self.detect_objects(frame)
        flagged_ids = self.update_tracking(boxes)

        return {
            "boxes": boxes,
            "flagged_ids": flagged_ids
        }


if __name__ == "__main__":
    # Quick manual test — TODO: replace with a real sample video path
    detector = AbandonedObjectDetector()
    cap = cv2.VideoCapture("data/sample_clips/test_clip.mp4")

    if not cap.isOpened():
        print("No test clip found. Add a test video to data/sample_clips/ to test this module.")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result = detector.detect_abandoned_objects(frame)
            if result["flagged_ids"]:
                print(f"Abandoned object(s) flagged: {result['flagged_ids']}")
        cap.release()
