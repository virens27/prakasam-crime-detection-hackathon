"""
fight_detection.py

Module: Fight / Assault Detection (Multi-Person Version)

Approach:
1. Use YOLOv8n to detect each "person" in the frame as a separate bounding box.
2. Run MediaPipe Pose individually on each person's cropped region, so each
   person gets their own tracked keypoints (avoids the single-pose model
   jumping between different people in a crowd, which caused false positives
   in the earlier single-person version).
3. Track each person across frames using a simple centroid tracker (matches
   detections to existing tracked people based on proximity).
4. Compute each tracked person's wrist movement velocity between frames.
5. Flag a "fight" only when TWO tracked people are close together AND BOTH
   show high velocity at the same time -- this is a much stronger signal
   than a single person's movement alone.

TODO (Team):
1. Tune VELOCITY_THRESHOLD, PROXIMITY_THRESHOLD, and MAX_TRACK_DISTANCE
   using your test clips. Crowd density and camera distance affect these.
2. (Stretch goal) Add a few consecutive-frame requirement before flagging,
   to reduce one-off noise (e.g. require 3 consecutive flagged frames).
3. (Stretch goal) Replace the simple centroid tracker with a more robust
   tracker (e.g. OpenCV's built-in trackers or a Kalman filter) if you have
   time -- the current tracker is intentionally simple for hackathon speed.
"""

import cv2
import time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

mp_pose = mp.solutions.pose

# --- Tunable thresholds (adjust based on testing with real footage) ---
PERSON_CONF_THRESHOLD = 0.4     # YOLO confidence to count as a "person"
VELOCITY_THRESHOLD = 150         # wrist movement (px/frame) considered "rapid" -- raised from 35
                                  # to account for 3-frame skipping + crowd pose noise
PROXIMITY_THRESHOLD = 150        # px distance between two people considered "close" -- tightened from 200
MAX_TRACK_DISTANCE = 80          # px -- max distance to match a detection to an existing tracked person
MAX_MISSED_FRAMES = 10           # frames a tracked person can go undetected before being dropped
MIN_CONSECUTIVE_FLAGS = 3        # require this many consecutive flagged checks for the same pair
                                  # before raising a real alert (filters out one-off noise spikes)

PERSON_DETECTOR_MODEL = "yolov8n.pt"  # base YOLOv8n already includes a 'person' class -- no custom training needed


class TrackedPerson:
    """Holds state for one tracked individual across frames."""
    def __init__(self, person_id, bbox, center):
        self.id = person_id
        self.bbox = bbox            # (x1, y1, x2, y2)
        self.center = center        # (cx, cy)
        self.prev_wrist_kpts = None # previous frame's wrist keypoints (left, right)
        self.velocity = 0.0
        self.missed_frames = 0


class FightDetector:
    def __init__(self):
        self.person_detector = YOLO(PERSON_DETECTOR_MODEL)
        self.pose = mp_pose.Pose(
            static_image_mode=True,  # processing independent crops, not a continuous stream per-person
            min_detection_confidence=0.5
        )
        self.tracked_people = {}
        self.next_id = 0
        self.pair_flag_counts = {}  # tracks how many consecutive times each pair has been flagged

    def detect_people(self, frame):
        """Detect person bounding boxes in the frame using YOLOv8n."""
        results = self.person_detector(frame, verbose=False, classes=[0])[0]  # class 0 = 'person' in COCO
        boxes = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf >= PERSON_CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2))

        return boxes

    def get_wrist_keypoints(self, frame, bbox):
        """
        Run MediaPipe Pose on a cropped person region and return wrist
        keypoint coordinates (in original frame coordinates, not crop-local).
        """
        x1, y1, x2, y2 = bbox
        crop = frame[max(0, y1):y2, max(0, x1):x2]

        if crop.size == 0:
            return None

        rgb_crop = crop[:, :, ::-1]
        results = self.pose.process(rgb_crop)

        if not results.pose_landmarks:
            return None

        h, w, _ = crop.shape
        landmarks = results.pose_landmarks.landmark

        # MediaPipe Pose indices: 15 = left wrist, 16 = right wrist
        left_wrist = (landmarks[15].x * w + x1, landmarks[15].y * h + y1)
        right_wrist = (landmarks[16].x * w + x1, landmarks[16].y * h + y1)

        return (left_wrist, right_wrist)

    def match_to_tracked(self, center):
        """Find the closest existing tracked person within MAX_TRACK_DISTANCE."""
        best_id = None
        best_dist = MAX_TRACK_DISTANCE

        for person_id, person in self.tracked_people.items():
            dist = np.linalg.norm(np.array(center) - np.array(person.center))
            if dist < best_dist:
                best_dist = dist
                best_id = person_id

        return best_id

    def update_tracking(self, frame, boxes):
        """
        Match current detections to tracked people (or create new ones),
        compute wrist velocity for each, and age out people who disappeared.
        """
        matched_ids = set()

        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            matched_id = self.match_to_tracked(center)

            if matched_id is not None:
                person = self.tracked_people[matched_id]
                person.bbox = bbox
                person.missed_frames = 0
            else:
                matched_id = self.next_id
                self.next_id += 1
                person = TrackedPerson(matched_id, bbox, center)
                self.tracked_people[matched_id] = person

            person.center = center
            matched_ids.add(matched_id)

            # Compute wrist velocity for this person
            wrists = self.get_wrist_keypoints(frame, bbox)
            if wrists and person.prev_wrist_kpts:
                left_v = np.linalg.norm(np.array(wrists[0]) - np.array(person.prev_wrist_kpts[0]))
                right_v = np.linalg.norm(np.array(wrists[1]) - np.array(person.prev_wrist_kpts[1]))
                person.velocity = max(left_v, right_v)
            else:
                person.velocity = 0.0

            person.prev_wrist_kpts = wrists

        # Age out people not seen this frame
        for person_id in list(self.tracked_people.keys()):
            if person_id not in matched_ids:
                self.tracked_people[person_id].missed_frames += 1
                if self.tracked_people[person_id].missed_frames > MAX_MISSED_FRAMES:
                    del self.tracked_people[person_id]

    def check_for_fights(self):
        """
        Check all pairs of currently tracked people: flag a fight only if a
        pair has been close together AND both showing high velocity for
        MIN_CONSECUTIVE_FLAGS consecutive checks in a row (filters out
        one-off noise spikes from pose estimation jitter).

        Returns:
            List of dicts: {"person_ids": (id1, id2), "distance": float}
        """
        confirmed_pairs = []
        people = list(self.tracked_people.values())
        current_pairs_flagged = set()

        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                p1, p2 = people[i], people[j]
                pair_key = tuple(sorted((p1.id, p2.id)))
                dist = np.linalg.norm(np.array(p1.center) - np.array(p2.center))

                is_flagged_this_frame = (
                    dist < PROXIMITY_THRESHOLD
                    and p1.velocity > VELOCITY_THRESHOLD
                    and p2.velocity > VELOCITY_THRESHOLD
                )

                if is_flagged_this_frame:
                    current_pairs_flagged.add(pair_key)
                    self.pair_flag_counts[pair_key] = self.pair_flag_counts.get(pair_key, 0) + 1

                    if self.pair_flag_counts[pair_key] >= MIN_CONSECUTIVE_FLAGS:
                        confirmed_pairs.append({
                            "person_ids": pair_key,
                            "distance": dist,
                            "velocities": (p1.velocity, p2.velocity),
                            "consecutive_flags": self.pair_flag_counts[pair_key]
                        })

        # Reset counters for pairs not flagged this frame (so the streak must be continuous)
        for pair_key in list(self.pair_flag_counts.keys()):
            if pair_key not in current_pairs_flagged:
                self.pair_flag_counts[pair_key] = 0

        return confirmed_pairs

    def detect_fight(self, frame):
        """
        Main entry point: detect people, update tracking, and check for
        fight-like behavior between any pair of nearby people.

        Returns:
            dict with "flagged" (bool) and "details" (list of flagged pairs)
        """
        boxes = self.detect_people(frame)
        self.update_tracking(frame, boxes)
        flagged_pairs = self.check_for_fights()

        return {
            "flagged": len(flagged_pairs) > 0,
            "details": flagged_pairs
        }


if __name__ == "__main__":
    import sys

    # Accept an optional video path as a command-line argument.
    # Usage: python src/modules/fight_detection.py [path/to/video.mp4]
    # Defaults to test_clip.mp4 if no argument given.
    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_clips/test_clip.mp4"

    detector = FightDetector()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        print("Add a test video to data/sample_clips/ to test this module.")
    else:
        print(f"Testing on: {video_path}\n")
        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Process every 3rd frame for CPU efficiency on longer clips
            if frame_count % 3 != 0:
                continue

            result = detector.detect_fight(frame)
            if result["flagged"]:
                print(f"[Frame {frame_count}] Potential fight detected! Details: {result['details']}")

        elapsed = time.time() - start_time
        print(f"\nFinished processing {frame_count} frames in {elapsed:.1f}s")

        cap.release()