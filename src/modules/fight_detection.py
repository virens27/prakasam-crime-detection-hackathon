"""
fight_detection.py

Module: Fight / Assault Detection
Uses MediaPipe Pose to estimate body keypoints for people in frame, then flags
"fight-like" behavior based on rapid limb movement between people in close proximity.

Approach (simple, CPU-friendly heuristic — not a trained action-recognition model):
1. Detect pose keypoints for each person in the frame (MediaPipe Pose works best
   single-person at a time — for multiple people, you may need to run it per
   cropped person region, e.g. using a person detector first, or test MediaPipe's
   multi-person capabilities depending on version).
2. Track keypoint positions across consecutive frames.
3. Compute movement velocity of wrists/arms between frames.
4. If two people are within a close distance AND show high limb velocity
   simultaneously -> flag as potential fight.

TODO (Team):
1. Decide on multi-person handling strategy (see note above)
2. Tune VELOCITY_THRESHOLD and PROXIMITY_THRESHOLD using real sample clips
3. Test against both real "fight" clips and normal "crowd walking" clips to
   check false positive rate (this is the trickiest part — expect to iterate)
"""

import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

# TODO: tune these thresholds based on testing with real sample footage
VELOCITY_THRESHOLD = 40    # pixels/frame movement considered "rapid"
PROXIMITY_THRESHOLD = 150  # pixels between two people considered "close"


class FightDetector:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.prev_keypoints = None  # store previous frame's keypoints for velocity calc

    def get_keypoints(self, frame):
        """
        Extract pose landmarks (keypoints) from a frame.

        Returns:
            List of (x, y) keypoint coordinates, or None if no person detected.
        """
        rgb_frame = frame[:, :, ::-1]  # BGR -> RGB for MediaPipe
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        h, w, _ = frame.shape
        keypoints = [(lm.x * w, lm.y * h) for lm in results.pose_landmarks.landmark]
        return keypoints

    def compute_velocity(self, current_kpts, prev_kpts):
        """Compute average movement velocity of wrist keypoints between frames."""
        if prev_kpts is None or current_kpts is None:
            return 0.0

        # MediaPipe Pose indices: 15 = left wrist, 16 = right wrist
        wrist_indices = [15, 16]
        velocities = []

        for idx in wrist_indices:
            if idx < len(current_kpts) and idx < len(prev_kpts):
                dx = current_kpts[idx][0] - prev_kpts[idx][0]
                dy = current_kpts[idx][1] - prev_kpts[idx][1]
                velocities.append(np.sqrt(dx**2 + dy**2))

        return np.mean(velocities) if velocities else 0.0

    def detect_fight(self, frame):
        """
        Analyze a single frame and flag potential fight behavior.

        NOTE: This single-frame version only handles ONE person's keypoints
        as a starting skeleton. TODO: extend to multi-person comparison
        (proximity + simultaneous high velocity between two people) for a
        true "fight" signal rather than just "rapid movement".

        Returns:
            dict with "flagged" (bool) and "velocity" (float)
        """
        keypoints = self.get_keypoints(frame)
        velocity = self.compute_velocity(keypoints, self.prev_keypoints)
        self.prev_keypoints = keypoints

        flagged = velocity > VELOCITY_THRESHOLD

        return {
            "flagged": flagged,
            "velocity": velocity
        }


if __name__ == "__main__":
    # Quick manual test — TODO: replace with a real sample video path
    import cv2

    detector = FightDetector()
    cap = cv2.VideoCapture("data/sample_clips/test_clip.mp4")

    if not cap.isOpened():
        print("No test clip found. Add a test video to data/sample_clips/ to test this module.")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result = detector.detect_fight(frame)
            if result["flagged"]:
                print(f"Potential fight detected! Velocity: {result['velocity']:.2f}")
        cap.release()
