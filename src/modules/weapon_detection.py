"""
weapon_detection.py

Module: Weapon Detection
Detects visible weapons (knife/gun) in video frames using a YOLOv8 model.

TODO (Team):
1. Download/fine-tune a YOLOv8n weapon-detection model
   - Option A: Find a pretrained weapon-detection .pt weights file (Roboflow Universe
     has several open weapon-detection datasets you can train YOLOv8n on quickly)
   - Option B: Fine-tune yolov8n.pt on a weapon dataset using Ultralytics' `model.train()`
2. Place the trained weights file at: src/modules/weights/weapon_yolov8n.pt
3. Test detect_weapons() on a sample frame and confirm bounding boxes + confidence scores look right
4. Tune CONFIDENCE_THRESHOLD based on false positive rate during testing
"""

from ultralytics import YOLO

# TODO: update path once you have trained/downloaded weights
MODEL_PATH = "src/modules/weights/weapon_yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5

# Classes we care about (depends on your training dataset's class names)
WEAPON_CLASSES = ["knife", "gun", "pistol", "rifle"]


class WeaponDetector:
    def __init__(self, model_path: str = MODEL_PATH, conf_threshold: float = CONFIDENCE_THRESHOLD):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_weapons(self, frame):
        """
        Run weapon detection on a single video frame.

        Args:
            frame: a single image/frame (numpy array, BGR format from OpenCV)

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

            if conf >= self.conf_threshold and class_name.lower() in WEAPON_CLASSES:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })

        return detections


if __name__ == "__main__":
    # Quick manual test — TODO: replace with a real sample frame/image path
    import cv2

    detector = WeaponDetector()
    test_frame = cv2.imread("data/sample_clips/test_frame.jpg")

    if test_frame is not None:
        results = detector.detect_weapons(test_frame)
        print("Detections:", results)
    else:
        print("No test frame found. Add a test image to data/sample_clips/ to test this module.")
