import cv2
import numpy as np
from ultralytics import YOLO

# What faces and landmarks exist in this frame?

class FaceDetection:
    """
    Lightweight container for one detected face
    """
    def __init__(self, bbox, confidence, landmarks):
        self.bbox = bbox              # (x1, y1, x2, y2)
        self.confidence = confidence  # float
        self.landmarks = landmarks    # np.ndarray shape (5, 2)


class YOLOv8FaceDetector:
    """
    YOLOv8 Face + Landmark Detector
    Model: yolov8n-faceL.pt
    """

    def __init__(self, model_path, conf_thres=0.4, device=None):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

        if device:
            self.model.to(device)

    def detect(self, frame):
        """
        Detect faces in a BGR frame

        Returns:
            List[FaceDetection]
        """
        results = self.model(
            frame,
            conf=self.conf_thres,
            verbose=False
        )[0]

        detections = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        if results.keypoints is None:
            return detections  # face-only fallback (should not happen with faceL)

        keypoints = results.keypoints.xy.cpu().numpy()

        for box, score, kps in zip(boxes, scores, keypoints):
            x1, y1, x2, y2 = map(int, box)

            detection = FaceDetection(
                bbox=(x1, y1, x2, y2),
                confidence=float(score),
                landmarks=kps.astype(np.float32)
            )

            detections.append(detection)

        return detections

    @staticmethod
    def draw(frame, detections, draw_landmarks=True):
        """
        Utility method for debugging / visualization
        """
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if draw_landmarks:
                for (x, y) in det.landmarks:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"{det.confidence:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        return frame
