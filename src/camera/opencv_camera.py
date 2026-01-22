#src/camera/opencv_camera.py
import cv2
from src.camera.camera_interface import Camera


class OpenCVCamera(Camera):
    def __init__(self, camera_index=0, backend=None):
        if backend is None:
            self.cap = cv2.VideoCapture(camera_index)
        else:
            self.cap = cv2.VideoCapture(camera_index, backend)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
