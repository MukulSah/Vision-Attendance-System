import onnxruntime as ort
import numpy as np
import cv2

class ArcFaceONNX:
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def preprocess(self, face_bgr):
        # face_bgr: aligned face, 112x112

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face = face_rgb.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5           # ArcFace normalization
        face = np.transpose(face, (2, 0, 1))  # HWC → CHW
        face = np.expand_dims(face, axis=0)   # (1,3,112,112)
        return face

    def get_embedding(self, aligned_face):
        blob = self.preprocess(aligned_face)

        emb = self.sess.run(
            [self.output_name],
            {self.input_name: blob}
        )[0][0]   # shape (512,)

        # L2 normalize (CRITICAL)
        emb = emb / np.linalg.norm(emb)
        return emb
