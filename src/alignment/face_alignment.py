#src/alignment/face_alignment.py
"""
This handles:
rotation,
scale,
translation
in one mathematically stable step.

No manual eye-angle hacks. This is the right way.
"""


import cv2
import numpy as np


# ArcFace standard reference landmarks (112x112)
ARCFACE_LANDMARKS = np.array([
    [38.29, 51.69],   # left eye
    [73.53, 51.69],   # right eye
    [56.02, 71.73],   # nose
    [39.95, 92.36],   # left mouth
    [72.11, 92.36]    # right mouth
], dtype=np.float32)


def align_face(img, landmarks, output_size=(112, 112)):
    """
    Align face using 5-point landmarks to ArcFace canonical geometry.

    img: raw BGR frame from camera
    landmarks: numpy array (5,2) from YOLOv8
    return: aligned 112x112 BGR face
    """

    src_pts = landmarks.astype(np.float32)
    dst_pts = ARCFACE_LANDMARKS.copy()

    # Estimate affine transform
    M, _ = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.LMEDS
    )

    if M is None:
        raise ValueError("Face alignment failed (affine transform)")

    aligned = cv2.warpAffine(
        img,
        M,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderValue=0
    )

    return aligned
