"""
MediaPipe pose landmark indices and helper functions.
33-point body skeleton reference.
"""

# MediaPipe Pose landmark indices
class LM:
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


import numpy as np


def get_point(landmarks, idx, w, h):
    """Get pixel coordinates of a landmark."""
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def get_point_norm(landmarks, idx):
    """Get normalized [0,1] coordinates."""
    lm = landmarks[idx]
    return np.array([lm.x, lm.y])


def angle_between(a, b, c):
    """
    Compute angle at point B formed by A-B-C.
    Returns angle in degrees.
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def midpoint(a, b):
    return (a + b) / 2


def distance(a, b):
    return np.linalg.norm(a - b)
