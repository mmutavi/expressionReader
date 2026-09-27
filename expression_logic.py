"""
Pure geometry, no trained model: distances between mediapipe face-mesh
landmarks decide a rough expression label. Not clinically accurate, just a
fun heuristic.
"""

import math

LEFT_EYE = [386, 385, 387, 373, 380, 362]
RIGHT_EYE = [159, 158, 160, 144, 153, 133]
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
LEFT_EYEBROW = 105
LEFT_EYE_TOP = 159


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _eye_aspect_ratio(landmarks, idxs):
    p = [landmarks[i] for i in idxs]
    vertical = _dist(p[1], p[5]) + _dist(p[2], p[4])
    horizontal = _dist(p[0], p[3])
    return vertical / (2.0 * horizontal + 1e-6)


def classify_expression(landmarks):
    left_ear = _eye_aspect_ratio(landmarks, LEFT_EYE)
    right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE)
    avg_ear = (left_ear + right_ear) / 2

    mouth_height = _dist(landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM])
    mouth_width = _dist(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT])
    mouth_ratio = mouth_height / (mouth_width + 1e-6)

    eyebrow_gap = _dist(landmarks[LEFT_EYEBROW], landmarks[LEFT_EYE_TOP])

    if avg_ear < 0.15:
        return "Eyes closed"
    if mouth_ratio > 0.45 and eyebrow_gap > 0.045:
        return "Surprised"
    if mouth_width > 0.34:
        return "Smiling"
    return "Neutral"
