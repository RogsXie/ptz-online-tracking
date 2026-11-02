"""Draw diagnostic telemetry on a copy, never on the detector's input frame."""
import cv2
import numpy as np


def draw_overlay(rgb, measurement, sim_time=0.0):
    image = rgb.copy()
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    green, blue, yellow = (35, 240, 95), (65, 170, 255), (255, 205, 40)
    cv2.drawMarker(image, center, blue, cv2.MARKER_CROSS, 22, 2)
    cv2.rectangle(image, (0, 0), (width - 1, 49), (15, 22, 30), -1)
    cv2.putText(image, "HSV IMAGE-FEEDBACK DEMO  |  NOT A LEARNED POLICY", (14, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    visible = measurement["visible"] and measurement["bbox"] is not None
    status = f"t={sim_time:.1f}s   {'DETECTED' if visible else 'NOT DETECTED'}"
    if visible:
        box = np.asarray(measurement["bbox"], dtype=float)
        left, top, right, bottom = np.rint(box * [width, height, width, height]).astype(int)
        left, right = np.clip([left, right], 0, width - 1)
        top, bottom = np.clip([top, bottom], 0, height - 1)
        target = ((left + right) // 2, (top + bottom) // 2)
        cv2.arrowedLine(image, center, target, yellow, 2, cv2.LINE_AA, tipLength=0.18)
        cv2.rectangle(image, (left, top), (right, bottom), green, 2, cv2.LINE_AA)
        cv2.circle(image, target, 4, green, -1, cv2.LINE_AA)
        cv2.putText(image, "HSV car-body", (left, max(66, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 1, cv2.LINE_AA)
        status += f"   center offset={measurement['center_error']:.3f}"
    cv2.putText(image, status, (14, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, green if visible else (255, 90, 90), 1, cv2.LINE_AA)
    return image
