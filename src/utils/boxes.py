import numpy as np
import cv2

CLASS_NAME = {
    0: "persion",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def draw_label(
        image: np.ndarray, 
        label: int, 
        x: int, 
        y: int, 
        color: tuple[int, int, int]
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, scale, thickness
    )
    label_y1 = max(0, y - text_height - baseline - 4)
    label_y2 = max(text_height + baseline + 4, y)
    label_x2 = min(image.shape[1] - 1, x + text_width + 6)
    cv2.rectangle(image, (x, label_y1), (label_x2, label_y2), color, -1)
    cv2.putText(
        image,
        label,
        (x + 3, label_y2 - baseline - 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = (
        (56, 56, 255), (151, 157, 255), (31, 112, 255),
        (29, 178, 255), (49, 210, 207), (10, 249, 72),
        (23, 204, 146), (134, 219, 61), (52, 147, 26),
        (187, 212, 0), (168, 153, 44), (255, 194, 0),
        (147, 69, 52), (255, 115, 100), (236, 24, 0),
        (255, 56, 132), (133, 0, 82), (255, 56, 203),
        (200, 149, 255), (199, 55, 255),
    )
    return palette[class_id % len(palette)]
