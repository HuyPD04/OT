from __future__ import annotations

import math

Box = tuple[float, float, float, float]


def box_iou(box_a: Box, box_b: Box) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def normalized_center_distance(box_a: Box, box_b: Box) -> float:
    center_a_x = (box_a[0] + box_a[2]) * 0.5
    center_a_y = (box_a[1] + box_a[3]) * 0.5
    center_b_x = (box_b[0] + box_b[2]) * 0.5
    center_b_y = (box_b[1] + box_b[3]) * 0.5

    distance = math.hypot(center_a_x - center_b_x, center_a_y - center_b_y)
    width = max(1.0, ((box_a[2] - box_a[0]) + (box_b[2] - box_b[0])) * 0.5)
    height = max(1.0, ((box_a[3] - box_a[1]) + (box_b[3] - box_b[1])) * 0.5)
    return distance / max(width, height)
