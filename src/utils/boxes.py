import numpy as np
import cv2
from ..models.detection import Detection
from ..models.track import Track
from collections.abc import Sequence


CLASS_NAME = {
    0: "persion",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def draw_label(
        image: np.ndarray, 
        label: str,
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

def preprocess(frame_bgr: np.ndarray, input_size: int):
        image, scale, pad_x, pad_y = _letterbox(frame_bgr, input_size)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2,0,1))
        image = np.expand_dims(image, axis=0)

        transport = {
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y
        }
        return image, transport

def _letterbox(image: np.ndarray, input_size: int):
    h, w = image.shape[:2]
    size = input_size

    scale = min(size / h, size / w)
    new_w = int(round(w*scale))
    new_h = int(round(h*scale))
    resized = cv2.resize(image, (new_w, new_h))

    pad_w = size - new_w
    pad_h = size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114,114,114)
    )
    return padded, scale, left, top

def postprocess(
        outputs: np.ndarray | Sequence[np.ndarray],
        frame_shape: tuple[int, int],
        transport: dict,
        confidence_threshold: float,
        iou_threshold: float,
        class_ids: list[int] | None = None,
):
    output = outputs[0] if isinstance(outputs, Sequence) else outputs

    if output.ndim == 3:
        output = output[0]

    if output.shape[0] < output.shape[1]:
        output = output.T

    if output.shape[1] <= 4:
        return []

    raw_boxes = output[:, :4]
    class_scores = output[:, 4:]
    class_ids_array = np.argmax(class_scores, axis=1)
    scores_array = class_scores[
        np.arange(class_scores.shape[0]),
        class_ids_array,
    ]

    keep_mask = scores_array >= confidence_threshold
    if class_ids is not None:
        keep_mask &= np.isin(class_ids_array, class_ids)

    if not np.any(keep_mask):
        return []

    raw_boxes = raw_boxes[keep_mask]
    scores_array = scores_array[keep_mask]
    class_ids_array = class_ids_array[keep_mask]

    h, w = frame_shape
    cx = raw_boxes[:, 0]
    cy = raw_boxes[:, 1]
    box_width = raw_boxes[:, 2]
    box_height = raw_boxes[:, 3]

    x1 = (cx - box_width / 2 - transport["pad_x"]) / transport["scale"]
    y1 = (cy - box_height / 2 - transport["pad_y"]) / transport["scale"]
    x2 = (cx + box_width / 2 - transport["pad_x"]) / transport["scale"]
    y2 = (cy + box_height / 2 - transport["pad_y"]) / transport["scale"]

    x1 = np.clip(x1, 0, w - 1).astype(np.int32)
    y1 = np.clip(y1, 0, h - 1).astype(np.int32)
    x2 = np.clip(x2, 0, w - 1).astype(np.int32)
    y2 = np.clip(y2, 0, h - 1).astype(np.int32)

    valid_boxes = (x2 > x1) & (y2 > y1)
    if not np.any(valid_boxes):
        return []

    x1 = x1[valid_boxes]
    y1 = y1[valid_boxes]
    x2 = x2[valid_boxes]
    y2 = y2[valid_boxes]
    scores_array = scores_array[valid_boxes]
    class_ids_array = class_ids_array[valid_boxes]

    boxes_array = np.column_stack((x1, y1, x2 - x1, y2 - y1))
    boxes = boxes_array.tolist()
    scores = scores_array.astype(float).tolist()
    class_ids = class_ids_array.astype(int).tolist()

    keep = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        confidence_threshold,
        iou_threshold
    )
    detections = []
    for idx in np.array(keep).flatten():
        x, y, box_w, box_h = boxes[idx]

        detections.append(
            Detection(
                x1=x,
                y1=y,
                x2=x + box_w,
                y2=y + box_h,
                conf=scores[idx],
                class_id=class_ids[idx]
            )
        )
    return detections 


def draw_detection(image: np.ndarray, detection: Detection) -> None:
    color = color_for_class(detection.class_id)
    cv2.rectangle(
        image,
        (detection.x1, detection.y1),
        (detection.x2, detection.y2),
        color,
        2,
    )
    class_name = CLASS_NAME.get(detection.class_id, str(detection.class_id))
    draw_label(image, f"{class_name} {detection.conf:.2f}", detection.x1, detection.y1, color)


def draw_track(image: np.ndarray, track: Track) -> None:
    color = color_for_class(track.track_id)
    cv2.rectangle(
        image,
        (track.x1, track.y1),
        (track.x2, track.y2),
        color,
        2,
    )
    label = f"ID {track.track_id}"
    if track.class_id is not None:
        label = f"{CLASS_NAME.get(track.class_id, track.class_id)} {label}"
    draw_label(image, label, track.x1, track.y1, color)
