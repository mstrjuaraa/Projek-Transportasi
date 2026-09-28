from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "yolo26n.pt",
)

PROCESS_FPS = float(
    os.getenv(
        "PROCESS_FPS",
        "8",
    )
)

CONF_THRESHOLD = float(
    os.getenv(
        "CONF_THRESHOLD",
        "0.55",
    )
)

TRACKER = os.getenv(
    "TRACKER",
    "bytetrack.yaml",
)


# COCO classes:
# bicycle = 1
# car = 2
# motorcycle = 3
# bus = 5
# truck = 7

TARGET_CLASSES = {
    1: "sepeda",
    2: "mobil",
    3: "motor",
    5: "bus",
    7: "truk",
}


MODEL = None


def get_model():
    global MODEL

    if MODEL is None:
        MODEL = YOLO(MODEL_NAME)

    return MODEL


def horizontal_line(
    y: float | None,
    width: int,
):
    if y is None:
        return None

    y = float(y)

    return (
        (0.0, y),
        (float(width), y),
    )


def line_side(
    point,
    a,
    b,
):
    px, py = point
    ax, ay = a
    bx, by = b

    return (
        (bx - ax) * (py - ay)
        - (by - ay) * (px - ax)
    )


def crossed(
    prev_point,
    curr_point,
    a,
    b,
):
    if prev_point is None or curr_point is None:
        return False

    s1 = line_side(
        prev_point,
        a,
        b,
    )

    s2 = line_side(
        curr_point,
        a,
        b,
    )

    return (
        (s1 == 0)
        or (s2 == 0)
        or ((s1 < 0) != (s2 < 0))
    )


def analyze_video(
    video_path: Path,
    route: str,
    calibration_distance_m: float | None,
    line_a_y: float | None,
    line_b_y: float | None,
):
    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Video tidak dapat dibuka oleh OpenCV."
        )

    fps = (
        cap.get(cv2.CAP_PROP_FPS)
        or 30.0
    )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
        or 0
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        or 0
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        or 0
    )

    duration = (
        frame_count / fps
        if frame_count > 0 and fps > 0
        else 0.0
    )

    process_fps = max(
        1.0,
        min(PROCESS_FPS, fps),
    )

    stride = max(
        1,
        round(fps / process_fps),
    )

    line_a = horizontal_line(
        line_a_y,
        width,
    )

    line_b = horizontal_line(
        line_b_y,
        width,
    )

    speed_distance = float(
        calibration_distance_m or 0.0
    )

    calibrated = bool(
        speed_distance > 0
        and line_a is not None
        and line_b is not None
        and line_a_y != line_b_y
    )

    model = get_model()

    counts = defaultdict(int)

    counted_ids: set[int] = set()

    previous_centers: dict[
        int,
        tuple[float, float],
    ] = {}

    line_a_times: dict[
        int,
        float,
    ] = {}

    line_b_times: dict[
        int,
        float,
    ] = {}

    speed_samples: list[float] = []

    timeline = []

    processed = 0
    frame_index = 0

    started = time.time()

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_index % stride != 0:
            frame_index += 1
            continue

        timestamp = frame_index / fps

        results = model.track(
            source=frame,
            persist=True,
            tracker=TRACKER,
            conf=CONF_THRESHOLD,
            classes=list(
                TARGET_CLASSES.keys()
            ),
            verbose=False,
        )

        r = results[0]

        detections = []

        if (
            r.boxes is not None
            and len(r.boxes) > 0
        ):
            xyxy = (
                r.boxes.xyxy
                .cpu()
                .numpy()
            )

            cls = (
                r.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            conf = (
                r.boxes.conf
                .cpu()
                .numpy()
            )

            ids = (
                r.boxes.id
                .cpu()
                .numpy()
                .astype(int)
                if r.boxes.id is not None
                else None
            )

            for i, (
                box,
                cls_id,
                score,
            ) in enumerate(
                zip(
                    xyxy,
                    cls,
                    conf,
                )
            ):
                if cls_id not in TARGET_CLASSES:
                    continue

                track_id = (
                    int(ids[i])
                    if ids is not None
                    else None
                )

                x1, y1, x2, y2 = map(
                    float,
                    box,
                )

                center = (
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                )

                name = TARGET_CLASSES[
                    cls_id
                ]

                detections.append(
                    {
                        "x1": round(
                            x1 / max(width, 1),
                            5,
                        ),
                        "y1": round(
                            y1 / max(height, 1),
                            5,
                        ),
                        "x2": round(
                            x2 / max(width, 1),
                            5,
                        ),
                        "y2": round(
                            y2 / max(height, 1),
                            5,
                        ),
                        "class": name,
                        "confidence": round(
                            float(score),
                            4,
                        ),
                        "track_id": track_id,
                        "time": round(
                            timestamp,
                            3,
                        ),
                    }
                )

                if track_id is None:
                    continue

                prev = previous_centers.get(
                    track_id
                )

                if (
                    line_a is not None
                    and line_b is not None
                ):
                    if (
                        track_id
                        not in counted_ids
                        and (
                            crossed(
                                prev,
                                center,
                                *line_a,
                            )
                            or crossed(
                                prev,
                                center,
                                *line_b,
                            )
                        )
                    ):
                        counted_ids.add(
                            track_id
                        )
                        counts[name] += 1

                elif track_id not in counted_ids:
                    counted_ids.add(
                        track_id
                    )
                    counts[name] += 1

                if calibrated and prev is not None:
                    if (
                        crossed(
                            prev,
                            center,
                            *line_a,
                        )
                        and track_id
                        not in line_a_times
                    ):
                        line_a_times[
                            track_id
                        ] = timestamp

                    if (
                        crossed(
                            prev,
                            center,
                            *line_b,
                        )
                        and track_id
                        not in line_b_times
                    ):
                        line_b_times[
                            track_id
                        ] = timestamp

                    if (
                        track_id
                        in line_a_times
                        and track_id
                        in line_b_times
                    ):
                        dt = abs(
                            line_b_times[
                                track_id
                            ]
                            - line_a_times[
                                track_id
                            ]
                        )

                        if (
                            0 < dt <= 60
                        ):
                            speed = (
                                speed_distance
                                / dt
                            )

                            if (
                                0
                                < speed
                                < 60
                            ):
                                speed_samples.append(
                                    speed
                                )

                        line_a_times.pop(
                            track_id,
                            None,
                        )

                        line_b_times.pop(
                            track_id,
                            None,
                        )

                previous_centers[
                    track_id
                ] = center

        timeline.append(
            {
                "time": round(
                    timestamp,
                    3,
                ),
                "detections": detections,
            }
        )

        processed += 1
        frame_index += 1

    cap.release()

    total = int(
        sum(counts.values())
    )

    flow = (
        total
        / (duration / 60.0)
        if duration > 0
        else None
    )

    avg_speed = (
        float(
            np.mean(speed_samples)
        )
        if speed_samples
        else None
    )

    if flow is None:
        status = "BELUM TERSEDIA"

    elif flow < 30:
        status = "RENDAH"

    elif flow < 60:
        status = "SEDANG"

    else:
        status = "TINGGI"

    return {
        "route": route,

        "video": {
            "duration_seconds": round(
                duration,
                3,
            ),
            "fps": round(
                fps,
                3,
            ),
            "analysis_fps": round(
                process_fps,
                3,
            ),
            "frames": frame_count,
            "frames_processed": processed,
            "resolution": {
                "width": width,
                "height": height,
            },
        },

        "vehicles": {
            "total": total,
            "motor": int(
                counts["motor"]
            ),
            "mobil": int(
                counts["mobil"]
            ),
            "bus": int(
                counts["bus"]
            ),
            "truk": int(
                counts["truk"]
            ),
            "sepeda": int(
                counts["sepeda"]
            ),
        },

        "traffic": {
            "flow_rate_vehicles_per_minute": (
                round(flow, 3)
                if flow is not None
                else None
            ),
            "status": status,
        },

        "speed": {
            "average_speed_mps": (
                round(
                    avg_speed,
                    3,
                )
                if avg_speed is not None
                else None
            ),
            "sample_count": len(
                speed_samples
            ),
            "status": (
                "terkalibrasi"
                if calibrated
                else "belum terkalibrasi"
            ),
        },

        "queue": {
            "available": False,
            "total": None,
            "status": "belum tersedia",
        },

        "calibration": {
            "distance_m": (
                speed_distance
                if speed_distance > 0
                else None
            ),
            "line_a_y": line_a_y,
            "line_b_y": line_b_y,
            "speed_calibrated": calibrated,
            "counting_line_configured": bool(
                line_a is not None
                and line_b is not None
            ),
        },

        "detections_timeline": timeline,

        "analysis_runtime_seconds": round(
            time.time() - started,
            3,
        ),

        "model": MODEL_NAME,

        "tracker": TRACKER,

        "confidence_threshold": (
            CONF_THRESHOLD
        ),

        "source": "video asli",

        "random_data": False,
    }
