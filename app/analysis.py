from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "yolo26s.pt",
)

PROCESS_FPS = float(
    os.getenv(
        "PROCESS_FPS",
        "10",
    )
)

CONF_THRESHOLD = float(
    os.getenv(
        "CONF_THRESHOLD",
        "0.55",
    )
)

TRACK_CONF_THRESHOLD = float(
    os.getenv(
        "TRACK_CONF_THRESHOLD",
        "0.40",
    )
)

MOTOR_MIN_CONF = float(
    os.getenv(
        "MOTOR_MIN_CONF",
        "0.40",
    )
)

OTHER_MIN_CONF = float(
    os.getenv(
        "OTHER_MIN_CONF",
        "0.55",
    )
)

INFER_SIZE = int(
    os.getenv(
        "INFER_SIZE",
        "1280",
    )
)

TRACKER = os.getenv(
    "TRACKER",
    "bytetrack.yaml",
)


# ============================================================
# COCO VEHICLE CLASSES
# ============================================================

TARGET_CLASSES = {
    1: "sepeda",
    2: "mobil",
    3: "motor",
    5: "bus",
    7: "truk",
}


# ============================================================
# MODEL
# ============================================================

MODEL = None


def get_model():
    global MODEL

    if MODEL is None:
        MODEL = YOLO(MODEL_NAME)

    return MODEL


# ============================================================
# LINE HELPERS
# ============================================================

def horizontal_line(
    y: float | None,
    width: int,
):
    if y is None:
        return None

    return (
        (0.0, float(y)),
        (float(width), float(y)),
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
    previous_point,
    current_point,
    a,
    b,
):
    if (
        previous_point is None
        or current_point is None
    ):
        return False

    side_previous = line_side(
        previous_point,
        a,
        b,
    )

    side_current = line_side(
        current_point,
        a,
        b,
    )

    return (
        side_previous == 0
        or side_current == 0
        or (
            (side_previous < 0)
            != (side_current < 0)
        )
    )


# ============================================================
# VIDEO ANALYSIS
# ============================================================

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

    # --------------------------------------------------------
    # VIDEO METADATA
    # --------------------------------------------------------

    fps = (
        cap.get(cv2.CAP_PROP_FPS)
        or 30.0
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
        or 0
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
        or 0
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
        or 0
    )

    duration = (
        frame_count / fps
        if frame_count > 0
        and fps > 0
        else 0.0
    )

    # --------------------------------------------------------
    # FRAME SAMPLING
    # --------------------------------------------------------

    process_fps = max(
        1.0,
        min(
            PROCESS_FPS,
            fps,
        ),
    )

    stride = max(
        1,
        round(
            fps / process_fps
        ),
    )

    # --------------------------------------------------------
    # COUNTING / SPEED LINES
    # --------------------------------------------------------

    line_a = horizontal_line(
        line_a_y,
        width,
    )

    line_b = horizontal_line(
        line_b_y,
        width,
    )

    distance_m = float(
        calibration_distance_m or 0.0
    )

    calibrated = bool(
        distance_m > 0
        and line_a is not None
        and line_b is not None
        and line_a_y != line_b_y
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = get_model()

    # --------------------------------------------------------
    # NORMAL TRACK DATA
    # --------------------------------------------------------

    previous_centers = {}

    # Setiap track ID mewakili satu kendaraan yang dihitung satu kali.
    # Line A/B hanya digunakan untuk crossing dan perhitungan kecepatan.
    track_class_votes = defaultdict(lambda: defaultdict(int))
    track_observations = defaultdict(int)
    track_confidence_sum = defaultdict(float)
    track_first_center = {}
    track_last_center = {}

    line_a_times = {}

    line_b_times = {}

    speed_samples = []

    # --------------------------------------------------------
    # FALLBACK MOTOR TRACKING
    # --------------------------------------------------------

    fallback_motor_tracks = {}

    next_fallback_motor_id = -1

    # --------------------------------------------------------
    # DEBUG RAW DETECTION COUNTS
    # --------------------------------------------------------

    raw_detection_counts = {
        "motor": 0,
        "mobil": 0,
        "bus": 0,
        "truk": 0,
        "sepeda": 0,
    }

    # --------------------------------------------------------
    # TIMELINE UNTUK FRONTEND
    # --------------------------------------------------------

    timeline = []

    processed_frames = 0
    frame_index = 0

    started_at = time.time()

    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Hanya proses sebagian frame agar CPU tidak terlalu
        # berat.
        if frame_index % stride != 0:
            frame_index += 1
            continue

        timestamp = (
            frame_index / fps
            if fps > 0
            else 0.0
        )

        # ====================================================
        # YOLO + BYTE TRACK
        # ====================================================

        results = model.track(
            source=frame,
            persist=True,
            tracker=TRACKER,
            conf=TRACK_CONF_THRESHOLD,
            imgsz=INFER_SIZE,
            max_det=100,
            classes=list(
                TARGET_CLASSES.keys()
            ),
            verbose=False,
        )

        result = results[0]

        detections = []

        # ====================================================
        # EXTRACT BOXES
        # ====================================================

        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            if result.boxes.id is not None:

                track_ids = (
                    result.boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

            else:

                track_ids = None

            # =================================================
            # EACH DETECTION
            # =================================================

            for i, (
                box,
                class_id,
                confidence,
            ) in enumerate(
                zip(
                    boxes,
                    classes,
                    confidences,
                )
            ):

                if class_id not in TARGET_CLASSES:
                    continue

                class_name = TARGET_CLASSES[
                    class_id
                ]

                # -------------------------------------------------
                # CLASS-SPECIFIC CONFIDENCE FILTER
                # -------------------------------------------------
                # Motor dibuat sedikit lebih permisif karena kendaraan
                # kecil pada CCTV sering memiliki confidence lebih rendah.
                min_conf = (
                    MOTOR_MIN_CONF
                    if class_name == "motor"
                    else OTHER_MIN_CONF
                )

                if float(confidence) < min_conf:
                    continue

                # -------------------------------------------------
                # RAW/ACCEPTED DETECTION COUNTER
                # -------------------------------------------------

                raw_detection_counts[
                    class_name
                ] += 1

                # -------------------------------------------------
                # BBOX
                # -------------------------------------------------

                x1, y1, x2, y2 = map(
                    float,
                    box,
                )

                center = (
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                )

                # -------------------------------------------------
                # TRACK ID
                # -------------------------------------------------

                if track_ids is not None:

                    track_id = int(
                        track_ids[i]
                    )

                else:

                    track_id = None

                # =================================================
                # TRACK ID SAFETY
                # =================================================
                # Untuk counting yang konsisten, hanya gunakan ID yang
                # benar-benar diberikan ByteTrack. Jangan membuat ID
                # fallback yang dapat menggandakan satu kendaraan.
                if track_id is None:
                    continue

                # Simpan statistik track untuk validasi akhir.
                track_observations[track_id] += 1
                track_confidence_sum[track_id] += float(confidence)
                track_first_center.setdefault(track_id, center)
                track_last_center[track_id] = center

                # =================================================
                # SAVE DETECTION
                # =================================================

                detections.append(
                    {
                        "x1": round(
                            x1 / max(width, 1),
                            6,
                        ),
                        "y1": round(
                            y1 / max(height, 1),
                            6,
                        ),
                        "x2": round(
                            x2 / max(width, 1),
                            6,
                        ),
                        "y2": round(
                            y2 / max(height, 1),
                            6,
                        ),
                        "class": class_name,
                        "confidence": round(
                            float(confidence),
                            4,
                        ),
                        "track_id": track_id,
                        "time": round(
                            timestamp,
                            3,
                        ),
                    }
                )

                # =================================================
                # PREVIOUS CENTER
                # =================================================

                previous_center = (
                    previous_centers.get(
                        track_id
                    )
                )

                # =================================================
                # COUNTING BERDASARKAN UNIQUE TRACK ID
                # =================================================
                # Jangan tunggu kendaraan melewati Line A/B.
                # Line A/B hanya untuk crossing dan speed.
                # Simpan suara kelas untuk setiap track agar perubahan
                # label sesaat tidak membuat satu kendaraan dihitung
                # sebagai dua kelas.
                track_class_votes[
                    track_id
                ][
                    class_name
                ] += 1

                # =================================================
                # SPEED
                # =================================================

                if (
                    calibrated
                    and previous_center is not None
                ):

                    crossed_a = crossed(
                        previous_center,
                        center,
                        *line_a,
                    )

                    crossed_b = crossed(
                        previous_center,
                        center,
                        *line_b,
                    )

                    if (
                        crossed_a
                        and track_id
                        not in line_a_times
                    ):

                        line_a_times[
                            track_id
                        ] = timestamp

                    if (
                        crossed_b
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

                        delta_t = abs(
                            line_b_times[
                                track_id
                            ]
                            -
                            line_a_times[
                                track_id
                            ]
                        )

                        if (
                            0
                            < delta_t
                            <= 60
                        ):

                            speed_mps = (
                                distance_m
                                / delta_t
                            )

                            if (
                                0
                                < speed_mps
                                < 60
                            ):

                                speed_samples.append(
                                    speed_mps
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

        # ========================================================
        # SAVE TIMELINE
        # ========================================================

        timeline.append(
            {
                "time": round(
                    timestamp,
                    3,
                ),
                "detections": detections,
            }
        )

        processed_frames += 1
        frame_index += 1

    cap.release()

    # ============================================================
    # FINAL UNIQUE TRACK COUNTER
    # ============================================================

    counted_classes = defaultdict(int)
    accepted_motor_tracks = []
    rejected_motor_tracks = []

    for track_id, votes in track_class_votes.items():
        if not votes:
            continue

        # Ambil kelas yang paling sering terlihat pada track tersebut.
        best_class = max(
            votes.items(),
            key=lambda item: item[1],
        )[0]

        observations = int(track_observations.get(track_id, 0))
        mean_conf = (
            track_confidence_sum.get(track_id, 0.0) / observations
            if observations > 0
            else 0.0
        )

        first_center = track_first_center.get(track_id)
        last_center = track_last_center.get(track_id)

        motion_px = 0.0
        if first_center is not None and last_center is not None:
            motion_px = float(
                (
                    (last_center[0] - first_center[0]) ** 2
                    + (last_center[1] - first_center[1]) ** 2
                ) ** 0.5
            )

        # Untuk motor gunakan filter konservatif:
        # harus terlihat minimal 3 frame, confidence rata-rata >= 0.40,
        # dan bergerak setidaknya 15 pixel agar track sangat singkat/
        # false positive tidak langsung menjadi kendaraan.
        if best_class == "motor":
            accepted = (
                observations >= 3
                and mean_conf >= 0.40
                and motion_px >= 15.0
            )

            debug_item = {
                "track_id": int(track_id),
                "observations": observations,
                "mean_confidence": round(mean_conf, 3),
                "motion_px": round(motion_px, 1),
            }

            if accepted:
                accepted_motor_tracks.append(debug_item)
                counted_classes["motor"] += 1
            else:
                rejected_motor_tracks.append(debug_item)
        else:
            counted_classes[best_class] += 1

    # ============================================================
    # FINAL METRICS
    # ============================================================

    total = int(
        sum(
            counted_classes.values()
        )
    )

    if duration > 0:

        flow_rate = (
            total
            / (duration / 60.0)
        )

    else:

        flow_rate = None

    if speed_samples:

        average_speed = float(
            np.mean(
                speed_samples
            )
        )

    else:

        average_speed = None

    # ============================================================
    # TRAFFIC STATUS
    # ============================================================

    if flow_rate is None:

        traffic_status = (
            "BELUM TERSEDIA"
        )

    elif flow_rate < 30:

        traffic_status = "RENDAH"

    elif flow_rate < 60:

        traffic_status = "SEDANG"

    else:

        traffic_status = "TINGGI"

    # ============================================================
    # RESULT
    # ============================================================

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
            "frames_processed": processed_frames,
            "resolution": {
                "width": width,
                "height": height,
            },
        },

        "vehicles": {
            "total": total,

            "motor": int(
                counted_classes[
                    "motor"
                ]
            ),

            "mobil": int(
                counted_classes[
                    "mobil"
                ]
            ),

            "bus": int(
                counted_classes[
                    "bus"
                ]
            ),

            "truk": int(
                counted_classes[
                    "truk"
                ]
            ),

            "sepeda": int(
                counted_classes[
                    "sepeda"
                ]
            ),
        },

        # Ini untuk membedakan:
        # detector menemukan motor atau tidak.
        "unique_track_count": int(len(track_class_votes)),
        "motor_track_debug": {
            "accepted": accepted_motor_tracks,
            "rejected": rejected_motor_tracks,
        },

        "tracker_confidence_threshold": TRACK_CONF_THRESHOLD,
        "motor_min_confidence": MOTOR_MIN_CONF,
        "other_min_confidence": OTHER_MIN_CONF,

        "raw_detection_counts": {
            "motor": int(
                raw_detection_counts[
                    "motor"
                ]
            ),

            "mobil": int(
                raw_detection_counts[
                    "mobil"
                ]
            ),

            "bus": int(
                raw_detection_counts[
                    "bus"
                ]
            ),

            "truk": int(
                raw_detection_counts[
                    "truk"
                ]
            ),

            "sepeda": int(
                raw_detection_counts[
                    "sepeda"
                ]
            ),
        },

        "traffic": {
            "flow_rate_vehicles_per_minute": (
                round(
                    flow_rate,
                    3,
                )
                if flow_rate is not None
                else None
            ),
            "status": traffic_status,
        },

        "speed": {
            "average_speed_mps": (
                round(
                    average_speed,
                    3,
                )
                if average_speed is not None
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
                distance_m
                if distance_m > 0
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
            time.time()
            - started_at,
            3,
        ),

        "model": MODEL_NAME,

        "tracker": TRACKER,

        "confidence_threshold": (
            CONF_THRESHOLD
        ),
        "tracker_confidence_threshold": TRACK_CONF_THRESHOLD,
        "motor_min_confidence": MOTOR_MIN_CONF,
        "other_min_confidence": OTHER_MIN_CONF,

        "inference_image_size": (
            INFER_SIZE
        ),

        "source": "video asli",

        "provenance": (
            "AI YOLO + ByteTrack"
        ),

        "random_data": False,
    }
