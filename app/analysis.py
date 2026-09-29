from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "yolo26s.pt",
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

# Ukuran inferensi diperbesar supaya kendaraan kecil
# seperti motor lebih mudah terdeteksi.
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
# COCO CLASS MAPPING
# ============================================================
#
# COCO:
# 1 = bicycle
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
#

TARGET_CLASSES = {
    1: "sepeda",
    2: "mobil",
    3: "motor",
    5: "bus",
    7: "truk",
}


# Model dibuat sekali dan dipakai kembali.
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
    if (
        prev_point is None
        or curr_point is None
    ):
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


# ============================================================
# MAIN VIDEO ANALYSIS
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

    fps = cap.get(
        cv2.CAP_PROP_FPS
    ) or 30.0

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

    # ========================================================
    # FRAME SAMPLING
    # ========================================================

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

    # ========================================================
    # CALIBRATION
    # ========================================================

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

    # ========================================================
    # MODEL
    # ========================================================

    model = get_model()

    # ========================================================
    # COUNT STORAGE
    # ========================================================

    counts = defaultdict(int)

    # Track ID yang sudah dihitung.
    counted_ids: set[int] = set()

    # Center terakhir dari setiap track.
    previous_centers = {}

    # ========================================================
    # MOTOR FALLBACK TRACKING
    # ========================================================
    #
    # Kadang detector menemukan motor tetapi ByteTrack belum
    # memberikan track_id. Karena itu motor diberi temporary
    # fallback ID berdasarkan kedekatan posisi antarfame.
    #

    fallback_motor_tracks = {}

    next_fallback_motor_id = -1

    # ========================================================
    # SPEED STORAGE
    # ========================================================

    line_a_times = {}
    line_b_times = {}

    speed_samples = []

    # ========================================================
    # DEBUG / DETECTION STORAGE
    # ========================================================

    # Jumlah raw detection dari YOLO, sebelum counting.
    raw_detection_counts = {
        "motor": 0,
        "mobil": 0,
        "bus": 0,
        "truk": 0,
        "sepeda": 0,
    }

    # Timeline untuk frontend bounding box.
    timeline = []

    processed = 0
    frame_index = 0

    started = time.time()

    # ========================================================
    # VIDEO LOOP
    # ========================================================

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        # Sampling frame.
        if frame_index % stride != 0:
            frame_index += 1
            continue

        timestamp = (
            frame_index / fps
            if fps > 0
            else 0.0
        )

        # ----------------------------------------------------
        # YOLO + BYTE TRACK
        # ----------------------------------------------------

        # ========================================================
# UPSCALE FRAME UNTUK OBJEK KECIL
# ========================================================

scale = 2.0

upscaled_frame = cv2.resize(
    frame,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_CUBIC,
)

results = model.track(
    source=upscaled_frame,
    persist=True,
    tracker=TRACKER,
    conf=CONF_THRESHOLD,
    imgsz=INFER_SIZE,
    max_det=100,
    classes=list(
        TARGET_CLASSES.keys()
    ),
    verbose=False,
)

        result = results[0]

        detections = []

        # ----------------------------------------------------
        # RAW BOXES
        # ----------------------------------------------------

        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):
            xyxy = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            class_ids = (
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

            # ------------------------------------------------
            # PROCESS EACH DETECTION
            # ------------------------------------------------

            for i, (
                box,
                class_id,
                score,
            ) in enumerate(
                zip(
                    xyxy,
                    class_ids,
                    confidences,
                )
            ):
                if class_id not in TARGET_CLASSES:
                    continue

                name = TARGET_CLASSES[
                    class_id
                ]

                # -----------------------------
                # RAW DETECTION DEBUG COUNTER
                # -----------------------------

                raw_detection_counts[
                    name
                ] += 1

                # -----------------------------
                # BBOX
                # -----------------------------

               # Kembalikan koordinat bbox ke ukuran frame asli
x1, y1, x2, y2 = map(
    float,
    box / scale,
)

                center = (
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                )

                # -----------------------------
                # TRACK ID
                # -----------------------------

                if track_ids is not None:
                    track_id = int(
                        track_ids[i]
                    )
                else:
                    track_id = None

                # ==================================================
                # MOTOR FALLBACK
                # ==================================================
                #
                # Hanya motor yang memakai fallback.
                #
                # Tujuannya:
                # YOLO mendeteksi motor
                # -> ByteTrack tidak memberi ID
                # -> motor tetap bisa dilacak sementara
                #

                if track_id is None:

                    if name == "motor":

                        # Hapus fallback lama yang sudah
                        # terlalu jauh tidak terlihat.
                        stale_ids = []

                        for (
                            fallback_id,
                            state,
                        ) in fallback_motor_tracks.items():
                            if (
                                timestamp
                                - state["last_seen"]
                                > 2.0
                            ):
                                stale_ids.append(
                                    fallback_id
                                )

                        for fallback_id in stale_ids:
                            fallback_motor_tracks.pop(
                                fallback_id,
                                None,
                            )

                        # Cari motor terdekat dari posisi
                        # motor pada frame sebelumnya.
                        best_id = None
                        best_distance = 100.0

                        for (
                            fallback_id,
                            state,
                        ) in fallback_motor_tracks.items():

                            last_center = (
                                state["center"]
                            )

                            distance_px = (
                                (
                                    center[0]
                                    - last_center[0]
                                )
                                ** 2
                                +
                                (
                                    center[1]
                                    - last_center[1]
                                )
                                ** 2
                            ) ** 0.5

                            if (
                                distance_px
                                < best_distance
                            ):
                                best_distance = (
                                    distance_px
                                )
                                best_id = (
                                    fallback_id
                                )

                        # Kalau tidak ada motor lama
                        # yang cukup dekat, buat ID baru.
                        if best_id is None:

                            track_id = (
                                next_fallback_motor_id
                            )

                            next_fallback_motor_id -= 1

                            fallback_motor_tracks[
                                track_id
                            ] = {
                                "center": center,
                                "last_seen": timestamp,
                                "counted": False,
                            }

                        else:

                            track_id = best_id

                            fallback_motor_tracks[
                                track_id
                            ]["center"] = center

                            fallback_motor_tracks[
                                track_id
                            ]["last_seen"] = timestamp

                    else:
                        # Untuk kelas lain tetap mengikuti
                        # tracking normal.
                        continue

                # ==================================================
                # SAVE DETECTION
                # ==================================================

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

                # ==================================================
                # PREVIOUS CENTER
                # ==================================================

                previous_center = (
                    previous_centers.get(
                        track_id
                    )
                )

                # ==================================================
                # COUNTING
                # ==================================================

                # Kalau Line A dan Line B tersedia,
                # kendaraan dihitung ketika melewati
                # salah satu garis.

                if (
                    line_a is not None
                    and line_b is not None
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
                        track_id
                        not in counted_ids
                        and (
                            crossed_a
                            or crossed_b
                        )
                    ):

                        counted_ids.add(
                            track_id
                        )

                        counts[name] += 1

                else:
                    # Bila line belum digunakan,
                    # hitung track unik sekali.

                    if (
                        track_id
                        not in counted_ids
                    ):

                        counted_ids.add(
                            track_id
                        )

                        counts[name] += 1

                # ==================================================
                # SPEED
                # ==================================================

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

                        dt = abs(
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
                            < dt
                            <= 60
                        ):

                            speed = (
                                speed_distance
                                / dt
                            )

                            # Hindari hasil ekstrim
                            # yang tidak masuk akal.
                            if (
                                0 < speed < 60
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

                # Simpan posisi terbaru.
                previous_centers[
                    track_id
                ] = center

        # ========================================================
        # TIMELINE
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

        processed += 1
        frame_index += 1

    cap.release()

    # ============================================================
    # FINAL METRICS
    # ============================================================

    total = int(
        sum(
            counts.values()
        )
    )

    flow = (
        total
        / (duration / 60.0)
        if duration > 0
        else None
    )

    average_speed = (
        float(
            np.mean(
                speed_samples
            )
        )
        if speed_samples
        else None
    )

    # ============================================================
    # TRAFFIC STATUS
    # ============================================================

    if flow is None:
        traffic_status = (
            "BELUM TERSEDIA"
        )

    elif flow < 30:
        traffic_status = "RENDAH"

    elif flow < 60:
        traffic_status = "SEDANG"

    else:
        traffic_status = "TINGGI"

    # ============================================================
    # RESULT
    # ============================================================

    return {
        "route": route,

        # --------------------------------------------------------
        # VIDEO
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # VEHICLES
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # RAW YOLO DETECTION DEBUG
        # --------------------------------------------------------
        #
        # Ini sangat penting untuk tes motor.
        #
        # Misalnya:
        # raw motor = 25
        # counted motor = 0
        #
        # berarti YOLO melihat motor tetapi counting gagal.
        #
        # Kalau:
        # raw motor = 0
        #
        # berarti YOLO memang tidak mendeteksi motor.
        #

        "raw_detection_counts": {
            "motor": int(
                raw_detection_counts["motor"]
            ),
            "mobil": int(
                raw_detection_counts["mobil"]
            ),
            "bus": int(
                raw_detection_counts["bus"]
            ),
            "truk": int(
                raw_detection_counts["truk"]
            ),
            "sepeda": int(
                raw_detection_counts["sepeda"]
            ),
        },

        # --------------------------------------------------------
        # TRAFFIC
        # --------------------------------------------------------

        "traffic": {
            "flow_rate_vehicles_per_minute": (
                round(
                    flow,
                    3,
                )
                if flow is not None
                else None
            ),
            "status": traffic_status,
        },

        # --------------------------------------------------------
        # SPEED
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # QUEUE
        # --------------------------------------------------------

        "queue": {
            "available": False,
            "total": None,
            "status": "belum tersedia",
        },

        # --------------------------------------------------------
        # CALIBRATION
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # BOUNDING BOX TIMELINE
        # --------------------------------------------------------

        "detections_timeline": timeline,

        # --------------------------------------------------------
        # ANALYSIS INFO
        # --------------------------------------------------------

        "analysis_runtime_seconds": round(
            time.time()
            - started,
            3,
        ),

        "model": MODEL_NAME,

        "tracker": TRACKER,

        "confidence_threshold": CONF_THRESHOLD,

        "inference_image_size": INFER_SIZE,

        "process_fps": process_fps,

        # --------------------------------------------------------
        # PROVENANCE
        # --------------------------------------------------------

        "source": "video asli",

        "provenance": (
            "AI YOLO + ByteTrack"
        ),

        "random_data": False,
    }
