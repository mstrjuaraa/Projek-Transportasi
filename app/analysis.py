from __future__ import annotations

import os
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = os.getenv("MODEL_NAME", "yolo26s.pt")
PROCESS_FPS = float(os.getenv("PROCESS_FPS", "10"))
TRACK_CONF_THRESHOLD = float(os.getenv("TRACK_CONF_THRESHOLD", "0.40"))
MOTOR_MIN_CONF = float(os.getenv("MOTOR_MIN_CONF", "0.40"))
OTHER_MIN_CONF = float(os.getenv("OTHER_MIN_CONF", "0.55"))
INFER_SIZE = int(os.getenv("INFER_SIZE", "1280"))
TRACKER = os.getenv("TRACKER", "bytetrack.yaml")

# Counting utama: kendaraan dihitung 1x ketika melewati Line A.
# Bila Line A tidak diberikan, fallback ke unique tracked IDs.
COUNT_ON_LINE_A = os.getenv("COUNT_ON_LINE_A", "true").lower() in {"1", "true", "yes"}

# Progress stream tiap beberapa frame yang diproses.
PROGRESS_EVERY_FRAMES = int(os.getenv("PROGRESS_EVERY_FRAMES", "5"))


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
def horizontal_line(y: float | None, width: int):
    if y is None:
        return None
    return ((0.0, float(y)), (float(width), float(y)))


def line_side(point, a, b):
    px, py = point
    ax, ay = a
    bx, by = b
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def crossed(previous_point, current_point, a, b):
    if previous_point is None or current_point is None:
        return False
    prev = line_side(previous_point, a, b)
    curr = line_side(current_point, a, b)
    return prev == 0 or curr == 0 or ((prev < 0) != (curr < 0))


# ============================================================
# HELPERS
# ============================================================
def _empty_classes():
    return {name: 0 for name in ("motor", "mobil", "bus", "truk", "sepeda")}


def _mean(values):
    return float(np.mean(values)) if values else None


def _traffic_status(flow_rate):
    if flow_rate is None:
        return "BELUM TERSEDIA"
    if flow_rate < 30:
        return "RENDAH"
    if flow_rate < 60:
        return "SEDANG"
    return "TINGGI"


def _make_progress(
    route: str,
    timestamp: float,
    duration: float,
    counted_classes: Counter,
    speed_samples: list[float],
    frame_index: int,
    processed_frames: int,
    final: bool = False,
):
    total = int(sum(counted_classes.values()))
    flow_rate = total / (duration / 60.0) if duration > 0 else None
    average_speed = _mean(speed_samples)
    return {
        "type": "final" if final else "progress",
        "route": route,
        "progress": round(min(max(timestamp / duration, 0.0), 1.0), 4) if duration > 0 else None,
        "frame_index": int(frame_index),
        "frames_processed": int(processed_frames),
        "timestamp_seconds": round(timestamp, 3),
        "vehicles": {
            "total": total,
            **{name: int(counted_classes[name]) for name in _empty_classes()},
        },
        "traffic": {
            "flow_rate_vehicles_per_minute": round(flow_rate, 3) if flow_rate is not None else None,
            "status": _traffic_status(flow_rate),
        },
        "speed": {
            "average_speed_mps": round(average_speed, 3) if average_speed is not None else None,
            "sample_count": len(speed_samples),
        },
    }


# ============================================================
# STREAMING ANALYSIS
# ============================================================
def analyze_video_stream(
    video_path: Path,
    route: str,
    calibration_distance_m: float | None,
    line_a_y: float | None,
    line_b_y: float | None,
) -> Iterator[dict]:
    """Yield JSON-serializable progress snapshots, then one final result.

    Counting is performed inside the frame loop. The old end-of-video
    motion/observation filter is intentionally removed: stationary or slow
    vehicles are not discarded just because they moved <15 px or appeared
    for <3 observations.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Video tidak dapat dibuka oleh OpenCV.")

    started_at = time.time()
    model = get_model()

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0

    process_fps = max(1.0, min(PROCESS_FPS, fps))
    stride = max(1, round(fps / process_fps))

    line_a = horizontal_line(line_a_y, width)
    line_b = horizontal_line(line_b_y, width)
    distance_m = float(calibration_distance_m or 0.0)
    calibrated = bool(
        distance_m > 0
        and line_a is not None
        and line_b is not None
        and line_a_y != line_b_y
    )

    previous_centers: dict[int, tuple[float, float]] = {}
    track_class_votes: defaultdict[int, Counter] = defaultdict(Counter)
    counted_track_ids: set[int] = set()
    line_a_times: dict[int, float] = {}
    line_b_times: dict[int, float] = {}
    speed_samples: list[float] = []
    counted_classes: Counter = Counter()
    timeline: list[dict] = []

    raw_detection_counts: Counter = Counter()
    accepted_detection_counts: Counter = Counter()

    processed_frames = 0
    frame_index = 0
    last_yield_frame = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % stride != 0:
                frame_index += 1
                continue

            timestamp = frame_index / fps if fps > 0 else 0.0

            results = model.track(
                source=frame,
                persist=True,
                tracker=TRACKER,
                conf=TRACK_CONF_THRESHOLD,
                imgsz=INFER_SIZE,
                max_det=100,
                classes=list(TARGET_CLASSES.keys()),
                verbose=False,
            )
            result = results[0]
            detections: list[dict] = []

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()

                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = None

                for i, (box, class_id, confidence) in enumerate(zip(boxes, classes, confidences)):
                    if class_id not in TARGET_CLASSES:
                        continue

                    class_name = TARGET_CLASSES[class_id]
                    raw_detection_counts[class_name] += 1

                    min_conf = MOTOR_MIN_CONF if class_name == "motor" else OTHER_MIN_CONF
                    if float(confidence) < min_conf:
                        continue
                    accepted_detection_counts[class_name] += 1

                    x1, y1, x2, y2 = map(float, box)
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                    # Tidak membuat track ID palsu. Hanya ID ByteTrack yang dipakai.
                    if track_ids is None:
                        continue
                    track_id = int(track_ids[i])

                    previous_center = previous_centers.get(track_id)
                    track_class_votes[track_id][class_name] += 1

                    # --------------------------------------------------------
                    # COUNTING DI DALAM LOOP
                    # --------------------------------------------------------
                    crossed_a = False
                    if line_a is not None and previous_center is not None:
                        crossed_a = crossed(previous_center, center, *line_a)

                    if COUNT_ON_LINE_A and line_a is not None:
                        # Satu track dihitung saat pertama kali menyeberangi Line A.
                        if crossed_a and track_id not in counted_track_ids:
                            best_class = track_class_votes[track_id].most_common(1)[0][0]
                            counted_classes[best_class] += 1
                            counted_track_ids.add(track_id)
                    elif line_a is None:
                        # Tanpa garis hitung, hitung sekali saat track pertama terlihat.
                        if track_id not in counted_track_ids:
                            counted_classes[class_name] += 1
                            counted_track_ids.add(track_id)

                    # --------------------------------------------------------
                    # SPEED: Line A -> Line B
                    # --------------------------------------------------------
                    if calibrated and previous_center is not None:
                        crossed_b = crossed(previous_center, center, *line_b)

                        if crossed_a and track_id not in line_a_times:
                            line_a_times[track_id] = timestamp

                        if crossed_b and track_id not in line_b_times:
                            line_b_times[track_id] = timestamp

                        if track_id in line_a_times and track_id in line_b_times:
                            delta_t = abs(line_b_times[track_id] - line_a_times[track_id])
                            if 0 < delta_t <= 60:
                                speed_mps = distance_m / delta_t
                                if 0 < speed_mps < 60:
                                    speed_samples.append(speed_mps)
                            line_a_times.pop(track_id, None)
                            line_b_times.pop(track_id, None)

                    previous_centers[track_id] = center

                    detections.append(
                        {
                            "x1": round(x1 / max(width, 1), 6),
                            "y1": round(y1 / max(height, 1), 6),
                            "x2": round(x2 / max(width, 1), 6),
                            "y2": round(y2 / max(height, 1), 6),
                            "class": class_name,
                            "confidence": round(float(confidence), 4),
                            "track_id": track_id,
                            "time": round(timestamp, 3),
                        }
                    )

            timeline.append({"time": round(timestamp, 3), "detections": detections})
            processed_frames += 1

            if processed_frames == 1 or processed_frames - last_yield_frame >= PROGRESS_EVERY_FRAMES:
                last_yield_frame = processed_frames
                yield _make_progress(
                    route,
                    timestamp,
                    duration,
                    counted_classes,
                    speed_samples,
                    frame_index,
                    processed_frames,
                    final=False,
                )

            frame_index += 1

    finally:
        cap.release()

    total = int(sum(counted_classes.values()))
    flow_rate = total / (duration / 60.0) if duration > 0 else None
    average_speed = _mean(speed_samples)

    final = {
        "route": route,
        "video": {
            "duration_seconds": round(duration, 3),
            "fps": round(fps, 3),
            "analysis_fps": round(process_fps, 3),
            "frames": frame_count,
            "frames_processed": processed_frames,
            "resolution": {"width": width, "height": height},
        },
        "vehicles": {
            "total": total,
            "motor": int(counted_classes["motor"]),
            "mobil": int(counted_classes["mobil"]),
            "bus": int(counted_classes["bus"]),
            "truk": int(counted_classes["truk"]),
            "sepeda": int(counted_classes["sepeda"]),
        },
        "raw_detection_counts": {
            "motor": int(raw_detection_counts["motor"]),
            "mobil": int(raw_detection_counts["mobil"]),
            "bus": int(raw_detection_counts["bus"]),
            "truk": int(raw_detection_counts["truk"]),
            "sepeda": int(raw_detection_counts["sepeda"]),
        },
        "accepted_detection_counts": {
            "motor": int(accepted_detection_counts["motor"]),
            "mobil": int(accepted_detection_counts["mobil"]),
            "bus": int(accepted_detection_counts["bus"]),
            "truk": int(accepted_detection_counts["truk"]),
            "sepeda": int(accepted_detection_counts["sepeda"]),
        },
        "unique_track_count": int(len(track_class_votes)),
        "counted_track_count": int(len(counted_track_ids)),
        "traffic": {
            "flow_rate_vehicles_per_minute": round(flow_rate, 3) if flow_rate is not None else None,
            "status": _traffic_status(flow_rate),
        },
        "speed": {
            "average_speed_mps": round(average_speed, 3) if average_speed is not None else None,
            "sample_count": len(speed_samples),
            "status": "terkalibrasi" if calibrated else "belum terkalibrasi",
        },
        "queue": {"available": False, "total": None, "status": "belum tersedia"},
        "calibration": {
            "distance_m": distance_m if distance_m > 0 else None,
            "line_a_y": line_a_y,
            "line_b_y": line_b_y,
            "speed_calibrated": calibrated,
            "counting_line_configured": line_a is not None,
        },
        "detections_timeline": timeline,
        "analysis_runtime_seconds": round(time.time() - started_at, 3),
        "model": MODEL_NAME,
        "tracker": TRACKER,
        "confidence_threshold": CONF_THRESHOLD if "CONF_THRESHOLD" in globals() else TRACK_CONF_THRESHOLD,
        "tracker_confidence_threshold": TRACK_CONF_THRESHOLD,
        "motor_min_confidence": MOTOR_MIN_CONF,
        "other_min_confidence": OTHER_MIN_CONF,
        "inference_image_size": INFER_SIZE,
        "counting_mode": "line_a_crossing" if COUNT_ON_LINE_A and line_a is not None else "unique_track_fallback",
        "source": "video asli",
        "provenance": "AI YOLO + ByteTrack",
        "random_data": False,
    }

    yield _make_progress(
        route,
        duration,
        duration,
        counted_classes,
        speed_samples,
        frame_index,
        processed_frames,
        final=True,
    )

    # The final detailed result is embedded in a second final message field.
    yield {"type": "result", "result": final}


# ============================================================
# BACKWARD-COMPATIBLE API FOR CURRENT main.py
# ============================================================
def analyze_video(
    video_path: Path,
    route: str,
    calibration_distance_m: float | None,
    line_a_y: float | None,
    line_b_y: float | None,
):
    """Return the final analysis dict for the existing job architecture.

    For realtime/job progress use analyze_video_stream(). Keeping this wrapper
    prevents the existing main.py from breaking while the frontend still polls
    /jobs/{job_id}.
    """
    final_result = None
    for item in analyze_video_stream(
        video_path=video_path,
        route=route,
        calibration_distance_m=calibration_distance_m,
        line_a_y=line_a_y,
        line_b_y=line_b_y,
    ):
        if item.get("type") == "result":
            final_result = item["result"]
    if final_result is None:
        raise RuntimeError("Analisis selesai tanpa menghasilkan result.")
    return final_result


# Backward-compatible name used by some older code.
def analyze_video_generator(
    video_path: Path,
    route: str,
    calibration_distance_m: float | None,
    line_a_y: float | None,
    line_b_y: float | None,
):
    yield from analyze_video_stream(
        video_path=video_path,
        route=route,
        calibration_distance_m=calibration_distance_m,
        line_a_y=line_a_y,
        line_b_y=line_b_y,
    )
