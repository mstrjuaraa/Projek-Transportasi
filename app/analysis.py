"""
Analisis lalu lintas: YOLO + ByteTrack, versi STREAMING (generator).

`analyze_video()` sekarang adalah generator. Setiap frame yang diproses
menghasilkan satu event (dict yang siap di-JSON-kan):

    {"type": "start", ...}   -> sekali di awal (metadata video)
    {"type": "frame", ...}   -> setiap frame yang diproses (total, kecepatan, status)
    {"type": "final", ...}   -> sekali di akhir (hasil lengkap, format lama)

Contoh pemakaian dengan FastAPI (NDJSON):

    import json
    from fastapi.responses import StreamingResponse

    def ndjson(events):
        for event in events:
            yield json.dumps(event) + "\\n"

    @app.post("/analyze")
    def analyze(...):
        gen = analyze_video(path, route, dist_m, line_a_y, line_b_y, emit_every=2)
        return StreamingResponse(ndjson(gen), media_type="application/x-ndjson")

Untuk kode lama yang butuh hasil akhir saja: `analyze_video_blocking(...)`.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = os.getenv("MODEL_NAME", "yolo26s.pt")
PROCESS_FPS = float(os.getenv("PROCESS_FPS", "10"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.55"))
TRACK_CONF_THRESHOLD = float(os.getenv("TRACK_CONF_THRESHOLD", "0.40"))
MOTOR_MIN_CONF = float(os.getenv("MOTOR_MIN_CONF", "0.40"))
OTHER_MIN_CONF = float(os.getenv("OTHER_MIN_CONF", "0.55"))
INFER_SIZE = int(os.getenv("INFER_SIZE", "1280"))
TRACKER = os.getenv("TRACKER", "bytetrack.yaml")

# Track dianggap kendaraan asli (dan langsung dihitung) setelah terlihat
# minimal sekian kali pada frame yang diproses. TIDAK ADA syarat gerakan,
# jadi kendaraan yang berhenti/merayap karena macet tetap terhitung.
# Pada PROCESS_FPS=10, 3 observasi ~ 0.3 detik.
MIN_TRACK_OBSERVATIONS = int(os.getenv("MIN_TRACK_OBSERVATIONS", "3"))

# Flow rate baru ditampilkan setelah video berjalan sekian detik,
# supaya status lalu lintas tidak meloncat-loncat di detik-detik awal.
MIN_FLOW_WINDOW_SECONDS = float(os.getenv("MIN_FLOW_WINDOW_SECONDS", "5"))


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

CLASS_NAMES = ["motor", "mobil", "bus", "truk", "sepeda"]


# ============================================================
# MODEL
# ============================================================

MODEL = None


def get_model():
    global MODEL

    if MODEL is None:
        MODEL = YOLO(MODEL_NAME)

    return MODEL


def reset_tracker(model) -> None:
    """
    Model disimpan global dan model.track(persist=True) menyimpan state
    tracker antar-panggilan. Tanpa reset, ID dan track dari video sebelumnya
    "bocor" ke video berikutnya. Panggil ini di awal setiap video.

    Catatan: karena state tracker menempel pada model, satu instance model
    tidak aman dipakai dua analisis yang berjalan bersamaan. Untuk request
    paralel gunakan satu instance YOLO per analisis (atau antrian/lock).
    """
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None) if predictor else None

    if trackers:
        for tracker in trackers:
            tracker.reset()


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

    side_previous = line_side(previous_point, a, b)
    side_current = line_side(current_point, a, b)

    return (
        side_previous == 0
        or side_current == 0
        or ((side_previous < 0) != (side_current < 0))
    )


# ============================================================
# TRACK STATE + COUNTING (PER FRAME)
# ============================================================

@dataclass
class TrackState:
    observations: int = 0
    class_votes: dict = field(default_factory=lambda: defaultdict(float))
    counted_class: str | None = None  # None = belum dihitung

    def observe(self, class_name: str, confidence: float) -> None:
        self.observations += 1
        # Vote dibobot confidence: label yang yakin lebih berpengaruh
        # daripada label ragu-ragu sesaat.
        self.class_votes[class_name] += confidence

    def best_class(self) -> str:
        return max(self.class_votes.items(), key=lambda kv: kv[1])[0]


def update_track_count(state: TrackState, counted_classes: dict) -> bool:
    """
    Dipanggil setiap kali sebuah track diamati pada frame ini.

    - Track baru dihitung SEKALI, tepat saat observasinya mencapai
      MIN_TRACK_OBSERVATIONS (return True = baru dikonfirmasi).
    - Jika kelas terbaik track berubah setelah dihitung (misal motor->mobil),
      hitungan dipindah antar kelas; total tidak berubah.
    - Tidak ada filter gerakan -> kendaraan berhenti tetap terhitung.
    """
    if state.observations < MIN_TRACK_OBSERVATIONS:
        return False

    best = state.best_class()

    if state.counted_class is None:
        counted_classes[best] += 1
        state.counted_class = best
        return True

    if state.counted_class != best:
        counted_classes[state.counted_class] -= 1
        counted_classes[best] += 1
        state.counted_class = best

    return False


# ============================================================
# SNAPSHOT HELPERS (dipakai event "frame" dan "final")
# ============================================================

def vehicles_snapshot(counted_classes: dict) -> dict:
    snapshot = {"total": int(sum(counted_classes.values()))}

    for name in CLASS_NAMES:
        snapshot[name] = int(counted_classes.get(name, 0))

    return snapshot


def compute_flow_rate(total: int, elapsed_seconds: float, min_seconds: float):
    if elapsed_seconds <= 0 or elapsed_seconds < min_seconds:
        return None

    return total / (elapsed_seconds / 60.0)


def traffic_status_from_flow(flow_rate: float | None) -> str:
    if flow_rate is None:
        return "BELUM TERSEDIA"

    if flow_rate < 30:
        return "RENDAH"

    if flow_rate < 60:
        return "SEDANG"

    return "TINGGI"


def traffic_snapshot(total: int, elapsed_seconds: float, min_seconds: float) -> dict:
    flow_rate = compute_flow_rate(total, elapsed_seconds, min_seconds)

    return {
        "flow_rate_vehicles_per_minute": (
            round(flow_rate, 3) if flow_rate is not None else None
        ),
        "status": traffic_status_from_flow(flow_rate),
    }


def speed_snapshot(speed_samples: list, calibrated: bool) -> dict:
    average = float(np.mean(speed_samples)) if speed_samples else None

    return {
        "average_speed_mps": round(average, 3) if average is not None else None,
        "average_speed_kmh": round(average * 3.6, 2) if average is not None else None,
        "sample_count": len(speed_samples),
        "status": "terkalibrasi" if calibrated else "belum terkalibrasi",
    }


# ============================================================
# VIDEO ANALYSIS (GENERATOR)
# ============================================================

def analyze_video(
    video_path: Path,
    route: str,
    calibration_distance_m: float | None,
    line_a_y: float | None,
    line_b_y: float | None,
    emit_every: int = 1,
    include_timeline_in_final: bool = False,
) -> Iterator[dict[str, Any]]:
    """
    Generator: yield event "start", lalu event "frame" (setiap `emit_every`
    frame yang diproses), lalu event "final".

    emit_every               : 1 = kirim setiap frame yang diproses; 2 = tiap
                               dua frame, dst. (mengurangi trafik ke frontend)
    include_timeline_in_final: True = event "final" memuat seluruh
                               detections_timeline seperti versi lama. Default
                               False karena detail per-frame sudah dikirim
                               lewat event "frame" (hemat memori & payload).
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Video tidak dapat dibuka oleh OpenCV.")

    # try/finally penting pada generator: kalau klien memutus koneksi,
    # server memanggil generator.close() -> blok finally tetap berjalan
    # dan video ter-release.
    try:
        # ----------------------------------------------------
        # VIDEO METADATA
        # ----------------------------------------------------
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0

        # ----------------------------------------------------
        # FRAME SAMPLING
        # ----------------------------------------------------
        process_fps = max(1.0, min(PROCESS_FPS, fps))
        stride = max(1, round(fps / process_fps))

        # ----------------------------------------------------
        # LINES / KALIBRASI
        # ----------------------------------------------------
        line_a = horizontal_line(line_a_y, width)
        line_b = horizontal_line(line_b_y, width)
        distance_m = float(calibration_distance_m or 0.0)

        calibrated = bool(
            distance_m > 0
            and line_a is not None
            and line_b is not None
            and line_a_y != line_b_y
        )

        video_info = {
            "duration_seconds": round(duration, 3),
            "fps": round(fps, 3),
            "analysis_fps": round(process_fps, 3),
            "frames": frame_count,
            "resolution": {"width": width, "height": height},
        }

        calibration_info = {
            "distance_m": distance_m if distance_m > 0 else None,
            "line_a_y": line_a_y,
            "line_b_y": line_b_y,
            "speed_calibrated": calibrated,
            "counting_line_configured": bool(
                line_a is not None and line_b is not None
            ),
        }

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------
        model = get_model()
        reset_tracker(model)

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------
        tracks: dict[int, TrackState] = {}
        counted_classes: dict[str, int] = defaultdict(int)

        previous_centers: dict[int, tuple[float, float]] = {}
        line_a_times: dict[int, float] = {}
        line_b_times: dict[int, float] = {}
        speed_samples: list[float] = []

        raw_detection_counts = {name: 0 for name in CLASS_NAMES}
        timeline: list[dict] = []

        processed_frames = 0
        frame_index = 0
        last_timestamp = 0.0
        emit_every = max(1, int(emit_every))
        started_at = time.time()

        # ----------------------------------------------------
        # EVENT: START
        # ----------------------------------------------------
        yield {
            "type": "start",
            "route": route,
            "video": video_info,
            "calibration": calibration_info,
            "model": MODEL_NAME,
            "tracker": TRACKER,
        }

        # ====================================================
        # FRAME LOOP
        # ====================================================
        while True:

            # Frame yang dilewati cukup di-grab (tanpa decode penuh)
            # supaya lebih ringan di CPU.
            if frame_index % stride != 0:
                if not cap.grab():
                    break

                frame_index += 1
                continue

            success, frame = cap.read()

            if not success:
                break

            timestamp = frame_index / fps if fps > 0 else 0.0
            last_timestamp = timestamp

            # ------------------------------------------------
            # YOLO + BYTETRACK
            # ------------------------------------------------
            result = model.track(
                source=frame,
                persist=True,
                tracker=TRACKER,
                conf=TRACK_CONF_THRESHOLD,
                imgsz=INFER_SIZE,
                max_det=100,
                classes=list(TARGET_CLASSES.keys()),
                verbose=False,
            )[0]

            detections: list[dict] = []
            newly_confirmed = 0
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()

                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = [None] * len(xyxy)

                for box, class_id, confidence, track_id in zip(
                    xyxy, class_ids, confidences, track_ids
                ):
                    if class_id not in TARGET_CLASSES:
                        continue

                    class_name = TARGET_CLASSES[class_id]
                    confidence = float(confidence)

                    # Filter confidence per kelas. Motor lebih permisif
                    # karena objek kecil di CCTV confidence-nya cenderung rendah.
                    min_conf = MOTOR_MIN_CONF if class_name == "motor" else OTHER_MIN_CONF

                    if confidence < min_conf:
                        continue

                    raw_detection_counts[class_name] += 1

                    # Hanya pakai ID asli dari ByteTrack (tanpa ID fallback,
                    # supaya satu kendaraan tidak terhitung ganda).
                    if track_id is None:
                        continue

                    track_id = int(track_id)

                    x1, y1, x2, y2 = map(float, box)
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                    # ----------------------------------------
                    # COUNTING PER FRAME
                    # ----------------------------------------
                    state = tracks.get(track_id)

                    if state is None:
                        state = TrackState()
                        tracks[track_id] = state

                    state.observe(class_name, confidence)

                    if update_track_count(state, counted_classes):
                        newly_confirmed += 1

                    # ----------------------------------------
                    # SAVE DETECTION
                    # ----------------------------------------
                    detections.append(
                        {
                            "x1": round(x1 / max(width, 1), 6),
                            "y1": round(y1 / max(height, 1), 6),
                            "x2": round(x2 / max(width, 1), 6),
                            "y2": round(y2 / max(height, 1), 6),
                            "class": class_name,
                            "confidence": round(confidence, 4),
                            "track_id": track_id,
                            "confirmed": state.counted_class is not None,
                            "time": round(timestamp, 3),
                        }
                    )

                    # ----------------------------------------
                    # SPEED (Line A -> Line B)
                    # ----------------------------------------
                    previous_center = previous_centers.get(track_id)

                    if calibrated and previous_center is not None:

                        if track_id not in line_a_times and crossed(
                            previous_center, center, *line_a
                        ):
                            line_a_times[track_id] = timestamp

                        if track_id not in line_b_times and crossed(
                            previous_center, center, *line_b
                        ):
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

            processed_frames += 1
            frame_index += 1

            if include_timeline_in_final:
                timeline.append(
                    {"time": round(timestamp, 3), "detections": detections}
                )

            # ------------------------------------------------
            # EVENT: FRAME (STREAM KE FRONTEND)
            # ------------------------------------------------
            if processed_frames % emit_every == 0:
                vehicles = vehicles_snapshot(counted_classes)

                yield {
                    "type": "frame",
                    "time": round(timestamp, 3),
                    "frame_index": frame_index,
                    "processed_frames": processed_frames,
                    "progress": (
                        round(min(frame_index / frame_count, 1.0), 4)
                        if frame_count > 0
                        else None
                    ),
                    "vehicles": vehicles,
                    "new_vehicles": newly_confirmed,
                    "traffic": traffic_snapshot(
                        vehicles["total"], timestamp, MIN_FLOW_WINDOW_SECONDS
                    ),
                    "speed": speed_snapshot(speed_samples, calibrated),
                    "detections": detections,
                }

        # ====================================================
        # EVENT: FINAL
        # ====================================================
        vehicles = vehicles_snapshot(counted_classes)

        # Final: pakai durasi video penuh seperti versi lama.
        final_seconds = duration if duration > 0 else last_timestamp
        traffic = traffic_snapshot(vehicles["total"], final_seconds, 0.0)

        unconfirmed_tracks = [
            {
                "track_id": int(track_id),
                "class": state.best_class(),
                "observations": state.observations,
            }
            for track_id, state in tracks.items()
            if state.counted_class is None
        ]

        final_video = dict(video_info)
        final_video["frames_processed"] = processed_frames

        yield {
            "type": "final",
            "route": route,
            "video": final_video,
            "vehicles": vehicles,
            "unique_track_count": len(tracks),
            "track_debug": {
                "confirmed": len(tracks) - len(unconfirmed_tracks),
                "unconfirmed_count": len(unconfirmed_tracks),
                # Track terlalu singkat (< MIN_TRACK_OBSERVATIONS) yang tidak dihitung.
                "unconfirmed": unconfirmed_tracks[:50],
            },
            "raw_detection_counts": {
                name: int(count) for name, count in raw_detection_counts.items()
            },
            "traffic": traffic,
            "speed": speed_snapshot(speed_samples, calibrated),
            "queue": {"available": False, "total": None, "status": "belum tersedia"},
            "calibration": calibration_info,
            "detections_timeline": timeline if include_timeline_in_final else None,
            "analysis_runtime_seconds": round(time.time() - started_at, 3),
            "model": MODEL_NAME,
            "tracker": TRACKER,
            "confidence_threshold": CONF_THRESHOLD,
            "tracker_confidence_threshold": TRACK_CONF_THRESHOLD,
            "motor_min_confidence": MOTOR_MIN_CONF,
            "other_min_confidence": OTHER_MIN_CONF,
            "min_track_observations": MIN_TRACK_OBSERVATIONS,
            "inference_image_size": INFER_SIZE,
            "source": "video asli",
            "provenance": "AI YOLO + ByteTrack",
            "random_data": False,
        }

    finally:
        cap.release()


# ============================================================
# BACKWARD-COMPATIBLE WRAPPER
# ============================================================

def analyze_video_blocking(*args, **kwargs) -> dict[str, Any]:
    """
    Menjalankan generator sampai habis dan mengembalikan event "final".
    Untuk kode lama yang belum mendukung streaming. Timeline penuh
    disertakan agar bentuk hasilnya setara dengan versi sebelumnya.
    """
    kwargs.setdefault("include_timeline_in_final", True)
    final = None

    for event in analyze_video(*args, **kwargs):
        if event["type"] == "final":
            final = event

    if final is None:
        raise RuntimeError("Analisis berakhir tanpa hasil akhir.")

    return final
