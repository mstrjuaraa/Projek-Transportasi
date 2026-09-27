
import os
from collections import defaultdict

import cv2
from ultralytics import YOLO


# =========================================================
# KONFIGURASI MODEL
# =========================================================

MODEL_NAME = os.getenv("MODEL_NAME", "yolo26n.pt")

# COCO:
# 1 = bicycle
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
CLASS_MAP = {
    1: "sepeda",
    2: "mobil",
    3: "motor",
    5: "bus",
    7: "truk",
}

TRACK_CLASSES = list(CLASS_MAP.keys())


# =========================================================
# ANALISIS VIDEO
# =========================================================

def analyze_video(
    video_path: str,
    route: str = "Cihampelas",
    calibration_distance_m: float = 0.0,
    line_a_y: float = 0.0,
    line_b_y: float = 0.0,
):
    """
    Analisis video lalu lintas menggunakan YOLO + ByteTrack.

    Output:
    - jumlah kendaraan yang melewati counting line
    - jumlah berdasarkan kelas
    - flow rate kendaraan/menit
    - estimasi kecepatan rata-rata m/s jika dikalibrasi
    - estimasi kendaraan lambat/berhenti
    - indikator kepadatan
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video tidak ditemukan: {video_path}"
        )

    # -----------------------------------------------------
    # Buka video
    # -----------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            "Video tidak dapat dibuka oleh OpenCV."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0:
        fps = 25.0

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration_seconds = (
        frame_count / fps
        if frame_count > 0
        else 0
    )

    # -----------------------------------------------------
    # Counting line
    # -----------------------------------------------------

    if line_a_y > 0:
        count_line_y = line_a_y
    else:
        count_line_y = height * 0.50

    # -----------------------------------------------------
    # YOLO
    # -----------------------------------------------------

    model = YOLO(MODEL_NAME)

    # -----------------------------------------------------
    # Data tracking
    # -----------------------------------------------------

    previous_centers = {}

    counted_ids = set()

    counted_classes = defaultdict(int)

    track_history = defaultdict(list)

    line_a_crossings = {}
    line_b_crossings = {}

    speed_measurements = []

    # kendaraan aktif pada frame terakhir
    current_active_ids = set()

    # -----------------------------------------------------
    # Queue estimation
    # -----------------------------------------------------

    slow_ids = set()

    frame_index = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_time = frame_index / fps

        # -------------------------------------------------
        # Deteksi + tracking
        # -------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=TRACK_CLASSES,
            conf=0.25,
            verbose=False,
        )

        if not results:
            frame_index += 1
            continue

        result = results[0]

        if result.boxes is None:
            frame_index += 1
            continue

        boxes = result.boxes

        if boxes.id is None:
            frame_index += 1
            continue

        ids = boxes.id.int().cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()

        current_active_ids = set(ids)

        # -------------------------------------------------
        # Proses setiap kendaraan
        # -------------------------------------------------

        for track_id, class_id, bbox in zip(
            ids,
            classes,
            xyxy,
        ):

            if class_id not in CLASS_MAP:
                continue

            label = CLASS_MAP[class_id]

            x1, y1, x2, y2 = bbox

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            track_history[track_id].append(
                (
                    frame_time,
                    center_x,
                    center_y,
                )
            )

            # batasi history
            if len(track_history[track_id]) > 30:
                track_history[track_id] = (
                    track_history[track_id][-30:]
                )

            # -------------------------------------------------
            # Crossing detection
            # -------------------------------------------------

            previous = previous_centers.get(track_id)

            if previous is not None:

                previous_y = previous[1]

                # Kendaraan melewati counting line
                crossed = (
                    previous_y < count_line_y <= center_y
                    or
                    previous_y > count_line_y >= center_y
                )

                if crossed and track_id not in counted_ids:

                    counted_ids.add(track_id)

                    counted_classes[label] += 1

                # -------------------------------------------------
                # Speed line A
                # -------------------------------------------------

                if line_a_y > 0:

                    crossed_a = (
                        previous_y < line_a_y <= center_y
                        or
                        previous_y > line_a_y >= center_y
                    )

                    if (
                        crossed_a
                        and track_id not in line_a_crossings
                    ):
                        line_a_crossings[track_id] = (
                            frame_time
                        )

                # -------------------------------------------------
                # Speed line B
                # -------------------------------------------------

                if line_b_y > 0:

                    crossed_b = (
                        previous_y < line_b_y <= center_y
                        or
                        previous_y > line_b_y >= center_y
                    )

                    if (
                        crossed_b
                        and track_id not in line_b_crossings
                    ):

                        line_b_crossings[track_id] = (
                            frame_time
                        )

                        # Hitung speed jika A sudah dilewati
                        if track_id in line_a_crossings:

                            t_a = line_a_crossings[
                                track_id
                            ]

                            t_b = line_b_crossings[
                                track_id
                            ]

                            delta_t = abs(
                                t_b - t_a
                            )

                            if (
                                calibration_distance_m > 0
                                and delta_t > 0
                            ):

                                speed_mps = (
                                    calibration_distance_m
                                    / delta_t
                                )

                                # buang nilai yang tidak masuk akal
                                if (
                                    0.1
                                    <= speed_mps
                                    <= 50
                                ):
                                    speed_measurements.append(
                                        speed_mps
                                    )

            previous_centers[track_id] = (
                center_x,
                center_y,
            )

            # -------------------------------------------------
            # Slow / stopped vehicle estimation
            # -------------------------------------------------

            history = track_history[track_id]

            if len(history) >= 5:

                old_time, old_x, old_y = history[-5]

                delta_time = (
                    frame_time - old_time
                )

                if delta_time > 0:

                    pixel_distance = (
                        (
                            center_x - old_x
                        ) ** 2
                        +
                        (
                            center_y - old_y
                        ) ** 2
                    ) ** 0.5

                    pixel_speed = (
                        pixel_distance
                        / delta_time
                    )

                    # Kendaraan sangat lambat
                    if pixel_speed < 20:
                        slow_ids.add(track_id)

                    else:
                        slow_ids.discard(track_id)

        frame_index += 1

    cap.release()

    # =====================================================
    # HASIL AKHIR
    # =====================================================

    total_vehicles = len(counted_ids)

    # -----------------------------------------------------
    # Flow rate
    # -----------------------------------------------------

    if duration_seconds > 0:

        flow_rate = (
            total_vehicles
            / (duration_seconds / 60.0)
        )

    else:
        flow_rate = 0.0

    # -----------------------------------------------------
    # Speed
    # -----------------------------------------------------

    if speed_measurements:

        average_speed_mps = (
            sum(speed_measurements)
            / len(speed_measurements)
        )

        speed_status = "terkalibrasi"

    else:

        average_speed_mps = None

        speed_status = "belum terkalibrasi"

    # -----------------------------------------------------
    # Queue
    # -----------------------------------------------------

    queued_vehicles = len(slow_ids)

    # -----------------------------------------------------
    # Density indicator
    # -----------------------------------------------------

    # Jumlah rata-rata kendaraan aktif relatif terhadap
    # kapasitas indikator sederhana pada frame.
    #
    # Ini adalah indikator heuristik dari video,
    # bukan kapasitas jalan resmi.

    if total_vehicles == 0:

        traffic_status = "TIDAK TERDETEKSI"

        density_index = 0

    elif flow_rate < 15:

        traffic_status = "RENDAH"

        density_index = 25

    elif flow_rate < 30:

        traffic_status = "SEDANG"

        density_index = 55

    else:

        traffic_status = "TINGGI"

        density_index = 85

    # =====================================================
    # RETURN
    # =====================================================

    return {
        "route": route,

        "video": {
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "duration_seconds": round(
                duration_seconds,
                2,
            ),
        },

        "vehicles": {
            "total": total_vehicles,
            "motor": counted_classes.get(
                "motor",
                0,
            ),
            "mobil": counted_classes.get(
                "mobil",
                0,
            ),
            "bus": counted_classes.get(
                "bus",
                0,
            ),
            "truk": counted_classes.get(
                "truk",
                0,
            ),
            "sepeda": counted_classes.get(
                "sepeda",
                0,
            ),
        },

        "traffic": {
            "flow_rate_vehicles_per_minute": round(
                flow_rate,
                2,
            ),

            "queued_vehicles_estimate": (
                queued_vehicles
            ),

            "density_index": density_index,

            "status": traffic_status,
        },

        "speed": {
            "average_speed_mps": (
                round(
                    average_speed_mps,
                    2,
                )
                if average_speed_mps is not None
                else None
            ),

            "status": speed_status,

            "samples": len(
                speed_measurements
            ),
        },

        "counting": {
            "counting_line_y": round(
                float(count_line_y),
                2,
            ),

            "line_a_y": (
                round(
                    float(line_a_y),
                    2,
                )
                if line_a_y > 0
                else None
            ),

            "line_b_y": (
                round(
                    float(line_b_y),
                    2,
                )
                if line_b_y > 0
                else None
            ),

            "calibration_distance_m": (
                calibration_distance_m
                if calibration_distance_m > 0
                else None
            ),
        },

        "method": {
            "detection": "Ultralytics YOLO",
            "tracking": "ByteTrack",
            "source": "video asli yang diunggah",
            "random_traffic_profile": False,
        },
    }
