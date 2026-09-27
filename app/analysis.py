import os
from collections import defaultdict

import cv2
from ultralytics import YOLO


MODEL_NAME = os.getenv("MODEL_NAME", "yolo26n.pt")

CLASS_MAP = {
    1: "sepeda",
    2: "mobil",
    3: "motor",
    5: "bus",
    7: "truk",
}

TRACK_CLASSES = list(CLASS_MAP.keys())

# Model dimuat sekali ketika backend aktif,
# bukan setiap kali user menekan tombol analisis.
MODEL = YOLO(MODEL_NAME)


def analyze_video(
    video_path: str,
    route: str = "Cihampelas",
    calibration_distance_m: float = 0.0,
    line_a_y: float = 0.0,
    line_b_y: float = 0.0,
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video tidak ditemukan: {video_path}"
        )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            "Video tidak dapat dibuka."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration = (
        total_frames / fps
        if total_frames > 0
        else 0
    )

    # -----------------------------------------------------
    # Counting line otomatis jika belum diatur
    # -----------------------------------------------------

    count_line_y = (
        line_a_y
        if line_a_y > 0
        else height * 0.50
    )

    # -----------------------------------------------------
    # Tracking data
    # -----------------------------------------------------

    previous_centers = {}

    counted_ids = set()

    vehicle_counts = defaultdict(int)

    line_a_times = {}
    line_b_times = {}

    speed_values = []

    frame_index = 0

    # -----------------------------------------------------
    # Pemrosesan video
    #
    # Untuk video 4K, frame diperkecil ketika masuk YOLO.
    # Video asli tetap disimpan sebagai sumber tampilan.
    # -----------------------------------------------------

    MAX_PROCESS_WIDTH = 1280

    while True:

        success, frame = cap.read()

        if not success:
            break

        current_time = frame_index / fps

        # Resize hanya untuk AI
        process_frame = frame

        if width > MAX_PROCESS_WIDTH:

            scale = (
                MAX_PROCESS_WIDTH
                / float(width)
            )

            new_height = int(
                height * scale
            )

            process_frame = cv2.resize(
                frame,
                (
                    MAX_PROCESS_WIDTH,
                    new_height,
                ),
            )

            process_count_line_y = (
                count_line_y * scale
            )

            process_line_a_y = (
                line_a_y * scale
                if line_a_y > 0
                else 0
            )

            process_line_b_y = (
                line_b_y * scale
                if line_b_y > 0
                else 0
            )

        else:

            process_count_line_y = count_line_y
            process_line_a_y = line_a_y
            process_line_b_y = line_b_y

        # -------------------------------------------------
        # YOLO + ByteTrack
        # -------------------------------------------------

        results = MODEL.track(
            process_frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=TRACK_CLASSES,
            conf=0.25,
            imgsz=640,
            verbose=False,
        )

        if not results:
            frame_index += 1
            continue

        result = results[0]

        if result.boxes is None:
            frame_index += 1
            continue

        if result.boxes.id is None:
            frame_index += 1
            continue

        ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        classes = (
            result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        boxes = (
            result.boxes.xyxy
            .cpu()
            .tolist()
        )

        for track_id, class_id, box in zip(
            ids,
            classes,
            boxes,
        ):

            if class_id not in CLASS_MAP:
                continue

            label = CLASS_MAP[class_id]

            x1, y1, x2, y2 = box

            center_x = (
                x1 + x2
            ) / 2

            center_y = (
                y1 + y2
            ) / 2

            previous = previous_centers.get(
                track_id
            )

            if previous is not None:

                previous_y = previous[1]

                # -----------------------------------------
                # Counting kendaraan
                # -----------------------------------------

                crossed_count_line = (
                    previous_y
                    < process_count_line_y
                    <= center_y
                    or
                    previous_y
                    > process_count_line_y
                    >= center_y
                )

                if (
                    crossed_count_line
                    and track_id not in counted_ids
                ):

                    counted_ids.add(
                        track_id
                    )

                    vehicle_counts[label] += 1

                # -----------------------------------------
                # Line A
                # -----------------------------------------

                if process_line_a_y > 0:

                    crossed_a = (
                        previous_y
                        < process_line_a_y
                        <= center_y
                        or
                        previous_y
                        > process_line_a_y
                        >= center_y
                    )

                    if (
                        crossed_a
                        and track_id
                        not in line_a_times
                    ):

                        line_a_times[
                            track_id
                        ] = current_time

                # -----------------------------------------
                # Line B
                # -----------------------------------------

                if process_line_b_y > 0:

                    crossed_b = (
                        previous_y
                        < process_line_b_y
                        <= center_y
                        or
                        previous_y
                        > process_line_b_y
                        >= center_y
                    )

                    if (
                        crossed_b
                        and track_id
                        not in line_b_times
                    ):

                        line_b_times[
                            track_id
                        ] = current_time

                        if (
                            track_id
                            in line_a_times
                        ):

                            delta_t = (
                                line_b_times[
                                    track_id
                                ]
                                -
                                line_a_times[
                                    track_id
                                ]
                            )

                            delta_t = abs(
                                delta_t
                            )

                            if (
                                calibration_distance_m
                                > 0
                                and delta_t
                                > 0
                            ):

                                speed = (
                                    calibration_distance_m
                                    / delta_t
                                )

                                if (
                                    0.1
                                    <= speed
                                    <= 40
                                ):

                                    speed_values.append(
                                        speed
                                    )

            previous_centers[
                track_id
            ] = (
                center_x,
                center_y,
            )

        frame_index += 1

    cap.release()

    # =====================================================
    # HASIL
    # =====================================================

    total = len(counted_ids)

    if duration > 0:

        flow_rate = (
            total
            / (duration / 60.0)
        )

    else:

        flow_rate = 0.0

    if speed_values:

        average_speed = (
            sum(speed_values)
            / len(speed_values)
        )

        speed_status = "terkalibrasi"

    else:

        average_speed = None

        speed_status = "belum terkalibrasi"

    # -----------------------------------------------------
    # Status lalu lintas
    # -----------------------------------------------------

    if flow_rate < 15:

        traffic_status = "RENDAH"

    elif flow_rate < 30:

        traffic_status = "SEDANG"

    else:

        traffic_status = "TINGGI"

    return {
        "route": route,

        "video": {
            "width": width,
            "height": height,
            "fps": round(
                fps,
                2,
            ),
            "frame_count": total_frames,
            "duration_seconds": round(
                duration,
                2,
            ),
        },

        "vehicles": {
            "total": total,
            "motor": vehicle_counts.get(
                "motor",
                0,
            ),
            "mobil": vehicle_counts.get(
                "mobil",
                0,
            ),
            "bus": vehicle_counts.get(
                "bus",
                0,
            ),
            "truk": vehicle_counts.get(
                "truk",
                0,
            ),
            "sepeda": vehicle_counts.get(
                "sepeda",
                0,
            ),
        },

        "traffic": {
            "flow_rate_vehicles_per_minute": round(
                flow_rate,
                2,
            ),
            "status": traffic_status,
        },

        "speed": {
            "average_speed_mps": (
                round(
                    average_speed,
                    2,
                )
                if average_speed is not None
                else None
            ),
            "status": speed_status,
            "samples": len(
                speed_values
            ),
        },

        "calibration": {
            "line_a_y": (
                line_a_y
                if line_a_y > 0
                else None
            ),
            "line_b_y": (
                line_b_y
                if line_b_y > 0
                else None
            ),
            "distance_m": (
                calibration_distance_m
                if calibration_distance_m > 0
                else None
            ),
        },

        "method": {
            "source": "video asli",
            "detection": "YOLO",
            "tracking": "ByteTrack",
            "random_data": False,
        },
    }
