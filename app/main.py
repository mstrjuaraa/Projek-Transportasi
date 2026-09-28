import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .analysis import analyze_video


BASE_DIR = Path(__file__).resolve().parent.parent
MEDIA_DIR = BASE_DIR / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "150"))
MAX_WORKERS = max(1, int(os.getenv("MAX_ANALYSIS_WORKERS", "1")))

app = FastAPI(
    title="Integrated Cihampelas Mobility AI Backend",
    version="0.2.0",
)

origins = [
    x.strip()
    for x in os.getenv("CORS_ORIGINS", "*").split(",")
    if x.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/media",
    StaticFiles(directory=str(MEDIA_DIR)),
    name="media",
)

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()


def set_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(updates)


def run_job(
    job_id: str,
    input_path: Path,
    route: str,
    config: dict[str, Any],
    source_url: str,
) -> None:
    set_job(
        job_id,
        status="processing",
        message="AI sedang memproses video.",
    )

    try:
        result = analyze_video(
            video_path=input_path,
            route=route,
            calibration_distance_m=config.get("calibration_distance_m"),
            line_a_y=config.get("line_a_y"),
            line_b_y=config.get("line_b_y"),
        )

        result["job_id"] = job_id
        result["source_video_url"] = source_url
        result["video_url"] = source_url

        set_job(
            job_id,
            status="completed",
            message="Analisis selesai.",
            result=result,
        )

    except Exception as exc:
        set_job(
            job_id,
            status="failed",
            message="Analisis gagal.",
            error=f"{type(exc).__name__}: {exc}",
        )


@app.on_event("shutdown")
def shutdown_event() -> None:
    executor.shutdown(
        wait=False,
        cancel_futures=True,
    )


@app.get("/")
def root():
    with jobs_lock:
        count = len(jobs)

    return {
        "status": "online",
        "service": "Integrated Cihampelas Mobility AI Backend",
        "version": "0.2.0",
        "jobs": count,
    }


@app.get("/health")
def health():
    with jobs_lock:
        counts = {
            "queued": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
        }

        for job in jobs.values():
            status = job.get("status")
            if status in counts:
                counts[status] += 1

    return {
        "status": "healthy",
        "jobs": len(jobs),
        **counts,
    }


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    route: str = Form("Cihampelas"),
    calibration_distance_m: str = Form(""),
    line_a_y: str = Form(""),
    line_b_y: str = Form(""),
):
    if not video.filename:
        raise HTTPException(
            400,
            "Nama file video tidak tersedia.",
        )

    suffix = Path(video.filename).suffix.lower()

    allowed = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
    }

    if suffix not in allowed:
        raise HTTPException(
            400,
            f"Format video tidak didukung: {sorted(allowed)}",
        )

    def parse_float(raw: str | None) -> float | None:
        value = (raw or "").strip()

        if not value:
            return None

        try:
            return float(value)
        except ValueError as exc:
            raise HTTPException(
                422,
                f"Nilai numerik tidak valid: {raw}",
            ) from exc

    distance = parse_float(calibration_distance_m)
    line_a = parse_float(line_a_y)
    line_b = parse_float(line_b_y)

    job_id = str(uuid.uuid4())

    job_dir = MEDIA_DIR / job_id
    job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_path = job_dir / f"source{suffix}"
    source_url = f"/media/{job_id}/source{suffix}"

    total = 0

    try:
        with input_path.open("wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)

                if not chunk:
                    break

                total += len(chunk)

                if total > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(
                        413,
                        f"Video melebihi batas {MAX_UPLOAD_MB} MB.",
                    )

                out.write(chunk)

    except HTTPException:
        shutil.rmtree(
            job_dir,
            ignore_errors=True,
        )
        raise

    except Exception as exc:
        shutil.rmtree(
            job_dir,
            ignore_errors=True,
        )

        raise HTTPException(
            500,
            f"Upload video gagal: {type(exc).__name__}: {exc}",
        ) from exc

    config = {
        "calibration_distance_m": distance,
        "line_a_y": line_a,
        "line_b_y": line_b,
    }

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "message": (
                "Video berhasil diunggah. "
                "Analisis AI sedang diproses."
            ),
            "source_filename": video.filename,
        }

    try:
        executor.submit(
            run_job,
            job_id,
            input_path,
            route,
            config,
            source_url,
        )

    except Exception as exc:
        set_job(
            job_id,
            status="failed",
            error=(
                f"Queue gagal: "
                f"{type(exc).__name__}: {exc}"
            ),
        )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": (
            "Video berhasil diunggah. "
            "Analisis AI sedang diproses."
        ),
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            raise HTTPException(
                404,
                "Job tidak ditemukan.",
            )

        return dict(job)
