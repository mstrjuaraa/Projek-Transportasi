import os
import shutil
import uuid
from pathlib import Path
from threading import Thread

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .analysis import analyze_video


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Integrated Cihampelas Mobility AI Backend",
    description=(
        "Backend analisis video lalu lintas "
        "Jalan Cihampelas dan Jalan Cipaganti."
    ),
    version="0.2.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# FOLDER
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MEDIA_DIR = BASE_DIR / "media"
UPLOAD_DIR = MEDIA_DIR / "uploads"

MEDIA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# STATIC MEDIA
# =========================================================

app.mount(
    "/media",
    StaticFiles(directory=str(MEDIA_DIR)),
    name="media",
)


# =========================================================
# JOB STORAGE
#
# Untuk prototype:
# status disimpan di memory.
# =========================================================

JOBS = {}


# =========================================================
# BACKGROUND ANALYSIS
# =========================================================

def run_analysis_job(
    job_id: str,
    video_path: str,
    route: str,
    calibration_distance_m: float,
    line_a_y: float,
    line_b_y: float,
):
    try:

        JOBS[job_id]["status"] = "processing"

        result = analyze_video(
            video_path=video_path,
            route=route,
            calibration_distance_m=calibration_distance_m,
            line_a_y=line_a_y,
            line_b_y=line_b_y,
        )

        filename = Path(video_path).name

        result["video_url"] = (
            f"/media/uploads/{filename}"
        )

        result["job_id"] = job_id

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = result

    except Exception as exc:

        JOBS[job_id]["status"] = "failed"

        JOBS[job_id]["error"] = str(exc)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Integrated Cihampelas Mobility AI Backend",
        "version": "0.2.0",
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "jobs": len(JOBS),
    }


# =========================================================
# DEMO
# =========================================================

@app.get("/demo")
def demo():

    demo_file = BASE_DIR / "app" / "demo.html"

    if not demo_file.exists():
        raise HTTPException(
            status_code=404,
            detail="demo.html tidak ditemukan.",
        )

    return FileResponse(
        str(demo_file)
    )


# =========================================================
# START ANALYSIS JOB
# =========================================================

@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    route: str = Form("Cihampelas"),
    calibration_distance_m: float = Form(0.0),
    line_a_y: float = Form(0.0),
    line_b_y: float = Form(0.0),
):

    if not video.filename:
        raise HTTPException(
            status_code=400,
            detail="Nama file video tidak tersedia.",
        )

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }

    suffix = Path(
        video.filename
    ).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Format video tidak didukung. "
                "Gunakan MP4, AVI, MOV, MKV, atau WEBM."
            ),
        )

    # -----------------------------------------------------
    # JOB ID
    # -----------------------------------------------------

    job_id = str(uuid.uuid4())

    filename = (
        f"{job_id}{suffix}"
    )

    saved_path = (
        UPLOAD_DIR / filename
    )

    # -----------------------------------------------------
    # SIMPAN VIDEO
    # -----------------------------------------------------

    try:

        with saved_path.open("wb") as buffer:

            shutil.copyfileobj(
                video.file,
                buffer,
            )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Gagal menyimpan video: {exc}"
            ),
        )

    # -----------------------------------------------------
    # DAFTARKAN JOB
    # -----------------------------------------------------

    JOBS[job_id] = {
        "status": "queued",
        "result": None,
        "error": None,
        "route": route,
    }

    # -----------------------------------------------------
    # JALANKAN ANALISIS DI BACKGROUND
    # -----------------------------------------------------

    thread = Thread(
        target=run_analysis_job,
        args=(
            job_id,
            str(saved_path),
            route,
            calibration_distance_m,
            line_a_y,
            line_b_y,
        ),
        daemon=True,
    )

    thread.start()

    # -----------------------------------------------------
    # LANGSUNG KEMBALIKAN JOB ID
    # -----------------------------------------------------

    return {
        "job_id": job_id,
        "status": "queued",
        "message": (
            "Video berhasil diunggah. "
            "Analisis AI sedang diproses."
        ),
    }


# =========================================================
# CHECK JOB STATUS
# =========================================================

@app.get("/jobs/{job_id}")
def get_job(job_id: str):

    if job_id not in JOBS:

        raise HTTPException(
            status_code=404,
            detail="JOB ID tidak ditemukan.",
        )

    job = JOBS[job_id]

    response = {
        "job_id": job_id,
        "status": job["status"],
        "route": job["route"],
    }

    if job["status"] == "completed":

        response["result"] = job["result"]

    if job["status"] == "failed":

        response["error"] = job["error"]

    return response
