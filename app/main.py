import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .analysis import analyze_video


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Integrated Cihampelas Mobility AI Backend",
    description=(
        "Backend prototype untuk analisis video lalu lintas "
        "Jalan Cihampelas dan Jalan Cipaganti."
    ),
    version="0.1.0",
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
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Integrated Cihampelas Mobility AI Backend",
        "version": "0.1.0",
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =========================================================
# DEMO PAGE
# =========================================================

@app.get("/demo")
def demo():
    demo_file = BASE_DIR / "app" / "demo.html"

    if not demo_file.exists():
        raise HTTPException(
            status_code=404,
            detail="demo.html tidak ditemukan."
        )

    return FileResponse(str(demo_file))


# =========================================================
# ANALYZE VIDEO
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
            detail="File video tidak memiliki nama."
        )

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }

    suffix = Path(video.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Format video tidak didukung. "
                "Gunakan MP4, AVI, MOV, MKV, atau WEBM."
            ),
        )

    # -----------------------------------------------------
    # Simpan video asli
    # -----------------------------------------------------

    video_id = str(uuid.uuid4())
    saved_filename = f"{video_id}{suffix}"
    saved_path = UPLOAD_DIR / saved_filename

    try:
        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal menyimpan video: {exc}",
        )

    # -----------------------------------------------------
    # Analisis AI
    # -----------------------------------------------------

    try:
        result = analyze_video(
            video_path=str(saved_path),
            route=route,
            calibration_distance_m=calibration_distance_m,
            line_a_y=line_a_y,
            line_b_y=line_b_y,
        )

    except Exception as exc:
        # hapus file jika analisis gagal
        try:
            saved_path.unlink(missing_ok=True)
        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail=f"Analisis video gagal: {exc}",
        )

    # -----------------------------------------------------
    # URL video asli
    # -----------------------------------------------------

    result["video_url"] = f"/media/uploads/{saved_filename}"
    result["route"] = route

    return result
