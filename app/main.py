import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .analysis import analyze_video


# =========================================================
# BASIC CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MEDIA_DIR = BASE_DIR / "media"
MEDIA_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "150"))


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

cors_origins = os.getenv("CORS_ORIGINS", "*")

origins = [
    origin.strip()
    for origin in cors_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "service": "Integrated Cihampelas Mobility AI Backend",
        "status": "online",
        "message": "Backend berhasil dijalankan.",
        "docs": "/docs",
        "health": "/health",
        "demo": "/demo",
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# =========================================================
# DEMO PAGE
# =========================================================

@app.get("/demo", response_class=HTMLResponse)
def demo():
    demo_path = BASE_DIR / "app" / "demo.html"

    if not demo_path.exists():
        return HTMLResponse(
            content="<h1>Demo page belum tersedia.</h1>",
            status_code=404,
        )

    return demo_path.read_text(encoding="utf-8")


# =========================================================
# VIDEO ANALYSIS
# =========================================================

@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    location: str = Form("Cihampelas"),
    config: str = Form("{}"),
):
    """
    Menerima video dan mengirimkannya ke video analysis engine.

    Form:
    - video     : file video
    - location  : Cihampelas / Cipaganti
    - config    : JSON konfigurasi kamera
    """

    # -----------------------------------------------------
    # VALIDATE FILE NAME
    # -----------------------------------------------------

    if not video.filename:
        raise HTTPException(
            status_code=400,
            detail="Nama file video tidak tersedia.",
        )

    # -----------------------------------------------------
    # VALIDATE EXTENSION
    # -----------------------------------------------------

    extension = Path(video.filename).suffix.lower()

    allowed_extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
    }

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Format video tidak didukung. "
                "Gunakan MP4, MOV, AVI, MKV, WEBM, atau M4V."
            ),
        )

    # -----------------------------------------------------
    # VALIDATE CONFIG
    # -----------------------------------------------------

    try:
        camera_config = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Config kamera tidak valid: {exc}",
        ) from exc

    # -----------------------------------------------------
    # CREATE UNIQUE JOB DIRECTORY
    # -----------------------------------------------------

    job_id = str(uuid.uuid4())

    job_dir = MEDIA_DIR / job_id
    job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_filename = f"source{extension}"
    input_path = job_dir / input_filename

    # -----------------------------------------------------
    # SAVE UPLOADED VIDEO
    # -----------------------------------------------------

    total_size = 0

    try:
        with input_path.open("wb") as output_file:

            while True:

                chunk = await video.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Ukuran video melebihi batas "
                            f"{MAX_UPLOAD_MB} MB."
                        ),
                    )

                output_file.write(chunk)

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
            status_code=500,
            detail=(
                "Gagal menyimpan video: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    # -----------------------------------------------------
    # RUN AI ANALYSIS
    # -----------------------------------------------------

    try:

        result = analyze_video(
            video_path=input_path,
            location=location,
            config=camera_config,
            job_id=job_id,
        )

    except Exception as exc:

        shutil.rmtree(
            job_dir,
            ignore_errors=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Analisis video gagal: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    # -----------------------------------------------------
    # ADD FILE INFORMATION
    # -----------------------------------------------------

    result["job_id"] = job_id

    result["source_filename"] = video.filename

    result["source_video_url"] = (
        f"/media/{job_id}/{input_filename}"
    )

    result["uploaded_size_bytes"] = total_size

    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return result
