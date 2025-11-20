from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from scipy.spatial.distance import cosine

from app.retinaface_detector import RetinaFaceDetector


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="RetinaFace Video Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

retinaface_detector: RetinaFaceDetector | None
detector_error: str | None = None
FACE_DATABASE: list[dict] = []
try:
    retinaface_detector = RetinaFaceDetector()
except Exception as exc:  # pragma: no cover - initialization failure path
    retinaface_detector = None
    detector_error = str(exc)


@app.get("/")
async def root() -> FileResponse:
    """Serve the SPA entrypoint."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Frontend not found.")
    return FileResponse(index_path)


@app.post("/detect")
async def detect_faces(file: UploadFile = File(...), mode: str = "search") -> dict:
    """Handle a video upload, run RetinaFace, and return cropped faces or matches."""
    if retinaface_detector is None:
        detail = detector_error or "RetinaFace detector is not initialized."
        raise HTTPException(status_code=500, detail=detail)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Please upload a video file.")

    normalized_mode = mode.lower().strip()
    if normalized_mode not in {"search", "enroll"}:
        raise HTTPException(status_code=400, detail="Mode must be either 'search' or 'enroll'.")

    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)

    try:
        results = _extract_faces(tmp_path, retinaface_detector, normalized_mode, file.filename)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    return {"faces": results["faces"], "count": len(results["faces"]), "matches_found": results["matches"]}


def _extract_faces(
    video_path: Path,
    detector: RetinaFaceDetector,
    mode: str,
    filename: str,
    max_faces: int = 60,
) -> dict:
    """Sample frames from a video and run RetinaFace detection and recognition."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(fps // 2) or 1, 1)  # inspect roughly twice per second

    faces: List[dict] = []
    matches_count = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        detections = detector.detect(frame)
        for detection in detections:
            x1, y1, x2, y2 = (int(v) for v in detection["box"])
            x1 = max(0, min(x1, frame.shape[1] - 1))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            x2 = max(x1 + 1, min(x2, frame.shape[1]))
            y2 = max(y1 + 1, min(y2, frame.shape[0]))
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            embedding = detector.get_embedding(face_img)
            match_info = None

            if embedding:
                if mode == "search":
                    best_score = 1.0
                    best_match = None
                    for record in FACE_DATABASE:
                        score = cosine(record["embedding"], embedding)
                        if score < 0.4 and score < best_score:
                            best_score = score
                            best_match = record
                    if best_match:
                        matches_count += 1
                        match_info = {
                            "origin_video": best_match["video_origin"],
                            "origin_timestamp": best_match["timestamp"],
                            "similarity_score": round((1 - best_score) * 100, 2),
                        }
                elif mode == "enroll":
                    FACE_DATABASE.append(
                        {
                            "video_origin": filename,
                            "embedding": embedding,
                            "timestamp": round(frame_idx / fps, 2),
                        }
                    )

            face_uri = _encode_image(face_img)
            faces.append(
                {
                    "timestamp": round(frame_idx / fps, 2),
                    "image": face_uri,
                    "score": detection["score"],
                    "match": match_info,
                }
            )
            if len(faces) >= max_faces:
                break
        frame_idx += 1
        if len(faces) >= max_faces:
            break

    cap.release()
    return {"faces": faces, "matches": matches_count}


def _encode_image(image: np.ndarray) -> str:
    """Convert a BGR image to a base64 data URI."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode face crop.")
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
