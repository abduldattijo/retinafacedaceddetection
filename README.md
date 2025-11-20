# RetinaFace Video Detector

Upload a short video and preview the faces found in its frames using the official [`biubug6/Pytorch_Retinaface`](https://github.com/biubug6/Pytorch_Retinaface) model, all inside a lightweight FastAPI web app. This repo already contains the upstream RetinaFace code under the same root (see `README_upstream.md` for the original docs).

Now supports **face recognition and re-identification**: enroll faces from one video, then search for the same people in another video using FaceNet embeddings (`facenet-pytorch`).

## Getting Started

1. **Download the pretrained weights (if not already present)**  
   Grab `mobilenet0.25_Final.pth` from the [official release page](https://github.com/biubug6/Pytorch_Retinaface#trained-model) and place it at `./weights/mobilenet0.25_Final.pth`.

2. **Install dependencies** (first run will download FaceNet weights for recognition)

   ```bash
   
   source .venv1/bin/activate
   
   pip install -r requirements.txt
   ```

3. **Run the server**

   ```bash
   uvicorn app.main:app --reload
   ```

4. **Open the UI**  
   Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) and upload an MP4 (or any supported video).

## Enrollment vs Search

- **Enroll mode**: Upload a clip with the toggle set to *Enroll* (or call `/detect?mode=enroll`). The app extracts faces, builds 512-dim FaceNet embeddings, and stores them in memory (`FACE_DATABASE`) along with the source video name and timestamp.
- **Search mode**: Upload another clip with the toggle set to *Search* (or call `/detect?mode=search`). Each detected face is embedded and compared against the in-memory database using cosine distance. Matches are returned with the originating video and timestamp and shown in the UI.

## How It Works

- The `/detect` endpoint saves the uploaded video temporarily, samples frames about twice per second, and runs the cloned PyTorch RetinaFace model on each sampled frame.
- For every detected face a PNG crop is encoded as a base64 data URI (capped at 60 faces for responsiveness) and returned to the browser. When recognition is enabled, each crop also carries an embedding and optional match info.
- The frontend grid displays each crop with its timestamp; matches show the originating video, timestamp, and similarity score.

## Tips

- Short clips (<20 MB) work best in the browser; larger videos will take longer to upload and process.
- You can tweak sampling density, the face cap, or the match threshold (defaults to cosine distance < 0.4) in `_extract_faces` in `app/main.py`.
- The in-memory `FACE_DATABASE` is reset when the server restarts; for production, persist embeddings in a proper store (Postgres, Redis, Milvus/FAISS, etc.).
- If you plan to deploy, consider moving uploads to object storage and persisting detections instead of returning base64 blobs.
- Running on CPU works out of the box. If you have a GPU available, adjust `RetinaFaceDetector` inside `app/retinaface_detector.py` to set `use_cpu=False`.
