from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The official RetinaFace implementation now lives at the project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import cfg_mnet, cfg_re50  # type: ignore  # noqa: E402
from layers.functions.prior_box import PriorBox  # type: ignore  # noqa: E402
from models.retinaface import RetinaFace  # type: ignore  # noqa: E402
from utils.box_utils import decode, decode_landm  # type: ignore  # noqa: E402
from utils.nms.py_cpu_nms import py_cpu_nms  # type: ignore  # noqa: E402


DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "mobilenet0.25_Final.pth"


def _remove_prefix(state_dict: dict, prefix: str) -> dict:
    """Strip DistributedDataParallel's 'module.' prefix if present."""
    if not prefix:
        return state_dict
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


class RetinaFaceDetector:
    """Wrapper around the official RetinaFace PyTorch implementation."""

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        network: str = "mobile0.25",
        use_cpu: bool = True,
        confidence_threshold: float = 0.6,
        top_k: int = 5000,
        keep_top_k: int = 750,
        nms_threshold: float = 0.4,
    ) -> None:
        cfg_template = cfg_mnet if network == "mobile0.25" else cfg_re50
        self.cfg = deepcopy(cfg_template)
        self.cfg["pretrain"] = False  # avoid looking for training-only backbone weights
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.resize = 1
        self.device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")

        weights = weights_path or DEFAULT_WEIGHTS
        if not weights.exists():
            raise FileNotFoundError(
                f"RetinaFace weights not found at {weights}. "
                "Download mobilenet0.25_Final.pth from the official repo and place it in ./weights."
            )

        self.model = RetinaFace(cfg=self.cfg, phase="test")
        self.model = self._load_model(weights)
        self.model.eval()
        self.model = self.model.to(self.device)

        # Face embedding model (FaceNet) for identification
        self.recognizer: InceptionResnetV1 | None = None
        self.face_preprocess: transforms.Compose | None = None
        self.recognizer_error: str | None = None
        try:
            self.recognizer = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            self.face_preprocess = transforms.Compose(
                [
                    transforms.Resize((160, 160)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        except Exception as exc:  # pragma: no cover - initialization failure path
            self.recognizer_error = f"Failed to load recognition model: {exc}"
            self.recognizer = None
            self.face_preprocess = None

    def _load_model(self, weights_path: Path) -> RetinaFace:
        pretrained = torch.load(weights_path, map_location=self.device)
        if "state_dict" in pretrained:
            pretrained = _remove_prefix(pretrained["state_dict"], "module.")
        else:
            pretrained = _remove_prefix(pretrained, "module.")
        self.model.load_state_dict(pretrained, strict=False)
        return self.model

    def detect(self, frame: np.ndarray) -> List[dict]:
        """Run detection on a single BGR frame."""
        img = np.float32(frame)
        im_height, im_width, _ = img.shape
        scale = torch.tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]],
            dtype=torch.float32,
            device=self.device,
        )

        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            loc, conf, landms = self.model(img)

        priors = PriorBox(self.cfg, image_size=(im_height, im_width)).forward()
        priors = priors.to(self.device)
        prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg["variance"])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg["variance"])
        scale1 = torch.tensor(
            [
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        order = scores.argsort()[::-1][: self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = dets[: self.keep_top_k, :]
        landms = landms[: self.keep_top_k, :]

        results: List[dict] = []
        for box, score, landmark in zip(dets[:, :4], dets[:, 4], landms):
            results.append(
                {
                    "box": box.tolist(),
                    "score": float(score),
                    "landmarks": landmark.tolist(),
                }
            )
        return results

    def get_embedding(self, face_img_numpy: np.ndarray) -> List[float] | None:
        """Convert a cropped BGR face into a FaceNet embedding."""
        if self.recognizer is None or self.face_preprocess is None:
            return None

        try:
            face_pil = Image.fromarray(cv2.cvtColor(face_img_numpy, cv2.COLOR_BGR2RGB))
            face_tensor = self.face_preprocess(face_pil).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                embedding = self.recognizer(face_tensor)
            return embedding[0].cpu().numpy().tolist()
        except Exception:
            return None
