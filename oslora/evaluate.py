"""Output-based quality judges, run on the training samples.

    likeness   face identity similarity to the training faces (YuNet + ArcFace w600k_r50, CPU)
    adherence  CLIP text-image similarity between each sample and its prompt
    diversity  1 - mean CLIP similarity between samples of different prompts (collapse -> 0)
    copy       mean of each sample's highest CLIP similarity to any training image (memorization)
    aesthetic  LAION aesthetic predictor on the CLIP embeddings (1-10; broken / overbaked -> down)

Flexibility checks (single renders, compared with the step-0 render of the same seed):
    sharpness  frame sharpness of a "shaky, grainy" prompt: a LoRA that scrubs the effect raises it
    text_match CLIP similarity to an attribute the dataset lacks ("platinum-blonde bob, glasses")
    likeness of a different person ("an elderly man"): identity bleed

Loss probes move by ~1% over a run; these move a lot (likeness ~0.1 -> 0.7), and they are
what a person actually looks at.
"""

import os

# OpenCV 5's "Targets are not supported by the new graph engine" notice for YuNet is harmless
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "models", "eval")
YUNET = ("face_detection_yunet_2023mar.onnx",
         "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
ARCFACE = "arcface_w600k_r50.onnx"
ARCFACE_ZIP = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
CLIP_REPO = "openai/clip-vit-large-patch14"
AESTHETIC = ("laion_aesthetic_l14.pth",
             "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth")


def frame_sharpness(image: Image.Image) -> float:
    """Variance of the Laplacian over the whole frame (motion blur / grain-smear -> low)."""
    import cv2

    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _aesthetic_head() -> Optional[torch.nn.Module]:
    """LAION improved aesthetic predictor: a linear MLP on normalized CLIP ViT-L/14 image
    embeddings (no activations, as published). None if it cannot be fetched."""
    path = os.path.join(EVAL_DIR, AESTHETIC[0])
    if not os.path.isfile(path):
        try:
            import requests

            r = requests.get(AESTHETIC[1], timeout=120)
            r.raise_for_status()
            os.makedirs(EVAL_DIR, exist_ok=True)
            with open(path, "wb") as f:
                f.write(r.content)
        except Exception:
            return None
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k.replace("layers.", ""): v for k, v in sd.items()}
    head = torch.nn.Sequential(torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2), torch.nn.Linear(1024, 128),
                               torch.nn.Dropout(0.2), torch.nn.Linear(128, 64), torch.nn.Dropout(0.1),
                               torch.nn.Linear(64, 16), torch.nn.Linear(16, 1))
    head.load_state_dict(sd)
    return head.eval().requires_grad_(False)

# ArcFace's canonical 112x112 positions of: right eye, left eye (as seen in the image: left
# then right), nose tip, right and left mouth corners -- the same order YuNet reports them.
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def ensure_face_models() -> Dict[str, str]:
    import requests

    os.makedirs(EVAL_DIR, exist_ok=True)
    det = os.path.join(EVAL_DIR, YUNET[0])
    if not os.path.isfile(det) or os.path.getsize(det) < 100_000:
        r = requests.get(YUNET[1], timeout=120)
        r.raise_for_status()
        with open(det, "wb") as f:
            f.write(r.content)
    rec = os.path.join(EVAL_DIR, ARCFACE)
    if not os.path.isfile(rec) or os.path.getsize(rec) < 100_000_000:
        # InsightFace buffalo_l (non-commercial research license): keep only the recognizer
        import io
        import zipfile

        r = requests.get(ARCFACE_ZIP, timeout=600)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            name = next(n for n in z.namelist() if n.endswith("w600k_r50.onnx"))
            with open(rec, "wb") as f:
                f.write(z.read(name))
    return {"det": det, "rec": rec}


class FaceID:
    """YuNet detection + 5-point alignment + ArcFace (InsightFace w600k_r50) identity
    embedding. ArcFace separates look-alikes far better than SFace did: SFace scored the
    calibration LoRA ~0.73 from step 235 while the face still looked only roughly right."""

    def __init__(self):
        import cv2
        import onnxruntime as ort

        paths = ensure_face_models()
        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN.create(paths["det"], "", (320, 320), 0.7)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.rec = ort.InferenceSession(paths["rec"], opts, providers=["CPUExecutionProvider"])
        self.rec_input = self.rec.get_inputs()[0].name

    def embed(self, image: Image.Image) -> Optional[np.ndarray]:
        r = self.analyze(image)
        return r["embedding"] if r else None

    def analyze(self, image: Image.Image) -> Optional[Dict]:
        """Largest face: identity embedding, box in original-image pixels, face count."""
        faces = self.analyze_all(image)
        return max(faces, key=lambda f: f["box"][2] * f["box"][3]) if faces else None

    def analyze_all(self, image: Image.Image) -> List[Dict]:
        """Every detected face: embedding, box (original pixels), size in px, face count."""
        cv2 = self.cv2
        img = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        s = min(1.0, 1024 / max(img.shape[:2]))
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        self.det.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.det.detect(img)
        if faces is None or len(faces) == 0:
            return []
        out = []
        for face in faces:
            v = self._embed_face(img, face)
            if v is None:
                continue
            x, y, w, h = (float(c) / s for c in face[:4])
            out.append({"embedding": v, "box": (x, y, w, h), "size": min(w, h), "count": int(len(faces)),
                        "score": float(face[-1])})
        return out

    def _embed_face(self, img, face) -> Optional[np.ndarray]:
        cv2 = self.cv2
        landmarks = face[4:14].reshape(5, 2).astype(np.float32)
        m, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_DST, method=cv2.LMEDS)
        if m is None:
            return None
        crop = cv2.warpAffine(img, m, (112, 112), borderValue=0)
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        v = self.rec.run(None, {self.rec_input: blob})[0].flatten().astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)


class ClipJudge:
    def __init__(self, device="cuda"):
        from transformers import CLIPModel, CLIPProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        self.processor = CLIPProcessor.from_pretrained(CLIP_REPO)
        self.model = CLIPModel.from_pretrained(CLIP_REPO, dtype=torch.float16).eval().requires_grad_(False)
        self.device = device

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        out = []
        for i in range(0, len(images), 16):
            px = self.processor(images=list(images[i : i + 16]), return_tensors="pt")["pixel_values"]
            e = self.model.get_image_features(pixel_values=px.to(self.device, torch.float16))
            e = e.pooler_output if hasattr(e, "pooler_output") else e
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def texts(self, texts: Sequence[str]) -> torch.Tensor:
        tok = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True, max_length=77)
        e = self.model.get_text_features(input_ids=tok["input_ids"].to(self.device),
                                         attention_mask=tok["attention_mask"].to(self.device))
        e = e.pooler_output if hasattr(e, "pooler_output") else e
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu()


class Evaluator:
    """Reference embeddings of the training set are computed once; ``score`` judges a set of
    samples (one per prompt) in a couple of seconds. CLIP lives on the CPU between calls."""

    def __init__(self, train_images: Sequence[str], device="cuda", log=print):
        self.device = device
        self.face = FaceID()
        self.clip = ClipJudge(device).to(device)
        faces, pics = [], []
        for p in train_images:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((1024, 1024))
                v = self.face.embed(im)
                if v is not None:
                    faces.append(v)
                pics.append(im.copy())
        self.train_faces = np.stack(faces) if faces else None
        if faces:
            # robust centroid: drop the farthest 20% once (odd angles / mislabelled images)
            c = self.train_faces.mean(0)
            c /= np.linalg.norm(c)
            if len(faces) >= 5:
                keep = np.argsort(self.train_faces @ c)[int(len(faces) * 0.2):]
                c = self.train_faces[keep].mean(0)
            self.face_centroid = c / np.linalg.norm(c)
            self.face_self = float((self.train_faces @ self.face_centroid).mean())
        else:
            self.face_centroid, self.face_self = None, None
        self.train_clip = self.clip.images(pics)
        self.aesthetic_head = _aesthetic_head()
        self.clip.to("cpu")
        torch.cuda.empty_cache()
        log(f"평가자 준비: 학습 이미지 얼굴 {len(faces)}/{len(train_images)}장 인식"
            + (f" (학습 이미지끼리 닮음 {self.face_self:.2f})" if faces else " - 얼굴이 없어 닮음 지표는 끕니다"))

    MIN_FACE_PX = 100  # below this ArcFace is unreliable (measured: 78-117px faces scored 0.04-0.58)

    def likeness_of(self, image: Image.Image) -> Optional[float]:
        """The subject's likeness in one image: among faces large enough to judge, the one most
        like the training faces (a busy scene can contain other people), or None."""
        faces = [f for f in self.face.analyze_all(image) if f["size"] >= self.MIN_FACE_PX]
        if not faces or self.face_centroid is None:
            return None
        return max(float(f["embedding"] @ self.face_centroid) for f in faces)

    def score(self, images: List[Image.Image], prompts: List[str]) -> Dict:
        out: Dict = {}
        if self.face_centroid is not None:
            sims = [s for s in (self.likeness_of(im) for im in images) if s is not None]
            # median: one odd render must not swing the whole round
            out["likeness"] = float(np.median(sims)) if sims else None
            out["faces"] = len(sims)
        self.clip.to(self.device)
        try:
            img = self.clip.images(images)
            txt = self.clip.texts(prompts)
        finally:
            self.clip.to("cpu")
            torch.cuda.empty_cache()
        out["adherence"] = float((img * txt).sum(-1).mean() * 100)
        if len(images) > 1:
            sim = img @ img.T
            n = len(images)
            out["diversity"] = float(1 - (sim.sum() - n) / (n * (n - 1)))
        out["copy"] = float((img @ self.train_clip.T).max(dim=1).values.mean())
        if self.aesthetic_head is not None:
            out["aesthetic"] = float(self.aesthetic_head(img.float()).mean())
        return out

    def likeness_spread(self, images: List[Image.Image]) -> Dict:
        """Median likeness over several renders and its interquartile range (how much of a
        round-to-round change could be luck)."""
        sims = [s for s in (self.likeness_of(im) for im in images) if s is not None]
        if not sims:
            return {}
        q1, med, q3 = np.percentile(sims, [25, 50, 75])
        return {"likeness": float(med), "likeness_iqr": float(q3 - q1), "faces": len(sims)}

    def text_match(self, image: Image.Image, text: str) -> float:
        """CLIP similarity (x100) between one image and a short description."""
        self.clip.to(self.device)
        try:
            v = float((self.clip.images([image]) * self.clip.texts([text])).sum() * 100)
        finally:
            self.clip.to("cpu")
            torch.cuda.empty_cache()
        return v

    def bleed(self, image: Image.Image) -> float:
        """How much a render of someone else looks like the character (any face, even small)."""
        faces = self.face.analyze_all(image)
        if not faces or self.face_centroid is None:
            return 0.0
        return max(float(f["embedding"] @ self.face_centroid) for f in faces)
