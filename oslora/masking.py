"""Masked training: the loss counts the subject, not the room around them.

A character's face is ~10-15 % of a bust shot's pixels, so with a plain loss most of every step goes
to the background, clothes and light of those particular photos -- what makes the LoRA copy scenes
(bleed) and drift on unrelated prompts (leak) once the holdout has stopped improving. A grey-scale
mask per image says what counts: white fully, black at ``floor``. The image itself still trains
whole, so the subject is learned in its scene; per image the weights are scaled to mean 1, so a
picture with a small mask counts as much as one without (the subject gets proportionally more).

Masks live in ``<dataset>/_masks/<image stem>.png``. Images without one get it automatically from a
person segmenter (Google's MediaPipe multiclass selfie model, Apache-2.0, 16 MB, downloaded once,
run on the CPU with LiteRT -- the approach of Fizgig's automask, Apache-2.0); a mask edited by hand
is kept. Deleting ``_masks/`` drops them all. The fit / holdout probes use the same weights, so the
Autopilot judges what is being trained.
"""

import hashlib
import os
import urllib.request
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .data import cover_crop

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK_DIR = "_masks"
MODEL_DIR = os.path.join(ROOT, "models", "masks")
MODEL_NAME = "selfie_multiclass_256x256.tflite"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/"
             "latest/" + MODEL_NAME)
MODEL_SHA256 = "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"

BACKGROUND, HAIR, BODY, FACE, CLOTHES, OTHER = range(6)  # the segmenter's classes
PRESETS: Dict[str, Tuple[str, Tuple[int, ...]]] = {
    "person": ("인물 전체 (머리~옷)", (HAIR, BODY, FACE, CLOTHES, OTHER)),
    "no_clothes": ("인물, 옷 제외", (HAIR, BODY, FACE)),
    "head": ("얼굴과 머리만", (HAIR, FACE)),
}


def mask_path(image: str) -> str:
    folder, name = os.path.split(image)
    return os.path.join(folder, MASK_DIR, os.path.splitext(name)[0] + ".png")


def _upright(path: str) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def ensure_model(log: Callable[[str], None] = print) -> str:
    """The segmenter file, downloaded (and its hash checked) the first time."""
    path = os.path.join(MODEL_DIR, MODEL_NAME)
    if os.path.isfile(path):
        return path
    os.makedirs(MODEL_DIR, exist_ok=True)
    log("마스크: 인물 분할 모델 받는 중 (16 MB, 처음 한 번)")
    tmp = path + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp)
    with open(tmp, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != MODEL_SHA256:
            os.remove(tmp)
            raise RuntimeError("받은 인물 분할 모델이 예상한 파일이 아니에요. 잠시 뒤 다시 해 주세요")
    os.replace(tmp, path)
    return path


class Segmenter:
    """Picture -> per-class probabilities (H, W, 6) at the picture's own size."""

    def __init__(self, path: str):
        from ai_edge_litert.interpreter import Interpreter

        self.it = Interpreter(model_path=path)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.h, self.w = int(self.inp["shape"][1]), int(self.inp["shape"][2])

    def classes(self, img: Image.Image) -> np.ndarray:
        x = np.asarray(img.resize((self.w, self.h), Image.BILINEAR), np.float32) / 255.0
        self.it.set_tensor(self.inp["index"], x[None])
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"])[0].astype(np.float32)
        if y.min() < 0 or y.max() > 1.0001:  # logits
            y = np.exp(y - y.max(-1, keepdims=True))
            y /= y.sum(-1, keepdims=True)
        return np.stack([np.asarray(Image.fromarray(y[..., c]).resize(img.size, Image.BILINEAR))
                         for c in range(y.shape[-1])], -1)

    def mask(self, img: Image.Image, preset: str) -> Image.Image:
        probs = self.classes(img)
        m = np.clip(probs[..., list(PRESETS[preset][1])].sum(-1), 0.0, 1.0)
        return Image.fromarray((m * 255).round().astype(np.uint8), "L")


def make_masks(images: Iterable[str], preset: str = "person", log: Callable[[str], None] = print,
               overwrite: bool = False) -> int:
    """Masks for the images that have none (all of them with ``overwrite``); how many were made."""
    todo = [p for p in images if overwrite or not os.path.isfile(mask_path(p))]
    if not todo:
        return 0
    seg = Segmenter(ensure_model(log))
    for p in todo:
        out = mask_path(p)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        seg.mask(_upright(p), preset).save(out)
    return len(todo)


def load(image: str, bucket: Tuple[int, int], latent_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
    """The image's mask fitted like the image (cover-resize + center-crop to its bucket), then
    averaged onto the latent grid: (H, W) float32 in [0, 1]; None without a usable mask."""
    path = mask_path(image)
    if not os.path.isfile(path):
        return None
    m = Image.open(path).convert("L")
    upright = ImageOps.exif_transpose(Image.open(image)).size
    if m.size != upright:
        if abs(m.width / m.height - upright[0] / upright[1]) > 0.02:  # made for another crop / rotation
            return None
        m = m.resize(upright, Image.BILINEAR)
    px = torch.from_numpy(np.asarray(cover_crop(m, bucket, Image.BILINEAR), np.float32) / 255.0)
    grid = F.interpolate(px[None, None], size=tuple(latent_hw), mode="area")[0, 0]
    if float(grid.mean()) < 0.01:  # all black: nothing to train on
        return None
    return grid.clamp(0.0, 1.0)


def weights(masks: torch.Tensor, floor: float) -> torch.Tensor:
    """(B, H, W) masks -> loss weights: never below ``floor``, scaled to mean 1 per image."""
    w = masks.clamp(min=float(floor), max=1.0)
    return w / w.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


class MaskedLoss:
    """The training / probe loss under the items' masks; items without a mask count every cell."""

    def __init__(self, floor: float):
        self.floor = float(floor)

    def batch_masks(self, items: List, shape: torch.Size, flip: bool = False) -> Optional[torch.Tensor]:
        """(B, H, W) weights for a batch whose latents have ``shape`` (B, C, H, W), or None when no
        item in it has a mask."""
        if not any(it.mask is not None for it in items):
            return None
        h, w = shape[-2], shape[-1]
        m = torch.stack([it.mask if it.mask is not None and tuple(it.mask.shape) == (h, w)
                         else torch.ones(h, w) for it in items])
        return m.flip(-1) if flip else m

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, masks: Optional[torch.Tensor]) -> torch.Tensor:
        se = (pred.float() - target.float()) ** 2
        if masks is None:
            return se.mean()
        w = weights(masks.to(se.device), self.floor)
        w = w.view(w.shape[0], *([1] * (se.dim() - 3)), w.shape[-2], w.shape[-1])
        return (se * w).mean()


def attach(items: List, preset: str, log: Callable[[str], None]) -> Dict[str, int]:
    """Make the missing masks, then put each item's mask on its latent grid (items need their
    latents). Returns counts for the log."""
    images = sorted({it.image for it in items})
    made = make_masks(images, preset, log)
    used = skipped = 0
    for it in items:
        it.mask = load(it.image, it.bucket, tuple(it.latent.shape[-2:])) if it.latent is not None else None
        if it.mask is None:
            skipped += 1
        else:
            used += 1
    cover = [float((it.mask >= 0.5).float().mean()) for it in items if it.mask is not None]
    return {"made": made, "used": used, "skipped": skipped,
            "cover": int(round(100 * float(np.median(cover)))) if cover else 0}
