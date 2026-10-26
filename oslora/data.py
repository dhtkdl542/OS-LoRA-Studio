"""Dataset scanning, aspect bucketing and on-disk latent / text-embedding caches."""

import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass
class Item:
    image: str
    caption: str
    key: str = ""
    bucket: Tuple[int, int] = (0, 0)  # (width, height) in pixels
    latent: Optional[torch.Tensor] = None
    latent_photo: Optional[torch.Tensor] = None  # same image with photo texture (tone + sensor noise)
    text: Optional[torch.Tensor] = None
    # long / medium / short captions (MOSS official recaption style) with their sampling weights;
    # empty = the single .txt caption
    alt: List[Tuple[str, float]] = field(default_factory=list)
    alt_text: List[Tuple[torch.Tensor, float]] = field(default_factory=list)
    # the same captions without their image-finish statements: paired with the photo-textured
    # latent, so "polished, no visible grain" isn't taught on a toned, grainy picture
    text_plain: Optional[torch.Tensor] = None
    alt_text_plain: List[Tuple[torch.Tensor, float]] = field(default_factory=list)
    holdout: bool = False


def scan_folder(folder: str, caption_ext: str = ".txt") -> List[Tuple[str, str]]:
    """(image_path, caption) pairs; missing captions come back as ''."""
    pairs = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(IMAGE_EXTS):
            continue
        path = os.path.join(folder, name)
        cap_path = os.path.splitext(path)[0] + caption_ext
        caption = ""
        if os.path.isfile(cap_path):
            with open(cap_path, encoding="utf-8") as f:
                caption = f.read().strip()
        pairs.append((path, caption))
    return pairs


def apply_trigger(caption: str, trigger: str) -> str:
    trigger = (trigger or "").strip()
    if not trigger or trigger.lower() in caption.lower():
        return caption
    return f"{trigger}, {caption}" if caption else trigger


def bucket_for(width: int, height: int, resolution: int, step: int = 16, max_ratio: float = 2.0) -> Tuple[int, int]:
    """Size with ~resolution^2 pixels at the image's aspect ratio (clamped), multiples of ``step``."""
    aspect = min(max(width / height, 1 / max_ratio), max_ratio)
    area = resolution * resolution
    bw = max(step, int(round(math.sqrt(area * aspect) / step)) * step)
    bh = max(step, int(round(math.sqrt(area / aspect) / step)) * step)
    return bw, bh


def load_image(path: str, size: Tuple[int, int], photo: bool = False, tone: bool = False) -> torch.Tensor:
    """Cover-resize + center-crop to ``size``; returns (3, H, W) in [-1, 1]. ``photo`` adds the
    photo texture (tone + phone sensor noise) at the training size."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    bw, bh = size
    scale = max(bw / img.width, bh / img.height)
    nw, nh = max(bw, round(img.width * scale)), max(bh, round(img.height * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - bw) // 2, (nh - bh) // 2
    img = img.crop((left, top, left + bw, top + bh))
    if photo:
        from . import photoreal

        img = photoreal.apply(img, os.path.basename(path), tone)
    t = torch.from_numpy(__import__("numpy").asarray(img).copy()).permute(2, 0, 1).float()
    return t / 127.5 - 1.0


def file_key(path: str, extra: str) -> str:
    st = os.stat(path)
    raw = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}|{extra}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def text_key(text: str, model_tag: str) -> str:
    return hashlib.sha1(f"{model_tag}|{text}".encode("utf-8")).hexdigest()[:16]


CAPTIONS_EXT = ".captions.json"  # {"long", "medium", "short"} next to the image; .txt holds long
# long gets the most weight: the prompts this is used with are long, detailed descriptions
LENGTH_WEIGHTS = (("long", 0.5), ("medium", 0.3), ("short", 0.2))


def load_caption_variants(image_path: str) -> Dict[str, str]:
    try:
        with open(os.path.splitext(image_path)[0] + CAPTIONS_EXT, encoding="utf-8") as f:
            data = json.load(f)
        return {k: str(data[k]).strip() for k, _ in LENGTH_WEIGHTS if str(data.get(k) or "").strip()}
    except (OSError, ValueError, TypeError):
        return {}


def save_caption_variants(image_path: str, variants: Dict[str, str], caption_ext: str = ".txt", **meta):
    """Write the three lengths (sidecar) and the long one as the regular caption."""
    base = os.path.splitext(image_path)[0]
    with open(base + CAPTIONS_EXT, "w", encoding="utf-8") as f:
        json.dump({**{k: variants[k] for k, _ in LENGTH_WEIGHTS if variants.get(k)}, **meta}, f, ensure_ascii=False, indent=1)
    with open(base + caption_ext, "w", encoding="utf-8") as f:
        f.write(variants.get("long") or variants.get("medium") or variants.get("short") or "")


CROP_PREFIX = "FaceCrop_"
AUG_PREFIX = "Aug_"  # Aug_<kind>_<original stem>.png: a degraded copy (grain, blur, ...)
DERIVED_PREFIXES = (CROP_PREFIX, AUG_PREFIX)


def is_derived(name: str) -> bool:
    """A face crop or an augmented copy: repeats an original's face."""
    return name.startswith(DERIVED_PREFIXES)


def crop_source(name: str, names: Sequence[str]) -> Optional[str]:
    """The original a derived copy came from: ``FaceCrop_<stem>.png`` / ``Aug_<kind>_<stem>.png``
    (ours), or a numbered ``FaceCrop_007.png`` (other tools) matched to the one original whose
    last number is 7."""
    originals = [n for n in names if not is_derived(n)]
    if name.startswith(AUG_PREFIX):
        rest = os.path.splitext(name[len(AUG_PREFIX):])[0]
        stem = rest.split("_", 1)[1] if "_" in rest else rest
        return next((n for n in originals if os.path.splitext(n)[0] == stem), None)
    if not name.startswith(CROP_PREFIX):
        return None
    stem = os.path.splitext(name[len(CROP_PREFIX):])[0]
    for n in originals:
        if os.path.splitext(n)[0] == stem:
            return n
    if stem.isdigit():
        same = [n for n in originals if (re.findall(r"\d+", n) or [None])[-1] is not None
                and int(re.findall(r"\d+", n)[-1]) == int(stem)]
        if len(same) == 1:
            return same[0]
    return None


def _pairs_path(folder: str) -> str:
    key = hashlib.md5(os.path.normcase(os.path.abspath(folder)).encode("utf-8")).hexdigest()[:16]
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "crop_pairs", key + ".json")


def save_crop_pairs(folder: str, pairs: Dict[str, str]):
    """Crop -> original pairs found by face matching (dataset check-up), for names that the
    naming rules cannot pair (e.g. originals renumbered after some were removed)."""
    path = _pairs_path(folder)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=1)


def crop_sources(folder: str, names: Sequence[str]) -> Dict[str, str]:
    """crop name -> original name for every crop whose original is in ``names``."""
    out = {n: src for n in names if (src := crop_source(n, names))}
    try:
        with open(_pairs_path(folder), encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        saved = {}
    present = set(names)
    for crop, src in saved.items():
        if crop in present and src in present and crop not in out:
            out[crop] = src
    return out


def pick_holdout(n: int, mode, seed: int = 0, names: Optional[Sequence[str]] = None,
                 sources: Optional[Dict[str, str]] = None) -> List[int]:
    """'auto': hold out ~8% (2-6 images) when there are >= 16 images, else none.

    With ``names``, a face crop and its original (``sources``: crop -> original) are held out
    or trained on together: a held-out photo whose crop is in training is not unseen."""
    if mode in (None, False, 0, "off", "none"):
        return []
    groups = [[i] for i in range(n)]
    if names is not None:
        if sources is None:
            sources = {nm: src for nm in names if (src := crop_source(nm, names))}
        index = {nm: i for i, nm in enumerate(names)}
        by_src: Dict[int, List[int]] = {}
        for i, nm in enumerate(names):
            by_src.setdefault(index.get(sources.get(nm), i), []).append(i)
        groups = [sorted(g) for g in by_src.values()]
    units = len(groups)
    if mode == "auto":
        if units < 16:
            return []
        count = min(6, max(2, round(units * 0.08)))
    else:
        count = int(mode)
    count = min(count, max(0, units - 4))
    rng = random.Random(seed)
    if names is None:
        return sorted(rng.sample(range(n), count))
    return sorted(i for g in rng.sample(range(units), count) for i in groups[g])


class LatentCache:
    def __init__(self, root: str, model_tag: str):
        self.root = root
        self.model_tag = model_tag
        os.makedirs(root, exist_ok=True)

    def latent_path(self, key: str) -> str:
        return os.path.join(self.root, f"lat_{key}.safetensors")

    def text_path(self, key: str) -> str:
        return os.path.join(self.root, f"te_{key}.safetensors")

    def get_latent(self, key: str) -> Optional[torch.Tensor]:
        p = self.latent_path(key)
        return load_file(p)["latent"] if os.path.isfile(p) else None

    def put_latent(self, key: str, latent: torch.Tensor):
        save_file({"latent": latent.to(torch.bfloat16).contiguous()}, self.latent_path(key))

    def get_text(self, text: str) -> Optional[torch.Tensor]:
        p = self.text_path(text_key(text, self.model_tag))
        return load_file(p)["embed"] if os.path.isfile(p) else None

    def put_text(self, text: str, embed: torch.Tensor):
        save_file({"embed": embed.to(torch.bfloat16).contiguous()}, self.text_path(text_key(text, self.model_tag)),
                  metadata={"text": text[:2000]})


class BucketSampler:
    """Infinite stream of batches: every epoch shuffles the items and groups them by bucket
    so a batch always shares one resolution."""

    def __init__(self, items: List[Item], batch_size: int, seed: int = 0):
        self.items = items
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.epoch = 0
        self._queue: List[List[Item]] = []

    def _refill(self):
        groups: Dict[Tuple[int, int], List[Item]] = {}
        for it in self.items:
            groups.setdefault(it.bucket, []).append(it)
        batches = []
        for group in groups.values():
            self.rng.shuffle(group)
            for i in range(0, len(group), self.batch_size):
                batches.append(group[i : i + self.batch_size])
        self.rng.shuffle(batches)
        self._queue = batches
        self.epoch += 1

    def steps_per_epoch(self) -> int:
        groups: Dict[Tuple[int, int], int] = {}
        for it in self.items:
            groups[it.bucket] = groups.get(it.bucket, 0) + 1
        return sum(math.ceil(n / self.batch_size) for n in groups.values())

    def next(self) -> List[Item]:
        if not self._queue:
            self._refill()
        return self._queue.pop()
