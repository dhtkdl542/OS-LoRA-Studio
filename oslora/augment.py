"""Caption-aware image degradations for a few dataset copies.

A dataset of clean, retouched images teaches the LoRA that the character *is* clean: prompts
asking for grain, shake or a cheap phone look get scrubbed (measured on Damdeari: frame
sharpness on a flash/motion-blur prompt 34-59 without the LoRA, 126-229 with it). A handful of
degraded copies whose caption starts by naming the degradation ties the look to words instead:
"grainy" means grainy, also on her face. The originals stay clean, so the default stays clean.

Each copy gets one quality effect (grain, motion, lowres, soft, film, selfie, nightselfie) and
every other copy also a colour filter (bw, warm, cool, faded, vivid):
``Aug_<kind>[-<filter>]_<original stem>.png`` with ``<phrases> <original caption>``. New copies
take the effects the folder has fewest of, so a dataset made before an effect existed gets it.
"""

import json
import os
import zlib
from typing import Dict, List

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .data import AUG_PREFIX, LENGTH_WEIGHTS, crop_source, load_caption_variants, save_caption_variants

# kind -> sentence put in front of the original caption (same register as the user's prompts)
PHRASES: Dict[str, str] = {
    "grain": ("A grainy low-light phone photo. Coarse, blotchy high-ISO noise covers the whole image and is "
              "smeared by noise reduction into soft, waxy patches, so fine detail is lost everywhere, "
              "including on her face and in her hair."),
    "motion": ("A blurry handheld photo. The camera moved during a slow exposure, so the whole frame, her face "
               "included, is smeared with directional motion blur and a faint doubled ghost image."),
    "lowres": ("A small, low-resolution, heavily compressed image. Edges are mushy, flat areas show blocky "
               "JPEG artifacts, and fine detail is smeared everywhere, including on her face."),
    "soft": ("An out-of-focus photo. The lens missed focus, so the whole image, her face included, is soft "
             "and blurry with no sharp detail."),
    "film": ("A 35mm film snapshot with visible film grain, slightly faded warm colors, soft contrast and a "
             "gentle dark vignette in the corners."),
    # measured on the user's reference selfies: detail (Laplacian variance) 5-10x below the
    # dataset, saturation ~0.15-0.2, lifted blacks; the softness comes from a small front-camera
    # sensor plus noise reduction rather than from shake
    "selfie": ("A washed-out front-camera phone selfie. It is low resolution and soft, fine detail is smoothed "
               "away by noise reduction, the blacks are lifted into a slight haze, and the contrast is low with "
               "muted, desaturated colors."),
    "nightselfie": ("A dim low-light front-camera phone selfie. The image is soft and slightly blurry, with smeared "
                    "noise, murky lifted gray shadows, a faint warm cast and muted colors."),
}
KINDS = list(PHRASES)

# colour filters (phone / SNS filter looks), stacked on every other copy
FILTER_PHRASES: Dict[str, str] = {
    "bw": "A black-and-white photo with no color at all, only shades of gray.",
    "warm": "A warm-toned filter washes the whole photo in golden orange, with yellowed highlights.",
    "cool": "A cool-toned filter tints the whole photo blue-teal, with muted, slightly desaturated colors.",
    "faded": "A faded, washed-out filter: lifted matte blacks, low contrast and pale, desaturated colors.",
    "vivid": "A punchy, high-contrast filter with deep shadows and strongly saturated colors.",
}
FILTERS = list(FILTER_PHRASES)
KIND_LABELS = {"grain": "노이즈", "motion": "흔들림", "lowres": "저해상도", "soft": "초점 흐림", "film": "필름",
               "selfie": "폰 셀카", "nightselfie": "밤 셀카"}
FILTER_LABELS = {"bw": "흑백", "warm": "따뜻한 톤", "cool": "차가운 톤", "faded": "빛바램", "vivid": "쨍한"}


def coverage(names) -> tuple:
    """How many copies of each effect / colour filter a folder already has: ({kind: n}, {filter: n})."""
    kinds, filters = {k: 0 for k in KINDS}, {f: 0 for f in FILTERS}
    for n in names:
        if n.startswith(AUG_PREFIX) and n.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            k, f = parse_kind(n)
            if k:
                kinds[k] += 1
            if f:
                filters[f] += 1
    return kinds, filters


def _rng(name: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(name.encode("utf-8")))


def _grain(im: Image.Image, rng) -> Image.Image:
    a = np.asarray(im).astype(np.float32)
    h, w = a.shape[:2]
    # blotchy: noise generated at a third of the size and enlarged, plus fine luma noise
    coarse = rng.normal(0, 16, (h // 3 + 1, w // 3 + 1, 3)).astype(np.float32)
    coarse = np.asarray(Image.fromarray(np.clip(coarse + 128, 0, 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
                        dtype=np.float32) - 128
    fine = rng.normal(0, 9, (h, w, 1)).astype(np.float32)
    out = Image.fromarray(np.clip(a * 0.92 + 10 + coarse + fine, 0, 255).astype(np.uint8))
    return out.filter(ImageFilter.MedianFilter(3))  # noise reduction smear


def _motion(im: Image.Image, rng) -> Image.Image:
    import cv2

    a = np.asarray(im)
    h, w = a.shape[:2]
    length = max(5, int(w * rng.uniform(0.01, 0.018)))
    angle = rng.uniform(-35, 35)
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= max(k.sum(), 1e-6)
    blurred = cv2.filter2D(a, -1, k)
    # flash + slow shutter: a translucent, more smeared copy offset to one side
    dx = int(w * rng.uniform(0.025, 0.045)) * (1 if rng.random() < 0.5 else -1)
    ghost = np.roll(cv2.filter2D(blurred, -1, k), dx, axis=1)
    return Image.fromarray(np.clip(blurred * 0.65 + ghost * 0.35, 0, 255).astype(np.uint8))


def _lowres(im: Image.Image, rng) -> Image.Image:
    import io

    w, h = im.size
    s = rng.uniform(0.4, 0.55)
    small = im.resize((max(64, int(w * s)), max(64, int(h * s))), Image.BILINEAR)
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=int(rng.integers(22, 36)))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB").resize((w, h), Image.BILINEAR)


def _soft(im: Image.Image, rng) -> Image.Image:
    r = im.width * rng.uniform(0.004, 0.007)
    return im.filter(ImageFilter.GaussianBlur(r))


def _film(im: Image.Image, rng) -> Image.Image:
    a = np.asarray(im).astype(np.float32) / 255
    a = 0.08 + a * 0.86  # lifted blacks, soft contrast
    a *= np.array([1.04, 1.0, 0.92], np.float32)  # faded warm
    h, w = a.shape[:2]
    y, x = np.ogrid[:h, :w]
    d = ((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2
    a *= (1 - 0.28 * np.clip(d - 0.35, 0, 1))[..., None]  # vignette
    a += rng.normal(0, 0.035, (h, w, 1)).astype(np.float32)  # film grain
    return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))


def _phone_soft(im: Image.Image, rng, scale) -> Image.Image:
    """Small-sensor capture: a fraction of the resolution, noise reduction (edge-preserving smoothing),
    then upscaled back -- soft, smooth detail without blocks."""
    import cv2

    w, h = im.size
    small = im.resize((max(64, int(w * scale)), max(64, int(h * scale))), Image.BILINEAR)
    a = cv2.bilateralFilter(np.asarray(small), 5, 25, 4)
    return Image.fromarray(a).resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(0.4))


def _tone_selfie(a: np.ndarray, sat: float, contrast: float, black: float, warm: float) -> np.ndarray:
    g = a.mean(2, keepdims=True)
    # muted colour, but warm hues (skin, cheeks, lips) keep more of theirs: phone selfies look
    # washed out, not pale -- the references keep rosy cheeks and lips
    warmth = np.clip((a[..., :1] - a[..., 2:3]) / 0.25, 0, 1)
    a = g + (a - g) * (sat + (1 - sat) * 0.6 * warmth)
    a = a.mean() + (a - a.mean()) * contrast  # flat contrast
    a = black + a * (1 - black)               # lifted, hazy blacks
    return a * np.array([1 + warm, 1 - 0.6 * warm, 1.0], np.float32)  # faint pink cast


def _jpeg(im: Image.Image, quality: int) -> Image.Image:
    import io

    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def _selfie(im: Image.Image, rng) -> Image.Image:
    soft = _phone_soft(im, rng, rng.uniform(0.45, 0.55))
    a = np.asarray(soft).astype(np.float32) / 255
    a = _tone_selfie(a, sat=rng.uniform(0.7, 0.82), contrast=rng.uniform(0.88, 0.94),
                     black=rng.uniform(0.03, 0.05), warm=rng.uniform(0.015, 0.035))
    a = 1 - (1 - a) ** 1.05  # a touch of over-exposure in the highlights
    return _jpeg(Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8)), int(rng.integers(55, 75)))


def _nightselfie(im: Image.Image, rng) -> Image.Image:
    import cv2

    a = np.asarray(im).astype(np.float32) / 255
    a = a ** 1.25 * 0.8  # under-exposed
    h, w = a.shape[:2]
    noise = rng.normal(0, 0.05, (h // 2 + 1, w // 2 + 1, 3)).astype(np.float32)
    a = a + cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
    im = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
    soft = _phone_soft(im, rng, rng.uniform(0.42, 0.5))  # noise reduction smears the grain
    a = np.asarray(soft).astype(np.float32) / 255
    a = _tone_selfie(a, sat=rng.uniform(0.6, 0.72), contrast=rng.uniform(0.9, 1.0),
                     black=rng.uniform(0.0, 0.02), warm=rng.uniform(0.03, 0.05))
    a = a / max(np.percentile(a, 99), 0.5) * 0.92  # phone auto-exposure pulls it back up
    length = int(w * rng.uniform(0.004, 0.008))
    out = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
    if length >= 3:  # slight hand shake
        k = np.zeros((length, length), np.float32)
        k[length // 2, :] = 1.0 / length
        out = Image.fromarray(cv2.filter2D(np.asarray(out), -1, k))
    return _jpeg(out, int(rng.integers(60, 78)))


FUNCS = {"grain": _grain, "motion": _motion, "lowres": _lowres, "soft": _soft, "film": _film,
         "selfie": _selfie, "nightselfie": _nightselfie}


def _tone(im: Image.Image, f: str) -> Image.Image:
    a = np.asarray(im).astype(np.float32) / 255
    if f == "bw":
        g = a @ np.array([0.299, 0.587, 0.114], np.float32)
        a = np.repeat((0.04 + g * 0.94)[..., None], 3, axis=2)
    elif f == "warm":
        a = a * np.array([1.08, 0.98, 0.80], np.float32) + np.array([0.04, 0.02, 0.0], np.float32)
    elif f == "cool":
        g = a.mean(2, keepdims=True)
        a = (g + (a - g) * 0.8) * np.array([0.88, 1.0, 1.12], np.float32) + np.array([0.0, 0.01, 0.03], np.float32)
    elif f == "faded":
        g = a.mean(2, keepdims=True)
        a = 0.14 + (g + (a - g) * 0.65) * 0.76
    elif f == "vivid":
        g = a.mean(2, keepdims=True)
        a = g + (a - g) * 1.45
        a = 0.5 + (a - 0.5) * 1.3
    return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))


def aug_path(path: str, kind: str) -> str:
    d, n = os.path.split(path)
    return os.path.join(d, f"{AUG_PREFIX}{kind}_{os.path.splitext(n)[0]}.png")


def _lead(kind: str, filt=None) -> str:
    return PHRASES[kind] + (" " + FILTER_PHRASES[filt] if filt else "")


def _write_captions(src: str, out: str, lead: str, caption_ext: str = ".txt"):
    """The copy's caption = the effect's sentences + the original's caption (every length)."""
    variants = load_caption_variants(src)
    if variants:
        save_caption_variants(out, {k: f"{lead} {v}" for k, v in variants.items()}, caption_ext)
        return
    with open(os.path.splitext(src)[0] + caption_ext, encoding="utf-8") as f:
        caption = f.read().strip()
    with open(os.path.splitext(out)[0] + caption_ext, "w", encoding="utf-8") as f:
        f.write(f"{lead} {caption}")


def parse_kind(name: str):
    """(kind, filter) from ``Aug_<kind>[-<filter>]_<stem>.png``."""
    tag = os.path.splitext(name[len(AUG_PREFIX):])[0].split("_", 1)[0]
    kind, _, filt = tag.partition("-")
    return (kind if kind in PHRASES else None), (filt if filt in FILTER_PHRASES else None)


def aug_lead(name: str) -> str:
    """The effect sentences for a copy's file name ("" if it is not one of ours)."""
    kind, filt = parse_kind(name)
    return _lead(kind, filt) if kind else ""


def refresh_captions(folder: str, caption_ext: str = ".txt") -> int:
    """Rebuild a copy's caption from its original's current caption, unless the copy was captioned
    from its own pixels (its sidecar names the caption model)."""
    names = sorted(os.listdir(folder))
    done = 0
    for n in names:
        if not n.startswith(AUG_PREFIX) or n.endswith((caption_ext, ".json")):
            continue
        kind, filt = parse_kind(n)
        src = crop_source(n, names)
        side = os.path.join(folder, os.path.splitext(n)[0] + ".captions.json")
        if os.path.isfile(side):
            try:
                with open(side, encoding="utf-8") as f:
                    if json.load(f).get("model"):
                        continue
            except ValueError:
                pass
        if kind is None or src is None or not os.path.isfile(os.path.join(folder, os.path.splitext(src)[0] + caption_ext)):
            continue
        _write_captions(os.path.join(folder, src), os.path.join(folder, n), _lead(kind, filt), caption_ext)
        done += 1
    return done


def make_copies(paths: List[str], caption_ext: str = ".txt") -> int:
    """One copy per image, every other one also with a colour filter; the caption gets the
    effects' sentences in front. Each copy takes the effect (and filter) the folder has fewest
    copies of, so effects added later get made before repeats. Images without a caption are
    skipped (caption them first)."""
    made = 0
    folder = os.path.dirname(paths[0]) if paths else ""
    have, have_f = coverage(os.listdir(folder) if folder and os.path.isdir(folder) else [])
    i = 0
    for p in paths:
        cap_path = os.path.splitext(p)[0] + caption_ext
        if not os.path.isfile(cap_path):
            continue
        kind = min(KINDS, key=lambda k: (have[k], KINDS.index(k)))
        filt = min(FILTERS, key=lambda f: (have_f[f], FILTERS.index(f))) if i % 2 == 1 else None
        i += 1
        out = aug_path(p, kind + (f"-{filt}" if filt else ""))
        if os.path.isfile(out):
            continue
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if filt:  # colour first, then the camera/compression damage on top
                im = _tone(im, filt)
            FUNCS[kind](im, _rng(os.path.basename(out))).save(out)
        _write_captions(p, out, _lead(kind, filt), caption_ext)
        have[kind] += 1
        if filt:
            have_f[filt] += 1
        made += 1
    return made
