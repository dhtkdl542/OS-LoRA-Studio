"""Caption-aware image degradations for a few dataset copies.

A dataset of clean, retouched images teaches the LoRA that the character *is* clean: prompts
asking for grain, shake or a cheap phone look get scrubbed (measured on Damdeari: frame
sharpness on a flash/motion-blur prompt 34-59 without the LoRA, 126-229 with it). A handful of
degraded copies whose caption starts by naming the degradation ties the look to words instead:
"grainy" means grainy, also on her face. The originals stay clean, so the default stays clean.

Each copy gets one effect and every other copy also a colour filter (bw, warm, cool, faded, vivid):
``Aug_<kind>[-<filter>]_<original stem>.png`` with ``<phrases> <original caption>``. Effects that
soften the image (motion, lowres, soft, selfie, nightselfie) are kept to about a third of the
copies; the rest add flash, light leaks, over-exposure, harsh contrast, HDR, grain or film looks,
which keep the detail. New copies take the effects the folder has fewest of, in an order that
alternates the two groups, so a dataset made before an effect existed gets it.
"""

import json
import os
import re
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
    # looks that keep (or add) sharp detail, so the copies don't pull the character towards soft
    "flash": ("A direct on-camera flash photo. The flash lights her face and the front of her body brightly and flatly, "
              "with crisp detail, slightly hot highlights on the skin and hard little shadows behind her, while the "
              "background falls off into darkness."),
    "lightleak": ("A photo with a warm light leak: a red-orange glow bleeds in from one edge of the frame and washes "
                  "over that side, while the rest of the image stays sharp."),
    "overexposed": ("A bright, overexposed high-key photo. The highlights are blown out to white, the shadows are light "
                    "and airy and the colors are pale, but the edges and hair stay sharp."),
    "harsh": ("A high-contrast photo in harsh, hard light: deep crushed shadows, bright highlights, punchy texture and "
              "crisp, sharp detail in her hair and skin."),
    "hdr": ("An over-processed phone photo with heavy HDR and over-sharpening: bright halos along the edges, crunchy "
            "detail in her hair and skin, and boosted local contrast."),
}
# blur / softening effects alternate with ones that keep detail, so the first copies of a dataset
# spread over both (7 copies -> 2 soft ones)
KINDS = ["grain", "flash", "motion", "hdr", "film", "lightleak", "lowres", "harsh", "selfie", "overexposed", "soft",
         "nightselfie"]
BLUR_KINDS = {"motion", "lowres", "soft", "selfie", "nightselfie"}
BLUR_SHARE = 0.35  # at most about a third of a dataset's copies soften the image

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
               "selfie": "폰 셀카", "nightselfie": "밤 셀카", "flash": "플래시", "lightleak": "빛샘",
               "overexposed": "과다 노출", "harsh": "강한 대비", "hdr": "폰 HDR 과보정"}
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


def _radial(h: int, w: int, cx: float, cy: float) -> np.ndarray:
    """Distance from (cx, cy) in image units, about 1 at the corners."""
    y, x = np.ogrid[:h, :w]
    return np.sqrt(((x / w - cx) / 0.75) ** 2 + ((y / h - cy) / 0.75) ** 2)


def _flash(im: Image.Image, rng) -> Image.Image:
    a = np.asarray(im).astype(np.float32) / 255
    h, w = a.shape[:2]
    d = _radial(h, w, 0.5 + rng.uniform(-0.05, 0.05), rng.uniform(0.35, 0.45))
    a = a * np.clip(1.35 - 0.95 * d, 0.35, 1.35)[..., None]  # bright centre, falls off into the dark
    a = 1 - (1 - np.clip(a, 0, 1)) ** 1.25                    # hot, flat highlights on the skin
    a = (a - 0.5) * 1.12 + 0.5
    a *= np.array([0.97, 1.0, 1.05], np.float32)             # cool flash white balance
    return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))


def _lightleak(im: Image.Image, rng) -> Image.Image:
    a = np.asarray(im).astype(np.float32) / 255
    h, w = a.shape[:2]
    side = int(rng.integers(4))
    t = np.linspace(1, 0, w if side < 2 else h, dtype=np.float32) ** rng.uniform(1.8, 2.6)
    t = t if side % 2 == 0 else t[::-1]
    g = t[None, :, None] if side < 2 else t[:, None, None]
    glow = np.array([1.0, rng.uniform(0.35, 0.55), 0.12], np.float32) * rng.uniform(0.55, 0.8)
    a = 1 - (1 - a) * (1 - g * glow)                          # screen blend: light added, detail kept
    return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))


def _overexposed(im: Image.Image, rng) -> Image.Image:
    a = np.asarray(im).astype(np.float32) / 255
    a = a * rng.uniform(1.35, 1.55) + 0.06                    # brighter, lifted shadows
    g = a.mean(2, keepdims=True)
    a = g + (a - g) * 0.8                                     # paler colour
    return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))


def _harsh(im: Image.Image, rng) -> Image.Image:
    a = np.asarray(im).astype(np.float32) / 255
    a = 1 / (1 + np.exp(-(a - 0.5) * rng.uniform(7.5, 9.0)))  # S-curve: crushed shadows, bright highlights
    a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6)
    out = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
    return out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))


def _hdr(im: Image.Image, rng) -> Image.Image:
    a = np.asarray(im).astype(np.float32) / 255
    base = np.asarray(im.filter(ImageFilter.GaussianBlur(im.width * 0.012)), dtype=np.float32) / 255
    a = a + (a - base) * rng.uniform(0.8, 1.1)               # local contrast boost (halos along edges)
    a = 0.5 + (a - 0.5) * 0.9                                 # tone-mapped: shadows lifted, highlights held
    g = a.mean(2, keepdims=True)
    a = g + (a - g) * 1.15
    out = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
    return out.filter(ImageFilter.UnsharpMask(radius=1.5, percent=110, threshold=1))  # over-sharpened


FUNCS = {"grain": _grain, "motion": _motion, "lowres": _lowres, "soft": _soft, "film": _film,
         "selfie": _selfie, "nightselfie": _nightselfie, "flash": _flash, "lightleak": _lightleak,
         "overexposed": _overexposed, "harsh": _harsh, "hdr": _hdr}


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


# the original caption's image-finish statements ("The image has a polished finish: sharp focus,
# retouched skin, minimal grain ...") contradict a degraded copy and would teach that those words
# also fit a blurry, noisy picture; the copy's effect sentences replace them
_QUALITY = re.compile(
    r"\b(grain|grainy|noise|noisy|retouch\w*|airbrush\w*|poreless|polished|render(?:ed|ing)?|compression|compressed|"
    r"jpe?g|artifacts?|sharp(?:ness|ly)?|crisp|in focus|out of focus|blurr?y|blur|high[- ]resolution|low[- ]resolution|"
    r"ai[- ]generated|digital(?:ly)?|snapshot|smooth\w*|clean|film(?:ic)?|pixel\w*|quality)\b", re.I)
_OPENER = re.compile(r"^(the (image|photo|picture|overall|whole image)|overall|it (has|is|looks))\b", re.I)
_MID = re.compile(r",?\s*(?:and|while|with)?\s*the (?:image|photo|overall look|whole image) (?:has|is|looks)\b.*$", re.I)
# the retouched-skin look ("her skin appears very smooth and lightly retouched with a gentle sheen")
_SKIN = re.compile(r"\b(smooth\w*|retouch\w*|airbrush\w*|poreless|flawless|sheen|glow\w*|glossy|dewy|luminous|porcelain)\b", re.I)
_SKIN_TAIL = re.compile(r",?\s*(?:and|with|while)?\s*(?:her|the|whose) (?:skin|complexion)\b.*$", re.I)
# trailing clauses that only describe the look: ", giving the image a clean, polished look",
# ", captured in a smooth, polished portrait style"
_TAILS = (_MID,
          re.compile(r",?\s*(?:and\s+)?(?:giving|lending|creating|producing|resulting in)\b.*$", re.I),
          re.compile(r",?\s*(?:captured|rendered|shot|presented)?\s*in an? (?:[\w-]+,\s+)*[^,;]*?\b(?:style|aesthetic|look)\b.*$", re.I))
# style adjectives in front of the subject: "A polished digital-render style portrait", "A smooth,
# highly retouched close-up portrait"
_HEAD_NOUN = re.compile(r"\b(portrait|photo\w*|image|shot|picture|close-up|selfie)\b", re.I)
_STYLE_ADJ = re.compile(r"\b(?:highly |very |lightly )?(?:polished|smooth|retouched|airbrushed|digital[- ]render|digital|"
                        r"rendered|render|glossy|flawless|crisp|sharp|clean|high[- ]end|high[- ]definition)(?:[- ]style)?\b,?\s*", re.I)


def _finishy(s: str) -> bool:
    s = s.strip()
    return ("finish" in s.lower() or bool(_OPENER.match(s) and _QUALITY.search(s)) or len(_QUALITY.findall(s)) >= 3
            or bool(re.match(r"(?:her|the) (?:skin|complexion)\b", s, re.I) and _SKIN.search(s))
            or bool(re.search(r"\bgives?\b.*\b(?:look|feel|aesthetic)\b", s, re.I) and _QUALITY.search(s)))


# look phrases inside a list: "... even lighting and a smooth, polished finish", "... a smooth dewy
# complexion, and glossy lips", "..., rendered with smooth skin and a studio-like finish"
_LIST = r"(?:,\s*(?:and\s+)?(?:with\s+)?|\s+and\s+(?:with\s+)?|\s+with\s+|\s+in\s+)"  # how a list item is joined on
_PHRASES = ((re.compile(r",?\s*(?:rendered|finished) with\b[^;.]*", re.I), _QUALITY),
            (re.compile(_LIST + r"an?\s+(?:[\w-]+,\s+)*[^,;.]*?\bfinish\b[^,;.]*", re.I), None),
            (re.compile(_LIST + r"an?\s+(?:[\w-]+,\s+)*[^,;.]*?\bcomplexion\b[^,;.]*", re.I), _SKIN),
            (re.compile(r"(?:,\s*which features|" + _LIST + r")\s*(?:(?:extremely |very |ultra-|highly )?smooth,?\s+)?"
                        r"(?:lightly |slightly |heavily |highly )?(?:retouched|airbrushed|poreless|flawless)\s+skin\b", re.I), None),
            (re.compile(r"\s+and the smoothness of (?:her|the) skin", re.I), None))


def _trim(s: str) -> str:
    """A content clause without its look statements ("..., and the image has a polished finish",
    "..., and her skin appears smooth and retouched", "... and a smooth, polished finish")."""
    for pat, words in _PHRASES:
        s = pat.sub(lambda m: "" if words is None or words.search(m.group(0)) else m.group(0), s)
    for pat, words in [(p, _QUALITY) for p in _TAILS] + [(_SKIN_TAIL, _SKIN)]:
        cut = pat.sub("", s)
        if cut != s and cut.strip() and words.search(s[len(cut):]):
            s = cut.rstrip(" ,;")
    return s


def _plain_head(s: str) -> str:
    """The subject without style adjectives in front of its noun."""
    m = _HEAD_NOUN.search(s)
    if not m:
        return s
    before = s[:m.start()]
    plain = _STYLE_ADJ.sub("", before)
    if plain == before:
        return s
    plain = re.sub(r"[,\s]+$", " ", plain).lstrip()  # "A warm, " -> "A warm "
    art = re.match(r"(an?)\s+", plain, re.I)
    if art:  # the article follows the new first word
        rest = plain[art.end():] + s[m.start():]
        return ("An " if rest[:1].lower() in "aeiou" else "A ") + rest
    out = plain + s[m.start():]
    return out[:1].upper() + out[1:]


def strip_finish(text: str) -> str:
    """Drop a caption's image-finish statements (sharpness, grain, retouching, render look), keeping
    the subject and everything that describes content. Works per clause, so a sentence that mixes
    both ("The lighting is even, and the image has a polished finish ...", "...gray background; the
    image has a smooth finish") keeps its content half -- the medium captions are one such sentence."""
    out = []
    for i, sent in enumerate(x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x):
        body = sent.rstrip(".!? ")
        kept = []
        for j, clause in enumerate(body.split(";")):
            clause = _trim(clause.strip())
            if (i, j) == (0, 0):  # the subject
                kept.append(_plain_head(clause))
            elif clause and not _finishy(clause):
                kept.append(clause)
        if kept and kept[0]:
            out.append("; ".join(kept) + (sent[len(body):].strip()[:1] or "."))
    return " ".join(out)


def _write_captions(src: str, out: str, lead: str, caption_ext: str = ".txt"):
    """The copy's caption = the effect's sentences + the original's caption without its finish
    statements (every length)."""
    variants = load_caption_variants(src)
    if variants:
        save_caption_variants(out, {k: f"{lead} {strip_finish(v)}" for k, v in variants.items()}, caption_ext)
        return
    with open(os.path.splitext(src)[0] + caption_ext, encoding="utf-8") as f:
        caption = f.read().strip()
    with open(os.path.splitext(out)[0] + caption_ext, "w", encoding="utf-8") as f:
        f.write(f"{lead} {strip_finish(caption)}")


def parse_kind(name: str):
    """(kind, filter) from ``Aug_<kind>[-<filter>]_<stem>.png``."""
    tag = os.path.splitext(name[len(AUG_PREFIX):])[0].split("_", 1)[0]
    kind, _, filt = tag.partition("-")
    return (kind if kind in PHRASES else None), (filt if filt in FILTER_PHRASES else None)


def aug_lead(name: str) -> str:
    """The effect sentences for a copy's file name ("" if it is not one of ours)."""
    kind, filt = parse_kind(name)
    return _lead(kind, filt) if kind else ""


def _clean_own_caption(path: str, lead: str, caption_ext: str) -> bool:
    """A copy captioned from its own pixels keeps that caption; only finish statements that
    contradict its effect go (the effect sentences in front stay). True if anything changed."""
    variants = load_caption_variants(path) or {}
    out = {}
    for k, v in variants.items():
        body = v[len(lead):].lstrip() if lead and v.startswith(lead) else v
        out[k] = (f"{lead} " if lead and v.startswith(lead) else "") + strip_finish(body)
    if not variants or out == variants:
        return False
    model = None
    side = os.path.splitext(path)[0] + ".captions.json"
    try:
        with open(side, encoding="utf-8") as f:
            model = json.load(f).get("model")
    except (OSError, ValueError):
        pass
    save_caption_variants(path, out, caption_ext, **({"model": model} if model else {}))
    return True


def refresh_captions(folder: str, caption_ext: str = ".txt") -> int:
    """Rebuild a copy's caption from its original's current caption (without the original's finish
    statements). A copy captioned from its own pixels (its sidecar names the caption model) keeps
    its caption, minus finish statements that contradict the effect."""
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
                        done += _clean_own_caption(os.path.join(folder, n), _lead(kind, filt) if kind else "", caption_ext)
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
    max_blur = max(1, int(BLUR_SHARE * (sum(have.values()) + len(paths))))
    i = 0
    for p in paths:
        cap_path = os.path.splitext(p)[0] + caption_ext
        if not os.path.isfile(cap_path):
            continue
        blur_full = sum(have[k] for k in BLUR_KINDS) >= max_blur
        kind = min(KINDS, key=lambda k: (have[k], blur_full and k in BLUR_KINDS, KINDS.index(k)))
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

def plan_rebalance(aug_names: List[str], target: int, cap: int):
    """Which copies to retire and which effects to add so every effect appears at most once and the
    softening ones stay within BLUR_SHARE. Keeps the dataset's copy count (at least ``target``, at
    most ``cap``). Returns (retire: [names], add: [kinds])."""
    total = min(max(target, len(aug_names)), cap)
    max_blur = max(1, int(BLUR_SHARE * total))
    by_kind = {}
    for n in sorted(aug_names):
        k = parse_kind(n)[0]
        if k:
            by_kind.setdefault(k, []).append(n)
    keep, blur = [], 0
    for k in KINDS:  # one copy per effect, in the interleaved order, softening ones up to the limit
        if k in by_kind and len(keep) < total:
            if k in BLUR_KINDS:
                if blur >= max_blur:
                    continue
                blur += 1
            keep.append(by_kind[k][0])
    add = []
    for k in KINDS:
        if len(keep) + len(add) >= total:
            break
        if k in by_kind and by_kind[k][0] in keep:
            continue
        if k in BLUR_KINDS:
            if blur >= max_blur:
                continue
            blur += 1
        add.append(k)
    retire = [n for n in aug_names if n not in keep]
    return retire, add

