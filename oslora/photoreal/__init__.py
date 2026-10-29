"""Photo texture for training images: the "natural photo" auto tone plus phone-camera sensor noise,
from the user's ComfyUI-PhotoReal nodes (core/ copied unchanged, MIT, (c) 2026 SKFRMSEHF).

A dataset of spotless AI renders teaches the LoRA that the character is spotless. Fine
Poisson-Gaussian noise on some of the steps makes a real photo's grain acceptable without
changing the look. Applied after resizing to the training bucket: noise added at the source
resolution would average away in the downscale. Measured at ~1 MP: the phone preset adds ~2/255
noise, which survives the VAE; the default preset (0.6/255) would not.

Version 1 also applied the "natural photo" auto tone (lifted blacks, lower contrast). Uncaptioned,
on half the steps, that became the LoRA's default: Qwen 2.1 outputs showed 25-45 % less edge
contrast and detail at hair / skin boundaries. Version 2 kept only the noise, on a quarter of
the steps -- which brought back the glossy AI skin. Version 3: the tone again, on a third of the
steps (the user's call, from the Qwen Judam results). Noise only (version 2) stays available for
the proxy experiments, which must all train the same way.
"""

import dataclasses
import zlib

import numpy as np
import torch
from PIL import Image

from .looks import LOOKS
from .noise import NOISE_PRESETS
from .tone import ToneCorrector

VERSION = 3  # tone + noise; bump when the recipe changes (part of the latent cache key)
NOISE_ONLY = 2  # the version-2 recipe, still used by the proxy experiments
TONE = "자연스러운 사진"
NOISE = "폰카 느낌"
RATE = 1 / 3  # share of training steps that see the textured version
# put in front of the caption on those steps (with the tone): the look is tied to words, like the
# degraded copies, instead of becoming the character's default (the user's call, 2026-10-29)
LEAD = ("A natural-looking phone photo with soft, true-to-life contrast, slightly lifted shadows and fine "
        "sensor noise over the whole image.")
ISO_JITTER = (0.7, 1.5)  # per image, so no single noise signature repeats over the dataset


def cache_tag(tone: bool) -> str:
    """Part of the latent cache key: the recipe the textured latent was made with."""
    return f"photo{VERSION if tone else NOISE_ONLY}"


def apply(img: Image.Image, name: str, tone: bool = True) -> Image.Image:
    """Tone (unless ``tone`` is off) + sensor noise, deterministic per file name."""
    seed = zlib.crc32(name.encode("utf-8"))
    x = torch.from_numpy(np.asarray(img.convert("RGB")).astype(np.float32) / 255)
    if tone:
        x, _ = ToneCorrector(LOOKS[TONE], 1.0).apply(x)
    rng = np.random.default_rng(seed)
    base = NOISE_PRESETS[NOISE]
    noise = dataclasses.replace(base, iso=base.iso * float(rng.uniform(*ISO_JITTER)))
    x = noise.apply(x[None], seed)[0]
    return Image.fromarray((x.clamp(0, 1).numpy() * 255).round().astype(np.uint8))
