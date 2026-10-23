"""Photo texture for training images: the "natural photo" auto tone plus phone-camera sensor
noise, from the user's ComfyUI-PhotoReal nodes (core/ copied unchanged, MIT, (c) 2026 SKFRMSEHF).

A dataset of spotless AI renders teaches the LoRA that the character is spotless. Giving the
images a real photo's finish (lifted blacks, a touch of vibrance, fine Poisson-Gaussian noise)
makes that the default instead. Applied after resizing to the training bucket: noise added at
the source resolution would average away in the downscale. Measured at ~1 MP: the phone preset
adds ~2/255 noise, which survives the VAE; the default preset (0.6/255) would not.
"""

import dataclasses
import zlib

import numpy as np
import torch
from PIL import Image

from .looks import LOOKS
from .noise import NOISE_PRESETS
from .tone import ToneCorrector

VERSION = 1  # bump when the recipe changes (part of the latent cache key)
TONE = "자연스러운 사진"
NOISE = "폰카 느낌"
ISO_JITTER = (0.7, 1.5)  # per image, so no single noise signature repeats over the dataset


def apply(img: Image.Image, name: str) -> Image.Image:
    """Tone + noise, deterministic per file name."""
    seed = zlib.crc32(name.encode("utf-8"))
    x = torch.from_numpy(np.asarray(img.convert("RGB")).astype(np.float32) / 255)
    x, _ = ToneCorrector(LOOKS[TONE], 1.0).apply(x)
    rng = np.random.default_rng(seed)
    base = NOISE_PRESETS[NOISE]
    noise = dataclasses.replace(base, iso=base.iso * float(rng.uniform(*ISO_JITTER)))
    x = noise.apply(x[None], seed)[0]
    return Image.fromarray((x.clamp(0, 1).numpy() * 255).round().astype(np.uint8))
