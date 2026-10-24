"""Photo texture for training images: phone-camera sensor noise, from the user's ComfyUI-PhotoReal
nodes (core/ copied unchanged, MIT, (c) 2026 SKFRMSEHF).

A dataset of spotless AI renders teaches the LoRA that the character is spotless. Fine
Poisson-Gaussian noise on some of the steps makes a real photo's grain acceptable without
changing the look. Applied after resizing to the training bucket: noise added at the source
resolution would average away in the downscale. Measured at ~1 MP: the phone preset adds ~2/255
noise, which survives the VAE; the default preset (0.6/255) would not.

Version 1 also applied the "natural photo" auto tone (lifted blacks, lower contrast). Uncaptioned,
on half the steps, that became the LoRA's default: Qwen 2.1 outputs showed 25-45 % less edge
contrast and detail at hair / skin boundaries. Version 2 keeps only the noise, on a quarter of
the steps.
"""

import dataclasses
import zlib

import numpy as np
import torch
from PIL import Image

from .noise import NOISE_PRESETS

VERSION = 2  # bump when the recipe changes (part of the latent cache key)
NOISE = "폰카 느낌"
RATE = 0.25  # share of training steps that see the noisy version
ISO_JITTER = (0.7, 1.5)  # per image, so no single noise signature repeats over the dataset


def apply(img: Image.Image, name: str) -> Image.Image:
    """Sensor noise only, deterministic per file name."""
    seed = zlib.crc32(name.encode("utf-8"))
    x = torch.from_numpy(np.asarray(img.convert("RGB")).astype(np.float32) / 255)
    rng = np.random.default_rng(seed)
    base = NOISE_PRESETS[NOISE]
    noise = dataclasses.replace(base, iso=base.iso * float(rng.uniform(*ISO_JITTER)))
    x = noise.apply(x[None], seed)[0]
    return Image.fromarray((x.clamp(0, 1).numpy() * 255).round().astype(np.uint8))
