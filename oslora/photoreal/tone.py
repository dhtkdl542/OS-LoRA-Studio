"""
Automatic tone and colour grading toward a preset look (DESIGN.md 2), in the order a photo editor applies it:
auto levels (measured per image), the look's RGB point curves, halation, bloom and vignette in linear light,
saturation in Oklab.
"""

from dataclasses import dataclass

import torch

from .color import linear_to_oklab, linear_to_srgb, luminance, oklab_to_linear, srgb_to_linear
from .filters import blur_wide
from .levels import AutoLevels


@dataclass
class ToneReport:
    """What a ToneCorrector did to one image."""
    input_blacks: list           # measured per-channel black points mapped to the look's output black

    def describe(self):
        r, g, b = self.input_blacks
        return f"input black R {r:.3f} G {g:.3f} B {b:.3f}"


class ToneCorrector:
    """Grades an sRGB image with a Look. `strength` blends between the input (0) and the full look (1)."""

    VIBRANCE_CHROMA = 0.15           # colours at this Oklab chroma or above get no saturation boost
    HALATION_COLOR = (1.0, 0.2, 0.06)  # red: an orange glow read as a yellow cast over neon scenes
    HALATION_RADIUS = 0.012          # fraction of the image's long side
    HALATION_START = 0.8             # linear luminance where the glow starts, so only strong lights glow
    BLOOM_COLOR = (1.0, 0.82, 0.55)  # warm, like sunlight scattered in old lens coatings
    BLOOM_RADIUS = 0.035
    BLOOM_START = 0.35

    def __init__(self, look, strength):
        self.look = look
        self.strength = strength

    def apply(self, image):
        """image (H, W, 3) sRGB in [0, 1]. Returns the graded image and a ToneReport."""
        out, blacks = AutoLevels(self.look.black).apply(image)
        out = self._curves(out)
        lin = self._vignette(self._bloom(self._halation(srgb_to_linear(out))))
        out = linear_to_srgb(oklab_to_linear(self._saturation(linear_to_oklab(lin))))
        return torch.lerp(image, out, self.strength), ToneReport(blacks)

    def _curves(self, image):
        look = self.look
        out = look.master(image)
        return torch.stack([curve(out[..., c]) for c, curve in enumerate((look.red, look.green, look.blue))], dim=-1)

    def _halation(self, lin):
        if not self.look.halation:
            return lin
        bright = ((luminance(lin) - self.HALATION_START) / (1 - self.HALATION_START)).clamp(0, 1)
        glow = blur_wide(bright, self.HALATION_RADIUS * max(bright.shape))
        return lin + self.look.halation * glow.unsqueeze(-1) * lin.new_tensor(self.HALATION_COLOR)

    def _bloom(self, lin):
        if not self.look.bloom:
            return lin
        bright = (luminance(lin) - self.BLOOM_START).clamp(min=0)
        glow = blur_wide(bright, self.BLOOM_RADIUS * max(bright.shape))
        return lin + self.look.bloom * glow.unsqueeze(-1) * lin.new_tensor(self.BLOOM_COLOR)

    def _vignette(self, lin):
        if not self.look.vignette:
            return lin
        h, w = lin.shape[:2]
        yy = torch.linspace(-1, 1, h, device=lin.device, dtype=lin.dtype).view(-1, 1)
        xx = torch.linspace(-1, 1, w, device=lin.device, dtype=lin.dtype).view(1, -1)
        r2 = (xx * xx + yy * yy) / 2                       # 0 at the centre, 1 in the corners
        return lin * (1 - self.look.vignette * r2 * r2).unsqueeze(-1)

    def _saturation(self, lab):
        s = self.look.saturation
        if s == 1:
            return lab
        ab = lab[..., 1:]
        if s > 1:  # vibrance: muted colours get the full boost, already saturated ones none
            ab = ab * (1 + (s - 1) * (1 - ab.norm(dim=-1) / self.VIBRANCE_CHROMA).clamp(0, 1)).unsqueeze(-1)
        else:
            ab = ab * s
        return torch.cat([lab[..., :1], ab], dim=-1)
