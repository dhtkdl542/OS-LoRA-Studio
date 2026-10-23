"""
Poisson-Gaussian camera sensor noise applied in linear light.

  - shot noise  : sigma = sqrt(L * gain / S0)        (photon statistics, signal dependent)
  - read noise  : sigma = R * gain / S0              (electronics, signal independent)
  where gain = ISO / 100, S0 = full-well electrons at base ISO, R = read noise in electrons.
  - luminance / chroma split: luma noise is fine-grained (slight demosaic correlation), chroma noise is low-pass
    filtered (like in-camera chroma NR) and scaled down.
  - area normalisation: an output pixel that represents several sensor pixels averages their noise
    (sigma / sqrt(sensor_px / output_px)).
  - noise is added in linear light, clipped (sensor saturation), then re-encoded to sRGB, so the gamma curve naturally
    makes shadows look noisier than highlights.
"""

from dataclasses import dataclass

import torch

from .color import linear_to_srgb, srgb_to_linear
from .filters import blur


@dataclass(frozen=True)
class SensorNoise:
    iso: float = 1600
    strength: float = 1.0        # multiplier on top of the physical model
    chroma: float = 0.3          # colour noise relative to luma noise
    chroma_blur: float = 1.5     # low-pass on colour noise, px
    grain_size: float = 0.6      # spatial correlation of luma noise, px
    sensor_mp: float = 24.0      # sensor megapixels for area normalisation; 0 = one sensor pixel per output pixel
    full_well_e: float = 25000.0  # saturation electrons at base ISO
    read_noise_e: float = 3.0    # read noise, electrons

    def apply(self, image, seed):
        """image (B, H, W, 3) sRGB in [0, 1]. Deterministic per seed and does not touch the global torch RNG."""
        if self.strength <= 0:
            return image
        x = image.permute(0, 3, 1, 2)
        n, _, h, w = x.shape
        lin = srgb_to_linear(x.clamp(0, 1))
        gain = self.iso / 100.0
        area = max(1.0, self.sensor_mp * 1e6 / (h * w)) if self.sensor_mp > 0 else 1.0
        var = lin * gain / self.full_well_e + (self.read_noise_e * gain / self.full_well_e) ** 2
        sigma = torch.sqrt(var / area) * self.strength

        # drawn on CPU so a seed gives the same grain on every device
        gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
        z_luma = self._unit_field((n, 1, h, w), self.grain_size, gen, image.device)
        z_chroma = self._unit_field((n, 3, h, w), self.chroma_blur, gen, image.device)
        z_chroma = z_chroma - z_chroma.mean(dim=1, keepdim=True)  # pure colour, no luma
        z = z_luma + self.chroma * z_chroma
        return linear_to_srgb(lin + sigma * z).permute(0, 2, 3, 1)

    @staticmethod
    def _unit_field(shape, sigma, gen, device):
        """White gaussian noise, blurred by sigma px, renormalised to unit std per channel."""
        z = torch.randn(shape, generator=gen, dtype=torch.float32).to(device)
        if sigma > 0.05:
            z = blur(z, sigma, mode="reflect")
        z = z - z.mean(dim=(2, 3), keepdim=True)
        return z / z.std(dim=(2, 3), keepdim=True).clamp(min=1e-8)


# Starting points for SensorNoise, selectable in the node. Every field is listed, so a preset fully defines the noise.
NOISE_PRESETS = {
    "범용 (기본)": SensorNoise(iso=800, chroma=0.2),
    "맑은 날 야외 (깨끗함)": SensorNoise(iso=300, chroma=0.2),
    "실내·카페": SensorNoise(iso=1600, chroma=0.2),
    "어두운 실내·밤": SensorNoise(iso=4000, chroma=0.3),
    "폰카 느낌": SensorNoise(iso=640, chroma=0.25, sensor_mp=12.0, full_well_e=6000.0),
    # coarse clumped grain with little colour noise, no area averaging. ISO 3200 matched film-scan examples (640-816 px)
    # viewed whole, but at 100% on a 2048 px render the skin looked dirty; 2000 sits between
    "필름 입자": SensorNoise(iso=2000, chroma=0.12, chroma_blur=2.0, grain_size=1.8, sensor_mp=0.0),
}
