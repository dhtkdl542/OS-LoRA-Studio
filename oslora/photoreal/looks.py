"""
Preset looks (DESIGN.md 2.3): an output black for auto levels, then point curves like a photo editor's RGB curves,
a fixed saturation and an optional halation glow.

The master curve shapes contrast for all channels; the red, green and blue curves grade colour. Lifting a channel
in a tonal range tints that range toward it: blue up in the shadows gives teal shadows, red up and blue down in the
highlights gives warm highlights. The shapes follow the published character of each film stock or style and the
measured reference images; they are fixed, so every preset looks the same way on every image.
"""

from dataclasses import dataclass, field

from .curves import Curve


@dataclass(frozen=True)
class Look:
    black: float                         # output black after auto levels
    master: Curve = field(default_factory=Curve)
    red: Curve = field(default_factory=Curve)
    green: Curve = field(default_factory=Curve)
    blue: Curve = field(default_factory=Curve)
    saturation: float = 1.0              # chroma scale; above 1 muted colours gain more (vibrance)
    halation: float = 0.0                # red glow around strong lights
    bloom: float = 0.0                   # warm soft glow spreading from all bright areas (backlight haze)
    vignette: float = 0.0                # corner darkening, share of light lost in the corners


LOOKS = {
    # auto levels and a touch of vibrance; renders are a little less saturated than photos (DESIGN.md 2.1)
    "자연스러운 사진": Look(black=0.03, saturation=1.06),
    # measured from three reference images: green-teal shadows, pink highlights held at 0.90, richer colour
    "레퍼런스 감성": Look(
        black=0.015,
        master=Curve([(0.25, 0.24), (0.5, 0.5), (0.75, 0.76), (1.0, 0.92)]),
        red=Curve([(0.25, 0.21), (0.75, 0.78)]),
        green=Curve([(0.25, 0.27), (0.75, 0.73)]),
        blue=Curve([(0.25, 0.29), (0.75, 0.73)]),
        saturation=1.1, halation=0.05),
    # Kodak Portra: soft but dense, warm creamy midtones, gently held highlights
    "포트라 (따뜻한 필름)": Look(
        black=0.03,
        master=Curve([(0.25, 0.26), (0.5, 0.51), (0.75, 0.76), (1.0, 0.97)]),
        red=Curve([(0.25, 0.26), (0.5, 0.53), (0.75, 0.78)]),
        blue=Curve([(0.25, 0.24), (0.5, 0.46), (0.75, 0.71)]),
        saturation=1.02),
    # Fuji Pro 400H: cool and clean, green-cyan shadows, slightly magenta highlights
    "후지 (청량한 필름)": Look(
        black=0.03,
        master=Curve([(0.25, 0.24), (0.75, 0.77)]),
        red=Curve([(0.25, 0.21), (0.75, 0.76)]),
        green=Curve([(0.25, 0.28), (0.75, 0.73)]),
        blue=Curve([(0.25, 0.29), (0.75, 0.78)]),
        saturation=1.05),
    # CineStill 800T: tungsten film, deep contrasty blacks, blue-teal deep shadows, red halation. The cool shift stays
    # below ~0.35 so skin keeps its warmth: blue lifted up to the midtones turned skin hue from 53 to 39 degrees, waxy
    "시네스틸 800T (밤·네온)": Look(
        black=0.01,
        master=Curve([(0.2, 0.16), (0.5, 0.5), (0.8, 0.84)]),
        red=Curve([(0.12, 0.09), (0.35, 0.34), (0.8, 0.82)]),
        green=Curve([(0.12, 0.125)]),
        blue=Curve([(0.12, 0.17), (0.35, 0.37), (0.6, 0.6)]),
        saturation=1.05, halation=0.15),
    # Kodak Gold: warm golden yellow, punchy and saturated
    "골드 (따뜻하고 진한)": Look(
        black=0.025,
        master=Curve([(0.25, 0.23), (0.75, 0.77)]),
        red=Curve([(0.25, 0.27), (0.5, 0.55)]),
        green=Curve([(0.5, 0.52)]),
        blue=Curve([(0.25, 0.21), (0.5, 0.43), (0.75, 0.69)]),
        saturation=1.12),
    # light and airy: lifted faded blacks, low contrast, bright midtones, soft pink, muted colour
    "화사한 감성 (밝고 뽀얀)": Look(
        black=0.06,
        master=Curve([(0.25, 0.29), (0.5, 0.56), (0.75, 0.8), (1.0, 0.98)]),
        red=Curve([(0.5, 0.51)]),
        green=Curve([(0.5, 0.48)]),
        blue=Curve([(0.5, 0.51)]),
        saturation=0.92),
    # measured from three heavily graded film-scan portraits: no true black (0.09-0.20), whites held at 0.82-0.90,
    # green shadows, golden highlights, rich colour, soft glow around the light, darker corners
    "필름 스캔 (빛바랜)": Look(
        black=0.12,
        master=Curve([(0.25, 0.26), (0.5, 0.5), (0.75, 0.74), (1.0, 0.88)]),
        red=Curve([(0.15, 0.13), (0.4, 0.39), (0.75, 0.77)]),
        green=Curve([(0.15, 0.17), (0.4, 0.41)]),
        blue=Curve([(0.15, 0.14), (0.5, 0.48), (0.75, 0.69), (1.0, 0.9)]),
        saturation=1.2, bloom=0.25, vignette=0.3),
}
