from .stats import percentiles


class AutoLevels:
    """Per-channel black point, measured from each image's histogram, like a photo editor's auto levels.

    Each channel's darkest 0.5% is mapped to the same output black: crushed AI blacks are lifted to a printable level
    and a colour cast in the shadows is neutralised. The input black point is capped, so an intentionally hazy image
    is not stretched into a contrasty one. The highlights are left to the look's curves: stretching them per channel
    blew out bright skin."""

    PERCENTILE = 0.005
    MAX_INPUT_BLACK = 0.06

    def __init__(self, output_black):
        self.output_black = output_black

    def apply(self, image):
        """image (H, W, 3) sRGB in [0, 1]. Returns the levelled image and the measured per-channel input blacks."""
        blacks = [min(percentiles(image[..., c], [self.PERCENTILE])[0], self.MAX_INPUT_BLACK) for c in range(3)]
        lo = image.new_tensor(blacks)
        x = ((image - lo) / (1 - lo)).clamp(0, 1)
        return self.output_black + (1 - self.output_black) * x, blacks
