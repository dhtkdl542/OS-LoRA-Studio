import torch

LUMA = (0.2126, 0.7152, 0.0722)

# Oklab (Ottosson 2020), from and to linear sRGB.
_LMS = ((0.4122214708, 0.5363325363, 0.0514459929), (0.2119034982, 0.6806995451, 0.1073969566), (0.0883024619, 0.2817188376, 0.6299787005))
_LAB = ((0.2104542553, 0.7936177850, -0.0040720468), (1.9779984951, -2.4285922050, 0.4505937099), (0.0259040371, 0.7827717662, -0.8086757660))


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=1e-12) ** (1 / 2.4) - 0.055)


def luminance(x):
    """Rec. 709 weighted sum over the last (RGB) dim."""
    return (x * x.new_tensor(LUMA)).sum(-1)


def linear_to_oklab(x):
    lms = (x @ x.new_tensor(_LMS).T).clamp(min=0) ** (1 / 3)
    return lms @ x.new_tensor(_LAB).T


def oklab_to_linear(lab):
    lms = (lab @ torch.linalg.inv(lab.new_tensor(_LAB)).T) ** 3
    return lms @ torch.linalg.inv(lab.new_tensor(_LMS)).T
