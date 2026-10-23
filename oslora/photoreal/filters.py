import math

import torch
import torch.nn.functional as F


def blur(x, sigma, mode="replicate"):
    """Separable gaussian blur over the last two dims of a (..., H, W) tensor."""
    r = max(1, math.ceil(sigma * 3))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    y = x.reshape(-1, 1, *x.shape[-2:])
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode=mode), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode=mode), k.view(1, 1, -1, 1))
    return y.reshape(x.shape)


def blur_wide(x, sigma):
    """Gaussian blur of a smooth (..., H, W) map computed at reduced resolution: a 30 px blur of a 3 MP image costs over
    a second on CPU at full size, and the result is smooth enough that a coarser grid is indistinguishable."""
    f = max(1, int(sigma // 6))
    if f == 1:
        return blur(x, sigma)
    small = F.avg_pool2d(x.reshape(-1, 1, *x.shape[-2:]), f, ceil_mode=True)
    out = F.interpolate(blur(small, sigma / f), size=x.shape[-2:], mode="bilinear", align_corners=False)
    return out.reshape(x.shape)
