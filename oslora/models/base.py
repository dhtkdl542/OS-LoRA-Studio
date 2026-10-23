"""What the trainer needs from a model family. One adapter per model (MOSS-Image, Krea 2, ...).

The trainer works in this model-independent frame:
    latents  (B, C, H/16, W/16) as produced by ``vae.encode`` (already normalized / packed)
    x_t      = (1 - t) * x0 + t * noise,  t in [0, 1]   (rectified flow)
    target   = noise - x0                               (velocity)
and every model-specific detail lives behind ``predict`` / ``sample`` / the loaders.
"""

from typing import Dict, List, Optional, Sequence

import torch
from torch import nn


class ModelAdapter:
    key: str = ""             # config.model / cache folder name
    label: str = ""           # shown in the UI
    base_model_tag: str = ""  # ss_base_model in the saved LoRA's metadata
    lora_prefix: str = "diffusion_model."  # key prefix of the saved LoRA (what ComfyUI maps)
    bucket_step: int = 16  # training image sizes are multiples of this (pixels)
    # LoRA targets by preset name (regexes over module names of the DiT)
    lora_targets: Dict[str, List[str]] = {}
    # what the inference model is called in the UI ("SFT", "Turbo", ...)
    inference_label: str = "추론용"
    # viewing samples / comparison on the inference model
    sample_steps: int = 30
    sample_cfg: float = 3.0
    # the inference model is step-distilled (Turbo): always render it with ``sample_steps``
    distilled_steps: bool = False
    # quality rounds (small renders)
    quality_steps: int = 16
    # renders on the training (base) model itself (leak probe references)
    base_sample_steps: int = 16
    base_sample_cfg: float = 3.0
    # DiT blocks kept in RAM and swapped in per forward (models that don't fit 16 GB otherwise)
    swap_blocks: int = 0

    # -- loaders ------------------------------------------------------------------------
    def text_encoder(self, path: str, log=print):
        """Object with ``encode(list[str]) -> list[Tensor (L, D)]`` and ``unload()``."""
        raise NotImplementedError

    def vae(self, path: str, log=print):
        """Object with ``encode(images[-1,1] (B,3,H,W)) -> latents``, ``decode(latents) -> [PIL]``
        and ``to(device)``."""
        raise NotImplementedError

    def load_dit(self, path: str, device="cuda", log=print) -> nn.Module:
        """The denoiser (frozen, int8 resident). Must expose ``gradient_checkpointing`` and
        ``device``."""
        raise NotImplementedError

    # -- maths --------------------------------------------------------------------------
    def predict(self, dit: nn.Module, x_t: torch.Tensor, t: torch.Tensor, text: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Velocity prediction (noise - x0) for t in [0, 1] (shape (B,))."""
        raise NotImplementedError

    def timestep_shift(self, width: int, height: int) -> float:
        """Training timestep shift for this resolution (logit-normal t, shifted)."""
        from ..config import auto_timestep_shift

        return auto_timestep_shift(width, height)

    def sample(self, dit: nn.Module, cond: torch.Tensor, uncond: Optional[torch.Tensor], width: int, height: int,
               steps: int, cfg: float, seed: int, distilled: bool = True) -> torch.Tensor:
        """Latents for one image. ``distilled``: the live weights are the inference model (Turbo
        schedule where the family has one), not the training base."""
        raise NotImplementedError

    # -- helpers ------------------------------------------------------------------------
    def weight_swap(self, live: nn.Module, path: str, log=print):
        """A second DiT parked in RAM, exchangeable with ``live`` (same layout)."""
        from .swap import WeightSwap

        return WeightSwap(live, self.load_dit(path, device="cpu", log=log), log=log)

    def block_offload(self, dit: nn.Module, log=print):
        """Swap ``swap_blocks`` of the DiT's blocks (spread evenly) in from RAM per forward, or None.
        Call after LoRA injection: new LoRA tensors go where their base layer lives."""
        if not self.swap_blocks:
            return None
        from .offload import BlockOffload

        blocks, n = dit.blocks, self.swap_blocks
        return BlockOffload([blocks[round((i + 0.5) * len(blocks) / n - 0.5)] for i in range(n)], log=log)

    def caption_prompts_extra(self) -> Sequence[str]:
        return ()
