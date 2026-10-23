"""Krea 2: text encoder (Qwen3-VL 4B), VAE (Wan 2.1), DiT loading, sampler, adapter.

Follows diffusers' ``pipeline_krea2.py`` (Apache-2.0):
  * text: Qwen-Image chat template, hidden states of layers (2, 5, ..., 35) stacked per token, the
    34-token system prefix dropped. diffusers pads the prompt to a fixed 512 in the middle of the
    template and masks the padding; padding only ever enters through that mask and text tokens all
    sit at RoPE position 0, so we cache the unpadded tokens -- identical output (tests/test_krea2_port.py).
  * latents: Wan 2.1 VAE (16 ch, /8), normalized with the VAE's mean/std, packed 2x2 -> 64 ch at /16
    (channel = c*4 + dy*2 + dx, diffusers' _pack_latents order).
  * schedule: flow-matching Euler on linspace(1, 1/n) with the exponential time shift
    sigma' = e^mu / (e^mu + 1/sigma - 1); mu from the token count for the base model (0.5 at 256
    tokens .. 1.15 at 6400), fixed 1.15 for the distilled Turbo. Guidance on the base only.
"""

import json
import math
import os
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from ..base import ModelAdapter
from ..quant import load_comfy_state_dict
from .transformer import Krea2Transformer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIG_DIR = os.path.join(ROOT, "models", "krea2")
TE_REPO = "Qwen/Qwen3-VL-4B-Instruct"
TE_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]

SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
PREFIX = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
          "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PREFIX_TOKENS = 34
MAX_TEXT = 512  # text tokens after the prefix, suffix included (diffusers' max_sequence_length)
TURBO_MU = 1.15


def calc_mu(seq_len: int, base_seq: int = 256, max_seq: int = 6400, base: float = 0.5, top: float = 1.15) -> float:
    m = (top - base) / (max_seq - base_seq)
    return seq_len * m + base - m * base_seq


def ensure_te_configs(log=print) -> str:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    missing = [f for f in TE_FILES if not os.path.isfile(os.path.join(CONFIG_DIR, f))]
    if missing:
        from huggingface_hub import hf_hub_download

        log("Qwen3-VL 4B 토크나이저/설정 파일 받는 중 (처음 한 번)")
        for f in missing:
            p = hf_hub_download(TE_REPO, f)
            with open(p, "rb") as src, open(os.path.join(CONFIG_DIR, f), "wb") as dst:
                dst.write(src.read())
    return CONFIG_DIR


class Krea2TextEncoder:
    def __init__(self, path: str, device="cuda", dtype=torch.bfloat16, log=print):
        from accelerate import init_empty_weights
        from transformers import AutoTokenizer
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextConfig, Qwen3VLTextModel

        d = ensure_te_configs(log)
        self.tokenizer = AutoTokenizer.from_pretrained(d)
        with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
            text_cfg = json.load(f)["text_config"]
        config = Qwen3VLTextConfig(**text_cfg)
        with init_empty_weights():
            model = Qwen3VLTextModel(config)
        log(f"텍스트 인코더 로딩: {os.path.basename(path)}")
        sd = load_file(path)
        pre = next((p for p in ("model.language_model.", "language_model.") if any(k.startswith(p) for k in sd)), "")
        sd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        missing = load_comfy_state_dict(model, sd, dtype, device)
        if missing:
            raise KeyError(f"text encoder weights missing: {missing[:8]}")
        self.model = model.eval().requires_grad_(False)
        self.device, self.dtype = device, dtype
        self.suffix_ids = self.tokenizer(SUFFIX)["input_ids"]

    @torch.no_grad()
    def encode(self, prompts: Sequence[str], max_length: int = MAX_TEXT) -> List[torch.Tensor]:
        """(L, 12, 2560) bf16 per prompt, padding-free."""
        out = []
        limit = max_length + PREFIX_TOKENS - len(self.suffix_ids)
        for p in prompts:
            ids = self.tokenizer(PREFIX + p, truncation=True, max_length=limit)["input_ids"] + self.suffix_ids
            x = torch.tensor([ids], device=self.device)
            hs = self.model(input_ids=x, output_hidden_states=True, use_cache=False).hidden_states
            h = torch.stack([hs[i] for i in SELECT_LAYERS], dim=2)[0, PREFIX_TOKENS:]
            out.append(h.to(torch.bfloat16).cpu())
        return out

    def unload(self):
        del self.model
        torch.cuda.empty_cache()


class Krea2VAE:
    """Wan 2.1 VAE (single frame), normalized and packed to the DiT's 64-channel /16 grid."""

    def __init__(self, path: str, device="cuda", dtype=torch.float32, log=print):
        from diffusers import AutoencoderKLWan

        sd = load_file(path)
        if "decoder.head.0.gamma" in sd or "encoder.conv1.weight" in sd:  # original Wan / ComfyUI layout
            from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

            sd = convert_wan_vae_to_diffusers(sd)
        vae = AutoencoderKLWan()
        vae.load_state_dict(sd, strict=True)
        self.vae = vae.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.device, self.dtype = device, dtype
        c = vae.config
        self.mean = torch.tensor(c.latents_mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(c.latents_std).view(1, -1, 1, 1).to(device)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images (B, 3, H, W) in [-1, 1] -> (B, 64, H/16, W/16)."""
        z = self.vae.encode(images.to(self.device, self.dtype)[:, :, None]).latent_dist.mode()[:, :, 0].float()
        z = (z - self.mean) / self.std
        b, c, h, w = z.shape
        z = z.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
        return z.reshape(b, c * 4, h // 2, w // 2)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> List[Image.Image]:
        z = latents.float().to(self.device)
        b, c4, h, w = z.shape
        z = z.reshape(b, c4 // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c4 // 4, h * 2, w * 2)
        z = z * self.std + self.mean
        img = self.vae.decode(z.to(self.dtype)[:, :, None], return_dict=False)[0][:, :, 0].float().clamp(-1, 1)
        arr = ((img + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(a) for a in arr]

    def to(self, device):
        self.vae.to(device)
        self.device = device
        self.mean, self.std = self.mean.to(device), self.std.to(device)
        return self


def load_dit(path: str, device="cuda", dtype=torch.bfloat16, log=print) -> Krea2Transformer:
    from accelerate import init_empty_weights

    log(f"DiT 로딩: {os.path.basename(path)}")
    sd = load_file(path)
    pre = "model.diffusion_model."
    if any(k.startswith(pre) for k in sd):
        sd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    cfg = Krea2Transformer.config_from_state_dict(sd)
    with init_empty_weights():
        model = Krea2Transformer(cfg)
    missing = load_comfy_state_dict(model, sd, dtype, device)
    if missing:
        raise KeyError(f"DiT weights missing: {missing[:8]}")
    return model.eval().requires_grad_(False)


def sigmas(steps: int, mu: float) -> torch.Tensor:
    s = np.linspace(1.0, 1.0 / steps, steps)
    s = math.exp(mu) / (math.exp(mu) + (1.0 / s - 1.0))
    return torch.tensor(np.append(s, 0.0), dtype=torch.float32)


@torch.no_grad()
def sample(dit, cond, uncond, width, height, steps=8, cfg=1.0, seed=0) -> torch.Tensor:
    """cfg <= 1: the distilled Turbo schedule (fixed mu, no guidance). cfg > 1: the base model with
    classifier-free guidance (uncond + cfg * (cond - uncond); Krea's guidance_scale = cfg - 1)."""
    device = dit.device
    gh, gw = height // 16, width // 16
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((1, dit.cfg.in_channels, gh, gw), generator=gen, device=device, dtype=torch.float32)
    distilled = cfg <= 1.0
    sig = sigmas(steps, TURBO_MU if distilled else calc_mu(gh * gw)).to(device)
    c = cond[None].to(device)
    u = uncond[None].to(device) if (uncond is not None and not distilled) else None
    for i in range(steps):
        t = sig[i].reshape(1)
        v = dit(x, t, c).float()
        if u is not None:
            vu = dit(x, t, u).float()
            v = vu + cfg * (v - vu)
        x = x + (sig[i + 1] - sig[i]) * v
    return x


class Krea2Adapter(ModelAdapter):
    key = "krea2"
    label = "Krea 2"
    base_model_tag = "krea2_raw"
    lora_targets = {
        "all": [r"blocks\.\d+\.(attn\.(wq|wk|wv|wo|gate)|mlp\.(gate|up|down))"],
        "attn": [r"blocks\.\d+\.attn\.(wq|wk|wv|wo|gate)"],
    }
    inference_label = "Turbo"
    sample_steps = 8
    sample_cfg = 1.0
    quality_steps = 8
    # 12.6 GB int8 DiT on 16 GB: 6 of 28 blocks in RAM -> training peak 12.3 GB at 512, 13.2 GB at
    # 768, ~4.2 / 8.4 s per step on a 5060 Ti (ai-toolkit: layer_offloading 20 %, ~4 s at 512)
    swap_blocks = 6
    base_sample_steps = 20  # renders on the training (Raw) model: guided
    base_sample_cfg = 5.5   # = Krea's default guidance_scale 4.5

    def text_encoder(self, path, log=print):
        return Krea2TextEncoder(path, log=log)

    def vae(self, path, log=print):
        return Krea2VAE(path, log=log)

    def load_dit(self, path, device="cuda", log=print):
        return load_dit(path, device=device, log=log)

    def predict(self, dit, x_t, t, text, mask):
        return dit(x_t, t, text, mask)

    def timestep_shift(self, width, height):
        return math.exp(calc_mu((width // 16) * (height // 16)))

    def sample(self, dit, cond, uncond, width, height, steps, cfg, seed):
        return sample(dit, cond, uncond, width, height, steps=steps, cfg=cfg, seed=seed)
