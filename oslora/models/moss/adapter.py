"""MOSS-Image components: text encoder (Qwen3.5-9B), FLUX.2 VAE, DiT and the sampler.

Everything is loaded from local ComfyUI-style files; only the small tokenizer/config files
are fetched from the vendor repo (once, into ``models/moss_image``).
"""

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

from ..quant import load_comfy_state_dict
from .transformer import MossTransformer

VENDOR_REPO = "OpenMOSS-Team/MOSS-Image-Base"
CONFIG_FILES = (
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/chat_template.jinja",
    "text_encoder/config.json",
    "vae/config.json",
)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIG_DIR = os.path.join(ROOT, "models", "moss_image")

DEFAULT_SHIFT = 5.0
DEFAULT_CFG = 3.0
DEFAULT_SKIP = (22,)
NUM_TRAIN_TIMESTEPS = 1000


def MossPaths():
    """The configured model files (settings.json), for scripts and tests."""
    from ...config import ModelPaths

    return ModelPaths()


def ensure_configs(log=print) -> str:
    missing = [f for f in CONFIG_FILES if not os.path.isfile(os.path.join(CONFIG_DIR, f))]
    if missing:
        from huggingface_hub import hf_hub_download

        log(f"MOSS 설정 파일 다운로드 중 ({len(missing)}개)...")
        for f in missing:
            hf_hub_download(VENDOR_REPO, f, local_dir=CONFIG_DIR)
    return CONFIG_DIR


# ---------------------------------------------------------------------------------
# Text encoder
# ---------------------------------------------------------------------------------
class MossTextEncoder:
    """Final-normed last hidden state of the chat-templated prompt (thinking on), one
    unpadded forward per prompt, exactly like the vendor's ``encode_prompt``."""

    def __init__(self, path: str, device="cuda", dtype=torch.bfloat16, log=print):
        from accelerate import init_empty_weights
        from transformers import Qwen2Tokenizer, Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        cfg_dir = ensure_configs(log)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(os.path.join(cfg_dir, "tokenizer"))
        with open(os.path.join(cfg_dir, "text_encoder", "config.json"), encoding="utf-8") as f:
            text_cfg = json.load(f)["text_config"]
        config = Qwen3_5TextConfig(**text_cfg)
        with init_empty_weights():
            model = Qwen3_5TextModel(config)
        log(f"텍스트 인코더 로딩: {os.path.basename(path)}")
        sd = {k[len("language_model.") :]: v for k, v in normalize_qwen35_keys(load_file(path)).items()
              if k.startswith("language_model.")}
        missing = load_comfy_state_dict(model, sd, dtype, device)
        if missing:
            raise KeyError(f"text encoder weights missing: {missing[:8]}")
        self.model = model.eval().requires_grad_(False)
        self.device = device
        self.dtype = dtype

    def _format(self, text: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    @torch.no_grad()
    def encode(self, prompts: Sequence[str], max_length: int = 4096) -> List[torch.Tensor]:
        out = []
        for text in prompts:
            ids = self.tokenizer(
                self._format(text), return_tensors="pt", truncation=True, max_length=max_length
            )["input_ids"].to(self.device)
            states = self.model(input_ids=ids, use_cache=False).last_hidden_state[0]
            out.append(states.to(torch.bfloat16).cpu())
        return out

    def unload(self):
        del self.model
        torch.cuda.empty_cache()


def normalize_qwen35_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Comfy-Org's ``qwen3.5_9b_int8_convrot`` prefixes everything with ``model.`` and adds
    ``lm_head`` / ``mtp``; the MOSS repack has bare ``language_model.`` / ``visual.`` keys."""
    if not any(k.startswith("model.") for k in sd):
        return sd
    return {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}


# ---------------------------------------------------------------------------------
# VAE (FLUX.2, 2x2 packed, BatchNorm-normalized -> 128 ch at /16)
# ---------------------------------------------------------------------------------
def convert_bfl_vae(sd: Dict[str, torch.Tensor], cfg: Dict) -> Dict[str, torch.Tensor]:
    """BFL/LDM key layout -> diffusers AutoencoderKLFlux2 (BatchNorm stats carried over)."""
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

    sd = dict(sd)
    for old, new in (("encoder.quant_conv.", "quant_conv."), ("decoder.post_quant_conv.", "post_quant_conv.")):
        for k in [k for k in sd if k.startswith(old)]:
            sd[new + k[len(old):]] = sd.pop(k)
    bn = {k: v for k, v in sd.items() if k.startswith("bn.")}
    out = convert_ldm_vae_checkpoint(sd, cfg)
    out.update(bn)
    return out


class MossVAE:
    def __init__(self, path: str, device="cuda", dtype=torch.float32, log=print):
        from diffusers import AutoencoderKLFlux2

        cfg_dir = ensure_configs(log)
        with open(os.path.join(cfg_dir, "vae", "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        sd = load_file(path)
        if "encoder.down.0.block.0.conv1.weight" in sd:  # original BFL layout (Comfy-Org flux2-vae)
            sd = convert_bfl_vae(sd, cfg)
        # the full-decoder (comfy flux2-vae) and small-decoder files differ in decoder width
        dec = sd.get("decoder.conv_in.weight")
        if dec is not None and dec.shape[0] != cfg["decoder_block_out_channels"][-1]:
            cfg["decoder_block_out_channels"] = cfg["block_out_channels"]
        vae = AutoencoderKLFlux2.from_config(cfg)
        vae.load_state_dict(sd, strict=True)
        self.vae = vae.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.device, self.dtype = device, dtype
        bn = self.vae.bn
        eps = self.vae.config.batch_norm_eps
        self.mean = bn.running_mean.float().view(1, -1, 1, 1).to(device)
        self.std = (bn.running_var.float() + eps).sqrt().view(1, -1, 1, 1).to(device)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images (B, 3, H, W) in [-1, 1] -> normalized latents (B, 128, H/16, W/16)."""
        z = self.vae.encode(images.to(self.device, self.dtype)).latent_dist.mode()
        b, c, h, w = z.shape
        z = z.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
        return (z.float() - self.mean) / self.std

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> List[Image.Image]:
        z = latents.float().to(self.device) * self.std + self.mean
        b, c, h, w = z.shape
        z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
        images = self.vae.decode(z.to(self.dtype), return_dict=False)[0].float().clamp(-1, 1)
        images = ((images + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(a) for a in images]

    def to(self, device):
        self.vae.to(device)
        self.device = device
        self.mean, self.std = self.mean.to(device), self.std.to(device)
        return self


# ---------------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------------
def load_dit(path: str, device="cuda", dtype=torch.bfloat16, log=print) -> MossTransformer:
    from accelerate import init_empty_weights

    log(f"DiT 로딩: {os.path.basename(path)}")
    sd = load_file(path)
    sd = {k: v for k, v in sd.items() if k != "rope.bands" and ".comfy_attention." not in k}
    sd = fuse_qkv(sd)
    cfg = MossTransformer.config_from_state_dict(sd)
    with init_empty_weights():
        model = MossTransformer(cfg)
    missing = load_comfy_state_dict(model, sd, dtype, device)
    if missing:
        raise KeyError(f"DiT weights missing: {missing[:8]}")
    return model.eval().requires_grad_(False)


def fuse_qkv(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Vendor (SFT) files keep q, k and v as separate int8 linears; the Comfy Base repack
    fuses them. Fuse here so every DiT has one layout and a LoRA fits either model. Exact:
    per-row scales, and the ConvRot rotation acts on the shared input dimension."""
    for qkey in [k for k in sd if k.endswith(".proj.q.weight")]:
        base = qkey[: -len(".q.weight")]
        parts = [f"{base}.{n}" for n in "qkv"]
        if not all(f"{p}.weight" in sd for p in parts):
            continue
        quant = [sd.get(f"{p}.comfy_quant") for p in parts]
        if quant[0] is not None and not all(torch.equal(quant[0], q) for q in quant[1:]):
            raise ValueError(f"{base}: q/k/v quantization differs, cannot fuse")
        for suffix in ("weight", "weight_scale", "bias"):
            if f"{parts[0]}.{suffix}" in sd:
                sd[f"{base}.qkv.{suffix}"] = torch.cat([sd.pop(f"{p}.{suffix}") for p in parts])
        if quant[0] is not None:
            sd[f"{base}.qkv.comfy_quant"] = quant[0]
            for p in parts:
                sd.pop(f"{p}.comfy_quant")
    return sd


# ---------------------------------------------------------------------------------
# Sampling (vendor schedule: shift-5 grid snapped to the training grid, fp32 Euler)
# ---------------------------------------------------------------------------------
def moss_sigmas(num_steps: int, shift: float = DEFAULT_SHIFT, total: int = NUM_TRAIN_TIMESTEPS):
    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(num_steps, shift)
    training = grid(total, 1.0).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where(
        (sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper
    )
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


@torch.no_grad()
def sample(
    dit: MossTransformer,
    cond: torch.Tensor,  # (L, D)
    uncond: Optional[torch.Tensor],
    width: int,
    height: int,
    steps: int = 30,
    cfg: float = DEFAULT_CFG,
    seed: int = 0,
    shift: float = DEFAULT_SHIFT,
    skip_layers: Tuple[int, ...] = DEFAULT_SKIP,
) -> torch.Tensor:
    device = dit.device
    gen = torch.Generator(device=device).manual_seed(seed)
    h, w = height // 16, width // 16
    x = torch.randn((1, dit.cfg.latent_channels, h, w), generator=gen, device=device, dtype=torch.float32)
    c = cond[None].to(device)
    u = uncond[None].to(device) if uncond is not None else None
    sigmas = moss_sigmas(steps, shift).to(device)
    for i in range(len(sigmas) - 1):
        s, s_next = sigmas[i], sigmas[i + 1]
        t = (s * NUM_TRAIN_TIMESTEPS).reshape(1)
        pred = dit(x, t, c).float()
        if u is not None and cfg != 1.0:
            pred_u = dit(x, t, u, skip_layers=skip_layers).float()
            pred = pred_u + cfg * (pred - pred_u)
        x = x + (s_next - s) * pred
    return x


def round_to(v: int, m: int = 16) -> int:
    return max(m, int(round(v / m)) * m)


# ---------------------------------------------------------------------------------
# Adapter (what the trainer sees)
# ---------------------------------------------------------------------------------
from ..base import ModelAdapter  # noqa: E402


class MossAdapter(ModelAdapter):
    key = "moss_image"
    label = "MOSS-Image"
    base_model_tag = "moss_image_base"
    lora_targets = {
        "all": [r"layers\.\d+\.proj\.(qkv|q|k|v|o|ffn_up_gate|ffn_down)"],
        "attn": [r"layers\.\d+\.proj\.(qkv|q|k|v|o)"],
    }
    inference_label = "SFT"
    sample_steps = 20
    sample_cfg = DEFAULT_CFG
    quality_steps = 16

    def text_encoder(self, path, log=print):
        return MossTextEncoder(path, log=log)

    def vae(self, path, log=print):
        return MossVAE(path, log=log)

    def load_dit(self, path, device="cuda", log=print):
        return load_dit(path, device=device, log=log)

    def predict(self, dit, x_t, t, text, mask):
        return dit(x_t, t * NUM_TRAIN_TIMESTEPS, text, mask)

    def sample(self, dit, cond, uncond, width, height, steps, cfg, seed):
        return sample(dit, cond, uncond, width, height, steps=steps, cfg=cfg, seed=seed)
