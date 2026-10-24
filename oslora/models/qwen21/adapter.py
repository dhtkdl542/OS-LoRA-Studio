"""Qwen-Image 2.1: text encoder (Qwen3-VL 8B), VAE (64 ch, /16, RGBA), DiT loading, sampler, adapter.

Follows diffusers' ``pipeline_qwenimage21.py`` (Apache-2.0):
  * text: the chat template with the system turn "Comprehend and analyze the provided prompt.",
    the last decoder layer's output *before* the final RMSNorm, the system turn dropped (everything
    before the second ``<|im_start|>``). An empty prompt is encoded as " " (Qwen has no BOS).
  * latents: the Qwen-Image 2.1 VAE takes RGBA; RGB photos get an opaque alpha (1.0, as ComfyUI
    pads it). 64 channels at /16, normalized with the VAE's mean/std, consumed unpatched.
  * schedule: the base model on linspace(1, 1/n) with the exponential shift (mu 0.5 at 256 tokens ..
    0.9 at 8192) stretched so the last sigma is 0.02; the distilled Turbo on its saved 8 sigmas.
    Both are sampled without guidance (the pipeline's default true_cfg_scale is 1).
"""

import json
import math
import os
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from ..base import ModelAdapter
from ..quant import load_comfy_state_dict
from .transformer import Qwen21Transformer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIG_DIR = os.path.join(ROOT, "models", "qwen21")
REPO = "Qwen/Qwen-Image-2.1"
REPO_FILES = {  # repo path -> local name
    "processor/tokenizer.json": "tokenizer.json",
    "processor/tokenizer_config.json": "tokenizer_config.json",
    "processor/vocab.json": "vocab.json",
    "processor/merges.txt": "merges.txt",
    "processor/added_tokens.json": "added_tokens.json",
    "processor/special_tokens_map.json": "special_tokens_map.json",
    "text_encoder/config.json": "config.json",
    "vae/config.json": "vae_config.json",
}

TEMPLATE = ("<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
IM_START = 151644
MAX_TEXT = 1024
# Qwen-Image-2.1-Turbo's saved schedule (model_index.json sample_sigmas), no shifting
TURBO_SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)
SHIFT_TERMINAL = 0.02


def calc_mu(seq_len: int, base_seq: int = 256, max_seq: int = 8192, base: float = 0.5, top: float = 0.9) -> float:
    m = (top - base) / (max_seq - base_seq)
    return seq_len * m + base - m * base_seq


def ensure_configs(log=print) -> str:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    missing = {r: n for r, n in REPO_FILES.items() if not os.path.isfile(os.path.join(CONFIG_DIR, n))}
    if missing:
        from huggingface_hub import hf_hub_download

        log("Qwen-Image 2.1 토크나이저/설정 파일 받는 중 (처음 한 번)")
        for repo_path, name in missing.items():
            p = hf_hub_download(REPO, repo_path)
            with open(p, "rb") as src, open(os.path.join(CONFIG_DIR, name), "wb") as dst:
                dst.write(src.read())
    return CONFIG_DIR


class Qwen21TextEncoder:
    def __init__(self, path: str, device="cuda", dtype=torch.bfloat16, log=print):
        from accelerate import init_empty_weights
        from transformers import AutoTokenizer
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextConfig, Qwen3VLTextModel

        d = ensure_configs(log)
        self.tokenizer = AutoTokenizer.from_pretrained(d)
        with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
            config = Qwen3VLTextConfig(**json.load(f)["text_config"])
        with init_empty_weights():
            model = Qwen3VLTextModel(config)
        log(f"텍스트 인코더 로딩: {os.path.basename(path)}")
        sd = load_file(path)
        # ComfyUI: model.layers.* / model.visual.* / lm_head; transformers: model.language_model.*
        pre = next(p for p in ("model.language_model.", "language_model.", "model.") if any(k.startswith(p) for k in sd))
        sd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre) and not k.startswith(pre + "visual.")}
        missing = load_comfy_state_dict(model, sd, dtype, device)
        if missing:
            raise KeyError(f"text encoder weights missing: {missing[:8]}")
        self.model = model.eval().requires_grad_(False)
        # the DiT reads the last decoder layer's output before the final norm
        model.norm.register_forward_hook(lambda module, args, output: args[0])
        self.device, self.dtype = device, dtype

    def token_ids(self, prompt: str, max_length: int = MAX_TEXT):
        """(ids, number of leading system-turn tokens to drop)."""
        ids = self.tokenizer(TEMPLATE.format(prompt or " "))["input_ids"]
        start = [i for i, t in enumerate(ids) if t == IM_START][1]  # drop the system turn
        head, body = ids[:start], ids[start:]
        if len(body) > max_length:  # keep the assistant cue at the end
            tail = self.tokenizer("<|im_end|>\n<|im_start|>assistant\n")["input_ids"]
            body = body[:max_length - len(tail)] + tail
        return head + body, len(head)

    @torch.no_grad()
    def encode(self, prompts: Sequence[str], max_length: int = MAX_TEXT) -> List[torch.Tensor]:
        """(L, 4096) bf16 per prompt, padding-free."""
        out = []
        for p in prompts:
            ids, drop = self.token_ids(p, max_length)
            h = self.model(input_ids=torch.tensor([ids], device=self.device), use_cache=False).last_hidden_state
            out.append(h[0, drop:].to(torch.bfloat16).cpu())
        return out

    def unload(self):
        del self.model
        torch.cuda.empty_cache()


def convert_vae(sd):
    """Original (Wan 2.2 layout, temporal kernel 1) -> diffusers AutoencoderKLQwenImage21 keys."""
    import re

    out = {}
    for k, v in sd.items():
        n = k
        n = re.sub(r"^conv1\.", "quant_conv.", n)
        n = re.sub(r"^conv2\.", "post_quant_conv.", n)
        n = re.sub(r"^(encoder|decoder)\.conv1\.", r"\1.conv_in.", n)
        n = re.sub(r"^(encoder|decoder)\.head\.0\.", r"\1.norm_out.", n)
        n = re.sub(r"^(encoder|decoder)\.head\.2\.", r"\1.conv_out.", n)
        n = re.sub(r"^(encoder|decoder)\.middle\.0\.", r"\1.mid_block.resnets.0.", n)
        n = re.sub(r"^(encoder|decoder)\.middle\.1\.", r"\1.mid_block.attentions.0.", n)
        n = re.sub(r"^(encoder|decoder)\.middle\.2\.", r"\1.mid_block.resnets.1.", n)
        m = re.match(r"^encoder\.downsamples\.(\d+)\.downsamples\.(\d+)\.(.*)$", n)
        if m:
            i, j, rest = m.groups()
            n = f"encoder.down_blocks.{i}." + ("downsampler." if rest.startswith(("resample", "time_conv")) else f"resnets.{j}.") + rest
        m = re.match(r"^decoder\.upsamples\.(\d+)\.upsamples\.(\d+)\.(.*)$", n)
        if m:
            i, j, rest = m.groups()
            n = f"decoder.up_blocks.{i}." + ("upsampler." if rest.startswith(("resample", "time_conv")) else f"resnets.{j}.") + rest
        n = (n.replace(".residual.0.", ".norm1.").replace(".residual.2.", ".conv1.")
             .replace(".residual.3.", ".norm2.").replace(".residual.6.", ".conv2.").replace(".shortcut.", ".conv_shortcut."))
        if v.ndim == 5 and v.shape[2] == 1:  # (out, in, 1, kh, kw): single-frame causal conv
            v = v[:, :, 0]
        out[n] = v
    return out


class Qwen21VAE:
    def __init__(self, path: str, device="cuda", dtype=torch.bfloat16, log=print):
        from diffusers import AutoencoderKLQwenImage21

        with open(os.path.join(ensure_configs(log), "vae_config.json"), encoding="utf-8") as f:
            cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        sd = load_file(path)
        if "encoder.conv1.weight" in sd or "decoder.head.0.gamma" in sd:
            sd = convert_vae(sd)
        vae = AutoencoderKLQwenImage21(**cfg)
        vae.load_state_dict(sd, strict=True)
        self.vae = vae.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.device, self.dtype = device, dtype
        self.mean = torch.tensor(cfg["latents_mean"]).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(cfg["latents_std"]).view(1, -1, 1, 1).to(device)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images (B, 3, H, W) in [-1, 1] -> (B, 64, H/16, W/16)."""
        x = images.to(self.device, self.dtype)
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)  # opaque alpha
        z = self.vae.encode(x[:, :, None]).latent_dist.mode()[:, :, 0].float()
        return (z - self.mean) / self.std

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> List[Image.Image]:
        z = (latents.float().to(self.device) * self.std + self.mean).to(self.dtype)[:, :, None]
        try:
            img = self.vae.decode(z, return_dict=False)[0]
        except torch.OutOfMemoryError:
            # tiles need +0.4 GB instead of +3.5 (640x864) / +4.9 GB (768x1024) but differ by ~3/255,
            # so only when the plain decode doesn't fit
            torch.cuda.empty_cache()
            self.vae.use_tiling = True
            try:
                img = self.vae.decode(z, return_dict=False)[0]
            finally:
                self.vae.use_tiling = False
        img = img[:, :3, 0].float().clamp(-1, 1)
        arr = ((img + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(a) for a in arr]

    def to(self, device):
        self.vae.to(device)
        self.device = device
        self.mean, self.std = self.mean.to(device), self.std.to(device)
        return self


def split_gate_up(sd):
    """ComfyUI's fused ``img_mlp.gate_up`` ([gate; up] rows) -> ``gate_layer`` + ``proj``. Rows split
    exactly, per-row int8 scales with them; the ConvRot rotation acts on the shared input."""
    for key in [k for k in sd if k.endswith(".img_mlp.gate_up.weight")]:
        base = key[: -len(".gate_up.weight")]
        fused = base + ".gate_up"
        w = sd.pop(fused + ".weight")
        half = w.shape[0] // 2
        sd[base + ".gate_layer.weight"], sd[base + ".proj.weight"] = w[:half].contiguous(), w[half:].contiguous()
        if fused + ".weight_scale" in sd:
            s = sd.pop(fused + ".weight_scale")
            if s.numel() == 1:
                sd[base + ".gate_layer.weight_scale"], sd[base + ".proj.weight_scale"] = s.clone(), s.clone()
            else:
                s = s.reshape(-1, 1)
                sd[base + ".gate_layer.weight_scale"], sd[base + ".proj.weight_scale"] = s[:half].contiguous(), s[half:].contiguous()
        if fused + ".comfy_quant" in sd:
            q = sd.pop(fused + ".comfy_quant")
            sd[base + ".gate_layer.comfy_quant"], sd[base + ".proj.comfy_quant"] = q.clone(), q.clone()
    return sd


def load_dit(path: str, device="cuda", dtype=torch.bfloat16, log=print) -> Qwen21Transformer:
    from accelerate import init_empty_weights

    log(f"DiT 로딩: {os.path.basename(path)}")
    sd = load_file(path)
    pre = next((p for p in ("model.diffusion_model.", "diffusion_model.", "transformer.") if any(k.startswith(p) for k in sd)), "")
    if pre:
        sd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    sd = split_gate_up(sd)
    cfg = Qwen21Transformer.config_from_state_dict(sd)
    with init_empty_weights():
        model = Qwen21Transformer(cfg)
    missing = load_comfy_state_dict(model, sd, dtype, device)
    if missing:
        raise KeyError(f"DiT weights missing: {missing[:8]}")
    return model.eval().requires_grad_(False)


def base_sigmas(steps: int, seq_len: int) -> np.ndarray:
    """FlowMatchEulerDiscreteScheduler with Qwen-Image 2.1's config: exponential dynamic shift, then
    stretched so the last sigma is 0.02. Terminal 0 appended."""
    s = np.linspace(1.0, 1.0 / steps, steps)
    mu = calc_mu(seq_len)
    s = math.exp(mu) / (math.exp(mu) + (1.0 / s - 1.0))
    one_minus = 1.0 - s
    s = 1.0 - one_minus / (one_minus[-1] / (1.0 - SHIFT_TERMINAL))
    return np.append(s, 0.0)


def turbo_sigmas(steps: int) -> np.ndarray:
    s = np.array(TURBO_SIGMAS)
    if steps != len(s):  # resample the distilled curve; the model was tuned for 8
        s = np.interp(np.linspace(0, len(s) - 1, steps), np.arange(len(s)), s)
    return np.append(s, 0.0)


@torch.no_grad()
def sample(dit, cond, uncond, width, height, steps=8, cfg=1.0, seed=0, distilled=True) -> torch.Tensor:
    device = dit.device
    gh, gw = height // 16, width // 16
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((1, dit.cfg.in_channels, gh, gw), generator=gen, device=device, dtype=torch.float32)
    sig = torch.tensor(turbo_sigmas(steps) if distilled else base_sigmas(steps, gh * gw), dtype=torch.float32, device=device)
    c = cond[None].to(device)
    u = uncond[None].to(device) if (uncond is not None and cfg > 1.0) else None
    for i in range(steps):
        t = sig[i].reshape(1)
        v = dit(x, t, c).float()
        if u is not None:
            vu = dit(x, t, u).float()
            v = vu + cfg * (v - vu)
        x = x + (sig[i + 1] - sig[i]) * v
    return x


class Qwen21Adapter(ModelAdapter):
    key = "qwen21"
    label = "Qwen Image 2.1"
    base_model_tag = "qwen_image_2.1"
    # ComfyUI maps transformer.*.img_mlp.gate_layer / proj onto the halves of its fused gate_up
    lora_prefix = "transformer."
    bucket_step = 32  # the pipeline only renders multiples of 32 (even latent grids)
    lora_targets = {
        "all": [r"transformer_blocks\.\d+\.(attn\.(to_q|to_k|to_v|to_out\.0)|img_mlp\.(gate_layer|proj|out))"],
        "attn": [r"transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)"],
    }
    inference_label = "Turbo"
    sample_steps = 8
    sample_cfg = 1.0
    distilled_steps = True
    quality_steps = 8
    base_sample_steps = 25  # the base model is also sampled without guidance
    base_sample_cfg = 1.0

    def text_encoder(self, path, log=print):
        return Qwen21TextEncoder(path, log=log)

    def vae(self, path, log=print):
        return Qwen21VAE(path, log=log)

    def load_dit(self, path, device="cuda", log=print):
        return load_dit(path, device=device, log=log)

    def predict(self, dit, x_t, t, text, mask):
        return dit(x_t, t, text, mask)

    def timestep_shift(self, width, height):
        return math.exp(calc_mu((width // 16) * (height // 16)))

    def sample(self, dit, cond, uncond, width, height, steps, cfg, seed, distilled=True):
        return sample(dit, cond, uncond, width, height, steps=steps, cfg=cfg, seed=seed, distilled=distilled)
