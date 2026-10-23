"""MOSS-Image DiT.

Ported from the vendor's ``pipeline_moss_image.py`` (OpenMOSS-Team/MOSS-Image, Apache-2.0).
Module names follow the checkpoint on disk so LoRA keys line up with ComfyUI:
the Comfy-Org repack fuses each block's q/k/v into ``proj.qkv``; the vendor layout keeps
``proj.q/k/v``. Both are supported (``fused_qkv`` is sniffed from the file).

One stream: image tokens then text tokens through 42 blocks sharing one AdaLN modulation;
the text first passes a 2-block refiner. Like the vendor, every sample runs alone at its
true text length (no padding inside attention).
"""

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class MossConfig:
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    freq_dim: int = 256
    latent_channels: int = 128
    rope_base: float = 10000.0
    fused_qkv: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rope_rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rope_rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rope_rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    out = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v)
    return out.transpose(1, 2).flatten(2)


class TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(2)])


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads, fused_qkv):
        super().__init__()
        self.heads = heads
        self.fused_qkv = fused_qkv
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        proj = {}
        if fused_qkv:
            proj["qkv"] = nn.Linear(dim, 3 * dim, bias=False)
        else:
            for name in ("q", "k", "v"):
                proj[name] = nn.Linear(dim, dim, bias=False)
        proj["o"] = nn.Linear(dim, dim, bias=False)
        proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)
        self.proj = nn.ModuleDict(proj)

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        if self.fused_qkv:
            q, k, v = self.proj["qkv"](y).chunk(3, -1)
        else:
            q, k, v = self.proj["q"](y), self.proj["k"](y), self.proj["v"](y)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](_attention(q, k, v, rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y).chunk(2, -1)
        return x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f[:, None]


def rope_bands(head_dim: int, base: float, device) -> torch.Tensor:
    num_bands = head_dim // 8
    exponent = torch.arange(num_bands, dtype=torch.float32) / num_bands
    return (1 / (base**exponent)).to(device)


class MossTransformer(nn.Module):
    def __init__(self, cfg: MossConfig):
        super().__init__()
        self.cfg = cfg
        dim, ffn, heads = cfg.dim, cfg.ffn_dim, cfg.num_heads
        self.video_embedder = nn.Linear(cfg.latent_channels, dim)
        self.text_embedder = TextProjection(cfg.text_dim, dim, ffn, heads)
        self.time_embedding = nn.Sequential(nn.Linear(cfg.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([Block(dim, ffn, heads, cfg.fused_qkv) for _ in range(cfg.num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, cfg.latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        self.gradient_checkpointing = False
        self._bands = {}

    # -- helpers ------------------------------------------------------------------
    @property
    def device(self):
        return self.final_norm_video.weight.device

    @property
    def dtype(self):
        return self.final_norm_video.weight.dtype

    @staticmethod
    def config_from_state_dict(sd) -> MossConfig:
        video = sd["video_embedder.weight"]
        dim = video.shape[0]
        return MossConfig(
            dim=dim,
            ffn_dim=sd["text_embedder.blocks.0.ffn_up.weight"].shape[0],
            num_heads=dim // sd["layers.0.q_norm.weight"].shape[0],
            num_layers=sum(1 for k in sd if k.startswith("layers.") and k.endswith(".norm1.weight")),
            text_dim=sd["text_embedder.proj.weight"].shape[1],
            freq_dim=sd["time_embedding.0.weight"].shape[1],
            latent_channels=video.shape[1],
            fused_qkv="layers.0.proj.qkv.weight" in sd,
        )

    def _rope(self, height, width, text_length, device):
        bands = self._bands.get(device)
        if bands is None:
            bands = rope_bands(self.cfg.dim // self.cfg.num_heads, self.cfg.rope_base, device)
            self._bands[device] = bands
        h, w = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
        )
        image = torch.stack(
            (torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1
        ).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)

    def _run(self, module, *args):
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    # -- forward ------------------------------------------------------------------
    def forward(
        self,
        latents: torch.Tensor,  # (B, C, H, W)
        timestep: torch.Tensor,  # (B,) or scalar, 0..1000 (1000 = pure noise)
        text: torch.Tensor,  # (B, S, text_dim)
        text_mask: Optional[torch.Tensor] = None,  # (B, S) bool
        skip_layers: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        skip = frozenset(skip_layers or ())
        batch, _, height, width = latents.shape
        device, dtype = latents.device, self.dtype
        if text_mask is None:
            text_mask = torch.ones(text.shape[:2], dtype=torch.bool, device=device)

        times = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1).expand(batch)
        half = self.cfg.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        emb = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(emb.to(dtype)).float()
        mod = self.shared_block_modulation(time.to(dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(dtype)).view(batch, 2, -1)

        outputs = []
        for i in range(batch):
            context = text[i : i + 1, text_mask[i].bool()].to(dtype)
            video = latents[i : i + 1].flatten(2).transpose(1, 2).to(dtype).contiguous()
            video = self.video_embedder(video)
            rope = self._rope(height, width, context.shape[1], device).to(dtype)
            t = self.text_embedder.proj(context)
            text_rope = rope[:, height * width :]
            for block in self.text_embedder.blocks:
                t = self._run(block, t, text_rope)
            x = torch.cat((video, t), 1)
            m = mod[i : i + 1]
            for idx, layer in enumerate(self.layers):
                if idx not in skip:
                    x = self._run(layer, x, rope, m)
            x = self.final_norm_video(x[:, : height * width])
            x = x * (1 + head[i : i + 1, 1, None]) + head[i : i + 1, 0, None]
            x = self.final_linear_video(x)
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        return torch.cat(outputs)
