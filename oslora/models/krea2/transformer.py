"""Krea 2 DiT, ported from diffusers' ``transformer_krea2.py`` (Apache-2.0, Krea AI / HuggingFace)
with the module tree laid out like the original checkpoint (``blocks.N.attn.wq``, ``txtfusion.*``,
``last.*``), so ComfyUI's int8 ConvRot files load without renaming and LoRA keys come out in the
same ``diffusion_model.blocks.N.attn.wq`` form ComfyUI already loads for Krea 2.

Single-stream MMDiT: the text-encoder hidden states of 12 tapped layers are fused per token
(``txtfusion``), projected (``txtmlp``) and concatenated in front of the patchified image tokens.
Grouped-query attention with q/k RMSNorm, 3-axis RoPE (text at the origin) and a sigmoid output
gate; one shared timestep modulation plus a learned additive table per block. All RMSNorm scales
are stored zero-centred (the multiplier is ``1 + scale``).
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class Krea2Config:
    in_channels: int = 64
    num_layers: int = 28
    head_dim: int = 128
    num_heads: int = 48
    num_kv_heads: int = 12
    intermediate: int = 16384
    t_embed: int = 256
    text_dim: int = 2560
    text_layers: int = 12
    text_heads: int = 20
    text_kv_heads: int = 20
    text_intermediate: int = 6912
    layerwise_blocks: int = 2
    refiner_blocks: int = 2
    rope_axes: Tuple[int, int, int] = (32, 48, 48)
    rope_theta: float = 1000.0
    eps: float = 1e-5

    @property
    def dim(self) -> int:
        return self.head_dim * self.num_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.rms_norm(x.float(), (self.dim,), weight=self.scale.float() + 1.0, eps=self.eps).to(x.dtype)


class QKNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float):
        super().__init__()
        self.qnorm, self.knorm = RMSNorm(head_dim, eps), RMSNorm(head_dim, eps)


def _rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved-pair rotation (diffusers ``apply_rotary_emb``, use_real, repeat_interleave):
    x (B, S, H, D), cos/sin (S, D)."""
    xf = x.float()
    re, im = xf.unflatten(-1, (-1, 2)).unbind(-1)
    rot = torch.stack([-im, re], dim=-1).flatten(-2)
    return (xf * cos[None, :, None] + rot * sin[None, :, None]).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int, eps: float):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, dim // heads
        self.wq = nn.Linear(dim, self.head_dim * heads, bias=False)
        self.wk = nn.Linear(dim, self.head_dim * kv_heads, bias=False)
        self.wv = nn.Linear(dim, self.head_dim * kv_heads, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.qknorm = QKNorm(self.head_dim, eps)

    def forward(self, x, rope=None, mask=None):
        q = self.qknorm.qnorm(self.wq(x).unflatten(-1, (self.heads, self.head_dim)))
        k = self.qknorm.knorm(self.wk(x).unflatten(-1, (self.kv_heads, self.head_dim)))
        v = self.wv(x).unflatten(-1, (self.kv_heads, self.head_dim))
        g = self.gate(x)
        if rope is not None:
            q, k = _rope_apply(q, *rope), _rope_apply(k, *rope)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if self.kv_heads != self.heads:
            # expand the grouped k/v heads: SDPA's enable_gqa path falls back to the math kernel on
            # Windows builds and materializes the full attention matrix (OOM at 768x1024)
            rep = self.heads // self.kv_heads
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=mask)
        out = out.transpose(1, 2).flatten(2)
        return self.wo(out * torch.sigmoid(g))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TextFusionBlock(nn.Module):
    def __init__(self, dim, heads, kv_heads, hidden, eps):
        super().__init__()
        self.prenorm, self.postnorm = RMSNorm(dim, eps), RMSNorm(dim, eps)
        self.attn = Attention(dim, heads, kv_heads, eps)
        self.mlp = MLP(dim, hidden)

    def forward(self, x, mask=None):
        x = x + self.attn(self.prenorm(x), mask=mask)
        return x + self.mlp(self.postnorm(x))


class TextFusion(nn.Module):
    """Per token: attend across the 12 tapped layers, collapse them (``projector``), then attend
    across the sequence (``refiner_blocks``)."""

    def __init__(self, c: Krea2Config):
        super().__init__()
        mk = lambda: TextFusionBlock(c.text_dim, c.text_heads, c.text_kv_heads, c.text_intermediate, c.eps)  # noqa: E731
        self.layerwise_blocks = nn.ModuleList([mk() for _ in range(c.layerwise_blocks)])
        self.projector = nn.Linear(c.text_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([mk() for _ in range(c.refiner_blocks)])

    def forward(self, x, mask=None):
        b, s, n, d = x.shape
        h = x.reshape(b * s, n, d)
        for blk in self.layerwise_blocks:
            h = blk(h.contiguous())
        h = self.projector(h.reshape(b, s, n, d).permute(0, 1, 3, 2)).squeeze(-1)
        for blk in self.refiner_blocks:
            h = blk(h, mask=mask)
        return h


class Modulation(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(shape))


class Block(nn.Module):
    def __init__(self, c: Krea2Config):
        super().__init__()
        self.prenorm, self.postnorm = RMSNorm(c.dim, c.eps), RMSNorm(c.dim, c.eps)
        self.attn = Attention(c.dim, c.num_heads, c.num_kv_heads, c.eps)
        self.mlp = MLP(c.dim, c.intermediate)
        self.mod = Modulation((6 * c.dim,))

    def forward(self, x, temb_mod, rope, mask):
        m = temb_mod.unflatten(-1, (6, -1)) + self.mod.lin.view(6, -1).to(temb_mod.dtype)
        prescale, preshift, pregate, postscale, postshift, postgate = m.unbind(-2)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, rope=rope, mask=mask)
        return x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)


class FinalLayer(nn.Module):
    def __init__(self, c: Krea2Config):
        super().__init__()
        self.norm = RMSNorm(c.dim, c.eps)
        self.modulation = Modulation((2, c.dim))
        self.linear = nn.Linear(c.dim, c.in_channels, bias=True)

    def forward(self, x, temb):
        scale, shift = (temb + self.modulation.lin.to(temb.dtype)).chunk(2, dim=1)
        return self.linear((1 + scale) * self.norm(x) + shift)


def rope_tables(text_len: int, gh: int, gw: int, axes, theta: float, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos/sin (text_len + gh*gw, head_dim): text tokens at (0,0,0), image tokens at (0, h, w)."""
    ids = torch.zeros(text_len + gh * gw, 3, device=device, dtype=torch.float64)
    hh, ww = torch.meshgrid(torch.arange(gh, device=device), torch.arange(gw, device=device), indexing="ij")
    ids[text_len:, 1] = hh.reshape(-1).double()
    ids[text_len:, 2] = ww.reshape(-1).double()
    cos, sin = [], []
    for i, d in enumerate(axes):
        freqs = 1.0 / (theta ** (torch.arange(0, d, 2, device=device, dtype=torch.float64) / d))
        ang = ids[:, i:i + 1] * freqs[None]
        cos.append(ang.cos().repeat_interleave(2, dim=-1))
        sin.append(ang.sin().repeat_interleave(2, dim=-1))
    return torch.cat(cos, -1).float(), torch.cat(sin, -1).float()


class Krea2Transformer(nn.Module):
    def __init__(self, c: Krea2Config):
        super().__init__()
        self.cfg = c
        self.first = nn.Linear(c.in_channels, c.dim, bias=True)
        self.tmlp = nn.Sequential(nn.Linear(c.t_embed, c.dim), nn.GELU(approximate="tanh"), nn.Linear(c.dim, c.dim))
        self.tproj = nn.Sequential(nn.GELU(approximate="tanh"), nn.Linear(c.dim, 6 * c.dim))
        self.txtfusion = TextFusion(c)
        self.txtmlp = nn.Sequential(RMSNorm(c.text_dim, c.eps), nn.Linear(c.text_dim, c.dim),
                                    nn.GELU(approximate="tanh"), nn.Linear(c.dim, c.dim))
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.num_layers)])
        self.last = FinalLayer(c)
        self.gradient_checkpointing = False
        self._rope_cache = {}

    @property
    def device(self):
        return self.first.weight.device

    @staticmethod
    def config_from_state_dict(sd) -> Krea2Config:
        dim = sd["first.weight"].shape[0]
        head = sd["blocks.0.attn.qknorm.qnorm.scale"].shape[0]
        t_head = sd["txtfusion.refiner_blocks.0.attn.qknorm.qnorm.scale"].shape[0]
        t_dim = sd["txtmlp.1.weight"].shape[1]
        return Krea2Config(
            text_heads=t_dim // t_head,
            text_kv_heads=sd["txtfusion.refiner_blocks.0.attn.wk.weight"].shape[0] // t_head,
            in_channels=sd["first.weight"].shape[1],
            num_layers=sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".prenorm.scale")),
            head_dim=head,
            num_heads=dim // head,
            num_kv_heads=sd["blocks.0.attn.wk.weight"].shape[0] // head,
            intermediate=sd["blocks.0.mlp.gate.weight"].shape[0],
            t_embed=sd["tmlp.0.weight"].shape[1],
            text_dim=sd["txtmlp.1.weight"].shape[1],
            text_layers=sd["txtfusion.projector.weight"].shape[1],
            text_intermediate=sd["txtfusion.refiner_blocks.0.mlp.gate.weight"].shape[0],
            layerwise_blocks=sum(1 for k in sd if k.startswith("txtfusion.layerwise_blocks.") and k.endswith(".prenorm.scale")),
            refiner_blocks=sum(1 for k in sd if k.startswith("txtfusion.refiner_blocks.") and k.endswith(".prenorm.scale")),
        )

    def _temb(self, t: torch.Tensor, dtype) -> torch.Tensor:
        half = self.cfg.t_embed // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = (t.float() * 1e3)[:, None, None] * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
        return self.tmlp(emb)  # (B, 1, dim)

    def forward(self, latents: torch.Tensor, t: torch.Tensor, text: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """latents (B, 64, h, w) packed (channel = c*4 + dy*2 + dx), t (B,) in [0, 1],
        text (B, L, 12, 2560), text_mask (B, L) bool. Returns the velocity in the same layout."""
        b, ch, gh, gw = latents.shape
        w = self.first.weight
        dtype = w.dtype if w.is_floating_point() else torch.bfloat16
        x = latents.permute(0, 2, 3, 1).reshape(b, gh * gw, ch).to(dtype)
        text = text.to(dtype)
        if text_mask is not None and bool(text_mask.all()):
            text_mask = None  # no padding: lets attention use the flash kernel
        temb = self._temb(t, dtype)
        temb_mod = self.tproj(temb)
        tmask = mask = None
        if text_mask is not None:
            tmask = text_mask[:, None, None, :]
            mask = torch.cat([text_mask, text_mask.new_ones(b, gh * gw)], 1)[:, None, None, :]
        txt = self.txtmlp(self.txtfusion(text, mask=tmask))
        h = torch.cat([txt, self.first(x)], dim=1)
        key = (txt.shape[1], gh, gw, h.device)
        if key not in self._rope_cache:
            if len(self._rope_cache) > 8:
                self._rope_cache.clear()
            self._rope_cache[key] = rope_tables(txt.shape[1], gh, gw, self.cfg.rope_axes, self.cfg.rope_theta, h.device)
        rope = self._rope_cache[key]
        for blk in self.blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                h = checkpoint(blk, h, temb_mod, rope, mask, use_reentrant=False)
            else:
                h = blk(h, temb_mod, rope, mask)
        out = self.last(h[:, txt.shape[1]:], temb)
        return out.reshape(b, gh, gw, ch).permute(0, 3, 1, 2)
