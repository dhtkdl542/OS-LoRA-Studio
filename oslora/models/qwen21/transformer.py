"""Qwen-Image 2.1 DiT, ported from diffusers' ``transformer_qwenimage21.py`` (Apache-2.0, Qwen-Image
Team / HuggingFace), text-to-image path only, with the checkpoint's module names
(``transformer_blocks.N.attn.to_q``, ``img_mlp.gate_layer`` / ``proj`` / ``out``, ...). ComfyUI files
store the two MLP input projections fused as ``img_mlp.gate_up``; ``load_dit`` splits them so LoRA
keys come out as ``transformer.transformer_blocks.N.img_mlp.gate_layer``, the layout ComfyUI maps
onto its fused weight and other Qwen 2.1 trainers write.

Single stream: the projected text tokens followed by the target image's tokens. Attention is
block-causal -- text tokens see only earlier text, image tokens see everything -- so the text
part never depends on the image. One modulation projection is shared by every block (scale and
gate only, no shift, tanh gates, affine-free LayerNorm); text tokens are modulated from t = 0
("causal_condition"), image tokens from the sampled t. 3-axis RoPE: text token i at (i, i, i), the
image at frame = text length on a height/width grid centred on zero.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class Qwen21Config:
    in_channels: int = 64
    num_layers: int = 32
    head_dim: int = 128
    num_heads: int = 32
    context_dim: int = 4096
    mlp_ratio: int = 3
    rope_axes: Tuple[int, int, int] = (16, 56, 56)
    rope_theta: float = 10000.0
    eps: float = 1e-6

    @property
    def dim(self) -> int:
        return self.head_dim * self.num_heads


class RMSNorm(nn.Module):
    """diffusers' RMSNorm: normalize in fp32, cast to the weight's dtype, then scale."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        h = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            h = h.to(self.weight.dtype)
        return h * self.weight


class ZeroCenteredRMSNorm(nn.Module):
    """Stored weight is ``scale - 1``, applied in fp32."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.rms_norm(x.float(), (self.dim,), weight=self.weight.float() + 1.0, eps=self.eps).to(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, c: Qwen21Config):
        super().__init__()
        self.text_norm = ZeroCenteredRMSNorm(c.context_dim, c.eps)
        self.in_layer = nn.Linear(c.context_dim, c.dim, bias=False)
        self.out_layer = nn.Linear(c.dim, c.dim, bias=False)

    def forward(self, x):
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(x)), approximate="tanh"))


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(256, dim, bias=False)
        self.linear_2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


class TimeTextEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.timestep_embedder = TimestepEmbedder(dim)

    def forward(self, t: torch.Tensor, dtype) -> torch.Tensor:
        half = 128
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = (1000.0 * t.float())[:, None] * freqs[None]
        return self.timestep_embedder(torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype))


def _rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved-pair complex rotation (diffusers ``apply_rotary_emb_qwen``, use_real=False):
    x (B, S, H, D), cos/sin (S, D)."""
    xf = x.float()
    re, im = xf.unflatten(-1, (-1, 2)).unbind(-1)
    rot = torch.stack([-im, re], dim=-1).flatten(-2)
    return (xf * cos[None, :, None] + rot * sin[None, :, None]).to(x.dtype)


def rope_tables(text_len: int, gh: int, gw: int, axes, theta: float, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos/sin (text_len + gh*gw, head_dim): text token i at (i, i, i), image token (y, x) at
    (text_len, y - (gh - gh//2), x - (gw - gw//2))."""
    ids = torch.empty(text_len + gh * gw, 3, device=device, dtype=torch.float64)
    ids[:text_len] = torch.arange(text_len, device=device, dtype=torch.float64)[:, None]
    hh = torch.arange(gh, device=device, dtype=torch.float64) - (gh - gh // 2)
    ww = torch.arange(gw, device=device, dtype=torch.float64) - (gw - gw // 2)
    ids[text_len:, 0] = text_len
    ids[text_len:, 1] = hh[:, None].expand(gh, gw).reshape(-1)
    ids[text_len:, 2] = ww[None, :].expand(gh, gw).reshape(-1)
    cos, sin = [], []
    for i, d in enumerate(axes):
        # the reference builds these in fp32 (outer(index, inv_freq) -> polar); match its rounding
        inv = 1.0 / torch.pow(torch.tensor(theta), torch.arange(0, d, 2, dtype=torch.float32).div(d))
        ang = ids[:, i:i + 1].float() * inv.to(device)[None]
        cos.append(ang.cos().repeat_interleave(2, dim=-1))
        sin.append(ang.sin().repeat_interleave(2, dim=-1))
    return torch.cat(cos, -1), torch.cat(sin, -1)


class Attention(nn.Module):
    def __init__(self, c: Qwen21Config):
        super().__init__()
        self.heads = c.num_heads
        self.to_q = nn.Linear(c.dim, c.dim, bias=False)
        self.to_k = nn.Linear(c.dim, c.dim, bias=False)
        self.to_v = nn.Linear(c.dim, c.dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(c.dim, c.dim, bias=False)])
        self.norm_q = RMSNorm(c.head_dim, c.eps)
        self.norm_k = RMSNorm(c.head_dim, c.eps)

    def forward(self, x, rope, text_len: int, text_mask: Optional[torch.Tensor]):
        """Block-causal attention: text queries see the text keys causally, image queries see all
        keys. ``text_mask`` (B, L) bool drops padded text keys."""
        v = self.to_v(x).unflatten(-1, (self.heads, -1))
        q = self.norm_q(self.to_q(x).unflatten(-1, (self.heads, -1))).to(v.dtype)
        k = self.norm_k(self.to_k(x).unflatten(-1, (self.heads, -1))).to(v.dtype)
        q, k = _rope_apply(q, *rope), _rope_apply(k, *rope)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        L = text_len
        if text_mask is None:
            txt = F.scaled_dot_product_attention(q[:, :, :L], k[:, :, :L], v[:, :, :L], is_causal=True)
            img = F.scaled_dot_product_attention(q[:, :, L:], k, v)
        else:
            keys = torch.cat([text_mask, text_mask.new_ones(text_mask.shape[0], x.shape[1] - L)], 1)[:, None, None]
            causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
            txt = F.scaled_dot_product_attention(q[:, :, :L], k[:, :, :L], v[:, :, :L],
                                                 attn_mask=causal & keys[..., :L])
            img = F.scaled_dot_product_attention(q[:, :, L:], k, v, attn_mask=keys)
        out = torch.cat([txt, img], dim=2).transpose(1, 2).flatten(2)
        return self.to_out[0](out)


class FeedForward(nn.Module):
    def __init__(self, c: Qwen21Config):
        super().__init__()
        hidden = c.dim * c.mlp_ratio
        self.gate_layer = nn.Linear(c.dim, hidden, bias=False)
        self.proj = nn.Linear(c.dim, hidden, bias=False)
        self.out = nn.Linear(hidden, c.dim, bias=False)

    def forward(self, x):
        return self.out(F.silu(self.gate_layer(x)) * self.proj(x))


def _rows(p: torch.Tensor, text_len: int, total: int) -> torch.Tensor:
    """(B+1, D) modulation -> (B, total, D): text rows from the trailing t = 0 row, image rows
    from each sample's own row."""
    b = p.shape[0] - 1
    zero, real = p[-1:].unsqueeze(1), p[:-1].unsqueeze(1)
    return torch.cat([zero.expand(b, text_len, -1), real.expand(b, total - text_len, -1)], dim=1)


class Block(nn.Module):
    def __init__(self, c: Qwen21Config):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(c.dim, elementwise_affine=False, eps=c.eps)
        self.attn = Attention(c)
        self.img_norm2 = nn.LayerNorm(c.dim, elementwise_affine=False, eps=c.eps)
        self.img_mlp = FeedForward(c)

    def forward(self, x, mod, rope, text_len, text_mask):
        scale1, gate1, scale2, gate2 = mod
        x = x + gate1 * self.attn(self.img_norm1(x) * (1 + scale1), rope, text_len, text_mask)
        return x + gate2 * self.img_mlp(self.img_norm2(x) * (1 + scale2))


class FinalLayer(nn.Module):
    def __init__(self, c: Qwen21Config):
        super().__init__()
        self.linear = nn.Linear(c.dim, c.dim, bias=False)
        self.norm = nn.LayerNorm(c.dim, elementwise_affine=False, eps=c.eps)

    def forward(self, x, temb):
        scale = self.linear(F.silu(temb).to(x.dtype))
        return self.norm(x) * (1 + scale[:, None])


class Qwen21Transformer(nn.Module):
    def __init__(self, c: Qwen21Config):
        super().__init__()
        self.cfg = c
        self.time_text_embed = TimeTextEmbed(c.dim)
        self.txt_in = TextProjection(c)
        self.img_in = nn.Linear(c.in_channels, c.dim, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(c.dim, 4 * c.dim, bias=False))
        self.transformer_blocks = nn.ModuleList([Block(c) for _ in range(c.num_layers)])
        self.norm_out = FinalLayer(c)
        self.proj_out = nn.Linear(c.dim, c.in_channels, bias=False)
        self.gradient_checkpointing = False
        self._rope_cache = {}

    @property
    def device(self):
        return self.img_in.weight.device

    @property
    def blocks(self):
        return self.transformer_blocks

    @staticmethod
    def config_from_state_dict(sd) -> Qwen21Config:
        dim, cin = sd["img_in.weight"].shape
        head = sd["transformer_blocks.0.attn.norm_q.weight"].shape[0]
        hidden = sd["transformer_blocks.0.img_mlp.out.weight"].shape[1]
        return Qwen21Config(
            in_channels=cin,
            num_layers=sum(1 for k in sd if k.startswith("transformer_blocks.") and k.endswith(".attn.norm_q.weight")),
            head_dim=head,
            num_heads=dim // head,
            context_dim=sd["txt_in.text_norm.weight"].shape[0],
            mlp_ratio=hidden // dim,
        )

    def forward(self, latents: torch.Tensor, t: torch.Tensor, text: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """latents (B, 64, h, w), t (B,) in [0, 1], text (B, L, 4096), text_mask (B, L) bool
        (right padding). Returns the velocity, same layout as ``latents``."""
        b, ch, gh, gw = latents.shape
        dtype = self.img_in.weight.dtype
        if text_mask is not None and bool(text_mask.all()):
            text_mask = None
        x = self.img_in(latents.flatten(2).transpose(1, 2).to(dtype))
        txt = self.txt_in(text.to(dtype))
        L = txt.shape[1]
        h = torch.cat([txt, x], dim=1)
        key = (L, gh, gw, h.device)
        if key not in self._rope_cache:
            if len(self._rope_cache) > 8:
                self._rope_cache.clear()
            self._rope_cache[key] = rope_tables(L, gh, gw, self.cfg.rope_axes, self.cfg.rope_theta, h.device)
        rope = self._rope_cache[key]
        # the pipeline hands the model t*1000 and t rounded to the compute dtype
        t = ((t.float() * 1000).to(dtype) / 1000).to(dtype)
        temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)  # extra t = 0 row for the text
        scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)
        n = h.shape[1]
        mod = (_rows(scale1, L, n), _rows(gate1.tanh(), L, n), _rows(scale2, L, n), _rows(gate2.tanh(), L, n))
        for blk in self.transformer_blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                h = checkpoint(blk, h, mod, rope, L, text_mask, use_reentrant=False)
            else:
                h = blk(h, mod, rope, L, text_mask)
        out = self.proj_out(self.norm_out(h[:, L:], temb[:-1]))
        return out.transpose(1, 2).reshape(b, ch, gh, gw)
