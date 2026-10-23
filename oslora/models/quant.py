"""ComfyUI int8 weights (``int8_tensorwise``, optionally ConvRot-rotated) kept int8 on the GPU.

Storage format (read from each layer's ``comfy_quant`` JSON blob):
    W_stored = round(R(W) / s)        int8 [out, in], s fp32 [out, 1]
where R is the block *regular* Hadamard transform along ``in`` (block = convrot_groupsize,
a power of 4). R is symmetric and orthogonal, so

    y = x W^T = (x R)(W R)^T ~= R(x) (W_stored * s)^T

i.e. we rotate the activation instead of un-rotating the weight. The weight is dequantized
per call (and again in backward) so only the int8 copy stays resident: a 5B DiT costs ~5 GB.
"""

import json
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

_HADAMARD: Dict[Tuple[int, str, torch.dtype], torch.Tensor] = {}


def regular_hadamard(n: int, device, dtype) -> torch.Tensor:
    key = (n, str(device), dtype)
    if key not in _HADAMARD:
        r4 = torch.tensor(
            [[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32
        )
        h = r4
        while h.shape[0] < n:
            h = torch.kron(h, r4)
        if h.shape[0] != n:
            raise ValueError(f"rotation size {n} is not a power of 4")
        _HADAMARD[key] = (h / n**0.5).to(device=device, dtype=dtype)
    return _HADAMARD[key]


def largest_pow4_divisor(d: int, cap: int) -> int:
    h = 1
    while d % (h * 4) == 0 and h * 4 <= cap:
        h *= 4
    return h


def rotate(x: torch.Tensor, n: int) -> torch.Tensor:
    """Block regular-Hadamard along the last dim (self-inverse)."""
    if n <= 1:
        return x
    shape = x.shape
    h = regular_hadamard(n, x.device, x.dtype)
    return torch.matmul(x.reshape(*shape[:-1], shape[-1] // n, n), h).reshape(shape)


def parse_comfy_quant(blob: torch.Tensor) -> dict:
    return json.loads(bytes(blob.cpu().tolist()).decode("utf-8"))


def _dequant(qweight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # int8 values are exact in bf16; only the scale is rounded first, so this stays within
    # about one bf16 ulp of the fp32 product, at a quarter of the memory traffic.
    return qweight.to(dtype) * scale.to(dtype)


class _Int8Matmul(torch.autograd.Function):
    """y = R(x) @ deq(W)^T + b. Saves only the int8 weight for backward."""

    @staticmethod
    def forward(ctx, x, qweight, scale, bias, rot):
        w = _dequant(qweight, scale, x.dtype)
        y = F.linear(rotate(x, rot), w, bias.to(x.dtype) if bias is not None else None)
        ctx.save_for_backward(qweight, scale)
        ctx.rot = rot
        return y

    @staticmethod
    def backward(ctx, grad_out):
        qweight, scale = ctx.saved_tensors
        w = _dequant(qweight, scale, grad_out.dtype)
        grad_x = rotate(grad_out @ w, ctx.rot)
        return grad_x, None, None, None, None


# int8 tensor-core compute (torch._int_mm): activations / output gradients are quantized per row
# (per token) to int8 and multiplied against the stored int8 weight directly -- the scheme of
# SwitchBack (Wortsman et al. 2023) for the forward and input-gradient matmuls; the ConvRot
# rotation is what keeps activations outlier-free enough for this (ComfyUI runs its int8 models
# the same way). RTX 40/50 int8 tensor throughput is ~2x bf16. Off by default: set per run.
INT8_COMPUTE = False


def _quant_rows(t: torch.Tensor):
    """Symmetric per-row int8: (int8 [M,K], fp32 scale [M,1])."""
    t = t.float()
    s = t.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-12) / 127.0
    return (t / s).round_().clamp_(-127, 127).to(torch.int8), s


class _Int8MatmulTC(torch.autograd.Function):
    """Same maths as _Int8Matmul, computed on int8 tensor cores."""

    @staticmethod
    def forward(ctx, x, qweight, scale, bias, rot):
        shape = x.shape
        xr = rotate(x, rot).reshape(-1, shape[-1])
        xq, sx = _quant_rows(xr)
        y = torch._int_mm(xq, qweight.t()).float().mul_(sx).mul_(scale.t())
        if bias is not None:
            y.add_(bias.float())
        ctx.save_for_backward(qweight, scale)
        ctx.rot, ctx.shape = rot, shape
        return y.to(x.dtype).reshape(*shape[:-1], qweight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        qweight, scale = ctx.saved_tensors
        g = grad_out.reshape(-1, grad_out.shape[-1]).float() * scale.t()  # fold the weight scale in
        gq, sg = _quant_rows(g)
        dx = torch._int_mm(gq, qweight).float().mul_(sg).to(grad_out.dtype)
        return rotate(dx, ctx.rot).reshape(ctx.shape), None, None, None, None


def _int8_ok(x: torch.Tensor, qweight: torch.Tensor) -> bool:
    # _int_mm needs CUDA, more than 16 rows and K, N multiples of 8
    return (INT8_COMPUTE and x.is_cuda and x.numel() // x.shape[-1] > 16
            and qweight.shape[0] % 8 == 0 and qweight.shape[1] % 8 == 0)


class Int8Linear(nn.Module):
    """Frozen int8 linear (per-row scales, optional ConvRot). LoRA wraps it like nn.Linear."""

    def __init__(self, in_features: int, out_features: int, bias: bool, rot: int, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rot = rot
        self.register_buffer(
            "qweight", torch.empty(out_features, in_features, dtype=torch.int8, device=device)
        )
        self.register_buffer("scale", torch.empty(out_features, 1, dtype=torch.float32, device=device))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, device=device))
        else:
            self.bias = None

    @property
    def weight_dtype(self):
        return torch.int8

    def forward(self, x):
        if _int8_ok(x, self.qweight):
            return _Int8MatmulTC.apply(x, self.qweight, self.scale, self.bias, self.rot)
        return _Int8Matmul.apply(x, self.qweight, self.scale, self.bias, self.rot)

    def dequantized_weight(self, dtype=torch.float32) -> torch.Tensor:
        """The un-rotated full-precision weight (for diagnostics / merging)."""
        w = _dequant(self.qweight, self.scale, torch.float32)
        return rotate(w, self.rot).to(dtype)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, rot={self.rot}, int8"


class Int8Embedding(nn.Module):
    """Per-row int8 embedding table; rows are dequantized (and un-rotated) per lookup."""

    def __init__(self, num: int, dim: int, rot: int, device=None):
        super().__init__()
        self.num_embeddings, self.embedding_dim, self.rot = num, dim, rot
        self.register_buffer("qweight", torch.empty(num, dim, dtype=torch.int8, device=device))
        self.register_buffer("scale", torch.empty(num, 1, dtype=torch.float32, device=device))
        self.out_dtype = torch.bfloat16

    def forward(self, ids):
        rows = self.qweight[ids].float() * self.scale[ids].float()
        return rotate(rows, self.rot).to(self.out_dtype)


def _get_parent(root: nn.Module, name: str):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    return parent, parts[-1]


def _set_child(parent, attr, module):
    if attr.isdigit():
        parent[int(attr)] = module
    else:
        setattr(parent, attr, module)


def load_comfy_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    dtype: torch.dtype,
    device,
    ignore_prefixes: Tuple[str, ...] = (),
):
    """Load a (possibly comfy-quantized) state dict into a meta-initialized ``model``.

    Quantized linears/embeddings are swapped for Int8 modules; everything else is cast to
    ``dtype``. Returns the list of keys the model expected but the file lacked.
    """
    quant_cfg = {}
    for key in list(state_dict):
        if key.endswith(".comfy_quant"):
            quant_cfg[key[: -len(".comfy_quant")]] = parse_comfy_quant(state_dict.pop(key))

    for name, cfg in quant_cfg.items():
        if name.startswith(ignore_prefixes):
            continue
        if cfg.get("format") != "int8_tensorwise":
            raise NotImplementedError(f"{name}: unsupported comfy quant format {cfg.get('format')}")
        parent, attr = _get_parent(model, name)
        old = getattr(parent, attr) if not attr.isdigit() else parent[int(attr)]
        group = int(cfg.get("convrot_groupsize", 256)) if cfg.get("convrot") else 1
        if isinstance(old, nn.Linear):
            rot = largest_pow4_divisor(old.in_features, group) if group > 1 else 1
            new = Int8Linear(old.in_features, old.out_features, old.bias is not None, rot, device="meta")
        elif isinstance(old, nn.Embedding):
            rot = largest_pow4_divisor(old.embedding_dim, group) if group > 1 else 1
            new = Int8Embedding(old.num_embeddings, old.embedding_dim, rot, device="meta")
            new.out_dtype = dtype
        else:
            raise TypeError(f"{name}: quantized weight on unsupported module {type(old).__name__}")
        _set_child(parent, attr, new)
        state_dict[f"{name}.qweight"] = state_dict.pop(f"{name}.weight")
        scale = state_dict.pop(f"{name}.weight_scale").float().reshape(-1, 1)
        rows = new.out_features if isinstance(new, Int8Linear) else new.num_embeddings
        if scale.shape[0] == 1 and rows > 1:  # one scale for the whole tensor (int8_tensorwise)
            scale = scale.expand(rows, 1).contiguous()
        state_dict[f"{name}.scale"] = scale

    for key in list(state_dict):
        if key.startswith(ignore_prefixes):
            del state_dict[key]
            continue
        t = state_dict[key]
        if t.is_floating_point() and not key.endswith(".scale"):
            state_dict[key] = t.to(dtype)

    result = model.load_state_dict(state_dict, strict=False, assign=True)
    if result.unexpected_keys:
        raise KeyError(f"unexpected keys in checkpoint: {result.unexpected_keys[:8]}")
    model.to(device)
    return result.missing_keys
