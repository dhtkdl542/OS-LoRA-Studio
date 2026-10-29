"""LoRA injection, saving (ComfyUI / kohya key layout) and EMA."""

import math
import re
from typing import Dict, Iterable, List, Optional

import torch
from safetensors.torch import save_file
from torch import nn

from .models.quant import Int8Linear


class LoRALinear(nn.Module):
    """y = base(x) + up(down(dropout(x))) * (alpha / rank) * multiplier"""

    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float = 0.0, rank_dropout: float = 0.0):
        super().__init__()
        self.base = base
        in_f, out_f = base.in_features, base.out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.multiplier = 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.rank_dropout = rank_dropout
        self.lora_down = nn.Linear(in_f, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.lora_adapter = None  # a frozen training adapter (adapter.py), on while training only

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        y = self.base(x)
        a = self.lora_adapter
        if a is not None and a.on:
            y = y + a(x).to(y.dtype)
        if self.multiplier == 0.0:
            return y
        h = self.lora_down(self.dropout(x).to(self.lora_down.weight.dtype))
        if self.training and self.rank_dropout > 0:
            keep = (torch.rand(h.shape[-1], device=h.device) > self.rank_dropout).to(h.dtype)
            h = h * keep / (1.0 - self.rank_dropout)
        return y + (self.lora_up(h) * (self.scale * self.multiplier)).to(y.dtype)


class LoRANetwork:
    def __init__(
        self,
        model: nn.Module,
        target_patterns: Iterable[str],
        rank: int = 32,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        dtype=torch.float32,
        key_prefix: str = "diffusion_model.",
    ):
        self.key_prefix = key_prefix
        self.modules: Dict[str, LoRALinear] = {}
        alpha = float(alpha if alpha is not None else rank)
        regexes = [re.compile(p) for p in target_patterns]
        targets = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, Int8Linear)) and any(r.fullmatch(name) for r in regexes):
                targets.append((name, module))
        if not targets:
            raise ValueError("LoRA target patterns matched no layers")
        for name, module in targets:
            lora = LoRALinear(module, rank, alpha, dropout, rank_dropout)
            lora.lora_down.to(device=module_device(module), dtype=dtype)
            lora.lora_up.to(device=module_device(module), dtype=dtype)
            parent, attr = _parent(model, name)
            setattr(parent, attr, lora)
            self.modules[name] = lora
        self.rank, self.alpha = rank, alpha

    @torch.no_grad()
    def delta_norm(self) -> float:
        """How far the LoRA moves the model: the sum over modules of ||scale * up @ down||_F, from
        the r x r factors (cheap). Grows smoothly with LR x steps; on Damdeari the user judged the
        LoRA with the larger value more alike in all three comparisons so far (v3 584 > v5 447;
        v5 2500 501 > 1500 361; 2000 447 > 1800 420)."""
        total = 0.0
        for m in self.modules.values():
            up, down = m.lora_up.weight.float(), m.lora_down.weight.float()
            r = torch.linalg.qr(up).R @ torch.linalg.qr(down.T).R.T
            total += float(torch.linalg.matrix_norm(r)) * m.scale
        return total

    def parameters(self) -> List[nn.Parameter]:
        params = []
        for m in self.modules.values():
            params += [m.lora_down.weight, m.lora_up.weight]
        return params

    def set_multiplier(self, value: float):
        for m in self.modules.values():
            m.multiplier = value

    def train(self, mode=True):
        for m in self.modules.values():
            m.train(mode)

    def state_dict(self, dtype=torch.bfloat16) -> Dict[str, torch.Tensor]:
        sd = {}
        for name, m in self.modules.items():
            key = self.key_prefix + name
            sd[f"{key}.lora_down.weight"] = m.lora_down.weight.detach().to("cpu", dtype).contiguous()
            sd[f"{key}.lora_up.weight"] = m.lora_up.weight.detach().to("cpu", dtype).contiguous()
            sd[f"{key}.alpha"] = torch.tensor(float(m.alpha))
        return sd

    def load_state_dict(self, sd: Dict[str, torch.Tensor]):
        """Load LoRA weights. Also maps a fused-qkv LoRA (Comfy repack layout, which the Base
        file uses) onto a model with split q/k/v (the vendor SFT layout): q, k and v share the
        fused down-projection and take their third of the up-projection's rows -- exact."""
        missing = []
        for name, m in self.modules.items():
            key = self.key_prefix + name
            if f"{key}.lora_down.weight" in sd:
                m.lora_down.weight.data.copy_(sd[f"{key}.lora_down.weight"])
                m.lora_up.weight.data.copy_(sd[f"{key}.lora_up.weight"])
                continue
            part = re.fullmatch(r"(.*\.proj)\.(q|k|v)", key)
            fused = f"{part.group(1)}.qkv" if part else None
            if fused and f"{fused}.lora_down.weight" in sd:
                up = sd[f"{fused}.lora_up.weight"]
                rows = up.shape[0] // 3
                i = "qkv".index(part.group(2))
                m.lora_down.weight.data.copy_(sd[f"{fused}.lora_down.weight"])
                m.lora_up.weight.data.copy_(up[i * rows : (i + 1) * rows])
                continue
            missing.append(name)
        if missing:
            raise KeyError(f"LoRA weights missing for {len(missing)} layers, e.g. {missing[:3]}")

    def save(self, path: str, metadata: Optional[Dict[str, str]] = None, weights: Optional[Dict[str, torch.Tensor]] = None):
        sd = weights if weights is not None else self.state_dict()
        save_file(sd, path, metadata={k: str(v) for k, v in (metadata or {}).items()})

    @torch.no_grad()
    def max_norm_(self, max_norm: float) -> float:
        """Scale each module's delta-W down so ||up @ down|| * scale <= max_norm (kohya's
        scale_weight_norms). Returns the largest norm seen, a cheap overbake indicator."""
        largest = 0.0
        for m in self.modules.values():
            up, down = m.lora_up.weight, m.lora_down.weight
            norm = float((up.float() @ down.float()).norm() * m.scale)
            largest = max(largest, norm)
            if max_norm > 0 and norm > max_norm:
                ratio = math.sqrt(max_norm / norm)
                up.mul_(ratio)
                down.mul_(ratio)
        return largest


def module_device(module: nn.Module):
    for t in list(module.parameters()) + list(module.buffers()):
        return t.device
    return torch.device("cpu")


def _parent(root: nn.Module, name: str):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


class EMA:
    """fp32 shadow of the LoRA weights with the usual warmup: decay_n = min(decay, (1+n)/(10+n))."""

    def __init__(self, params: List[nn.Parameter], decay: float = 0.99):
        self.params = params
        self.decay = decay
        self.shadow = [p.detach().float().clone() for p in params]
        self.n = 0

    @torch.no_grad()
    def update(self):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for s, p in zip(self.shadow, self.params):
            s.lerp_(p.detach().float(), 1 - d)

    @torch.no_grad()
    def swap(self):
        """Exchange live and shadow weights (call twice to restore)."""
        for s, p in zip(self.shadow, self.params):
            tmp = p.detach().float().clone()
            p.copy_(s.to(p.dtype))
            s.copy_(tmp)

    def state(self):
        return {"n": self.n, "shadow": [s.cpu() for s in self.shadow]}

    def load(self, state):
        self.n = state["n"]
        for s, v in zip(self.shadow, state["shadow"]):
            s.copy_(v)
