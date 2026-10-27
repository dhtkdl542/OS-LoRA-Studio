"""Training adapter: a frozen LoRA that sits in the model while a LoRA trains, and nowhere else.

Some models do not take a plain LoRA well. On Qwen Image 2.1 a LoRA trained alone wobbles (the face
drifts from render to render) or collapses into texture, and the loss does not show it; Fizgig and
SimpleTuner both train with a frozen "training adapter / assistant" in place, which the LoRA then
learns on top of. Fizgig's A/B: likeness 77.4 with it, 55.8 without.

The adapter is on for the training steps and the loss probes, off for every render (samples,
quality checks, the base pictures of the leak probes) and never part of the saved LoRA, so the
result works on the plain model, the one people generate with.

It is a child module named ``lora_adapter`` on each Linear it covers: the weight swap and the block
offload skip ``.lora_`` modules, so it stays on the GPU and is never exchanged.
"""
import contextlib
import re
from typing import Dict

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from .lora import LoRALinear
from .models.quant import Int8Linear


class FrozenLoRA(nn.Module):
    """delta(x) = up(down(x)) * scale, weights as buffers (no gradients, not in the optimizer)."""

    def __init__(self, down: torch.Tensor, up: torch.Tensor, scale: float):
        super().__init__()
        self.register_buffer("down", down)
        self.register_buffer("up", up)
        self.scale = scale
        self.on = True

    def forward(self, x):
        return F.linear(F.linear(x.to(self.down.dtype), self.down), self.up) * self.scale


class AdapterLinear(nn.Module):
    """A Linear the trained LoRA does not cover, with the adapter on it (same ``base`` naming as
    LoRALinear, so the weight swap finds its frozen weights)."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.lora_adapter = None

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        y = self.base(x)
        a = self.lora_adapter
        return y + a(x).to(y.dtype) if a is not None and a.on else y


def _pairs(sd: Dict[str, torch.Tensor], prefix: str):
    """{module name: (down, up, alpha or None)} from kohya (lora_down/up) or diffusers/PEFT
    (lora_A/B) keys, the model's key prefix taken off."""
    out = {}
    for k in sd:
        m = re.fullmatch(r"(.+)\.(lora_down|lora_A)(?:\.default)?\.weight", k)
        if not m:
            continue
        base = m.group(1)
        up_key = next((f"{base}.{u}.weight" for u in ("lora_up", "lora_B") if f"{base}.{u}.weight" in sd), None)
        if up_key is None:
            continue
        name = base[len(prefix):] if prefix and base.startswith(prefix) else base
        alpha = sd.get(f"{base}.alpha")
        out[name] = (sd[k], sd[up_key], float(alpha) if alpha is not None else None)
    return out


def _parent(root: nn.Module, name: str):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


class TrainingAdapter:
    def __init__(self, model: nn.Module, path: str, strength: float = 1.0, key_prefix: str = "",
                 device="cuda", log=print):
        pairs = _pairs(load_file(path), key_prefix)
        modules = dict(model.named_modules())
        self.parts = []
        missing = []
        for name, (down, up, alpha) in pairs.items():
            target = modules.get(name)
            if target is None:
                missing.append(name)
                continue
            rank = down.shape[0]
            adapter = FrozenLoRA(down.to(device, torch.bfloat16), up.to(device, torch.bfloat16),
                                 (alpha if alpha is not None else rank) / rank * strength)
            if isinstance(target, (LoRALinear, AdapterLinear)):
                target.lora_adapter = adapter
            elif isinstance(target, (nn.Linear, Int8Linear)):
                wrap = AdapterLinear(target)
                wrap.lora_adapter = adapter
                parent, attr = _parent(model, name)
                setattr(parent, attr, wrap)
            else:
                missing.append(name)
                continue
            self.parts.append(adapter)
        if not self.parts:
            raise ValueError(f"training adapter matched no layers of this model: {path}")
        if missing:
            log(f"⚠ training adapter: 모델에 없는 레이어 {len(missing)}개는 건너뛰어요 (예: {missing[:2]})")
        log(f"training adapter: {len(self.parts)}개 레이어, rank {pairs[next(iter(pairs))][0].shape[0]}, "
            f"세기 {strength:g} (학습 중에만 켜고 그림·저장에는 빼요)")

    def set(self, on: bool):
        for a in self.parts:
            a.on = on

    @contextlib.contextmanager
    def off(self):
        """Renders and the base pictures: the model as people use it, without the adapter."""
        self.set(False)
        try:
            yield
        finally:
            self.set(True)
