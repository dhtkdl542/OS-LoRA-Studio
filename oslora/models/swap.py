"""In-place exchange of a live DiT's frozen weights with a second copy parked in RAM."""

import torch


class WeightSwap:
    """A second DiT (same architecture) parked in RAM whose frozen weights can be exchanged
    with the live model's in place: samples and quality checks render on the inference model
    (MOSS SFT, Krea 2 Turbo) while training runs on the base, without holding two DiTs in VRAM. LoRA modules are not touched.

    Slots are (module, attribute) rather than tensor references: ``Module.to`` replaces
    buffer objects (pause/resume moves the DiT), so tensors are looked up on every swap."""

    def __init__(self, live: torch.nn.Module, other: torch.nn.Module, log=print):
        theirs = dict(other.named_parameters())
        theirs.update(other.named_buffers())
        self.slots = []
        for mod_name, module in live.named_modules():
            if ".lora_" in f".{mod_name}.":
                continue
            local = list(module._parameters.items()) + list(module._buffers.items())
            for attr, t in local:
                if t is None:
                    continue
                name = f"{mod_name}.{attr}" if mod_name else attr
                src = theirs.get(name.replace(".base.", "."))
                if src is None or src.shape != t.shape or src.dtype != t.dtype:
                    raise ValueError(f"추론용 DiT 구조가 학습 모델과 달라요: {name}")
                self.slots.append([module, attr, src])
        del other, theirs
        self.active = False

    @torch.no_grad()
    def swap(self):
        for slot in self.slots:
            module, attr, parked = slot
            if attr in module._parameters:
                p = module._parameters[attr]
                slot[2] = p.data.to("cpu")
                p.data = parked.to(p.device)
            else:
                b = module._buffers[attr]
                slot[2] = b.to("cpu")
                module._buffers[attr] = parked.to(b.device)
        self.active = not self.active
