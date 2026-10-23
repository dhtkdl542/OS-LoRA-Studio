"""Block swapping: keep the frozen weights of some transformer blocks in pinned RAM and copy them
to the GPU only while that block runs (ai-toolkit's layer_offloading, Fizgig's block swap)."""

import torch
from torch import nn


def _get(module, attr):
    return module._parameters[attr] if attr in module._parameters else module._buffers[attr]


def _set(module, attr, t):
    if attr in module._parameters:
        # a new Parameter object, never ``.data =``: autograd may have saved the old one for backward
        module._parameters[attr] = t if isinstance(t, nn.Parameter) else nn.Parameter(t, requires_grad=False)
    else:
        module._buffers[attr] = t


class BlockOffload:
    """A forward pre-hook copies a block's frozen weights to the GPU and a forward hook puts the RAM
    tensors back. With gradient checkpointing the backward recompute runs the block again, so the
    weights come back for it too; tensors an op saved for its own backward stay alive only until
    that backward. LoRA modules are trainable and stay on the GPU.

    Slots are (module, attribute), looked up on every call, so WeightSwap (which keeps each tensor
    on the device it finds it on) works unchanged; after ``Module.to(gpu)`` call ``place()``."""

    def __init__(self, blocks, device="cuda", log=print):
        self.blocks = list(blocks)
        self.device = torch.device(device)
        self.slots = {}
        self.stash = {}
        self.handles = []
        for blk in self.blocks:
            slots = []
            for mod_name, module in blk.named_modules():
                if ".lora_" in f".{mod_name}.":
                    continue
                for attr, t in list(module._parameters.items()) + list(module._buffers.items()):
                    if t is not None:
                        slots.append((module, attr))
            self.slots[id(blk)] = slots
            self.handles.append(blk.register_forward_pre_hook(self._load))
            # always_call: the non-reentrant checkpoint ends a recompute early with an exception
            self.handles.append(blk.register_forward_hook(self._unload, always_call=True))
        self.place()
        gb = sum(_get(m, a).nbytes for s in self.slots.values() for m, a in s) / 2**30
        log(f"블록 스왑: DiT 블록 {len(self.blocks)}개({gb:.1f} GB)를 RAM에 두고 쓸 때만 GPU로 올려요")

    @torch.no_grad()
    def place(self):
        """Move the swapped blocks' frozen weights (back) to pinned RAM."""
        for slots in self.slots.values():
            for module, attr in slots:
                t = _get(module, attr)
                if t.device.type != "cpu" or not t.is_pinned():
                    cpu = t.detach().to("cpu")
                    try:
                        cpu = cpu.pin_memory()
                    except RuntimeError:
                        pass  # pinning only makes the copy faster
                    _set(module, attr, cpu)
        torch.cuda.empty_cache()

    def _load(self, blk, args):
        key = id(blk)
        if key in self.stash:  # the previous call never reached its forward hook
            self._unload(blk, args, None)
        stash = []
        for module, attr in self.slots[key]:
            t = _get(module, attr)
            stash.append(t)
            _set(module, attr, t.to(self.device, non_blocking=True))
        self.stash[key] = stash

    def _unload(self, blk, args, out):
        stash = self.stash.pop(id(blk), None)
        if stash is not None:
            for (module, attr), t in zip(self.slots[id(blk)], stash):
                _set(module, attr, t)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
