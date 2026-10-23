"""Time one Krea 2 LoRA training step (fwd + bwd) at 512 / 768 and report peak VRAM.

    venv\\Scripts\\python.exe tests\\profile_krea2_step.py [512 768 ...] [--swap N]
"""

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oslora.config import RunConfig  # noqa: E402
from oslora.lora import LoRANetwork  # noqa: E402
from oslora.models import get_adapter  # noqa: E402


def step(dit, net, res, text):
    x = torch.randn(1, 64, res // 16, res // 16, device="cuda")
    t = torch.tensor([0.5], device="cuda")
    pred = dit(x, t, text)
    loss = F.mse_loss(pred.float(), x)
    loss.backward()
    for p in net.parameters():
        p.grad = None


def main():
    argv = sys.argv[1:]
    swap = int(argv[argv.index("--swap") + 1]) if "--swap" in argv else 0
    if swap:
        argv.remove("--swap")
        argv.remove(str(swap))
    sizes = [int(a) for a in argv if a.isdigit()] or [512, 768]
    ad = get_adapter("krea2")
    dit = ad.load_dit(RunConfig(model="krea2").paths.dit)
    dit.gradient_checkpointing = True
    net = LoRANetwork(dit, ad.lora_targets["all"], rank=32)
    for p in net.parameters():
        p.requires_grad_(True)
    if swap:
        ad.swap_blocks = swap
        ad.block_offload(dit)
    print(f"resident {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)
    text = torch.randn(1, 120, 12, 2560, device="cuda", dtype=torch.bfloat16)
    for res in sizes:
        step(dit, net, res, text)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(3):
            step(dit, net, res, text)
        torch.cuda.synchronize()
        print(f"{res}px: {(time.time() - t0) / 3:.2f} s/step, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB",
              flush=True)


if __name__ == "__main__":
    main()
