"""Time one LoRA training step (fwd + bwd) at a few resolutions and show the hottest ops."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from oslora.lora import LoRANetwork
from oslora.models.moss.adapter import MossPaths, load_dit
from oslora.models.moss.adapter import MossAdapter

TARGETS = MossAdapter.lora_targets


def step(dit, net, res, text):
    x = torch.randn(1, 128, res // 16, res // 16, device="cuda")
    t = torch.tensor([500.0], device="cuda")
    pred = dit(x, t, text)
    loss = F.mse_loss(pred.float(), x)
    loss.backward()
    for p in net.parameters():
        p.grad = None


def main():
    dit = load_dit(MossPaths().dit)
    dit.gradient_checkpointing = True
    net = LoRANetwork(dit, TARGETS["all"], rank=32)
    for p in net.parameters():
        p.requires_grad_(True)
    from oslora.models import quant

    text = torch.randn(1, 250, 4096, device="cuda", dtype=torch.bfloat16)
    for res in (768, 1024):
        for int8 in (False, True):  # bf16 dequant path vs int8 tensor cores
            quant.INT8_COMPUTE = int8
            step(dit, net, res, text)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(3):
                step(dit, net, res, text)
            torch.cuda.synchronize()
            print(f"{res}px {'int8' if int8 else 'bf16'}: {(time.time() - t0) / 3:.2f} s/step, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    if "--prof" in sys.argv:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step(dit, net, 768, text)
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))


if __name__ == "__main__":
    main()
