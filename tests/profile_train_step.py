"""Time one LoRA training step (fwd + bwd) of a model at a few resolutions and report peak VRAM.

    venv\\Scripts\\python.exe tests\\profile_train_step.py --model krea2 [512 768 ...] [--swap N]

Text is random with the model's embedding shape (150 tokens); the DiT is the training model.
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

TEXT_SHAPE = {"moss_image": (4096,), "krea2": (12, 2560), "qwen21": (4096,)}
LATENT_CH = {"moss_image": 128, "krea2": 64, "qwen21": 64}


def step(dit, net, model, res, text):
    x = torch.randn(1, LATENT_CH[model], res // 16, res // 16, device="cuda")
    t = torch.tensor([0.5], device="cuda")
    pred = dit(x, t, text)
    loss = F.mse_loss(pred.float(), x)
    loss.backward()
    for p in net.parameters():
        p.grad = None


def option(argv, name, default):
    if name not in argv:
        return default
    i = argv.index(name)
    value = argv[i + 1]
    del argv[i:i + 2]
    return value


def main():
    argv = sys.argv[1:]
    model = option(argv, "--model", "krea2")
    swap = int(option(argv, "--swap", -1))
    sizes = [int(a) for a in argv if a.isdigit()] or [512, 768]
    ad = get_adapter(model)
    dit = ad.load_dit(RunConfig(model=model).paths.dit)
    dit.gradient_checkpointing = True
    net = LoRANetwork(dit, ad.lora_targets["all"], rank=32, key_prefix=ad.lora_prefix)
    for p in net.parameters():
        p.requires_grad_(True)
    if swap >= 0:
        ad.swap_blocks = swap
    ad.block_offload(dit)
    print(f"resident {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)
    text = torch.randn(1, 150, *TEXT_SHAPE[model], device="cuda", dtype=torch.bfloat16)
    for res in sizes:
        step(dit, net, model, res, text)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(3):
            step(dit, net, model, res, text)
        torch.cuda.synchronize()
        print(f"{res}px: {(time.time() - t0) / 3:.2f} s/step, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB",
              flush=True)


if __name__ == "__main__":
    main()
