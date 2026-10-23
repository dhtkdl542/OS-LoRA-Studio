"""CPU check of the SFT rendering path on a tiny MOSS-shaped model:

1. fuse_qkv is exact (split q/k/v file == fused load)
2. a fused-qkv LoRA loads exactly onto a split-q/k/v model
3. WeightSwap: Base+LoRA -> swap == SFT+LoRA, swap back == Base+LoRA, also after the
   module's buffers were replaced (what Module.to does on pause/resume)

    venv\\Scripts\\python.exe tests\\test_sft_swap.py
"""

import os
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oslora.lora import LoRANetwork  # noqa: E402
from oslora.models.moss import adapter as moss  # noqa: E402
from oslora.models.moss.transformer import MossConfig, MossTransformer  # noqa: E402
from oslora.models.swap import WeightSwap  # noqa: E402
TARGETS = moss.MossAdapter.lora_targets

REAL_SFT = r"E:/ComfyUI_Model/diffusion_models/MOSS/moss-image-sft-dit-int8-convrot.safetensors"


def make_file(path, seed, quant_bytes):
    """Random tiny model in the vendor layout (split q/k/v), int8 comfy-quantized blocks."""
    torch.manual_seed(seed)
    cfg = MossConfig(dim=256, ffn_dim=512, num_heads=2, num_layers=2, text_dim=64, freq_dim=64,
                     latent_channels=128, fused_qkv=False)
    sd = {k: v.clone() for k, v in MossTransformer(cfg).state_dict().items()}
    for k in list(sd):
        if k.startswith("layers.") and ".proj." in k and k.endswith(".weight"):
            base = k[: -len(".weight")]
            sd[k] = torch.randint(-127, 128, sd[k].shape, dtype=torch.int8)
            sd[f"{base}.weight_scale"] = torch.rand(sd[k].shape[0], 1) * 1e-3
            sd[f"{base}.comfy_quant"] = quant_bytes.clone()
        elif sd[k].is_floating_point():
            sd[k] = (torch.randn_like(sd[k]) * 0.05).to(torch.bfloat16)
    save_file({k: v.contiguous() for k, v in sd.items()}, path)
    return sd


def run(model, x, t, text, mask):
    with torch.no_grad():
        return model(x, t, text, mask).float()


def main():
    with safe_open(REAL_SFT, "pt") as f:
        quant = f.get_tensor("layers.0.proj.q.comfy_quant")
    tmp = tempfile.mkdtemp()
    sft_path, base_path = os.path.join(tmp, "sft.safetensors"), os.path.join(tmp, "base.safetensors")
    make_file(sft_path, 1, quant)
    base_sd = moss.fuse_qkv(make_file(base_path, 2, quant))
    save_file({k: v.contiguous() for k, v in base_sd.items()}, base_path)  # Base ships fused

    log = lambda m: None  # noqa: E731
    dev = "cpu"
    torch.manual_seed(0)
    x = torch.randn(1, 128, 8, 8)
    t = torch.tensor([500.0])
    text, mask = moss_text(dev)

    # 1. fusion exact
    sft_fused = moss.load_dit(sft_path, device=dev, log=log)
    real_fuse = moss.fuse_qkv
    moss.fuse_qkv = lambda sd: sd
    try:
        sft_split = moss.load_dit(sft_path, device=dev, log=log)
    finally:
        moss.fuse_qkv = real_fuse
    assert sft_fused.cfg.fused_qkv and not sft_split.cfg.fused_qkv
    a, b = run(sft_fused, x, t, text, mask), run(sft_split, x, t, text, mask)
    print(f"1. fused vs split SFT: max diff {float((a - b).abs().max()):.2e} (output scale {float(a.abs().mean()):.2e})")
    assert torch.allclose(a, b, atol=1e-3, rtol=1e-2)

    # 2. fused LoRA onto split model
    net_f = LoRANetwork(sft_fused, TARGETS["all"], rank=4, alpha=4)
    for p in net_f.parameters():
        torch.nn.init.normal_(p, std=0.05)
    lora_sd = net_f.state_dict(dtype=torch.float32)
    net_s = LoRANetwork(sft_split, TARGETS["all"], rank=4, alpha=4)
    net_s.load_state_dict(lora_sd)
    a, b = run(sft_fused, x, t, text, mask), run(sft_split, x, t, text, mask)
    print(f"2. LoRA fused vs split: max diff {float((a - b).abs().max()):.2e}")
    assert torch.allclose(a, b, atol=1e-3, rtol=1e-2)
    sft_with_lora = a
    del sft_split, net_s

    # 3. swap
    base = moss.load_dit(base_path, device=dev, log=log)
    net_b = LoRANetwork(base, TARGETS["all"], rank=4, alpha=4)
    net_b.load_state_dict(lora_sd)
    before = run(base, x, t, text, mask)
    assert not torch.allclose(before, sft_with_lora, atol=1e-3), "test models must differ"
    swap = WeightSwap(base, moss.load_dit(sft_path, device="cpu", log=log), log=log)
    swap.swap()
    on_sft = run(base, x, t, text, mask)
    print(f"3a. swapped Base -> SFT vs SFT+LoRA: max diff {float((on_sft - sft_with_lora).abs().max()):.2e}")
    assert torch.allclose(on_sft, sft_with_lora, atol=1e-3, rtol=1e-2)
    swap.swap()
    back = run(base, x, t, text, mask)
    print(f"3b. swapped back vs Base+LoRA: max diff {float((back - before).abs().max()):.2e}")
    assert torch.equal(back, before)
    base._apply(lambda v: v.clone())  # Module.to replaces buffer objects (pause/resume)
    swap.swap()
    again = run(base, x, t, text, mask)
    swap.swap()
    print(f"3c. after buffer replacement: SFT diff {float((again - sft_with_lora).abs().max()):.2e}, "
          f"back diff {float((run(base, x, t, text, mask) - before).abs().max()):.2e}")
    assert torch.allclose(again, sft_with_lora, atol=1e-3, rtol=1e-2)
    assert torch.equal(run(base, x, t, text, mask), before)
    assert all(p.grad is None for p in net_b.parameters())
    print("OK")


def moss_text(dev):
    text = torch.randn(1, 12, 64, device=dev).to(torch.bfloat16)
    mask = torch.ones(1, 12, dtype=torch.bool, device=dev)
    return text, mask


if __name__ == "__main__":
    main()
