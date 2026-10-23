"""CPU checks of the Qwen-Image 2.1 port (no GPU, no model weights beyond file headers):

  * our DiT == diffusers' QwenImage21Transformer2DModel on a small random model, with and without
    a right-padded text mask
  * base / Turbo sigmas == FlowMatchEulerDiscreteScheduler with the two repos' scheduler configs
  * splitting ComfyUI's fused img_mlp.gate_up (plain and per-row int8 scales) is exact
  * the VAE key conversion covers diffusers' AutoencoderKLQwenImage21 (names and shapes)
  * the prompt template drops exactly the system turn
  * LoRA keys match the layout ComfyUI and existing Qwen 2.1 LoRAs use

    venv\\Scripts\\python -m tests.test_qwen21_port
"""

import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oslora.models.qwen21 import adapter as qa  # noqa: E402
from oslora.models.qwen21.transformer import Qwen21Transformer  # noqa: E402

VAE_FILE = "E:/ComfyUI_Model/vae/Qwen/qwen_image_2.1_vae_bf16.safetensors"
DIT_FILE = "E:/ComfyUI_Model/diffusion_models/Qwen/qwen_image_2.1_int8_convrot.safetensors"
TE_FILE = "E:/ComfyUI_Model/text_encoders/Qwen/qwen3vl_8b_int8_convrot.safetensors"


def meta_state_dict(path):
    """Every tensor of a safetensors file as a meta tensor, except the small comfy_quant blobs."""
    from safetensors import safe_open

    with open(path, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    header.pop("__metadata__", None)
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I8": torch.int8, "U8": torch.uint8}
    sd = {k: torch.empty(v["shape"], dtype=dtypes[v["dtype"]], device="meta") for k, v in header.items()}
    with safe_open(path, "pt") as f:
        for k in sd:
            if k.endswith(".comfy_quant"):
                sd[k] = f.get_tensor(k)
    return sd


def rel(a, c):
    return float((a - c).norm() / c.norm())


def check_dit():
    from diffusers import QwenImage21Transformer2DModel

    torch.manual_seed(0)
    ref = QwenImage21Transformer2DModel(num_layers=2, attention_head_dim=32, num_attention_heads=4,
                                        context_in_dim=64, mlp_ratio=3, axes_dims_rope=(8, 12, 12))
    with torch.no_grad():
        for p in ref.parameters():  # zero-initialised norms would hide mistakes
            p.add_(torch.randn_like(p) * 0.1)
    ref.eval()
    sd = ref.state_dict()
    cfg = Qwen21Transformer.config_from_state_dict(sd)
    cfg.rope_axes = (8, 12, 12)
    ours = Qwen21Transformer(cfg)
    missing, unexpected = ours.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    ours.eval()

    b, gh, gw, L = 2, 4, 6, 7
    lat = torch.randn(b, 64, gh, gw)
    text = torch.randn(b, L, 64)
    t = torch.tensor([0.37, 0.81])
    img_mask = torch.cat([torch.zeros(b, L, dtype=torch.bool), torch.ones(b, gh * gw // 4, dtype=torch.bool)], 1)

    def run_ref(mask):
        out = ref(hidden_states=lat.flatten(2).transpose(1, 2), encoder_hidden_states=text, timestep=t,
                  img_shapes=[[(1, gh, gw)]] * b, img_mask=img_mask, encoder_hidden_states_mask=mask,
                  return_dict=False)[0][:, -gh * gw:]  # the pipeline keeps the target rows
        return out.transpose(1, 2).reshape(b, 64, gh, gw)

    mask = torch.tensor([[True] * L, [True] * 5 + [False] * 2])
    with torch.no_grad():
        e_full = rel(ours(lat, t, text), run_ref(None))
        e_mask = rel(ours(lat, t, text, mask), run_ref(mask))
    print(f"DiT: ours vs diffusers rel err {e_full:.2e} (no mask), {e_mask:.2e} (padded text)")
    assert e_full < 1e-5 and e_mask < 1e-5
    return ours


def check_lora_keys(dit):
    from oslora.lora import LoRANetwork

    ad = qa.Qwen21Adapter()
    net = LoRANetwork(dit, ad.lora_targets["all"], rank=4, key_prefix=ad.lora_prefix)
    keys = {k.rsplit(".", 2)[0] for k in net.state_dict() if k.endswith(".lora_down.weight")}
    want = {f"transformer.transformer_blocks.{i}.{m}" for i in range(2) for m in
            ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out")}
    assert keys == want, sorted(keys)[:8]
    print(f"LoRA: {len(keys)} layers, keys like {sorted(keys)[0]}")


def check_sigmas():
    from diffusers import FlowMatchEulerDiscreteScheduler

    base = FlowMatchEulerDiscreteScheduler(base_image_seq_len=256, base_shift=0.5, max_image_seq_len=8192, max_shift=0.9,
                                           shift=1.0, shift_terminal=0.02, time_shift_type="exponential",
                                           use_dynamic_shifting=True)
    for steps, seq in ((25, 48 * 64), (40, 64 * 64), (20, 40 * 54)):
        base.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=qa.calc_mu(seq))
        err = float(np.abs(base.sigmas.numpy() - qa.base_sigmas(steps, seq)).max())
        print(f"base sigmas {steps} steps, {seq} tokens: max diff {err:.1e}")
        assert err < 1e-6
    turbo = FlowMatchEulerDiscreteScheduler(base_image_seq_len=256, base_shift=0.5, max_image_seq_len=8192,
                                            max_shift=0.9, shift=1.0, time_shift_type="exponential")
    turbo.set_timesteps(sigmas=list(qa.TURBO_SIGMAS))
    err = float(np.abs(turbo.sigmas.numpy() - qa.turbo_sigmas(8)).max())
    print(f"turbo sigmas: max diff {err:.1e}")
    assert err < 1e-6


def check_gate_up():
    torch.manual_seed(1)
    gate, up = torch.randn(12, 8), torch.randn(12, 8)
    sd = qa.split_gate_up({"transformer_blocks.0.img_mlp.gate_up.weight": torch.cat([gate, up])})
    assert torch.equal(sd["transformer_blocks.0.img_mlp.gate_layer.weight"], gate)
    assert torch.equal(sd["transformer_blocks.0.img_mlp.proj.weight"], up)
    q = torch.randint(-127, 128, (24, 8), dtype=torch.int8)
    s = torch.rand(24, 1)
    blob = torch.tensor(list(b'{"format": "int8_tensorwise"}'), dtype=torch.uint8)
    sd = qa.split_gate_up({"b.img_mlp.gate_up.weight": q, "b.img_mlp.gate_up.weight_scale": s,
                           "b.img_mlp.gate_up.comfy_quant": blob})
    assert torch.equal(sd["b.img_mlp.proj.weight"], q[12:]) and torch.equal(sd["b.img_mlp.proj.weight_scale"], s[12:])
    assert torch.equal(sd["b.img_mlp.gate_layer.comfy_quant"], blob)
    print("gate_up split: exact")


def check_vae_keys():
    if not os.path.isfile(VAE_FILE):
        print("VAE file not found, skipped:", VAE_FILE)
        return
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLQwenImage21

    with open(VAE_FILE, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    header.pop("__metadata__", None)
    fake = {k: torch.empty(v["shape"], device="meta") for k, v in header.items()}
    conv = qa.convert_vae(fake)
    with open(os.path.join(qa.ensure_configs(), "vae_config.json"), encoding="utf-8") as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    with init_empty_weights():
        want = AutoencoderKLQwenImage21(**cfg).state_dict()
    missing = [k for k in want if k not in conv]
    extra = [k for k in conv if k not in want]
    shape = [k for k in want if k in conv and tuple(conv[k].shape) != tuple(want[k].shape)]
    print(f"VAE keys: {len(conv)} converted, missing {len(missing)}, extra {len(extra)}, shape mismatch {len(shape)}")
    assert not missing and not extra and not shape, (missing[:5], extra[:5], shape[:5])


def check_real_files():
    """The real ComfyUI files load into our modules with no missing or unexpected keys."""
    from accelerate import init_empty_weights
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextConfig, Qwen3VLTextModel

    from oslora.models.quant import Int8Linear, load_comfy_state_dict

    if os.path.isfile(DIT_FILE):
        sd = qa.split_gate_up(meta_state_dict(DIT_FILE))
        cfg = Qwen21Transformer.config_from_state_dict(sd)
        with init_empty_weights():
            dit = Qwen21Transformer(cfg)
        missing = load_comfy_state_dict(dit, sd, torch.bfloat16, "meta")
        assert not missing, missing[:8]
        n8 = sum(isinstance(m, Int8Linear) for m in dit.modules())
        print(f"DiT file: {cfg.num_layers} blocks, dim {cfg.dim}, {n8} int8 linears, all keys matched")
    if os.path.isfile(TE_FILE):
        with open(os.path.join(qa.ensure_configs(), "config.json"), encoding="utf-8") as f:
            config = Qwen3VLTextConfig(**json.load(f)["text_config"])
        with init_empty_weights():
            te = Qwen3VLTextModel(config)
        sd = meta_state_dict(TE_FILE)
        sd = {k[6:]: v for k, v in sd.items() if k.startswith("model.") and not k.startswith("model.visual.")}
        missing = load_comfy_state_dict(te, sd, torch.bfloat16, "meta")
        assert not missing, missing[:8]
        print(f"text encoder file: {config.num_hidden_layers} layers, all keys matched")


def check_template():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(qa.ensure_configs())
    te = qa.Qwen21TextEncoder.__new__(qa.Qwen21TextEncoder)
    te.tokenizer = tok
    ids, drop = te.token_ids("a cat")
    assert tok.decode(ids[:drop]) == "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
    assert tok.decode(ids[drop:]) == "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"
    ids, drop = te.token_ids("word " * 2000, max_length=64)
    assert len(ids) - drop == 64 and tok.decode(ids[-5:]).endswith("<|im_start|>assistant\n")
    print(f"template: system turn = {drop} tokens dropped, long prompts keep the assistant cue")


def main():
    check_lora_keys(check_dit())
    check_sigmas()
    check_gate_up()
    check_vae_keys()
    check_real_files()
    check_template()
    print("OK")


if __name__ == "__main__":
    main()
