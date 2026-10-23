"""Our Krea 2 port == diffusers' Krea2Transformer2DModel on a small random model (CPU), with and
without a text padding mask, and with padding removed (the trimmed text cache is exact)."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel  # noqa: E402

from oslora.models.krea2.transformer import Krea2Transformer  # noqa: E402

PREFIX = [("img_in.", "first."), ("time_embed.linear_1.", "tmlp.0."), ("time_embed.linear_2.", "tmlp.2."),
          ("time_mod_proj.", "tproj.1."), ("txt_in.norm.weight", "txtmlp.0.scale"), ("txt_in.linear_1.", "txtmlp.1."),
          ("txt_in.linear_2.", "txtmlp.3."), ("text_fusion.", "txtfusion."), ("transformer_blocks.", "blocks."),
          ("final_layer.linear.", "last.linear."), ("final_layer.norm.weight", "last.norm.scale"),
          ("final_layer.scale_shift_table", "last.modulation.lin")]
INNER = [(".attn.to_q.", ".attn.wq."), (".attn.to_k.", ".attn.wk."), (".attn.to_v.", ".attn.wv."),
         (".attn.to_out.0.", ".attn.wo."), (".attn.to_gate.", ".attn.gate."),
         (".attn.norm_q.weight", ".attn.qknorm.qnorm.scale"), (".attn.norm_k.weight", ".attn.qknorm.knorm.scale"),
         (".ff.", ".mlp."), (".norm1.weight", ".prenorm.scale"), (".norm2.weight", ".postnorm.scale"),
         (".scale_shift_table", ".mod.lin")]


def to_raw(sd):
    out = {}
    for k, v in sd.items():
        n = k
        for a, b in PREFIX:
            if n.startswith(a):
                n = b + n[len(a):]
                break
        for a, b in INNER:
            n = n.replace(a, b)
        if n.startswith("blocks.") and n.endswith(".mod.lin"):
            v = v.reshape(-1)
        out[n] = v
    return out


def main():
    torch.manual_seed(0)
    ref = Krea2Transformer2DModel(in_channels=64, num_layers=2, attention_head_dim=32, num_attention_heads=4,
                                  num_key_value_heads=2, intermediate_size=192, timestep_embed_dim=64,
                                  text_hidden_dim=64, num_text_layers=3, text_num_attention_heads=2,
                                  text_num_key_value_heads=2, text_intermediate_size=96, axes_dims_rope=(8, 12, 12))
    with torch.no_grad():
        for p in ref.parameters():  # zero-initialised norms / tables would hide mistakes
            p.add_(torch.randn_like(p) * 0.1)
    ref.eval()
    sd = to_raw(ref.state_dict())
    cfg = Krea2Transformer.config_from_state_dict(sd)
    cfg.rope_axes = (8, 12, 12)
    ours = Krea2Transformer(cfg)
    missing, unexpected = ours.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    ours = ours.float()
    ours.eval()

    b, gh, gw, L = 1, 4, 6, 7
    lat = torch.randn(b, 64, gh, gw)
    text = torch.randn(b, L, 3, 64)
    t = torch.tensor([0.37])
    mask = torch.tensor([[True] * 5 + [False] * 2])
    packed = lat.permute(0, 2, 3, 1).reshape(b, gh * gw, 64)
    ids = Krea2PipelineIds(L, gh, gw)
    with torch.no_grad():
        r = ref(packed, text, t, ids, encoder_attention_mask=mask, return_dict=False)[0]
        r = r.reshape(b, gh, gw, 64).permute(0, 3, 1, 2)
        o = ours(lat, t, text, mask).float()
        o_trim = ours(lat, t, text[:, :5], torch.ones(b, 5, dtype=torch.bool)).float()
    rel = lambda a, c: float((a - c).norm() / c.norm())  # noqa: E731
    print(f"masked: ours vs diffusers rel err {rel(o, r):.2e}")
    print(f"padding removed vs masked: rel err {rel(o_trim, o):.2e}")
    assert rel(o, r) < 1e-4 and rel(o_trim, o) < 1e-4
    print("OK")


def Krea2PipelineIds(text_len, gh, gw):
    from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline

    return Krea2Pipeline.prepare_position_ids(text_len, gh, gw, torch.device("cpu"))


if __name__ == "__main__":
    main()
