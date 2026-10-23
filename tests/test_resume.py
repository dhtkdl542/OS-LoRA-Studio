"""CPU check of save_state / restore / prune with a tiny model (no DiT, no GPU work).

    venv\\Scripts\\python -m tests.test_resume
"""

import json
import os
import random
import tempfile

import torch
from torch import nn

from oslora.autopilot import Autopilot, AutopilotConfig
from oslora.config import RunConfig
from oslora.data import BucketSampler, Item
from oslora.lora import EMA, LoRANetwork
from oslora.train import RunLog, Trainer, resumable_steps


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].proj = nn.ModuleDict({"qkv": nn.Linear(8, 24, bias=False)})


def make(run_dir):
    tr = Trainer.__new__(Trainer)
    tr.cfg = RunConfig(name="t")
    tr.run_dir = run_dir
    tr.rl = RunLog(run_dir)
    tr.device = torch.device("cpu")
    from oslora.models import get_adapter

    tr.model = get_adapter("moss_image")
    tr.dit = Tiny()
    tr.net = LoRANetwork(tr.dit, [r"layers\.\d+\.proj\.qkv"], rank=4)
    for p in tr.net.parameters():
        p.requires_grad_(True)
    tr.opt = torch.optim.AdamW(tr.net.parameters(), lr=1e-3)
    tr.ema = EMA(tr.net.parameters(), 0.9)
    return tr


def train_steps(tr, n):
    for _ in range(n):
        x = torch.randn(4, 8)
        loss = tr.dit.layers[0].proj["qkv"](x).pow(2).mean()
        loss.backward()
        tr.opt.step()
        tr.opt.zero_grad()
        tr.ema.update()


def main():
    d = tempfile.mkdtemp()
    tr = make(d)
    ap = Autopilot(AutopilotConfig(), 3000, 100)
    ap.set_baseline(0.6, 0.6)
    sampler = BucketSampler([Item(image="x", caption="", bucket=(8, 8), key="k")], 1)
    os.makedirs(os.path.join(d, "samples"), exist_ok=True)

    train_steps(tr, 5)
    with tr.using_ema():
        tr.save_lora(os.path.join(d, "t_00100.safetensors"))
    ap.strikes = 1
    tr.save_state(100, ap, sampler, None, 0, full=True)
    w100_ema = [s.clone() for s in tr.ema.shadow]
    open(os.path.join(d, "samples", "00100_0.jpg"), "w").close()

    train_steps(tr, 5)
    with tr.using_ema():
        tr.save_lora(os.path.join(d, "t_00200.safetensors"))
    ap.strikes = 2
    tr.save_state(200, ap, sampler, tr.lora_weights(), 200, full=True)
    raw200 = tr.lora_weights()
    opt200 = tr.opt.state_dict()["state"][0]["exp_avg"].clone()
    open(os.path.join(d, "samples", "00200_0.jpg"), "w").close()
    tr.rl.metric(type="eval", step=100)
    tr.rl.metric(type="eval", step=200)

    print("이어하기 지점:", resumable_steps(d))

    # exact resume at 200 in a fresh trainer
    tr2 = make(d)
    ap2 = Autopilot(AutopilotConfig(), 4000, 100)
    best_step, best = tr2.restore(200, ap2, sampler, 4000)
    assert all(torch.equal(a, b) for a, b in zip(tr2.lora_weights(), raw200)), "raw weights differ"
    assert torch.equal(tr2.opt.state_dict()["state"][0]["exp_avg"], opt200), "optimizer differs"
    assert ap2.strikes == 2 and ap2.max_steps == 4000 and best_step == 200 and best is not None
    print("정확한 이어하기 OK: 가중치·옵티마이저·Autopilot 동일, 최대 step 4000, 마무리 시작", ap2.anneal_start)

    # checkpoint resume at 100: weights = EMA at 100, later files stashed, metrics trimmed
    tr3 = make(d)
    ap3 = Autopilot(AutopilotConfig(), 3000, 100)
    tr3.restore(100, ap3, sampler, 3000)
    assert all(torch.allclose(p.float(), s.to(torch.bfloat16).float()) for p, s in zip(tr3.net.parameters(), w100_ema))
    assert ap3.strikes == 1 and getattr(ap3, "rewarm", None) == (100, 50)
    print("재워밍업 LR:", [f"{ap3.lr_at(s):.1e}" for s in (100, 125, 150, 151)])
    left = sorted(os.listdir(d))
    assert "t_00200.safetensors" not in left and not os.path.exists(os.path.join(d, "samples", "00200_0.jpg"))
    steps_left = [json.loads(l)["step"] for l in open(os.path.join(d, "metrics.jsonl"), encoding="utf-8")]
    assert steps_left == [100], steps_left
    print("체크포인트 이어하기 OK: step 100 가중치, 이후 파일 이동, 그래프 기록 정리 →", [f for f in left if f.startswith("_")])
    print("이어하기 지점(정리 후):", resumable_steps(d))


if __name__ == "__main__":
    main()
