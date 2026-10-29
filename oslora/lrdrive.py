"""Epoch-level adaptive LR, ported from Fizgig (Apache-2.0: fizgig/krea2/trainer.py ``AdaptiveLR``,
itself a port of Klein's adaptive_lr), used while no sensor is validated (docs/PRINCIPLES.md).

The LR starts at the geometric mean of [min_lr, max_lr] and only moves at epoch ends, on the epoch's
mean training loss:
  * two epochs in a row with a new best mean   -> x1.25 (up to max_lr)
  * no new best for 1-2 epochs (patience 2 at the 2nd epoch and from the 5th, 1 at the 3rd-4th,
    2 once a stability event happened)        -> x0.5 (down to min_lr)
  * the LoRA weight norm grew > 30 % in one epoch (stability) -> x0.5, the weights blended 70/30 back
    toward the previous epoch's and the optimizer state restored
No warm-up, no anneal, no early stop. The Fizgig LoRA trained this way (Damdeari v3, 1e-4..2e-4)
looked more like the person than ours with the Autopilot's brakes (2026-10-28/29 blind picks).
"""

import math
from typing import Dict, List, Optional, Tuple

import torch


class EpochAdaptiveLR:
    BLEND = 0.7
    WEIGHT_GROWTH = 0.30

    def __init__(self, min_lr: float, max_lr: float):
        self.min_lr, self.max_lr = float(min_lr), float(max_lr)
        self.lr = math.sqrt(self.min_lr * self.max_lr)
        self.best_loss: Optional[float] = None
        self.good = self.bad = self.unstable = 0
        self.unstable_seen = False
        self.prev_norm: Optional[float] = None
        self.snapshot = None  # (weights, optimizer state) on the CPU, in memory only (as in Fizgig)

    # ------------------------------------------------------------------ state (resume)
    def state_dict(self) -> Dict:
        return {"lr": self.lr, "best_loss": self.best_loss, "good": self.good, "bad": self.bad,
                "unstable": self.unstable, "unstable_seen": self.unstable_seen, "prev_norm": self.prev_norm}

    def load_state_dict(self, d: Optional[Dict]):
        for k, v in (d or {}).items():
            setattr(self, k, v)

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def _norm(params: List[torch.Tensor]) -> float:
        with torch.no_grad():
            return math.sqrt(sum(float(p.detach().float().norm()) ** 2 for p in params))

    def _take_snapshot(self, params, opt):
        def cpu(o):
            if torch.is_tensor(o):
                return o.detach().cpu().clone()
            if isinstance(o, dict):
                return {k: cpu(v) for k, v in o.items()}
            if isinstance(o, list):
                return [cpu(v) for v in o]
            return o
        self.snapshot = ([p.detach().cpu().clone() for p in params], cpu(opt.state_dict()))

    def _blend_back(self, params, opt):
        weights, state = self.snapshot
        with torch.no_grad():
            for p, prev in zip(params, weights):
                p.copy_(self.BLEND * prev.to(p.device, p.dtype) + (1 - self.BLEND) * p)
        try:
            opt.load_state_dict(state)
        except Exception:
            pass

    # ------------------------------------------------------------------ the rule
    def epoch_end(self, epoch: int, loss: float, params: List[torch.Tensor], opt) -> Tuple[str, str]:
        """``epoch`` is 0-based. Returns (action, reason); ``self.lr`` holds the LR to use next."""
        norm = self._norm(params)
        if epoch == 0 or self.best_loss is None:  # the first epoch only arms the baseline
            self.best_loss, self.prev_norm = loss, norm
            self._take_snapshot(params, opt)
            return "ARMED", "첫 epoch 기준"
        patience_up = 2
        patience_down = 2 if (self.unstable_seen or epoch == 1 or epoch >= 4) else 1
        growth = (norm - self.prev_norm) / self.prev_norm if self.prev_norm else None
        action, reason = "HOLD", ""
        if growth is not None and growth > self.WEIGHT_GROWTH:
            self.unstable += 1
            if self.unstable >= (2 if self.unstable_seen else 1):
                if self.snapshot is not None:
                    self._blend_back(params, opt)
                new = max(self.lr * 0.5, self.min_lr)
                action = "REDUCE+ROLLBACK" if new < self.lr else "HOLD (floored)"
                self.lr = new
                reason = f"가중치 크기 +{growth:.0%} (> {self.WEIGHT_GROWTH:.0%}): 이전 epoch 쪽으로 70% 되돌림"
                self.good = self.bad = self.unstable = 0
                self.unstable_seen = True
            else:
                action, reason = "WAIT", f"가중치 크기 +{growth:.0%}, 한 번 더 보면 내림"
        elif loss < self.best_loss:
            self.unstable, self.best_loss, self.bad = 0, loss, 0
            self.good += 1
            if self.good >= patience_up:
                new = min(self.lr * 1.25, self.max_lr)
                action = "UP" if new > self.lr else "HOLD (capped)"
                reason = f"loss 최저 {self.good}회 연속"
                self.lr, self.good = new, 0
            else:
                reason = f"loss 최저 ({self.good}/{patience_up})"
        else:
            self.unstable, self.good = 0, 0
            self.bad += 1
            if self.bad >= patience_down:
                new = max(self.lr * 0.5, self.min_lr)
                action = "DOWN" if new < self.lr else "HOLD (floored)"
                reason = f"loss 정체 {self.bad}회"
                self.lr, self.bad = new, 0
            else:
                reason = f"loss 정체 ({self.bad}/{patience_down})"
        self.prev_norm = norm  # as in Fizgig: the norm before any blend-back
        self._take_snapshot(params, opt)
        return action, reason
