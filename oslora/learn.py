"""Learn the user's taste from the checkpoints they actually pick.

Every "use this one" / "make this the recommended file" appends a record to
cache/picks.jsonl: the picked step, the step the Autopilot recommended, the number of
training images and (from the comparison table) the scores of the pick and of the
highest-likeness column. From those:

    lateness    median log2(picked / recommended): > 0 means the user prefers later
                (more trained) checkpoints than the Autopilot picks
    per_img     median picked_step / training images: how many looks per image the user's
                favourite LoRAs got (drives the automatic step count)
    keep_frac   the Autopilot's tolerance for val giving back gains before it stops
                recommending later checkpoints (scaled with lateness)

Every learned value is blended with the default by the amount of evidence
(w = n / (n + 3)), so one or two picks only nudge it.
"""

import json
import math
import os
import statistics
import time
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICKS = os.path.join(ROOT, "cache", "picks.jsonl")

DEFAULT_PER_IMG = 35.0
DEFAULT_KEEP_FRAC = 0.25


def record(entry: Dict):
    os.makedirs(os.path.dirname(PICKS), exist_ok=True)
    entry = dict(entry, time=time.time())
    with open(PICKS, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load(kind: Optional[str] = None, model: Optional[str] = None) -> List[Dict]:
    out = []
    if os.path.isfile(PICKS):
        with open(PICKS, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("picked") and (kind is None or r.get("kind", "character") == kind) \
                        and (model is None or r.get("model", "moss_image") == model):
                    out.append(r)
    # one vote per run: the latest pick wins
    latest = {}
    for r in out:
        latest[r.get("run")] = r
    return list(latest.values())


def summarize(kind: str = "character", model: str = "moss_image") -> Dict:
    """Taste for one model and kind: a model's picks never steer another model's defaults."""
    picks = load(kind, model)
    n = len(picks)
    w = n / (n + 3.0)
    lat = [math.log2(p["picked"] / p["autopilot_best"]) for p in picks
           if p.get("autopilot_best") and p["picked"] > 0]
    lateness = statistics.median(lat) if lat else 0.0
    per = [p["picked"] / p["n_train"] for p in picks if p.get("n_train")]
    per_img_learned = statistics.median(per) if per else DEFAULT_PER_IMG
    # the run must go somewhat past the favourite point so the Autopilot can see it was the
    # best (and anneal into it): 30% headroom over where the picks actually were
    per_img_cap = per_img_learned * 1.3
    per_img = (1 - w) * DEFAULT_PER_IMG + w * min(max(per_img_cap, 10.0), 80.0) if per else DEFAULT_PER_IMG
    keep = DEFAULT_KEEP_FRAC * (2 ** max(min(lateness, 1.0), -1.0)) if lat else DEFAULT_KEEP_FRAC
    keep_frac = (1 - w) * DEFAULT_KEEP_FRAC + w * min(max(keep, 0.1), 0.6)
    like_first = [p for p in picks if p.get("picked_likeness") is not None and p.get("top_likeness") is not None]
    likes_top = sum(1 for p in like_first if p["picked_likeness"] >= p["top_likeness"] - 1e-6)

    notes = []
    if n:
        if lat and abs(lateness) >= 0.15:
            way = "늦은(더 학습된)" if lateness > 0 else "이른(덜 학습된)"
            notes.append(f"지난 {n}번 고르신 걸 보면 Autopilot 추천보다 {way} 체크포인트를 좋아하셨어요 "
                         f"(평균 {2 ** lateness:.1f}배 지점).")
        elif lat:
            notes.append(f"지난 {n}번은 대체로 Autopilot 추천 근처를 고르셨어요.")
        if per:
            notes.append(f"고르신 체크포인트는 이미지 한 장당 평균 {per_img_learned:.0f}번 학습된 것이었어요.")
        if like_first:
            notes.append(f"비교표에서 {likes_top}/{len(like_first)}번은 닮음 점수가 가장 높은 쪽을 고르셨어요.")
    return {"n": n, "weight": w, "lateness": lateness, "per_img": per_img, "per_img_learned": per_img_learned,
            "per_img_cap": per_img_cap if per else DEFAULT_PER_IMG,
            "keep_frac": keep_frac, "notes": notes}
