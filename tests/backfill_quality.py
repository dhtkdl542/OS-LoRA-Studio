"""Score an existing run's samples with the quality judges and append them to metrics.jsonl.

    venv\\Scripts\\python -m tests.backfill_quality <run_dir>
"""

import json
import os
import sys
from collections import defaultdict

import yaml
from PIL import Image

from oslora.autopilot import Autopilot, AutopilotConfig
from oslora.data import pick_holdout, scan_folder
from oslora.evaluate import Evaluator


def main():
    rd = sys.argv[1]
    cfg = yaml.safe_load(open(os.path.join(rd, "config.yaml"), encoding="utf-8"))
    pairs = scan_folder(cfg["dataset"]["folder"])
    hold = set(pick_holdout(len(pairs), cfg["dataset"].get("holdout", "auto"), cfg["train"].get("seed", 42)))
    ev = Evaluator([p for i, (p, _) in enumerate(pairs) if i not in hold])
    ap = Autopilot(AutopilotConfig(), 1500, 47)
    ap.like_target = 0.75 * ev.face_self if ev.face_self else None
    prompts = cfg["sample"]["prompts"]
    by_step = defaultdict(dict)
    for n in os.listdir(os.path.join(rd, "samples")):
        if n[0].isdigit():
            step, idx = n[:-4].split("_")
            by_step[int(step)][int(idx)] = os.path.join(rd, "samples", n)
    mp = os.path.join(rd, "metrics.jsonl")
    lines = [l for l in open(mp, encoding="utf-8") if '"quality"' not in l]
    for step in sorted(by_step):
        imgs = [Image.open(by_step[step][i]) for i in range(len(prompts))]
        q = ev.score(imgs, prompts)
        note = ap.quality(step, q)
        lines.append(json.dumps({"type": "quality", "step": step, "final": False, "note": note, **q}, ensure_ascii=False) + "\n")
        print(step, q, note)
    open(mp, "w", encoding="utf-8").writelines(lines)


if __name__ == "__main__":
    main()
