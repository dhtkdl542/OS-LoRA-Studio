"""Training settings chosen from the dataset check-up, each with a plain-language reason.

recommend(report, kind, trigger, output_dir) -> {"settings": {...}, "reasons": [...], "name": str}

The numbers are starting points calibrated on the user's runs (Damdeari, 88-94 images):
LR 1e-4 (their proven ai-toolkit value) as the ceiling, down to 3e-5, ~3000 steps for ~90
images, rank 32. The Autopilot then adjusts LR and stops on real evidence, so these only need
to be in the right neighbourhood.
"""

import os
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

SEC_PER_STEP = {512: 2.3, 768: 4.5, 1024: 8.8, 1280: 14.5}
# per model: seconds per step by resolution, the largest resolution worth defaulting to on 16 GB,
# and the LoRA name prefix (measured on a 5060 Ti; Krea 2 with 6 blocks swapped)
MODEL_PROFILE = {
    "moss_image": {"sec": SEC_PER_STEP, "max_res": 1024, "prefix": "OS_Moss_", "label": "MOSS"},
    "krea2": {"sec": {512: 4.2, 768: 8.4, 1024: 17.0}, "max_res": 768, "prefix": "OS_Krea2_", "label": "Krea 2"},
    # 7B single stream, int8 DiT 7.3 GB, no block swap: peak 8.0 / 8.7 / 9.9 GB (measured)
    "qwen21": {"sec": {512: 2.6, 768: 5.6, 1024: 10.6}, "max_res": 1024, "prefix": "OS_Qwen_", "label": "Qwen 2.1"},
}
KINDS = {"character": "캐릭터", "style": "화풍", "object": "사물·컨셉"}


def _round(v: float, step: int) -> int:
    return int(round(v / step) * step)


def _common_prefix(captions: List[str]) -> Optional[str]:
    """A token sequence that starts >= 90% of captions (an existing trigger word)."""
    heads = [re.split(r"[,.]", c.strip(), maxsplit=1)[0].strip().lower() for c in captions if c.strip()]
    if len(heads) < 4:
        return None
    word, n = Counter(heads).most_common(1)[0]
    return word if n >= 0.9 * len(heads) and 0 < len(word.split()) <= 4 else None


def _next_name(base: str, output_dir: str) -> str:
    v = 1
    while os.path.isfile(os.path.join(output_dir, f"{base}_v{v}", f"{base}_v{v}.safetensors")) or \
            os.path.isdir(os.path.join(output_dir, f"{base}_v{v}")):
        v += 1
    return f"{base}_v{v}"


def recommend(report: Dict, kind: str = "character", trigger: str = "", output_dir: str = "",
              model: str = "moss_image") -> Dict:
    items = [it for it in report["items"] if "error" not in it]
    n = len(items)
    faces = [it for it in items if "face" in it]
    reasons: List[str] = []
    s: Dict = {}

    # ---- resolution: as large as most images can feed without heavy upscaling
    areas = np.array([it["w"] * it["h"] for it in items]) if items else np.array([0])
    res = 1024
    if (areas >= 0.6 * 1024 * 1024).mean() < 0.7:
        res = 768
    if (areas >= 0.6 * 768 * 768).mean() < 0.7:
        res = 512
    prof = MODEL_PROFILE.get(model, MODEL_PROFILE["moss_image"])
    if res > prof["max_res"]:
        res = prof["max_res"]
        reasons.append(f"{prof['label']}는 모델이 커서 16GB에서 {res}px까지 권장해요 (더 크게 하면 느리고 메모리가 빠듯해요).")
    s["resolution"] = res
    med_mp = float(np.median(areas)) / 1e6
    reasons.append(f"학습 해상도 {res}px: 이미지 중간 크기가 {med_mp:.1f}MP라서 "
                   + ("MOSS 권장(1MP)대로 1024로 학습해요." if res == 1024 else
                      f"1024로 학습하면 대부분 확대돼서 흐려져요. {res}px이 이 데이터에 맞아요."))

    # ---- network size
    if kind == "style":
        rank = 32
        why = "화풍은 색·질감·필치를 넓게 배워야 해서 넉넉하게 잡아요."
    elif n < 20:
        rank = 16
        why = f"이미지가 {n}장으로 적어서, 너무 크면 사진을 통째로 외워요."
    else:
        rank = 32
        why = f"이미지 {n}장이면 정체성과 디테일을 담기에 32가 적당해요 (지금까지 쓰시던 값)."
    s["rank"], s["alpha"] = rank, rank
    reasons.append(f"Rank {rank} / Alpha {rank}: {why}")

    # ---- learning rate (Autopilot range)
    if kind == "style":
        s.update(start_lr=5e-5, min_lr=1e-5, max_lr=8e-5)
        reasons.append("LR 시작 5e-5 (1e-5~8e-5): 화풍은 loss가 꾸준히 내려가서 LR이 높으면 쉽게 과하게 구워져요.")
    else:
        s.update(start_lr=1e-4, min_lr=3e-5, max_lr=1e-4)
        reasons.append("LR 1e-4 (내려갈 때만 3e-5까지): ai-toolkit에서 검증된 1e-4가 상한이에요. 1.5e-4는 담다리 v2에서 "
                       "누출을 키우고 홀드아웃을 멈추게 해서, 올리지 않고 상황에 따라 내리기만 해요. "
                       "Autopilot이 상황 보고 조절해요.")

    # ---- holdout (same rule as data.pick_holdout "auto") and length over the training images
    # a face crop / degraded copy repeats its original's face: count photos, copies at half weight
    n_crops = sum(1 for it in items if it.get("crop") or it.get("aug"))
    units = n - sum(1 for it in items if it.get("source"))  # crop+original pairs count once
    hold = "auto" if units >= 16 else "off"
    n_hold = min(6, max(2, round(units * 0.08))) if hold == "auto" else 0
    n_train = max(1, round((n - n_crops) + 0.5 * n_crops - n_hold))
    from . import learn

    taste = learn.summarize(kind, model)
    per_img = 35 if kind != "style" else 30
    if taste["n"]:  # the user's picks pull the default toward what they actually used
        per_img = round((1 - taste["weight"]) * per_img + taste["weight"] * taste["per_img_cap"])
        per_img = min(max(per_img, 10), 80)
    steps = min(4000, max(1500, _round(n_train * per_img, 100)))
    s["max_steps"] = str(steps)
    counted = f"학습 이미지 {n_train}장" + (f"(얼굴 크롭·화질 변형 사본 {n_crops}장은 원본 얼굴의 반복이라 절반으로 셈)" if n_crops else "")
    reasons.append(f"최대 {steps} step: {counted}을 한 장당 약 {per_img}번 보도록 잡았어요. "
                   "과적합이 확인되면 Autopilot이 브레이크를 걸거나 일찍 끝내고, 가장 좋았던 시점을 골라줘요.")
    if taste["n"]:
        reasons.append(f"취향 반영 ({taste['n']}번의 선택 기록, 반영 비중 {taste['weight']:.0%}): "
                       + " ".join(taste["notes"])
                       + f" 그래서 step 수와 Autopilot의 추천 기준(허용치 {taste['keep_frac']:.2f})을 맞췄어요.")
    else:
        reasons.append("취향 반영: 아직 선택 기록이 없어요. 결과 탭에서 실제로 쓰신 체크포인트에 [이걸 썼어요]를 "
                       "누르면 다음 추천부터 반영돼요.")

    s["holdout"] = hold
    reasons.append("검증 홀드아웃 자동: 몇 장을 떼어 과적합을 감시해요." if hold == "auto" else
                   f"검증 홀드아웃 끄기: {n}장은 떼어낼 여유가 없어서, 대신 품질 확인과 누출 지표로 감시해요.")

    # ---- captions
    s["caption_dropout"] = 0.1 if kind == "style" else 0.05
    caps = []
    for it in items:
        p = os.path.splitext(it["path"])[0] + ".txt"
        if os.path.isfile(p):
            caps.append(open(p, encoding="utf-8").read())
    existing = _common_prefix(caps)
    trig = trigger.strip()
    if trig:
        s["trigger"] = trig
        reasons.append(f"트리거 '{trig}': 캡션에 없으면 앞에 자동으로 붙여 학습해요.")
    elif existing:
        s["trigger"] = ""
        reasons.append(f"캡션 대부분이 '{existing}'(으)로 시작해요. 이미 트리거처럼 쓰이고 있어서 따로 추가하지 않아요.")
    else:
        s["trigger"] = ""
        reasons.append("트리거 없음: 캡션의 'a woman' 같은 표현에 캐릭터가 묶여요 (지금까지 쓰시던 방식). "
                       "프롬프트에서 켜고 끌 단어가 필요하면 트리거 워드를 넣으세요.")

    missing = sum(1 for it in items if not it.get("caption_words"))
    if missing:
        reasons.append(f"⚠ 캡션 없는 이미지가 {missing}장 있어요. 데이터셋 탭 진단에서 먼저 자동 캡션하세요.")

    # ---- samples: keep the user's prompts; orient the sample size to the data
    portrait = sum(1 for it in items if it["h"] > it["w"] * 1.1)
    landscape = sum(1 for it in items if it["w"] > it["h"] * 1.1)
    if portrait >= landscape and portrait >= 0.4 * n:
        s["sample_w"], s["sample_h"] = 640, 864
        orient = "세로"
    elif landscape > portrait and landscape >= 0.4 * n:
        s["sample_w"], s["sample_h"] = 864, 640
        orient = "가로"
    else:
        s["sample_w"], s["sample_h"] = 736, 736
        orient = "정사각"
    s["sample_every"] = 500
    reasons.append(f"샘플 {s['sample_w']}×{s['sample_h']}, 500 step마다: 데이터셋이 주로 {orient}형이라 같은 비율로 그려요. "
                   "진행 확인용이라 작게 그려서 한 장에 1분 정도예요.")

    # ---- the rest
    s.update(autopilot=True, early_stop=True, ema="0.99", optimizer="adamw", kind=kind)
    if kind == "character" and len(faces) < 0.5 * n:
        reasons.append(f"⚠ 얼굴이 보이는 이미지가 {len(faces)}/{n}장이에요. 캐릭터 LoRA라면 얼굴이 잘 보이는 사진을 늘리면 좋아요.")

    eta = steps * prof["sec"].get(res, max(prof["sec"].values())) * 1.12 / 3600
    reasons.append(f"예상 소요: 최대 약 {eta:.1f}시간 (조기 종료되면 더 짧아요).")

    folder = os.path.basename(os.path.normpath(report["folder"]))
    base = f"{prof['prefix']}{re.sub(r'[^0-9A-Za-z가-힣_-]+', '_', folder).strip('_') or 'LoRA'}"
    name = _next_name(base, output_dir) if output_dir else f"{base}_v1"
    s["name"] = name
    reasons.insert(0, f"이름 {name}: 폴더 이름을 따서 붙였어요 (이미 있는 버전은 건너뛰어요).")
    return {"settings": s, "reasons": reasons, "name": name, "kind": kind}
