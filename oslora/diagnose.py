"""Dataset check-up and one-click fixes (CPU only, safe to run while training).

analyze(folder) inspects every image:
    resolution, caption, sharpness (variance of Laplacian), perceptual hash (duplicates),
    face (YuNet) with its size in the frame, ArcFace identity similarity to the dataset

and turns that into a short list of findings, each with an optional fix:
    missing captions      -> caption just those (the regular caption job)
    looks like someone else, duplicates, blurry -> move to excluded/<reason>/ (never deleted)
    few face close-ups    -> write FaceCrop_<name>.png crops of the best-likeness faces only
                             (~20% of the dataset, not one per photo)
    too many face crops   -> keep the best ~20% (likeness, detail gained, sharpness), move the
                             rest to excluded/크롭과다/: each crop repeats its original's face

Face crops are paired with their originals (by name, else by the face) so the holdout never
splits a pair and the step count treats a crop as half an image.
"""

import hashlib
import json
import os
import shutil
import time
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps

from .data import AUG_PREFIX, CROP_PREFIX, IMAGE_EXTS, is_derived

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache", "diagnosis")
EXCLUDED = "excluded"
REASON_DIRS = {"look": "다른사람같음", "duplicate": "중복", "blurry": "흐림", "crop": "크롭과다", "aug": "변형과다",
               "aug_rebalance": "변형정리"}
REPORT_VERSION = 3  # 2: face crops paired with their originals, 3: augmented copies


def _images(folder: str) -> List[str]:
    return [os.path.join(folder, n) for n in sorted(os.listdir(folder)) if n.lower().endswith(IMAGE_EXTS)]


def _dhash(img: Image.Image, size: int = 16) -> str:
    """256-bit difference hash as hex. Calibrated on a real dataset: different photos of the
    same character are >= 38 bits apart, a resized/re-saved copy 1, a slight crop 12."""
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):064x}"


def _hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


DUP_BITS = 20


def _sharpness(img: Image.Image) -> float:
    import cv2

    g = np.asarray(img.convert("L"))
    s = 768 / max(g.shape)
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _cache_path(folder: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, hashlib.sha1(os.path.abspath(folder).lower().encode()).hexdigest()[:16] + ".json")


def _signature(paths: List[str]) -> str:
    h = hashlib.sha1()
    for p in paths:
        st = os.stat(p)
        h.update(f"{os.path.basename(p)}|{st.st_size}|{int(st.st_mtime)}".encode())
        cap = os.path.splitext(p)[0] + ".txt"
        if os.path.isfile(cap):
            h.update(f"c{int(os.stat(cap).st_mtime)}".encode())
    return h.hexdigest()


def analyze(folder: str, resolution: int = 1024, progress: Optional[Callable[[int, int, str], None]] = None) -> Dict:
    from .evaluate import FaceID

    paths = _images(folder)
    sig = _signature(paths)
    cp = _cache_path(folder)
    if os.path.isfile(cp):
        try:
            with open(cp, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("signature") == sig and cached.get("resolution") == resolution                     and cached.get("version") == REPORT_VERSION:
                # the per-image analysis is reusable; the findings are cheap, so newer rules apply
                cached["findings"] = _findings(cached["items"], resolution)
                return cached
        except Exception:
            pass

    face = FaceID()
    items, embeds = [], []
    for i, p in enumerate(paths):
        if progress:
            progress(i, len(paths), os.path.basename(p))
        it = {"path": p, "name": os.path.basename(p)}
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                it["w"], it["h"] = im.size
                it["sharp"] = _sharpness(im)
                it["hash"] = _dhash(im)
                f = face.analyze(im)
        except Exception as e:
            it["error"] = str(e)
            items.append(it)
            continue
        cap = os.path.splitext(p)[0] + ".txt"
        text = open(cap, encoding="utf-8").read().strip() if os.path.isfile(cap) else ""
        it["caption_words"] = len(text.split())
        it["crop"] = it["name"].startswith(CROP_PREFIX)
        it["aug"] = it["name"].startswith(AUG_PREFIX)
        if f:
            x, y, w, h = f["box"]
            it["face"] = {"box": [x, y, w, h], "count": f["count"], "ratio": (w * h) / (it["w"] * it["h"])}
            embeds.append((len(items), f["embedding"]))
        items.append(it)

    # identity: robust centroid (drop the farthest 20% once), then similarity per image
    if len(embeds) >= 3:
        e = np.stack([v for _, v in embeds])
        c = e.mean(0)
        c /= np.linalg.norm(c)
        keep = np.argsort(e @ c)[int(len(e) * 0.2):]
        c = e[keep].mean(0)
        c /= np.linalg.norm(c)
        for (idx, v) in embeds:
            items[idx]["face"]["sim"] = float(v @ c)
    _pair_crops(folder, items, {items[idx]["name"]: v for idx, v in embeds})

    report = {"folder": folder, "signature": sig, "resolution": resolution, "items": items,
              "analyzed": time.time(), "version": REPORT_VERSION}
    report["findings"] = _findings(items, resolution)
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)
    return report


def _pair_crops(folder: str, items: List[Dict], embeds: Dict[str, np.ndarray]):
    """Find each face crop's original: by name, else by the face (a crop and its original
    share the exact face, ArcFace ~0.9+, clearly above any other photo of the character)."""
    from .data import crop_source, save_crop_pairs

    names = [it["name"] for it in items if "error" not in it]
    originals = [n for n in names if not is_derived(n) and n in embeds]
    pairs = {}
    for it in items:
        if not (it.get("crop") or it.get("aug")):
            continue
        src = crop_source(it["name"], names)
        if src is None and it["name"] in embeds and len(originals) >= 2:
            sims = sorted(((float(embeds[o] @ embeds[it["name"]]), o) for o in originals), reverse=True)
            if sims[0][0] >= 0.85 and sims[0][0] - sims[1][0] >= 0.05:
                src = sims[0][1]
        if src:
            it["source"] = src
            pairs[it["name"]] = src
    save_crop_pairs(folder, pairs)


def _crop_target(n_originals: int) -> int:
    """How many face crops a dataset of this size should carry: ~20% of the originals.
    One crop per photo shows every face twice and the face sets before anything else."""
    return max(2, min(12, round(n_originals * 0.2)))


def _rank_crops(crops: List[Dict], by_name: Dict[str, Dict]) -> List[Dict]:
    """Best crops first: most like the character, then the most detail gained (the face was
    small in the original), then sharpest."""
    sharp_rank = {id(it): r / max(1, len(crops) - 1)
                  for r, it in enumerate(sorted(crops, key=lambda it: it.get("sharp", 0)))}

    def score(it):
        src = by_name.get(it.get("source", ""))
        like = it.get("face", {}).get("sim")
        if like is None:  # the detector often misses tight crops: use the original's face
            like = src["face"]["sim"] - 0.02 if src and src.get("face", {}).get("sim") is not None else 0.0
        ratio = src["face"]["ratio"] if src and "face" in src else 0.08
        gain = min(max((0.15 - ratio) / 0.15, 0.0), 1.0)
        return like + 0.08 * gain + 0.03 * sharp_rank[id(it)]

    return sorted(crops, key=score, reverse=True)


def _aug_target(n_originals: int) -> int:
    """Degraded copies for a dataset of this size: ~15% of the originals (2-10)."""
    return max(2, min(10, round(n_originals * 0.15)))


def _findings(items: List[Dict], resolution: int) -> List[Dict]:
    # augmented copies are blurry / noisy / look-degraded on purpose: kept out of those checks
    augs = [it for it in items if "error" not in it and it.get("aug")]
    ok = [it for it in items if "error" not in it and not it.get("aug")]
    out = []

    def add(fid, level, title, detail, files, fix=None, fix_label=None):
        out.append({"id": fid, "level": level, "title": title, "detail": detail,
                    "files": [os.path.basename(f) for f in files], "fix": fix, "fix_label": fix_label})

    bad = [it["path"] for it in items if "error" in it]
    if bad:
        add("broken", "bad", f"열 수 없는 이미지 {len(bad)}장", "파일이 깨졌거나 이미지가 아니에요. 확인 후 지워주세요.", bad)

    missing = [it["path"] for it in ok if not it.get("caption_words")]
    if missing:
        add("caption", "bad", f"캡션 없는 이미지 {len(missing)}장",
            "캡션이 없으면 학습에 트리거 워드만 쓰여서 옷·배경까지 캐릭터로 외워버려요.", missing,
            "caption", "이 이미지들만 자동 캡션")

    sims = [it["face"]["sim"] for it in ok if it.get("face", {}).get("sim") is not None]
    if len(sims) >= 6:
        # ArcFace: the same person across photos is ~0.5-0.85, different people < ~0.35.
        # Clear mismatches (< 0.45 and a statistical outlier) get a fix; milder outliers
        # of a very consistent dataset are only listed for a look.
        q1, med, q3 = np.percentile(sims, [25, 50, 75])
        stat_cut = max(med - 1.5 * (q3 - q1), 0.25)
        hard_cut = min(stat_cut, 0.45)
        odd = [it["path"] for it in ok if it.get("face", {}).get("sim", 1) < hard_cut]
        mild = [it["path"] for it in ok if hard_cut <= it.get("face", {}).get("sim", 1) < stat_cut]
        if odd:
            add("look", "warn", f"다른 사람처럼 보이는 이미지 {len(odd)}장",
                f"얼굴 인식 닮음이 {hard_cut:.2f} 미만이에요 (데이터셋 중간값 {med:.2f}). 다른 사람이거나 얼굴이 "
                "크게 달라 보여서 정체성이 흐려질 수 있어요.", odd, "exclude:look", "excluded 폴더로 빼기")
        if mild:
            add("look_mild", "info", f"조금 덜 닮은 이미지 {len(mild)}장",
                f"같은 사람으로 보이지만 나머지보다 닮음이 낮아요 ({hard_cut:.2f}~{stat_cut:.2f}). 각도·표정·화장 "
                "차이일 수 있으니 눈으로 확인만 해보세요.", mild)

    groups, seen = [], set()
    for i, a in enumerate(ok):
        if i in seen:
            continue
        grp = [i]
        for j in range(i + 1, len(ok)):
            if j not in seen and _hamming(a["hash"], ok[j]["hash"]) <= DUP_BITS:
                grp.append(j)
                seen.add(j)
        if len(grp) > 1:
            groups.append(grp)
    if groups:
        extra = []
        for g in groups:
            g = sorted(g, key=lambda k: (-(ok[k]["w"] * ok[k]["h"]), -ok[k]["sharp"]))  # keep the largest, then sharpest
            extra += [ok[k]["path"] for k in g[1:]]
        add("duplicate", "warn", f"거의 같은 이미지 {len(extra)}장 ({len(groups)}묶음)",
            "같은 사진이 여러 장이면 그 사진만 과하게 외워요. 묶음마다 가장 큰 한 장만 남겨요.", extra,
            "exclude:duplicate", "중복만 빼기")

    sharp = [it["sharp"] for it in ok]
    if len(sharp) >= 6:
        med = float(np.median(sharp))
        blurry = [it["path"] for it in ok if it["sharp"] < med * 0.25]
        if blurry:
            add("blurry", "warn", f"흐린 이미지 {len(blurry)}장",
                "다른 이미지보다 많이 흐려요. 흐림까지 캐릭터 특징으로 배울 수 있어요.", blurry,
                "exclude:blurry", "excluded 폴더로 빼기")

    small = [it["path"] for it in ok if it["w"] * it["h"] < (resolution * resolution) * 0.5 and not it.get("crop")]
    if small:
        add("small", "info", f"학습 해상도({resolution}px)보다 많이 작은 이미지 {len(small)}장",
            "확대돼서 학습돼요. 결과가 조금 흐려질 수 있어요. 가능하면 더 큰 원본으로 바꿔주세요.", small)

    # face crops fill the frame and the detector often misses them: not a problem
    noface = [it["path"] for it in ok if "face" not in it and not it.get("crop")]
    if ok and noface and len(noface) < len(ok):
        add("noface", "info", f"얼굴이 안 보이는 이미지 {len(noface)}장",
            "뒷모습·소품 사진이면 괜찮아요. 캐릭터 LoRA라면 너무 많지 않은지만 확인하세요.", noface)

    originals = [it for it in ok if not it.get("crop")]
    crops = [it for it in ok if it.get("crop")]
    target = _crop_target(len(originals))
    by_name = {it["name"]: it for it in ok}
    if len(crops) > max(3, target * 1.5):
        ranked = _rank_crops(crops, by_name)
        extra = [it["path"] for it in ranked[target:]]
        add("crop_many", "warn", f"얼굴 크롭이 너무 많아요 ({len(crops)}장, 원본 {len(originals)}장)",
            "원본과 크롭으로 같은 얼굴을 두 번씩 보게 돼서 얼굴만 먼저 굳어요. 프롬프트의 흔들림·화장·표정이 "
            f"얼굴에만 잘 안 먹는 과적합의 원인이 돼요. 데이터셋 크기에 맞춰 닮음이 가장 좋은 {target}장만 남기고 "
            f"나머지 {len(extra)}장은 excluded/크롭과다 폴더로 옮겨요 (지우지 않아요).", extra,
            "exclude:crop", f"닮은 크롭 {target}장만 남기기")

    # face close-ups: offered whenever the dataset carries fewer crops than its size calls for
    # (urgent when most faces are small, optional otherwise), best-likeness faces only
    faces = [it for it in originals if "face" in it]
    closeups = [it for it in ok if it.get("face", {}).get("ratio", 0) >= 0.06 or it.get("crop")]
    has_crop = {it.get("source") for it in crops} | {it.get("source") for it in augs}

    def zooms_in(it):
        return _face_crop_box(it) is not None

    croppable = [it for it in faces if it["name"] not in has_crop and zooms_in(it)
                 and not os.path.isfile(_crop_path(it["path"]))]
    want = target - len(crops)
    if croppable and want > 0:
        def gain(it):  # how much a crop enlarges the face
            return min(max((0.15 - it["face"]["ratio"]) / 0.15, 0.0), 1.0)

        best = sorted(croppable, key=lambda it: it["face"].get("sim", 0) + 0.08 * gain(it), reverse=True)[:want]
        files = [it["path"] for it in best]
        why = (f"데이터셋 {len(originals)}장에는 얼굴 클로즈업 {target}장 정도가 알맞아요 (지금 크롭 {len(crops)}장). "
               f"닮음이 가장 좋은 {len(best)}장의 얼굴 부분을 잘라 FaceCrop_ 사본을 만들어요 (원본은 그대로). "
               "사진마다 만들지 않는 건, 같은 얼굴을 두 번씩 보면 얼굴만 먼저 굳어서예요.")
        if faces and len(closeups) < 0.3 * len(ok):
            add("closeup", "warn", f"얼굴 클로즈업이 적어요 ({len(closeups)}/{len(ok)}장)",
                "얼굴이 작게 나온 사진이 대부분이면 이목구비를 잘 못 배워요. " + why,
                files, "facecrop", f"얼굴 클로즈업 {len(best)}장 만들기")
        else:
            add("closeup", "info", f"얼굴 클로즈업 {len(best)}장을 더할 수 있어요",
                "얼굴이 이미 잘 보이는 데이터라 꼭 필요하진 않지만, 눈·입 같은 디테일을 더 또렷하게 배워요. " + why,
                files, "facecrop", f"얼굴 클로즈업 {len(best)}장 만들기")

    # degraded copies: a clean dataset makes the LoRA scrub grain / blur / phone looks from
    # prompts; a few copies captioned with their degradation tie that look to words instead
    aug_target = _aug_target(len(originals))
    if len(augs) > max(3, aug_target * 1.5):
        extra = sorted(augs, key=lambda it: by_name.get(it.get("source", ""), {}).get("face", {}).get("sim", 0),
                       reverse=True)[aug_target:]
        add("aug_many", "warn", f"화질 변형 사본이 너무 많아요 ({len(augs)}장, 원본 {len(originals)}장)",
            f"변형 사본도 원본 얼굴의 반복이라 많으면 그 얼굴만 굳어요. {aug_target}장만 남기고 나머지 "
            f"{len(extra)}장은 excluded/변형과다 폴더로 옮겨요.", [it["path"] for it in extra],
            "exclude:aug", f"변형 사본 {aug_target}장만 남기기")
    want = aug_target - len(augs)
    used = has_crop | {it.get("source") for it in augs}
    candidates = [it for it in originals if "face" in it and it["name"] not in used and it.get("caption_words")]
    from .augment import BLUR_KINDS, KIND_LABELS, coverage, plan_rebalance

    have, _ = coverage([it["name"] for it in augs])
    if augs and len(augs) <= max(3, aug_target * 1.5):
        # the copies should spread over many looks, not pile up on one (duplicates) or on the
        # softening ones; effects added later count as missing. Same copy count, rearranged.
        retire, new_kinds = plan_rebalance([it["name"] for it in augs], aug_target, int(max(3, aug_target * 1.5)))
        new_kinds = new_kinds[:len(candidates)]
        if retire or new_kinds:
            best = sorted(candidates, key=lambda it: it["face"].get("sim", 0), reverse=True)[:len(new_kinds)]
            blur = sum(c for k, c in have.items() if k in BLUR_KINDS)
            done = ", ".join(f"{KIND_LABELS[k]} {c}" for k, c in have.items() if c)
            title = (f"변형 효과가 흐림 쪽으로 쏠려 있어요 (흐림 계열 {blur}/{len(augs)}장)"
                     if blur > max(1, int(0.35 * len(augs))) else "변형 효과를 더 다양하게 바꿀 수 있어요")
            why = (f"지금 있는 변형 사본: {done}장. 사본은 여러 가지 모습을 한 장씩 보여줘야 효과가 있어요. 같은 효과가 "
                   "여러 장이거나 흐림 계열(흔들림·저해상도·초점 흐림·셀카)이 많으면, 캐릭터가 그 모습으로 끌려가서 잔머리·피부 "
                   "경계가 무뎌질 수 있어요. ")
            if retire:
                why += f"중복되거나 넘치는 사본 {len(retire)}장은 excluded/변형정리 폴더로 옮기고(지우지 않아요), "
            if best:
                why += (f"닮음이 좋은 원본 {len(best)}장으로 {', '.join(KIND_LABELS[k] for k in new_kinds[:len(best)])} "
                        "사본을 새로 만들어요. ")
            why += ("흐림 계열은 전체의 3분의 1 정도만 남겨요. 남는 사본의 캡션도 원본에서 다시 만들면서, 효과와 "
                    "어긋나는 원본의 화질 문장(선명함·노이즈 없음·보정된 피부 등)은 빼요.")
            add("augment", "warn" if blur > max(1, int(0.35 * len(augs))) else "info", title, why,
                [it["path"] for it in augs if it["name"] in retire] + [it["path"] for it in best],
                "aug_rebalance", "변형 효과 다시 맞추기")
            want = 0
    if want > 0 and candidates:
        best = sorted(candidates, key=lambda it: it["face"].get("sim", 0), reverse=True)[:want]
        add("augment", "info", f"화질 변형 사본 {len(best)}장을 더할 수 있어요",
            "데이터가 전부 깨끗하면 LoRA가 '이 사람은 늘 깨끗하다'까지 배워서, 노이즈·흔들림·저화질을 요청해도 "
            f"지워버려요. 크롭하지 않은 사진 중 닮음이 가장 좋은 {len(best)}장에 효과를 하나씩 걸어 Aug_ 사본을 만들어요. "
            "효과는 노이즈·플래시·폰 HDR·필름·빛샘·강한 대비·과다 노출 같은 또렷한 것과 흔들림·저해상도·초점 흐림·셀카 같은 "
            "흐린 것을 섞되, 흐린 것은 3분의 1 정도만 써요. 두 장에 한 장은 흑백·따뜻한 톤·차가운 톤·빛바램·쨍한 필터도 함께 "
            "걸고, 캡션 맨 앞에 어떤 효과인지 적어요 (원본은 그대로).",
            [it["path"] for it in best], "augment", f"화질 변형 {len(best)}장 만들기")
    elif want > 0 and any("face" in it and it["name"] not in used for it in originals):
        # the copies reuse the original's caption (effect sentences + caption): nothing to offer yet
        add("augment", "info", "캡션을 만들면 화질 변형 사본을 더할 수 있어요",
            f"변형 사본은 원본 캡션 앞에 효과를 적어서 만들어요. 캡션이 있는 원본이 없어서 아직 만들 수 없어요. "
            f"자동 캡션이 끝나면 진단이 다시 돌면서 변형 사본 {want}장 정도를 제안해요.", [])

    if not out:
        add("ok", "ok", "문제 없어 보여요", f"이미지 {len(ok)}장 모두 확인했어요.", [])
    return out


def _crop_path(path: str) -> str:
    d, n = os.path.split(path)
    return os.path.join(d, "FaceCrop_" + os.path.splitext(n)[0] + ".png")


def _face_crop_box(it: Dict):
    """A crop worth making: hair-to-neck framing (padding 0.9) if that actually zooms in
    (box <= 80% of the short side), else a tighter face framing (0.6); None if neither does."""
    for padding in (0.9, 0.6):
        box = _crop_box(it, padding)
        if box is not None and (box[2] - box[0]) <= 0.8 * min(it["w"], it["h"]):
            return box
    return None


def _crop_box(it: Dict, padding: float = 0.9, min_side: int = 512):
    """Square box around the face (+padding for hair/neck), inside the image; None if the
    crop would be too small to be useful."""
    x, y, w, h = it["face"]["box"]
    side = max(w, h) * (1 + 2 * padding)
    side = min(side, it["w"], it["h"])
    if side < min_side:
        return None
    cx, cy = x + w / 2, y + h / 2 + h * 0.1  # a little below centre: include chin/neck
    left = min(max(cx - side / 2, 0), it["w"] - side)
    top = min(max(cy - side / 2, 0), it["h"] - side)
    return int(left), int(top), int(left + side), int(top + side)


def _move_with_captions(path: str, dst_dir: str):
    """Move an image with its caption files (.txt and the 3-length .captions.json)."""
    stem = os.path.splitext(path)[0]
    for src in (path, stem + ".txt", stem + ".captions.json"):
        if os.path.isfile(src):
            shutil.move(src, os.path.join(dst_dir, os.path.basename(src)))


def fix(folder: str, action: str, files: List[str]) -> Dict:
    """Apply a finding's fix to the given file names (relative to ``folder``)."""
    paths = [os.path.join(folder, f) for f in files if os.path.isfile(os.path.join(folder, f))]
    if action.startswith("exclude:"):
        reason = REASON_DIRS.get(action.split(":", 1)[1], "기타")
        dst_dir = os.path.join(folder, EXCLUDED, reason)
        os.makedirs(dst_dir, exist_ok=True)
        moved = 0
        for p in paths:
            _move_with_captions(p, dst_dir)
            moved += 1
        return {"done": moved, "message": f"{moved}장을 '{EXCLUDED}/{reason}' 폴더로 옮겼어요 (되돌리려면 다시 꺼내면 돼요)"}
    if action == "aug_rebalance":
        from .augment import make_copies, refresh_captions

        retire = [p for p in paths if os.path.basename(p).startswith(AUG_PREFIX)]
        dst_dir = os.path.join(folder, EXCLUDED, REASON_DIRS["aug_rebalance"])
        os.makedirs(dst_dir, exist_ok=True)
        for p in retire:
            _move_with_captions(p, dst_dir)
        made = make_copies([p for p in paths if p not in retire])
        refresh_captions(folder)  # the kept copies get captions without the original's finish statements
        return {"done": len(retire) + made,
                "message": f"변형 사본 {len(retire)}장을 '{EXCLUDED}/{REASON_DIRS['aug_rebalance']}'로 옮기고 새 효과 {made}장을 만들었어요"}
    if action == "facecrop":
        report = analyze(folder)
        by_name = {it["name"]: it for it in report["items"]}
        made = 0
        for p in paths:
            it = by_name.get(os.path.basename(p))
            if not it or "face" not in it:
                continue
            box = _face_crop_box(it)
            out = _crop_path(p)
            if box is None or os.path.isfile(out):
                continue
            with Image.open(p) as im:
                crop = ImageOps.exif_transpose(im).convert("RGB").crop(box)
                if crop.width > 1024:
                    crop = crop.resize((1024, 1024), Image.LANCZOS)
                crop.save(out)
            made += 1
        return {"done": made, "message": f"얼굴 클로즈업 {made}장을 만들었어요. 캡션이 없으니 '캡션 없는 이미지'를 자동 캡션하세요"}
    if action == "augment":
        from .augment import make_copies

        made = make_copies(paths)
        return {"done": made, "message": f"화질 변형 사본 {made}장을 만들었어요 (캡션 맨 앞에 어떤 변형인지 적었어요)"}
    raise ValueError(f"알 수 없는 수정: {action}")
