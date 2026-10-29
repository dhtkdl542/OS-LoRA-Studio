"""One-click dataset cleanup (docs/PRINCIPLES.md, 1): the check-up's fixes in a fixed order, with one undo.

    1 shrink very large originals (first, so everything after opens small files)
    2 analyse once (faces, likeness, duplicates, blur)
    3 take out broken files, someone else, duplicates, blurry ones
    4 face crops: too many -> keep the best, loose -> cut again, too few -> cut from the best faces
    5 captions for the images without one (the regular caption job; skipped while the GPU is busy)
    6 degraded copies: too many -> trim, one-sided -> rearrange, too few -> make
    7 analyse again and say in one line what is left

Never taken out automatically: "a little less alike" (mostly angle and light), "no face", "small".
Nothing is deleted: files go to excluded/<reason>/ as with the single fixes. Every move, every new file
and the captions as they were are recorded in excluded/_정리기록/<time>.json; undo() puts the folder back
(and parks the files the cleanup made in excluded/_정리기록/<time>_되돌림/).

    python -m oslora.cleanup <folder>           (CPU steps only, no captions)
    python -m oslora.cleanup <folder> --undo
"""

import argparse
import json
import os
import shutil
import time
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from . import diagnose as dg
from .data import IMAGE_EXTS

STEPS = ("큰 이미지 줄이기", "분석", "문제 이미지 빼기", "얼굴 크롭 정리", "캡션", "화질 변형 정리", "다시 분석")
TAKE_OUT = ("broken", "look", "duplicate", "blurry")  # finding ids, all fixed by moving to excluded/


def _top_files(folder: str) -> List[str]:
    return sorted(n for n in os.listdir(folder) if os.path.isfile(os.path.join(folder, n)))


def _park(path: str, bin_dir: str):
    """Move a file out of the way without overwriting anything there."""
    os.makedirs(bin_dir, exist_ok=True)
    name = os.path.basename(path)
    shutil.move(path, os.path.join(bin_dir, dg._with_suffix(name, dg._free_suffix(bin_dir, [name]))))


class Journal:
    """What one cleanup changed in a folder, enough to put it back: moves (src, dst), files it
    made, and a copy of every caption as it was before. Saved after each move, so a cleanup
    that crashed half-way can still be undone."""

    def __init__(self, folder: str, stamp: Optional[str] = None):
        self.folder = folder
        self.stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
        self.moves: List[Tuple[str, str]] = []  # relative to the folder
        self.before: List[str] = []
        self.created: Optional[List[str]] = None  # set when the cleanup finishes
        self.undone = False

    @staticmethod
    def dir_of(folder: str) -> str:
        return os.path.join(folder, dg.EXCLUDED, dg.JOURNAL_DIR)

    @property
    def path(self) -> str:
        return os.path.join(self.dir_of(self.folder), self.stamp + ".json")

    @property
    def caption_dir(self) -> str:
        return os.path.join(self.dir_of(self.folder), self.stamp + "_캡션")

    @property
    def bin_dir(self) -> str:
        return os.path.join(self.dir_of(self.folder), self.stamp + "_되돌림")

    def begin(self):
        self.before = _top_files(self.folder)
        os.makedirs(self.caption_dir, exist_ok=True)
        for n in self.before:
            if n.endswith(dg.CAPTION_EXTS):
                shutil.copy2(os.path.join(self.folder, n), os.path.join(self.caption_dir, n))
        self.save()

    def moved(self, src: str, dst: str):
        self.moves.append((os.path.relpath(src, self.folder), os.path.relpath(dst, self.folder)))
        self.save()

    def finish(self):
        self.created = sorted(set(_top_files(self.folder)) - set(self.before))
        if self.moves or self.created or self._captions_changed():
            self.save()
        else:  # nothing to undo: drop this record (our own copies) so an earlier cleanup stays undoable
            shutil.rmtree(self.caption_dir, ignore_errors=True)
            if os.path.isfile(self.path):
                os.remove(self.path)

    def _captions_changed(self) -> bool:
        for n in os.listdir(self.caption_dir) if os.path.isdir(self.caption_dir) else []:
            cur = os.path.join(self.folder, n)
            if not os.path.isfile(cur):
                return True
            with open(os.path.join(self.caption_dir, n), "rb") as fb, open(cur, "rb") as fc:
                if fb.read() != fc.read():
                    return True
        return False

    def save(self):
        os.makedirs(self.dir_of(self.folder), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"stamp": self.stamp, "moves": self.moves, "before": self.before,
                       "created": self.created, "undone": self.undone}, f, ensure_ascii=False, indent=1)

    @classmethod
    def latest(cls, folder: str) -> Optional["Journal"]:
        """The most recent cleanup of this folder that has not been undone yet."""
        d = cls.dir_of(folder)
        if not os.path.isdir(d):
            return None
        for name in sorted((n for n in os.listdir(d) if n.endswith(".json")), reverse=True):
            try:
                with open(os.path.join(d, name), encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            if data.get("undone"):
                continue  # undo walks back one cleanup at a time, newest first
            j = cls(folder, data["stamp"])
            j.moves = [tuple(m) for m in data.get("moves", [])]
            j.before, j.created = data.get("before", []), data.get("created")
            return j
        return None

    def undo(self) -> Dict:
        """Park the files the cleanup made, move every file back, restore the captions."""
        created = self.created if self.created is not None else sorted(set(_top_files(self.folder)) - set(self.before))
        parked = restored = missing = 0
        for n in created:
            p = os.path.join(self.folder, n)
            if os.path.isfile(p):
                _park(p, self.bin_dir)
                parked += 1
        for src, dst in reversed(self.moves):
            s, d = os.path.join(self.folder, src), os.path.join(self.folder, dst)
            if not os.path.isfile(d):
                missing += 1
                continue
            if os.path.isfile(s):  # a shrunk copy or a re-cut crop took its place
                _park(s, self.bin_dir)
            shutil.move(d, s)
            restored += 1
        stems = {os.path.splitext(n)[0] for n in _top_files(self.folder) if n.lower().endswith(IMAGE_EXTS)}
        captions = 0
        if os.path.isdir(self.caption_dir):
            for n in os.listdir(self.caption_dir):
                stem = n[:-len(".captions.json")] if n.endswith(".captions.json") else os.path.splitext(n)[0]
                if stem not in stems:
                    continue
                b, cur = os.path.join(self.caption_dir, n), os.path.join(self.folder, n)
                if os.path.isfile(cur):
                    with open(b, "rb") as fb, open(cur, "rb") as fc:
                        if fb.read() == fc.read():
                            continue
                    _park(cur, self.bin_dir)
                shutil.copy2(b, cur)
                captions += 1
        self.undone = True
        self.save()
        msg = f"정리 전으로 되돌렸어요: 파일 {restored}개를 제자리로, 정리 때 만든 {parked}개는 excluded/{dg.JOURNAL_DIR}로 옮겼어요"
        if captions:
            msg += f", 캡션 {captions}개 복원"
        if missing:
            msg += f" (그 뒤에 옮겨지거나 지워져서 못 찾은 파일 {missing}개)"
        return {"restored": restored, "parked": parked, "captions": captions, "missing": missing, "message": msg}


# a caption runner: returns None when the missing captions were made, else a short reason it could not
CaptionRunner = Callable[[], Optional[str]]
Progress = Callable[[int, str, str], None]  # step index (0-based), step name, detail


class Cleanup:
    """Runs the seven steps on one folder. ``caption`` makes the missing captions (the server
    passes one that runs the caption job and waits); without it step 5 is skipped."""

    def __init__(self, folder: str, resolution: int = 1024, caption: Optional[CaptionRunner] = None,
                 progress: Optional[Progress] = None):
        self.folder, self.resolution = folder, resolution
        self.caption, self.progress = caption, progress
        self.journal = Journal(folder)
        self.done: Dict[str, int] = {}
        self.notes: List[str] = []
        self._step = 0

    def _say(self, step: int, detail: str = ""):
        self._step = step
        if self.progress:
            self.progress(step, STEPS[step], detail)

    def _analyze(self) -> Dict:
        def prog(i, n, name):
            self._say(self._step, f"{i + 1}/{n} {name}")
        return dg.analyze(self.folder, self.resolution, prog)

    def _apply(self, report: Dict, ids, tally: str) -> int:
        """Apply the fixes of these findings (in this order); count what they did under ``tally``."""
        n = 0
        for f in report["findings"]:
            if f["id"] in ids and f.get("fix") and f["fix"] != "caption" and f["files"]:
                n += dg.fix(self.folder, f["fix"], f["files"], self.journal)["done"]
        self.done[tally] = self.done.get(tally, 0) + n
        return n

    def _shrink_large(self):
        for n in _top_files(self.folder):
            p = os.path.join(self.folder, n)
            if not n.lower().endswith(IMAGE_EXTS) or n.startswith(dg.CROP_PREFIX):
                continue
            try:
                with Image.open(p) as im:  # header only
                    big = im.size[0] * im.size[1] > dg.LARGE_MP * 1e6
            except Exception:
                continue  # broken files are taken out in step 3
            if big:
                self._say(0, n)
                self.done["shrunk"] = self.done.get("shrunk", 0) + dg._shrink(p, self.folder, journal=self.journal)

    def run(self) -> Dict:
        self.journal.begin()
        try:
            self._say(0)
            self._shrink_large()
            self._say(1)
            report = self._analyze()
            self._say(2)
            self._apply(report, TAKE_OUT, "out")
            self._say(3)
            # too many first (keeps the best), then on what is left: re-cut loose ones, fill up to the target
            self._apply(self._analyze(), ("crop_many",), "crops_out")
            self._apply(self._analyze(), ("recrop", "closeup"), "crops")
            self._say(4)
            missing = next((f for f in self._analyze()["findings"] if f["id"] == "caption"), None)
            if missing:
                why = self.caption() if self.caption else "자동 캡션 없이 실행했어요"
                if why:
                    self.notes.append(f"캡션 없는 이미지 {len(missing['files'])}장: {why}")
                else:
                    self.done["captions"] = len(missing["files"])
            self._say(5)
            self._apply(self._analyze(), ("aug_many", "augment"), "augs")
            self._say(6)
            report = self._analyze()
        finally:
            self.journal.finish()
        return {"findings": report["findings"], "summary": self.summary(report["findings"]),
                "done": self.done, "stamp": self.journal.stamp}

    def summary(self, findings: List[Dict]) -> str:
        labels = (("shrunk", "줄임"), ("out", "뺌"), ("crops_out", "크롭 정리"), ("crops", "크롭"),
                  ("captions", "캡션"), ("augs", "화질 변형"))
        did = ", ".join(f"{lab} {self.done[k]}" for k, lab in labels if self.done.get(k))
        left = [f["title"] for f in findings if f["level"] in ("bad", "warn")]
        line = f"정리 끝 ({did})" if did else "정리 끝 (바꿀 게 없었어요)"
        if self.notes:
            line += ". " + "; ".join(self.notes)
        line += (". 남은 문제: " + ", ".join(left)) if left else ". 남은 문제 없음"
        return line


def main():
    ap = argparse.ArgumentParser(description="One-click dataset cleanup (CPU steps; captions via the web UI)")
    ap.add_argument("folder")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--undo", action="store_true", help="put the folder back as it was before the last cleanup")
    args = ap.parse_args()
    if args.undo:
        j = Journal.latest(args.folder)
        print(j.undo()["message"] if j else "되돌릴 정리가 없어요")
        return
    res = Cleanup(args.folder, args.resolution, progress=lambda i, step, d: d or print(f"[{i + 1}/7] {step}", flush=True)).run()
    print(res["summary"])


if __name__ == "__main__":
    main()
