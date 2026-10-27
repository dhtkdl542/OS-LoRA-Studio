"""Take the face out of a character's captions: makeup, lip colour, blush, brows, lashes, liner.

Whatever a caption names, the model learns as the prompt's job; a face that the captions keep
describing ("glossy pink lips, a touch of rosy blush, defined brows") is learned as "whatever the
prompt says", so at generation time, where the prompt says none of it, those parts drift. The
character's face must stay unnamed to be learned as hers.

What stays: expression and mouth action ("lips slightly parted", "a pout", "a wink"), where she
looks, which way the head turns, hair, and light falling on the face ("highlights on the nose and
cheeks"). Works on the list items, not whole sentences, so "She wears subtle makeup, including pink
lips, and a cream knit top" keeps the top.

    python -m oslora.facewords <dataset folder> [--apply]   # dry run prints every change
"""
import argparse
import json
import os
import re
import time

# always the face's look
_MAKEUP = (r"make-?up(?: look)?|blush|lipstick|lip ?gloss|lip (?:colou?r|tint)|eyeliner|liner|mascara|"
           r"eye ?shadow|eye make-?up|eye definition|brows|eyebrows|(?:(?:upper|lower) and (?:upper|lower) )?(?:eye)?lashes|"
           r"complexion|freckles|moles?")
# the look only when coloured or made up: "glossy pink lips" goes, "slightly parted lips" stays
_COLOURED = re.compile(r"\b(pink\w*|red\w*|nude|ros[ey]\w*|coral|berry|peach\w*|mauve|plum|glossy|gloss|matte|tint\w*|"
                       r"full|plump|natural(?:-looking|-toned)?|muted|warm-toned|cherry|crimson|scarlet|flush\w*|blush\w*|"
                       r"defined|lined|made-up|smoky|smokey|winged|painted)\b", re.I)
_SOMETIMES = r"lips|lip|eyes|cheeks|eyelids|lids"
_COLOUR = r"(?:pink|red|nude|rose|coral|berry|peach|mauve|plum|crimson|cherry)[\w-]*"
# words that end a list item rather than describe it
_STOP = r"(?:and|with|while|as|she|he|her|his|wears?|wearing|has|have|is|are|was|were|including|featuring|includes|features|in|that|which|plus|or)"
_WORD = r"(?:(?!" + _STOP + r"\b)[\w'-]+\s+)"
# makeup adjectives may run across commas ("soft, natural-looking makeup"); nothing else does, so "a
# white top, glossy lips" keeps the top
_MOD = (r"(?:(?:very |softly |lightly |subtly |neatly )?(?:soft|subtle|light|natural(?:-looking|-toned)?|glossy|shiny|fine|thin|"
        r"delicate|minimal|polished|dewy|warm|cool|smooth|matte|rosy|muted|pinkish|pink|defined|shaped|bold|dark|faint|"
        r"gentle|understated|sheer|luminous|fresh|clean|groomed|arched|full|plump)),\s+")
_FACE_PART = r"(?:cheeks?|eyes|lips|lids|eyelids|face|cheekbones|nose)"
_ON_FACE = (r"(?:\s+(?:on|across|along|around|over|at|framing|defining|lining|accentuating|enhancing|highlighting)\s+"
            r"(?:her|the|both)\s+" + _FACE_PART + r"(?:\s+and\s+(?:the\s+)?" + _FACE_PART + r")?|\s+on them)")
_PLACE = "(?:" + _ON_FACE + ")?"
_LEAD = r"\b(?:(?:a|an|the|her|some)\s+)?(?:" + _MOD + r")*"
_NP = re.compile(_LEAD + _WORD + r"{0,6}?(?:(?P<m>" + _MAKEUP + r")|(?P<s>" + _SOMETIMES + r"))\b" + _PLACE, re.I)
# "Her makeup is soft and natural" (the items after it go one by one below), "Her makeup includes",
# "..., lips slightly parted and makeup kept soft and understated"
_STATEMENT = re.compile(r"(?:,?\s*(?:and\s+)?(?:her|the)\s+)?\bmakeup(?: look)? (?:(?:is|looks|appears|kept|done|applied|remains|stays)\b"
                        r"[^,;.:]*?(?=\s+(?:with|featuring|including)\b|[,;.:]|$)|(?:includes|features|consists of)\b)", re.I)
# colour put on the face: "a soft pink tint on her cheeks", "a pinkish color on her lips"
_ON_CHEEKS = re.compile(_LEAD + _WORD + r"{0,3}?(?:tint|flush|wash|hint|touch|glow|colou?r|shade|hue|tone)\b(?:\s+of\s+[\w-]+)?"
                        + _ON_FACE, re.I)
# "her lips carry a soft pinkish tint", "lips tinted a soft pink", "lips coated in a coral-pink tint"
_LIPS_WEAR = re.compile(r",?\s*(?:and\s+)?(?:her\s+)?lips (?:carry|have|wear|show|bear)\s+" + _LEAD + _WORD
                        + r"{0,4}?(?:tint|tone|hue|colou?r|shade|sheen|gloss)\b", re.I)
_TINTED = re.compile(r",?\s*(?:and\s+)?(?:tinted|painted|glossed|coated|coloured|colored)\s+(?:in\s+)?" + _LEAD + _WORD
                     + r"{0,3}?" + _COLOUR + r"(?:\s+(?:tint|shade|colou?r|hue))?", re.I)
_LIP_SHADE = re.compile(r"(\blips\b(?:\s+[\w-]+){0,3}?)\s+(?:in|with)\s+" + _LEAD + _WORD + r"{0,3}?(?:tint|shade|colou?r|hue)\b", re.I)
# a mouth or eyes doing something is expression: "glossy pink lips pressed together" -> "lips pressed together"
_ACTION = re.compile(r"\s+(?:that |which )?(?:are |were |is )?(?:slightly |softly |gently |barely )?(?:parted|pressed|closed|pursed|pushed|open|"
                     r"curl\w*|curv\w*|pucker\w*|form\w*|in an? [\w\s,-]*?expression|squint\w*|crinkl\w*|narrow\w*)", re.I)
# "with dark updo hair and soft makeup gazes at the camera": the "and" goes with it
_AND_MAKEUP = re.compile(r"\s+and\s+" + _LEAD + _WORD + r"{0,4}?make-?up(?: look)?(?=\s+[a-z]+s\b)", re.I)
_DANGLING = re.compile(r"\b(?:with|including|featuring|includes|features|and|plus|of|such as)\s*(?=[,;:.]|$)", re.I)
_DANGLING_END = re.compile(r",?\s*\b(?:and\s+)?(?:she\s+)?wears?\s*$", re.I)  # "..., and she wears" (its makeup went)


def _np_out(m: re.Match) -> str:
    if m.group("m"):
        return ""
    if not _COLOURED.search(m.group(0)):
        return m.group(0)
    return m.group("s") if _ACTION.match(m.string, m.end()) else ""


def _tidy(s: str) -> str:
    for _ in range(4):
        s = _DANGLING.sub("", s)
        s = _DANGLING_END.sub("", s) if not re.fullmatch(r"\s*(?:she|he)\s+wears?\s*", s, re.I) else s
        s = re.sub(r"\s*,(?:\s*,)+", ",", s)                 # ", , ," -> ","
        s = re.sub(r",\s*(?=[;:.]|$)", "", s)                # "lips, ." -> "lips."
        s = re.sub(r"(?<=\w)\s*:\s*(?=[;.]|$)", "", s)       # "She wears:" left empty
        s = re.sub(r"\b(wears?|has|have|with)\s*,\s*", r"\1 ", s, flags=re.I)  # "She wears , and a top"
        s = re.sub(r"\b(wears?|has|have|with) and\s+", r"\1 ", s, flags=re.I)
        s = re.sub(r"^[,;:\s]*(?:(?:and|with|while|but)\s+)*(?=\S)", "", s) if re.match(r"[,;:\s]|(?:and|with)\b", s, re.I) else s
        s = re.sub(r"^lips\s+(?:that\s+|which\s+)?(?:are\s+)?", "her lips are ", s, flags=re.I)  # "Lips slightly parted"
        s = re.sub(r"\b(?:and\s+)?lips\s+(?:that|which)\s+are\b", "and her lips are", s, flags=re.I)
        s = re.sub(r"(?:^|(?<=[,;:]))\s*(?:and\s+)?(?:her\s+)?(?:lips|eyes|cheeks)\s*(?=[,;:.]|$)", "", s, flags=re.I)  # bare "lips" left of an item
        s = re.sub(r"\s+([,;:])", r"\1", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
    return s


_EMPTY = re.compile(r"^(?:and\s+)?(?:she|he|her|his|the person)?\s*(?:wears?|has|have|is|makeup|features?)?\s*$", re.I)


def strip_face(text: str) -> str:
    """The caption without its face description (makeup, lip colour, blush, brows, lashes)."""
    out = []
    for sent in (x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x):
        body, end = sent.rstrip(".!? "), (sent[len(sent.rstrip(".!? ")):].strip()[:1] or ".")
        kept = []
        for clause in body.split(";"):
            c = _AND_MAKEUP.sub("", clause)
            c = _STATEMENT.sub("", c)
            c = _TINTED.sub("", c)
            c = _LIPS_WEAR.sub("", c)
            c = _ON_CHEEKS.sub("", c)
            c = _LIP_SHADE.sub(r"\1", c)
            c = _NP.sub(_np_out, c)
            c = _tidy(c)
            if c and not _EMPTY.match(c):
                kept.append(c[:1].upper() + c[1:] if not kept else c)
        if kept:
            out.append("; ".join(kept) + end)
    return " ".join(out)


def _caption_files(folder: str):
    for n in sorted(os.listdir(folder)):
        p = os.path.join(folder, n)
        if os.path.isfile(p) and (n.endswith(".txt") or n.endswith(".captions.json")):
            yield p


def strip_folder(folder: str, apply: bool = False, log=print) -> int:
    """Strip every caption in the folder (each length in .captions.json too). The old files move
    to excluded/캡션백업_<time>/ first. Returns how many files change."""
    backup = os.path.join(folder, "excluded", time.strftime("캡션백업_%Y%m%d-%H%M"))
    changed = 0
    for p in _caption_files(folder):
        with open(p, encoding="utf-8") as f:
            raw = f.read()
        if p.endswith(".json"):
            data = json.loads(raw)
            new = dict(data)
            for k in ("long", "medium", "short"):
                if isinstance(data.get(k), str):
                    new[k] = strip_face(data[k])
            text = json.dumps(new, ensure_ascii=False, indent=1)
            if new == data:
                continue
        else:
            text = strip_face(raw)
            if text == raw.strip():
                continue
            log(f"--- {os.path.basename(p)}")
            _show_diff(raw, text, log)
        changed += 1
        if apply:
            os.makedirs(backup, exist_ok=True)
            os.replace(p, os.path.join(backup, os.path.basename(p)))
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
    return changed


def _show_diff(old: str, new: str, log):
    a = re.split(r"(?<=[.!?])\s+", old.strip())
    b = set(re.split(r"(?<=[.!?])\s+", new.strip()))
    for s in a:
        if s not in b:
            log(f"  - {s}")
            t = strip_face(s)
            log(f"  + {t}" if t else "  + (문장 삭제)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--apply", action="store_true", help="write the stripped captions (old ones to excluded/)")
    a = ap.parse_args()
    n = strip_folder(a.folder, a.apply)
    print(f"{'바꿈' if a.apply else '바뀔 파일'}: {n}개")


if __name__ == "__main__":
    main()
