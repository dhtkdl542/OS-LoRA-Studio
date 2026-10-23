"""Auto-captioning (Qwen3.8 / Gemma 4 through llama.cpp, Qwen3.5 int8, Qwen3-VL).

    python -m oslora.caption --folder D:/data --model qwen3vl-4b --style medium [--subject character]
                             [--trigger name] [--overwrite] [--status status.json]

Runs as its own process so its VRAM is released when it exits.
"""

import argparse
import json
import os
import re
import sys
import time

import torch
from PIL import Image, ImageOps

from .data import IMAGE_EXTS

MODELS = {
    # llama.cpp (the user's llama-server + GGUF): measured on 4 dataset images with the MOSS recaption
    # prompt: Qwen3.8 27B was the most accurate (30-45 s/image); Gemma 4 12B invented a pose once;
    # Qwen3.5 9B int8 with thinking took 240-370 s/image (our int8 path is slow at generation)
    "qwen38-27b-gguf": {"gguf": "qwen38", "quant": "llama.cpp", "label": "Qwen3.8 27B (llama.cpp, 가장 정확 ★추천, ~14GB)"},
    "gemma4-12b-gguf": {"gguf": "gemma4", "quant": "llama.cpp", "label": "Gemma 4 12B (llama.cpp, ~11GB)"},
    "qwen35-9b-int8": {"repo": "Qwen/Qwen3.5-9B", "quant": "comfy-int8",
                       "label": "Qwen3.5 9B int8 (MOSS 텍스트 인코더 파일 재사용, ~11GB)"},
    "qwen3vl-4b": {"repo": "Qwen/Qwen3-VL-4B-Instruct", "quant": None, "label": "Qwen3-VL 4B (빠름, ~9GB)"},
    "qwen3vl-8b-4bit": {"repo": "Qwen/Qwen3-VL-8B-Instruct", "quant": "4bit", "label": "Qwen3-VL 8B 4bit (~7GB)"},
    "qwen3vl-8b": {"repo": "Qwen/Qwen3-VL-8B-Instruct", "quant": None, "label": "Qwen3-VL 8B (~18GB)"},
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN35_DIR = os.path.join(ROOT, "models", "qwen35_captioner")


def ensure_qwen35_lm_head(log=print) -> str:
    """The comfy text-encoder file has no output head (an encoder never needs one). Fetch
    just ``lm_head.weight`` (~2 GB bf16) out of the stock Qwen3.5-9B shard with HTTP range
    requests and store it as per-row int8 (~1 GB)."""
    import struct

    import requests
    from huggingface_hub import hf_hub_download, hf_hub_url
    from safetensors.torch import save_file

    out = os.path.join(QWEN35_DIR, "lm_head_int8.safetensors")
    if os.path.isfile(out):
        return out
    os.makedirs(QWEN35_DIR, exist_ok=True)
    index = json.load(open(hf_hub_download("Qwen/Qwen3.5-9B", "model.safetensors.index.json"), encoding="utf-8"))
    shard = index["weight_map"]["lm_head.weight"]
    url = hf_hub_url("Qwen/Qwen3.5-9B", shard)
    head = requests.get(url, headers={"Range": "bytes=0-7"}, timeout=60).content
    n = struct.unpack("<Q", head)[0]
    header = json.loads(requests.get(url, headers={"Range": f"bytes=8-{7 + n}"}, timeout=60).content)
    meta = header["lm_head.weight"]
    if meta["dtype"] != "BF16":
        raise RuntimeError(f"unexpected lm_head dtype {meta['dtype']}")
    a, b = meta["data_offsets"]
    start, end = 8 + n + a, 8 + n + b - 1
    total = end - start + 1
    buf = bytearray(total)
    pos = 0
    log(f"Qwen3.5 출력층(lm_head) 다운로드 중 ({total / 2**30:.1f} GB, 처음 한 번만)")
    with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=60) as r:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 22):
            buf[pos : pos + len(chunk)] = chunk
            pos += len(chunk)
    if pos != total:
        raise RuntimeError("lm_head 다운로드가 중간에 끊겼습니다. 다시 시도하세요.")
    w = torch.frombuffer(buf, dtype=torch.bfloat16).reshape(meta["shape"]).float()
    scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-12)
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    save_file({"lm_head.weight": q, "lm_head.weight_scale": scale.float()}, out)
    return out


class Qwen35Captioner:
    """Qwen3.5-9B VLM assembled from the comfy int8 ConvRot text-encoder file (language model
    + vision tower) plus an int8 lm_head; everything stays int8 on the GPU."""

    def __init__(self, path: str = "", device="cuda", log=print):
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

        from .models.quant import load_comfy_state_dict

        from .models.moss.adapter import normalize_qwen35_keys
        from .settings import require_model

        path = path or require_model("text_encoder")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
        config = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B")
        with init_empty_weights():
            model = Qwen3_5ForConditionalGeneration(config)
        log("Qwen3.5 9B int8 로딩")
        # the MOSS repack lacks the LM head (fetched once from Qwen); Comfy-Org's file has it
        sd = normalize_qwen35_keys(load_file(path))
        head = {k: sd.pop(k) for k in [k for k in sd if k.startswith("lm_head.")]}
        sd = {("model." + k): v for k, v in sd.items() if not k.startswith("mtp.")}
        if not head:
            head = load_file(ensure_qwen35_lm_head(log))
            head["lm_head.comfy_quant"] = torch.tensor(list(b'{"format": "int8_tensorwise"}'), dtype=torch.uint8)
        sd.update(head)
        missing = load_comfy_state_dict(model, sd, torch.bfloat16, device)
        if missing:
            raise KeyError(f"Qwen3.5 weights missing: {missing[:8]}")
        self.model = model.eval().requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def caption(self, image, instruction, long=False, max_pixels=1024 * 1024, temperature=0.5):
        # thinking stays off here: our int8 path generates at ~4 tok/s, thinking took 4-6 min/image
        return _generate(self, image, instruction, max_pixels, temperature, 1500 if long else 400,
                         template_kwargs={"enable_thinking": False})

STYLES = {
    "medium": (
        "Write a caption of 2 to 4 natural English sentences for this image, to be used as a "
        "text-to-image training caption. Cover, in order: the main subject and what they are doing, "
        "the camera framing and angle (close-up, half body, full body; from above, eye level) and "
        "whether the face is visible, the pose and expression, the clothing and accessories, the "
        "setting and background, and the lighting and overall photographic style."
    ),
    "short": (
        "Write one natural English sentence describing this image as a text-to-image training caption: "
        "subject, camera framing, pose, clothing, setting and lighting."
    ),
    "detailed": (
        "Describe this image in one detailed English paragraph (5 to 8 sentences) for a text-to-image "
        "training caption: subject and action, composition and camera angle, pose and expression, "
        "clothing and accessories with colors and materials, background objects and environment, "
        "lighting, color palette and photographic or artistic style. Transcribe any visible text exactly."
    ),
    "tags": (
        "List comma-separated booru-style tags for this image: subject count, framing, pose, expression, "
        "clothing items with colors, background, lighting, style. Output only the tags."
    ),
    # MOSS-Image's official recaption prompt (USERGUIDE.md, "Image Captioning"): long / medium /
    # short at once, as JSON. The subject rule and the finish rule are inserted by build_instruction.
    "moss": (
        "You are an image recaptioner. Return exactly one JSON object with this schema:\n"
        '{"caption": {"long": "a detailed image description, including all visible text", '
        '"medium": "a one-paragraph image description", "short": "a one-sentence image description"}}\n\n'
        "Rules:\n"
        "- caption.long: Describe the composition, key subjects, actions, setting, colors, lighting, and "
        "photographic or artistic style in detail. Include only information visible in the image. Preserve "
        "visible text exactly; do not translate or paraphrase it.\n"
        "- caption.medium: compress long into one paragraph.\n"
        "- caption.short: summarize the main subject, action, or relationship in one sentence.\n"
        "{rules}"
        "- All three caption fields must be non-empty strings. Return only the JSON object, without Markdown "
        "or explanations."
    ),
}
LONG_STYLES = {"moss"}  # JSON with three lengths: more tokens, thinking on

# the clean "AI render" look must be named in words, or the LoRA bakes it into the character
FINISH_RULE = ("In long and medium, state the image's finish plainly: how sharp or soft it is, any noise or "
               "grain, how smooth or retouched the skin looks, compression, and whether it looks like a phone "
               "snapshot, a studio photo, film, or a polished digital render.")

SUBJECT_RULES = {
    "general": "",
    "character": (
        " The person is a specific recurring character: refer to them only as 'a woman', 'a man' or "
        "'a person'. Do NOT describe their face shape, eye shape or color, skin tone, ethnicity, age, or "
        "other innate identifying features, and do not guess their name. Do describe hairstyle, makeup, "
        "clothing, accessories, pose, expression and everything around them."
    ),
    "style": (
        " Describe only the content of the image. Do not mention the art style, medium, rendering, "
        "brushwork, color grading or any style words at all."
    ),
}

RULES = (
    " Start directly with the description; do not begin with phrases like 'This image shows' or "
    "'The image depicts'. No lists, no markdown, no opinions."
)

PREAMBLE = re.compile(
    r"^\s*(?:the|this)\s+(?:image|photo|photograph|picture|illustration)\s+(?:shows|depicts|features|"
    r"captures|presents|is of|displays)\s+",
    re.IGNORECASE,
)


def clean(text: str) -> str:
    text = text.strip().strip('"').strip()
    text = re.sub(r"\s+", " ", text)
    m = PREAMBLE.match(text)
    if m:
        text = text[m.end():]
        text = text[:1].upper() + text[1:]
    return text


def build_instruction(style: str, subject: str, extra: str = "") -> str:
    if style == "moss":
        rules = [SUBJECT_RULES.get(subject, "").strip()]
        if subject != "style":  # style LoRAs keep style words out of captions
            rules.append(FINISH_RULE)
        out = STYLES["moss"].replace("{rules}", "".join(f"- {r}\n" for r in rules if r))
        if extra.strip():
            out = out.replace("Return only the JSON", extra.strip() + " Return only the JSON")
        return out
    out = STYLES.get(style, STYLES["medium"]) + SUBJECT_RULES.get(subject, "") + RULES
    if extra.strip():
        out += " " + extra.strip()
    return out


class Captioner:
    def __init__(self, model_key: str = "qwen3vl-4b", device="cuda"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        spec = MODELS[model_key]
        kwargs = {"dtype": torch.bfloat16, "device_map": device}
        if spec["quant"] == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.processor = AutoProcessor.from_pretrained(spec["repo"])
        self.model = AutoModelForImageTextToText.from_pretrained(spec["repo"], **kwargs).eval()
        self.device = device

    @torch.no_grad()
    def caption(self, image: Image.Image, instruction: str, long: bool = False, max_pixels: int = 1024 * 1024,
                temperature: float = 0.5) -> str:
        return _generate(self, image, instruction, max_pixels, temperature, 1500 if long else 400)


def _generate(cap, image, instruction, max_pixels, temperature, max_new_tokens, template_kwargs=None) -> str:
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = (max_pixels / (image.width * image.height)) ** 0.5
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]}]
    inputs = cap.processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        **(template_kwargs or {}),
    ).to(cap.device)
    # captioning must not disturb anyone else's RNG stream
    state = torch.random.get_rng_state()
    out = cap.model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-3),
        top_p=0.9,
    )
    torch.random.set_rng_state(state)
    text = cap.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return clean(text)


class LlamaServerCaptioner:
    """A GGUF vision model served by the user's llama.cpp ``llama-server`` (started here, stopped
    by ``close``). Talks to its OpenAI-compatible endpoint; thinking on for the JSON styles, as
    the MOSS guide recommends."""

    def __init__(self, model_key: str, log=print):
        import socket
        import subprocess

        import requests

        from .settings import require_llama

        paths = require_llama(MODELS[model_key]["gguf"])
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        log(f"{MODELS[model_key]['label']} 로딩 (llama-server)")
        self.log_path = os.path.join(ROOT, "cache", "llama_server.log")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [paths["server"], "--model", paths["model"], "--mmproj", paths["mmproj"], "--host", "127.0.0.1",
             "--port", str(self.port), "--n-gpu-layers", "99", "--ctx-size", "12288", "--flash-attn", "on",
             "--parallel", "1", "--jinja"],
            stdout=self._log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.url = f"http://127.0.0.1:{self.port}"
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server가 시작하지 못했어요. 로그: {self.log_path}")
            try:
                if requests.get(self.url + "/health", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        self.close()
        raise RuntimeError("llama-server가 15분 안에 준비되지 않았어요.")

    def caption(self, image: Image.Image, instruction: str, long: bool = False, max_pixels: int = 1024 * 1024) -> str:
        import base64
        import io

        import requests

        image = ImageOps.exif_transpose(image).convert("RGB")
        scale = (max_pixels / (image.width * image.height)) ** 0.5
        if scale < 1:
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=92)
        body = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}]}],
            "max_tokens": 8192 if long else 1024, "temperature": 0.7 if long else 0.5, "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": bool(long)}}
        r = requests.post(self.url + "/v1/chat/completions", json=body, timeout=1800)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"].get("content") or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def close(self):
        if getattr(self, "proc", None) and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except Exception:
                self.proc.kill()
        if getattr(self, "_log", None):
            self._log.close()


def parse_variants(text: str):
    """{"long", "medium", "short"} from a recaptioner reply; tolerant of fences, extra prose and
    a broken JSON tail (each field is then read on its own). None if nothing usable."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            data = data.get("caption", data)
            out = {k: clean(str(data.get(k) or "")) for k in ("long", "medium", "short")}
            if all(out.values()):
                return out
        except (ValueError, AttributeError):
            pass
    out = {}
    for k in ("long", "medium", "short"):
        m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % k, text, flags=re.DOTALL)
        if m:
            try:
                out[k] = clean(json.loads('"' + m.group(1) + '"'))
            except ValueError:
                out[k] = clean(m.group(1))
    return out if out.get("long") and (out.get("medium") or out.get("short")) else None


def make_captioner(model_key: str, device="cuda", log=print):
    if MODELS[model_key]["quant"] == "llama.cpp":
        return LlamaServerCaptioner(model_key, log=log)
    if MODELS[model_key]["quant"] == "comfy-int8":
        return Qwen35Captioner(device=device, log=log)
    return Captioner(model_key, device)


def caption_folder(folder: str, model_key: str, style: str, subject: str, trigger: str = "",
                   overwrite: bool = False, extra: str = "", status_path: str = "", caption_ext: str = ".txt"):
    from .augment import aug_lead, refresh_captions
    from .data import AUG_PREFIX, CAPTIONS_EXT, save_caption_variants

    names = sorted(os.listdir(folder))
    # degraded copies are captioned from their own pixels too (the original's caption would call a
    # blurry black-and-white copy "sharp, cream-toned"), with the effect's sentences in front
    images = [os.path.join(folder, n) for n in names if n.lower().endswith(IMAGE_EXTS)]
    long = style in LONG_STYLES

    def needs(p):
        base = os.path.splitext(p)[0]
        if overwrite or not os.path.isfile(base + caption_ext):
            return True
        return long and not os.path.isfile(base + CAPTIONS_EXT)  # three lengths still missing

    todo = [p for p in images if needs(p)]

    def status(**kw):
        if status_path:
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump(kw, f, ensure_ascii=False)
        print(json.dumps(kw, ensure_ascii=False), flush=True)

    status(phase="loading", done=0, total=len(todo), message=f"{MODELS[model_key]['label']} 로딩 중")
    if not todo:
        n = refresh_captions(folder, caption_ext)
        status(phase="done", done=0, total=0, message="캡션이 필요한 이미지가 없습니다 (덮어쓰기 꺼짐)"
               + (f", 화질 변형 사본 {n}장 캡션은 원본에 맞춰 갱신" if n else ""))
        return
    # whatever happens, the captioner's weight files leave the Windows file cache afterwards
    from . import memclean
    from .settings import llama_paths, model_path

    spec = MODELS[model_key]
    if spec["quant"] == "llama.cpp":
        lp = llama_paths(spec["gguf"])
        weight_files = [lp["model"], lp["mmproj"]]
    elif spec["quant"] == "comfy-int8":
        weight_files = [model_path("text_encoder"), os.path.join(QWEN35_DIR, "lm_head_int8.safetensors")]
    else:
        weight_files = memclean.hf_cached_files(spec["repo"])
    memclean.schedule_after_exit([f for f in weight_files if f])
    cap = make_captioner(model_key, log=lambda m: status(phase="loading", done=0, total=len(todo), message=m))
    instruction = build_instruction(style, subject, extra)
    trig = trigger.strip()

    def with_trigger(text):
        return f"{trig}, {text}" if trig and trig.lower() not in text.lower() else text

    def finish(path, text):
        name = os.path.basename(path)
        lead = aug_lead(name) if name.startswith(AUG_PREFIX) else ""
        return f"{lead} {with_trigger(text)}".strip()

    t0, failed = time.time(), []
    try:
        for i, path in enumerate(todo):
            with Image.open(path) as im:
                if long:
                    variants = None
                    for _ in range(3):  # a malformed reply is asked again
                        variants = parse_variants(cap.caption(im, instruction, long=True))
                        if variants:
                            break
                    if variants is None:
                        failed.append(os.path.basename(path))
                        continue
                    variants = {k: finish(path, v) for k, v in variants.items() if v}
                    save_caption_variants(path, variants, caption_ext, model=model_key)
                    text = variants.get("long", "")
                else:
                    text = finish(path, cap.caption(im, instruction))
                    with open(os.path.splitext(path)[0] + caption_ext, "w", encoding="utf-8") as f:
                        f.write(text)
            eta = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
            status(phase="captioning", done=i + 1, total=len(todo), last=os.path.basename(path), caption=text,
                   message=f"캡션 {i + 1}/{len(todo)} (남은 시간 ~{eta / 60:.1f}분)")
    finally:
        if hasattr(cap, "close"):
            cap.close()
    n = refresh_captions(folder, caption_ext)
    msg = f"완료: {len(todo) - len(failed)}개 캡션 생성" + (f", 화질 변형 사본 {n}장 캡션도 원본에 맞춰 갱신" if n else "")
    if failed:
        msg += f" (실패 {len(failed)}장: {', '.join(failed[:5])} - 다시 실행하면 그것만 해요)"
    status(phase="done", done=len(todo), total=len(todo), message=msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--model", default="qwen38-27b-gguf", choices=list(MODELS))
    ap.add_argument("--style", default="medium", choices=list(STYLES))
    ap.add_argument("--subject", default="general", choices=list(SUBJECT_RULES))
    ap.add_argument("--trigger", default="")
    ap.add_argument("--extra", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--status", default="")
    a = ap.parse_args()
    try:
        caption_folder(a.folder, a.model, a.style, a.subject, a.trigger, a.overwrite, a.extra, a.status)
    except Exception as e:
        if a.status:
            with open(a.status, "w", encoding="utf-8") as f:
                json.dump({"phase": "error", "message": str(e)}, f, ensure_ascii=False)
        raise


if __name__ == "__main__":
    main()
