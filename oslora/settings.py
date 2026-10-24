"""Per-install settings (settings.json next to run.bat, not in git): the model file paths.

Model weights are not downloaded automatically: each user points the paths at their own files
(usually in their ComfyUI models folder). ``MODEL_FILES`` lists where to get each one.
"""

import json
import os
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = os.path.join(ROOT, "settings.json")

HF = "https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/"
HF_KREA2 = "https://huggingface.co/Comfy-Org/Krea-2/blob/main/"
HF_QWEN21 = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/"

# model -> file key -> what it is and where to get it (Comfy-Org repacks; files from other ComfyUI
# repacks in the same int8 ConvRot / BFL / diffusers layouts work too)
MOSS_FILES: Dict[str, Dict[str, str]] = {
    "dit": {
        "label": "DiT (학습용 Base)",
        "file": "moss_image_base_int8_convrot.safetensors",
        "url": HF + "diffusion_models/moss_image_base_int8_convrot.safetensors",
        "size": "5.4 GB",
        "folder": "models/diffusion_models",
    },
    "inference_dit": {
        "label": "샘플용 DiT (SFT)",
        "file": "moss_image_sft_int8_convrot.safetensors",
        "url": HF + "diffusion_models/moss_image_sft_int8_convrot.safetensors",
        "size": "5.4 GB",
        "folder": "models/diffusion_models",
    },
    "text_encoder": {
        "label": "텍스트 인코더 (Qwen3.5 9B)",
        "file": "qwen3.5_9b_int8_convrot.safetensors",
        "url": HF + "text_encoders/qwen3.5_9b_int8_convrot.safetensors",
        "size": "9.7 GB",
        "folder": "models/text_encoders",
    },
    "vae": {
        "label": "VAE (FLUX.2)",
        "file": "flux2-vae.safetensors",
        "url": HF + "vae/flux2-vae.safetensors",
        "size": "0.3 GB",
        "folder": "models/vae",
    },
}

KREA2_FILES: Dict[str, Dict[str, str]] = {
    "dit": {
        "label": "DiT (학습용 Raw)",
        "file": "krea2_raw_int8_convrot.safetensors",
        "url": HF_KREA2 + "diffusion_models/krea2_raw_int8_convrot.safetensors",
        "size": "13.5 GB",
        "folder": "models/diffusion_models",
    },
    "inference_dit": {
        "label": "샘플용 DiT (Turbo)",
        "file": "krea2_turbo_int8_convrot.safetensors",
        "url": HF_KREA2 + "diffusion_models/krea2_turbo_int8_convrot.safetensors",
        "size": "13.5 GB",
        "folder": "models/diffusion_models",
    },
    "text_encoder": {
        "label": "텍스트 인코더 (Qwen3-VL 4B)",
        "file": "qwen3vl_4b_int8_convrot.safetensors",
        "url": HF_KREA2 + "text_encoders/qwen3vl_4b_int8_convrot.safetensors",
        "size": "4.8 GB",
        "folder": "models/text_encoders",
    },
    "vae": {
        "label": "VAE (Qwen Image / Wan 2.1)",
        "file": "qwen_image_vae.safetensors",
        "url": HF_KREA2 + "vae/qwen_image_vae.safetensors",
        "size": "0.25 GB",
        "folder": "models/vae",
    },
}

QWEN21_FILES: Dict[str, Dict[str, str]] = {
    "dit": {
        "label": "DiT (학습용 기본 모델)",
        "file": "qwen_image_2.1_int8_convrot.safetensors",
        "url": HF_QWEN21 + "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
        "size": "7.3 GB",
        "folder": "models/diffusion_models",
    },
    "inference_dit": {
        "label": "샘플용 DiT (Turbo)",
        "file": "qwen_image_2.1_turbo_int8_convrot.safetensors",
        "url": HF_QWEN21 + "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
        "size": "7.3 GB",
        "folder": "models/diffusion_models",
    },
    "text_encoder": {
        "label": "텍스트 인코더 (Qwen3-VL 8B)",
        "file": "qwen3vl_8b_int8_convrot.safetensors",
        "url": HF_QWEN21 + "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
        "size": "9.4 GB",
        "folder": "models/text_encoders",
    },
    # the texture-fix finetune (madebyollin): same encoder output (exactly), a decoder without the
    # official VAE's 2 px checkerboard (measured: grid energy 73x -> 4x its neighbourhood). The
    # official qwen_image_2.1_vae_bf16 works too, samples just show the grid.
    "vae": {
        "label": "VAE (Texture-Fix, Qwen Image 2.1)",
        "file": "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors",
        "url": "https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1/blob/main/"
               "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors",
        "size": "0.7 GB",
        "folder": "models/vae",
    },
}

ALL_FILES: Dict[str, Dict[str, Dict[str, str]]] = {"moss_image": MOSS_FILES, "krea2": KREA2_FILES, "qwen21": QWEN21_FILES}
MODEL_FILES = MOSS_FILES  # (MOSS-only callers)
FILE_KEYS = ("dit", "inference_dit", "text_encoder", "vae")


# caption models run by llama.cpp's llama-server (GGUF + vision projector). Paths in settings.json
# under "llama_cpp"; empty = not set up (the UI says where to get them).
LLAMA_CPP_RELEASES = "https://github.com/ggml-org/llama.cpp/releases"
LLAMA_MODELS: Dict[str, Dict[str, str]] = {
    "qwen38": {
        "label": "Qwen3.8 27B (Huihui abliterated) UD-Q3_K_XL",
        "url": "https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF",
        "model_file": "Huihui-Qwen3.8-27B-abliterated-UD-Q3_K_XL.gguf",
        "mmproj_file": "mmproj-model-bf16.gguf",
        "model_size": "12.4 GB",
        "mmproj_size": "0.9 GB",
    },
    "gemma4": {
        "label": "Gemma 4 12B it (heretic) Q6_K",
        "url": "https://huggingface.co/prithivMLmods/gemma-4-12B-it-heretic_decensored-GGUF",
        "model_file": "gemma-4-12B-it-heretic_decensored.Q6_K.gguf",
        "mmproj_file": "gemma-4-12B-it-heretic_decensored.mmproj-f16.gguf",
        "model_size": "9.1 GB",
        "mmproj_size": "0.2 GB",
    },
}


def llama_paths(key: str) -> Dict[str, str]:
    """{"server", "model", "mmproj"} for a llama.cpp caption model (may be empty strings)."""
    cfg = load().get("llama_cpp") or {}
    m = cfg.get(key) or {}
    return {"server": str(cfg.get("server") or ""), "model": str(m.get("model") or ""), "mmproj": str(m.get("mmproj") or "")}


def set_llama_paths(key: str, server=None, model=None, mmproj=None):
    """Remember llama.cpp paths: the server is shared by every GGUF model, model/mmproj are per model."""
    data = load()
    cfg = data.get("llama_cpp") or {}
    if server is not None:
        cfg["server"] = str(server).strip().strip('"')
    m = cfg.get(key) or {}
    for k, v in (("model", model), ("mmproj", mmproj)):
        if v is not None:
            m[k] = str(v).strip().strip('"')
    cfg[key] = m
    data["llama_cpp"] = cfg
    save(data)


def llama_status(key: str) -> Dict[str, Dict]:
    """Per file (server / model / mmproj): path, whether it exists, what to download."""
    info, paths = LLAMA_MODELS[key], llama_paths(key)
    files = {
        "server": {"label": "llama-server.exe (llama.cpp, CUDA)", "file": "llama-server.exe", "size": "약 0.6 GB",
                   "url": LLAMA_CPP_RELEASES},
        "model": {"label": "모델 (GGUF)", "file": info["model_file"], "size": info.get("model_size", ""), "url": info["url"]},
        "mmproj": {"label": "비전 프로젝터 (mmproj)", "file": info["mmproj_file"], "size": info.get("mmproj_size", ""),
                   "url": info["url"]},
    }
    for k, f in files.items():
        f["path"] = paths[k]
        f["ok"] = bool(paths[k]) and os.path.isfile(paths[k])
    return {"label": info["label"], "files": files}


def require_llama(key: str) -> Dict[str, str]:
    paths = llama_paths(key)
    missing = [k for k, v in paths.items() if not v or not os.path.isfile(v)]
    if missing:
        info = LLAMA_MODELS[key]
        raise FileNotFoundError(
            f"{info['label']} 캡션에 필요한 파일이 아직 없어요 ({', '.join(missing)}). 자동 캡션의 [파일] 버튼에서 "
            "📁로 지정하거나 ⬇로 받으세요.")
    return paths


def load() -> Dict:
    try:
        with open(SETTINGS, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save(data: Dict):
    tmp = SETTINGS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    from .fsutil import replace

    replace(tmp, SETTINGS)


def _stored_paths() -> Dict[str, Dict[str, str]]:
    stored = load().get("model_paths") or {}
    if any(k in stored for k in FILE_KEYS):  # before per-model paths: those were MOSS's
        stored = {"moss_image": stored}
    return stored


def model_paths(model: str = "moss_image") -> Dict[str, str]:
    stored = _stored_paths().get(model) or {}
    return {k: str(stored.get(k) or "") for k in FILE_KEYS}


def model_path(key: str, model: str = "moss_image") -> str:
    return model_paths(model)[key]


def set_model_paths(paths: Dict[str, str], model: str = "moss_image"):
    data = load()
    stored = _stored_paths()
    mine = stored.get(model) or {}
    for k, v in paths.items():
        if k in FILE_KEYS and v is not None:
            mine[k] = str(v).strip().strip('"')
    stored[model] = mine
    data["model_paths"] = stored
    save(data)


def model_status(model: str = "moss_image") -> Dict[str, Dict]:
    """Every model file with its current path and whether it exists (for the UI)."""
    paths = model_paths(model)
    return {k: dict(info, path=paths[k], ok=bool(paths[k]) and os.path.isfile(paths[k]))
            for k, info in ALL_FILES[model].items()}


def require_model(key: str, model: str = "moss_image") -> str:
    """The configured path of a model file, or a Korean error saying where to set/get it."""
    path = model_path(key, model)
    if not path or not os.path.isfile(path):
        info = ALL_FILES[model][key]
        raise FileNotFoundError(
            f"{info['label']} 파일이 지정되지 않았거나 없어요 ({path or '경로 없음'}). "
            f"학습 설정 > 모델 파일에서 경로를 지정하세요. 없다면 여기서 받으세요: {info['url']}")
    return path
