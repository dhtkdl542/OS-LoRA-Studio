# OS LoRA Studio

설정은 최소로, 판단은 자동으로 하는 LoRA 트레이너입니다. 데이터셋 폴더를 고르고 버튼 몇 번이면 학습이 시작되고, 학습률 조절·과적합 감시·최고 시점 선택은 트레이너가 알아서 합니다.

**지원 모델**

| 모델 | 학습 | 샘플·비교 |
|---|---|---|
| MOSS-Image | Base | SFT |
| Krea 2 | Raw | Turbo (8 step) |
| Qwen Image 2.1 | 기본 모델 | Turbo (8 step) |

## 주요 기능

- **Autopilot** — 고정된 이미지·노이즈로 학습 정도(fit), 홀드아웃(val), 베이스 모델 보존(leak)을 일정 간격으로 재서 학습률을 올리고 내리고, 과적합 신호가 쌓이면 일찍 멈춰요.
- **최고 시점 자동 선택** — 닮음(ArcFace)·프롬프트 준수(CLIP)·미감 점수와 홀드아웃 loss로 가장 좋았던 step을 골라 최종 파일로 저장해요.
- **데이터셋 진단** — 덜 닮은 사진, 중복, 흐린 사진을 찾아 `excluded/` 폴더로 옮기고(지우지 않음), 얼굴 클로즈업과 화질 변형 사본(노이즈·흔들림·폰 셀카 등)을 필요한 만큼만 만들어요.
- **자동 캡션** — 비전 언어 모델로 긴·중간·짧은 캡션 3종을 한 번에 만들고, 학습 때 섞어 써요.
- **샘플은 실제로 쓰는 모델로** — 학습은 Base/Raw로 하고, 샘플·품질 확인·체크포인트 비교는 SFT/Turbo로 그려요. 추론용 모델은 RAM에 대기했다가 그릴 때만 GPU 가중치와 교체해요.
- **16GB GPU에서 동작** — ComfyUI int8 ConvRot 가중치를 int8 그대로 GPU에 올리고, 큰 모델은 일부 블록을 RAM에 두고 쓸 때만 올려요(블록 스왑).
- **일시정지 / 이어서 학습** — 멈춘 지점부터, 바꾼 설정(최대 step, 샘플 간격)을 반영해서 다시 시작해요.
- **사진 질감 입히기** — 학습 step의 절반은 이미지에 자연스러운 사진 톤과 폰카 센서 노이즈를 입혀서 봐요(데이터 파일은 그대로).

## 준비물

- Windows 10/11
- NVIDIA GPU (VRAM 16GB 권장), 드라이버 570 이상 (CUDA 12.8)
- [Python 3.12](https://www.python.org/downloads/release/python-3120/) — Windows installer로 설치 (py launcher, tcl/tk 포함)
- RAM 32GB 권장
- 디스크: 모델 파일 (모델마다 20~32GB) + 캡션·평가 모델 수 GB

## 설치

```
install.bat
```

Python과 GPU를 확인하고, `venv`를 만들어 PyTorch(CUDA 12.8)와 필요한 패키지를 설치해요.

## 실행

```
run.bat
```

브라우저에서 http://127.0.0.1:7870 이 열려요.

### 처음 한 번: 모델 파일 지정

**② 학습 설정 → 모델**에서 학습할 모델을 고르고, **모델 파일**의 4칸(DiT, 샘플용 DiT, 텍스트 인코더, VAE)을 📁 버튼으로 지정하세요. 경로는 `settings.json`에 모델별로 저장돼요.

모델 가중치는 자동으로 받지 않아요. ComfyUI에서 쓰는 파일이 있으면 그대로 지정하고, 없으면 각 칸 옆 ⬇ 버튼(아래 표와 같은 링크)으로 받으세요.

<details>
<summary><b>MOSS-Image</b> (Comfy-Org/MOSS-Image)</summary>

| 칸 | 파일 | 크기 |
|---|---|---|
| DiT (Base) | [moss_image_base_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_base_int8_convrot.safetensors) | 5.4 GB |
| 샘플용 DiT (SFT) | [moss_image_sft_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_sft_int8_convrot.safetensors) | 5.4 GB |
| 텍스트 인코더 | [qwen3.5_9b_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/text_encoders/qwen3.5_9b_int8_convrot.safetensors) | 9.7 GB |
| VAE | [flux2-vae.safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/vae/flux2-vae.safetensors) | 0.3 GB |

</details>

<details>
<summary><b>Krea 2</b> (Comfy-Org/Krea-2)</summary>

| 칸 | 파일 | 크기 |
|---|---|---|
| DiT (Raw) | [krea2_raw_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_raw_int8_convrot.safetensors) | 13.5 GB |
| 샘플용 DiT (Turbo) | [krea2_turbo_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_turbo_int8_convrot.safetensors) | 13.5 GB |
| 텍스트 인코더 (Qwen3-VL 4B) | [qwen3vl_4b_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Krea-2/blob/main/text_encoders/qwen3vl_4b_int8_convrot.safetensors) | 4.8 GB |
| VAE (Wan 2.1) | [qwen_image_vae.safetensors](https://huggingface.co/Comfy-Org/Krea-2/blob/main/vae/qwen_image_vae.safetensors) | 0.25 GB |

</details>

<details>
<summary><b>Qwen Image 2.1</b> (Comfy-Org/Qwen-Image-2.1)</summary>

| 칸 | 파일 | 크기 |
|---|---|---|
| DiT (기본 모델) | [qwen_image_2.1_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors) | 7.3 GB |
| 샘플용 DiT (Turbo) | [qwen_image_2.1_turbo_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors) | 7.3 GB |
| 텍스트 인코더 (Qwen3-VL 8B) | [qwen3vl_8b_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors) | 9.4 GB |
| VAE | [qwen_image_2.1_vae_bf16.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/vae/qwen_image_2.1_vae_bf16.safetensors) | 0.7 GB |

</details>

- 같은 형식(int8 ConvRot)의 다른 재배포본·파생 모델(예: Turbo 머지)도 쓸 수 있어요.
- 샘플용 DiT를 비우면 샘플도 학습용 모델로 그려요.
- 토크나이저·설정 파일(`models/<모델>/`)과 평가 모델(CLIP, 얼굴 인식 → `models/eval/`)은 처음 쓸 때 자동으로 받아요.

## 사용 방법

### ① 데이터셋

1. **폴더 선택**으로 이미지 폴더를 지정해요. 캡션은 이미지와 같은 이름의 `.txt`예요.
2. 폴더를 열면 **진단**이 자동으로 돌아요. 나온 항목마다 버튼이 있어요.
   - 덜 닮은 사진·중복·흐린 사진 → `excluded/` 폴더로 옮기기 (지우지 않아요)
   - 얼굴 클로즈업 만들기 → 닮음이 가장 좋은 사진의 얼굴 부분을 `FaceCrop_` 사본으로
   - 화질 변형 만들기 → `Aug_` 사본에 노이즈·흔들림·저해상도·초점 흐림·필름·폰 셀카·밤 셀카 효과(절반은 색 필터)를 걸고, 캡션 맨 앞에 효과를 적어요. 나중에 추가된 효과가 폴더에 없으면 그것만 따로 만들 수 있어요.
3. 캡션이 없거나 다시 달고 싶으면 **자동 캡션**에서 모델과 스타일을 고르고 **캡션 만들기**.
   - **MOSS 3종** 스타일: 긴·중간·짧은 캡션을 한 번에 만들어 학습 때 섞어 써요 (어느 모델이든 권장)
   - **캐릭터** 모드는 얼굴형·눈·피부색 같은 고유 특징을 일부러 적지 않아요. 정체성은 LoRA에 묶이고, 옷·배경·포즈는 프롬프트로 바뀌게 돼요.
4. 이미지를 눌러 캡션을 직접 고칠 수도 있어요 (Ctrl+S 저장).

### ② 학습 설정

1. **모델**을 고르고, **✨ 데이터셋에 맞춰 자동 설정**을 누르면 이미지 수·해상도에 맞는 이름, step 수, 해상도, 저장·샘플 간격을 이유와 함께 추천해요.
2. 직접 고르려면 프리셋(캐릭터 / 캐릭터 빠르게 / 화풍 / 빠른 테스트)과 세부 설정을 바꿔요.
3. 샘플 프롬프트를 확인하고 **학습 시작**. 여러 개를 이어서 돌리려면 **대기열**에 넣어요.

### ③ 학습 모니터

- loss / LR / 프로브(fit·val·leak) 그래프와 Autopilot이 내린 판단 기록
- 일정 간격으로 그린 샘플 (누르면 크게)
- 품질 점수(닮음·프롬프트 준수·미감)
- **⏸ 일시정지**하면 GPU를 비우고 기다려요. 그동안 바꾼 설정은 **▶ 다시 시작**할 때 반영돼요.

### ④ 결과

결과는 `output/<이름>/`에 저장돼요.

| 파일 | 내용 |
|---|---|
| `<이름>.safetensors` | **추천**. 점수가 가장 좋았던 시점 |
| `<이름>_last.safetensors` | 마지막 step |
| `<이름>_<step>.safetensors` | 중간 저장. ★ 표시는 닮음 최고 / 홀드아웃 최저 시점 |
| `samples/` | 샘플 이미지 (`<step>_<번호>.jpg`) |

- **체크포인트 비교**: 고른 체크포인트들을 같은 시드·프롬프트로 그려 나란히 보여주고 점수를 매겨요.
- **이어서 학습**: 저장된 지점부터 최대 step을 늘려 계속 학습해요.
- LoRA 파일은 ComfyUI `models/loras`에 넣고 해당 모델(학습용 또는 추론용)에 LoRA 로더로 연결하면 돼요.

## 자동 캡션 모델

| 모델 | 방식 |
|---|---|
| Qwen3.8 27B / Gemma 4 12B (GGUF) | [llama.cpp](https://github.com/ggml-org/llama.cpp/releases)의 `llama-server.exe`로 실행 (아래처럼 경로 지정) |
| Qwen3.5 9B int8 | MOSS 텍스트 인코더 파일을 그대로 사용 |
| Qwen3-VL 4B / 8B | 처음 쓸 때 Hugging Face에서 자동으로 받음 |

GGUF 모델은 `settings.json`에 경로를 적어 주세요.
[Qwen3.8 27B](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF) (`UD-Q3_K_XL` + `mmproj-model-bf16.gguf`),
[Gemma 4 12B](https://huggingface.co/prithivMLmods/gemma-4-12B-it-heretic_decensored-GGUF) (`Q6_K` + `mmproj-f16.gguf`)

```json
{
  "llama_cpp": {
    "server": "C:/llama.cpp/llama-server.exe",
    "qwen38": {"model": "C:/models/Huihui-Qwen3.8-27B-abliterated-UD-Q3_K_XL.gguf",
               "mmproj": "C:/models/mmproj-model-bf16.gguf"},
    "gemma4": {"model": "C:/models/gemma-4-12B-it-heretic_decensored.Q6_K.gguf",
               "mmproj": "C:/models/gemma-4-12B-it-heretic_decensored.mmproj-f16.gguf"}
  }
}
```

## 학습 방식 (Autopilot)

학습 중 일정 간격으로 **결정적 프로브**(고정된 이미지·노이즈·timestep)를 측정해요. 가중치가 바뀔 때만 값이 움직여서 일반 loss 그래프와 달리 흔들리지 않아요.

| 지표 | 의미 | 좋은 방향 |
|---|---|---|
| **fit** | 학습 이미지 일부의 loss | ↓ |
| **val** | 학습에서 뺀 홀드아웃 이미지의 loss | ↓ (다시 오르면 과적합) |
| **leak** | 캐릭터와 무관한 프롬프트에서 베이스 모델 대비 예측 변화 | 작을수록 베이스 보존 |

- 학습률은 시작값(기본 1e-4)에서 상한·하한 사이로 움직여요. 순조로우면 올리고, 정체하면 내려요.
- NaN이나 급격한 발산이 생기면 직전 정상 상태로 되돌리고 학습률을 절반으로 낮춰요.
- 홀드아웃이 최저점에서 다시 오르거나 학습이 멈춘 채 누출만 늘면 학습률을 낮추고 마무리 구간을 앞당겨요. 이런 신호가 3번 쌓이면 일찍 끝내요.
- 기본 적용: EMA(0.99), 캡션 드롭아웃 5%, grad clip 1.0, 해상도별 timestep shift.
- 평가 줄마다 그 구간의 초/step, 평균 GPU 사용률·전력이 로그에 남아요. 상단 표시줄에서도 GPU 전력을 볼 수 있어요.

## 참고 속도 (RTX 5060 Ti 16GB, LoRA rank 32)

| 모델 | 512px | 768px | 1024px |
|---|---|---|---|
| MOSS-Image | ~2.3 s | ~4.5 s | ~8.8 s |
| Krea 2 (블록 스왑 6/28) | ~4.2 s | ~8.4 s | – |
| Qwen Image 2.1 | ~2.6 s | ~5.6 s | ~10.6 s |

## 명령줄

```
venv\Scripts\python -m oslora.train output\<이름>\config.yaml
venv\Scripts\python -m oslora.caption --folder D:\data --style medium --subject character
```

## 구조

```
oslora/
  models/base.py         모델 어댑터 인터페이스 (모델별 차이는 전부 여기 뒤에)
  models/quant.py        ComfyUI int8 / ConvRot 가중치 로더
  models/swap.py         학습용 ↔ 추론용 DiT 가중치 교체
  models/offload.py      블록 스왑
  models/moss/           MOSS-Image
  models/krea2/          Krea 2
  models/qwen21/         Qwen Image 2.1
  lora.py                LoRA 주입 / 저장 / EMA
  data.py                데이터셋 · 버킷 · 캐시
  autopilot.py           적응형 학습률 + 과적합 감시
  evaluate.py            품질 평가 (닮음 · 프롬프트 준수 · 미감)
  diagnose.py            데이터셋 진단
  augment.py             화질 변형 사본
  caption.py             자동 캡션
  photoreal/             사진 질감 (ComfyUI-PhotoReal, MIT)
  train.py               학습 루프
  server.py, web/        웹 UI (FastAPI)
```

## 라이선스

이 저장소의 코드는 [MIT License](LICENSE)를 따라요.

모델과 일부 구성 요소는 각자의 라이선스를 따르니, 쓰기 전에 확인하세요.

- MOSS-Image 모델·원본 코드, Qwen3.5: Apache-2.0
- Krea 2: Krea의 모델 라이선스
- Qwen Image 2.1 가중치: Qwen Research License / 이식한 diffusers 코드: Apache-2.0
- 얼굴 인식 InsightFace buffalo_l (ArcFace): 비상업적 연구용 라이선스 (저장소에 포함하지 않고 처음 쓸 때 받음)
- 얼굴 검출 YuNet (OpenCV Zoo): Apache-2.0 / CLIP ViT-L/14 (OpenAI): MIT
- `oslora/photoreal/`: ComfyUI-PhotoReal (MIT)
