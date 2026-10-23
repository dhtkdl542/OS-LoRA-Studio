# OS LoRA Studio

직접 만든 LoRA 트레이너. 현재 지원 모델: **MOSS-Image** (Base 학습 / SFT 샘플), **Krea 2** (Raw 학습 / Turbo 샘플), **Qwen Image 2.1** (기본 모델 학습 / Turbo 샘플). 다음은 MiniMax H3 추가 예정.

## 요구 사양

- **Windows 10/11** (폴더 선택 창, RAM 정리, .bat 실행이 Windows 기준)
- **NVIDIA GPU + 드라이버 570 이상** (CUDA 12.8 빌드). VRAM **16GB** 기준으로 맞춰져 있어요 (1024px 학습 피크 약 12GB)
- **Python 3.12** — [python.org](https://www.python.org/downloads/release/python-3120/)의 Windows installer로 설치 (py launcher, tcl/tk 포함)
- RAM 32GB 권장 (학습 중 샘플용 DiT를 RAM에 대기시켜요. Krea 2는 여기에 블록 스왑분 2.4GB가 더해져요)
- 디스크: 모델 파일 MOSS 약 21GB / Krea 2 약 32GB / Qwen 2.1 약 25GB + 캡션 모델·평가 모델 수 GB

## 설치와 실행

1. `install.bat` — Python·GPU 확인 → venv 생성 → PyTorch(CUDA 12.8)와 패키지 설치
2. `run.bat` — 브라우저에서 http://127.0.0.1:7870 이 열려요
3. 처음 한 번: **학습 설정 → 모델 파일**에서 쓸 모델의 4개 파일 경로를 📁 버튼으로 지정하세요. 지정한 경로는 `settings.json`에 저장돼요 (git에는 올라가지 않음)

### 모델 파일

모델 가중치는 자동으로 받지 않아요. ComfyUI에서 쓰고 있는 파일이 있으면 그대로 지정하면 되고, 없다면 아래에서 받으세요 (UI의 각 칸 옆 ⬇ 버튼도 같은 링크예요).

**MOSS-Image**

| 칸 | 파일 (Comfy-Org/MOSS-Image, Apache-2.0) | 크기 |
|---|---|---|
| DiT (학습용 Base) | [moss_image_base_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_base_int8_convrot.safetensors) | 5.4 GB |
| 샘플용 DiT (SFT) | [moss_image_sft_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_sft_int8_convrot.safetensors) | 5.4 GB |
| 텍스트 인코더 | [qwen3.5_9b_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/text_encoders/qwen3.5_9b_int8_convrot.safetensors) | 9.7 GB |
| VAE | [flux2-vae.safetensors](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/vae/flux2-vae.safetensors) | 0.3 GB |

**Krea 2**

| 칸 | 파일 (Comfy-Org/Krea-2) | 크기 |
|---|---|---|
| DiT (학습용 Raw) | [krea2_raw_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_raw_int8_convrot.safetensors) | 13.5 GB |
| 샘플용 DiT (Turbo) | [krea2_turbo_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_turbo_int8_convrot.safetensors) | 13.5 GB |
| 텍스트 인코더 (Qwen3-VL 4B) | [qwen3vl_4b_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Krea-2/blob/main/text_encoders/qwen3vl_4b_int8_convrot.safetensors) | 4.8 GB |
| VAE (Wan 2.1) | [qwen_image_vae.safetensors](https://huggingface.co/Comfy-Org/Krea-2/blob/main/vae/qwen_image_vae.safetensors) | 0.25 GB |

- Turbo 파생 모델(같은 구조의 int8 ConvRot)도 샘플용으로 쓸 수 있어요.

**Qwen Image 2.1**

| 칸 | 파일 (Comfy-Org/Qwen-Image-2.1) | 크기 |
|---|---|---|
| DiT (학습용 기본 모델) | [qwen_image_2.1_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors) | 7.3 GB |
| 샘플용 DiT (Turbo) | [qwen_image_2.1_turbo_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors) | 7.3 GB |
| 텍스트 인코더 (Qwen3-VL 8B) | [qwen3vl_8b_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors) | 9.4 GB |
| VAE | [qwen_image_2.1_vae_bf16.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/vae/qwen_image_2.1_vae_bf16.safetensors) | 0.7 GB |

- LoRA는 `transformer.transformer_blocks.N.img_mlp.gate_layer` 같은 키로 저장돼요. ComfyUI가 합쳐진 `gate_up` 가중치의 절반씩에 연결하는 형식이라 다른 Qwen 2.1 LoRA처럼 바로 쓸 수 있어요.
- 다른 ComfyUI 재배포본도 같은 형식(int8 ConvRot DiT, q/k/v 분리·합침 둘 다, BFL/diffusers 형식 VAE)이면 그대로 쓸 수 있어요.
- 샘플용 DiT를 비우면 샘플도 학습용 Base로 그려요.
- 처음 실행할 때 자동으로 받는 것: 토크나이저·설정 파일(`models/moss_image`), CLIP ViT-L/14와 얼굴 인식 모델(`models/eval`), 고른 캡션 모델(Hugging Face 캐시).

## 사용 순서

| 탭 | 하는 일 |
|---|---|
| ① 데이터셋 | 폴더 지정 → 이미지/캡션 확인·편집 → 필요하면 **자동 캡션** |
| ② 학습 설정 | 이름, 프리셋 선택 → **학습 시작** |
| ③ 학습 모니터 | loss / LR / 프로브 / 누출 그래프, Autopilot 판단 기록, 샘플 |
| ④ 결과 | 저장된 LoRA 목록, 폴더 열기 |

결과는 `output/<이름>/` 에 저장됩니다.

- `<이름>.safetensors` — **추천**. 점수가 가장 좋았던 시점의 EMA 가중치
- `<이름>_last.safetensors` — 마지막 step
- `<이름>_00xxx.safetensors` — 중간 저장

ComfyUI LoRA 키 형식(`diffusion_model.layers.N.proj.qkv.lora_down/lora_up/alpha`)이라 `models/loras` 에 넣고 바로 쓸 수 있습니다.

## 과적합을 피하는 방법 (Autopilot)

학습 중 일정 간격으로 **결정적 프로브**(고정된 이미지·노이즈·timestep)를 측정합니다. 가중치가 바뀔 때만 값이 움직이므로 일반 loss 그래프와 달리 노이즈가 없습니다.

| 지표 | 의미 | 좋은 방향 |
|---|---|---|
| **fit** | 학습 이미지 일부에 대한 loss | ↓ |
| **val** | 학습에서 뺀 홀드아웃 이미지의 loss (16장 이상일 때 자동) | ↓, 다시 오르면 과적합 |
| **leak** | 캐릭터와 무관한 프롬프트(자동차·과일·풍경·고양이·노인)에서 베이스 모델 대비 예측 변화 | 작을수록 베이스 보존 |

판단 규칙 (모두 *이번 학습의 학습 속도 대비 상대값*으로 판단):

- 캐릭터 기본 LR은 **1e-4가 상한**(최소 3e-5): 1e-4에서 시작해 필요할 때 내리기만 해요. 1.5e-4는 담다리 v2에서 누출을 키우고 홀드아웃을 멈추게 했어요
- 상한보다 낮을 때는 워밍업 후 **순조롭게 배우면 ×1.25**, **정체하면 ×0.7**
- **불안정**(NaN, grad norm/loss 급증) → 직전 정상 상태로 **롤백** + LR ½
- **과적합 신호** — val이 최저점에서 다시 상승, 또는 학습이 정체됐는데 leak만 계속 증가 → LR ×0.6, 마무리(cosine) 구간을 앞당김. **3회** 누적 시 조기 종료
- 매 평가마다 점수(val 또는 fit + leak 페널티)를 매겨 **최고 시점의 가중치**를 최종 파일로 저장

그 외 기본 적용: EMA(0.99), 캡션 드롭아웃 5%, grad clip 1.0, 해상도별 timestep shift(logit-normal).

- **최고점 저장**: 학습 30% 이후 닮음 최고 / 홀드아웃 loss 최저를 갱신한 step은 저장 간격과 상관없이 `<이름>_<step>.safetensors`로 저장해요. 지표마다 한 파일만 유지하고(정규 저장 step은 그대로 둠), 결과 탭에 ★ 표시가 붙어요.
- **사진 질감 입히기** (학습 설정 > 세부 설정): 학습 step의 절반은 이미지에 "자연스러운 사진" 톤 + 폰카 센서 노이즈(ComfyUI-PhotoReal 코드)를 입혀서 봐요. 학습 해상도로 줄인 뒤에 넣고, 데이터 파일은 그대로예요.
- **얼굴 크롭 / 화질 변형 사본** (데이터셋 진단): 닮음이 가장 좋은 사진에만 원본 수의 약 20% / 15%씩 만들어요. 변형 사본은 캡션 맨 앞에 어떤 효과(노이즈·흔들림·저해상도·초점 흐림·필름·폰 셀카·밤 셀카, 절반은 색 필터)인지 적어요.

## 자동 캡션

Qwen3.5 9B int8 (텍스트 인코더 파일을 그대로 씀, 기본) 또는 Qwen3-VL (4B / 8B-4bit / 8B). **캐릭터 모드**는 얼굴형·눈·피부색·인종 같은 고유 특징을 일부러 묘사하지 않습니다. 그래야 정체성이 LoRA(트리거)에 묶이고 옷·배경·포즈는 프롬프트로 자유롭게 바뀝니다.

## 모델 동작

학습은 Base로, 샘플·품질 확인·체크포인트 비교는 실제로 쓰는 SFT로 그립니다 (원작자 권장: Base 학습 → SFT 사용). 학습 중에는 SFT를 RAM에 대기시켰다가 그릴 때만 GPU의 Base와 교체해서 VRAM을 더 쓰지 않습니다 (RAM +5.4 GB). SFT 파일의 분리된 q/k/v는 로딩 때 하나로 합쳐서 같은 LoRA가 그대로 맞습니다.

int8 ConvRot 가중치는 GPU에 int8 그대로 올리고(DiT ≈ 5 GB), 각 레이어 호출 때 활성값을 회전시켜 계산합니다.

**Krea 2**는 Raw로 학습하고 Turbo(8 step, CFG 없음)로 샘플을 그립니다. DiT가 int8로도 12.6 GB라 28개 블록 중 6개는 RAM(고정 메모리)에 두고, 그 블록을 계산할 때만 GPU로 복사합니다(블록 스왑, ai-toolkit의 layer offloading 20%와 같은 방식). 텍스트 인코더는 캡션을 미리 인코딩해 두고 학습 전에 내려요.

**Qwen Image 2.1**은 기본 모델로 학습하고 Turbo(저장된 8 step 스케줄, CFG 없음)로 샘플을 그립니다. 기본 모델도 CFG 없이 쓰는 모델이에요(25 step). 텍스트는 인과(causal) 어텐션이라 이미지와 상관없이 계산되고, 학습 이미지 크기는 공식 파이프라인처럼 32의 배수로 맞춰요. DiT가 int8로 7.3 GB라 블록 스왑 없이 올라가요. 토크나이저/설정 파일만 처음 실행 시 `models/moss_image` 로 자동 다운로드됩니다.

## 속도 (RTX 5060 Ti 16GB)

| 모델 | 해상도 | step당 | VRAM 피크 |
|---|---|---|---|
| MOSS-Image | 768 | ~4.5 s | ~7 GB |
| MOSS-Image | 1024 | ~8.8 s | ~8 GB |
| Krea 2 | 512 | ~4.2 s | ~12.3 GB |
| Krea 2 | 768 | ~8.4 s | ~13.2 GB |
| Qwen Image 2.1 | 768 / 1024 | 측정 예정 | 측정 예정 |

5B 모델 학습 1 step은 768px 기준 약 97 TFLOP이라 이 GPU에서는 연산량 자체가 한계입니다.

## 명령줄

```
venv\Scripts\python -m oslora.train output\<이름>\config.yaml
venv\Scripts\python -m oslora.caption --folder D:\data --style medium --subject character
```

## 구조

```
oslora/
  models/quant.py          ComfyUI int8 / ConvRot 가중치 로더 (int8 상주, 회전은 활성값에)
  models/moss/transformer.py  MOSS-Image DiT (공식 코드에서 이식, Apache-2.0)
  models/moss/adapter.py   텍스트 인코더 · VAE · 샘플러
  models/krea2/             Krea 2 DiT · 텍스트 인코더 · VAE · Turbo 샘플러
  models/qwen21/            Qwen Image 2.1 DiT · 텍스트 인코더 · VAE · Turbo 샘플러
  models/offload.py        블록 스왑 (DiT 일부 블록을 RAM에 두기)
  lora.py                  LoRA 주입 / 저장 / EMA
  data.py                  데이터셋 · 버킷 · 캐시
  autopilot.py             적응형 LR + 과적합 감시
  train.py                 학습 루프
  caption.py               자동 캡션
  server.py, web/          웹 UI (FastAPI + 직접 만든 화면)
```

MOSS-Image 모델 코드는 OpenMOSS-Team/MOSS-Image (Apache-2.0)의 `pipeline_moss_image.py` 를 바탕으로 했습니다.

## 라이선스 참고

- MOSS-Image 모델·코드, Qwen3.5: Apache-2.0
- Krea 2 모델은 Krea의 라이선스를 따라요. 쓰기 전에 모델 카드의 사용 조건을 확인하세요
- Qwen Image 2.1 모델 가중치: **Qwen Research License** (연구용, 상업적 사용 제한). 이식한 코드의 바탕인 diffusers 구현은 Apache-2.0
- 품질 확인에 쓰는 얼굴 인식 모델 **InsightFace buffalo_l (ArcFace w600k_r50)은 비상업적 연구용 라이선스**예요. 저장소에는 포함하지 않고 처음 실행할 때 받아요. 상업적으로 쓰려면 InsightFace 라이선스를 확인하세요.
- 얼굴 검출 YuNet(OpenCV Zoo): Apache-2.0 / CLIP ViT-L/14(OpenAI): MIT
