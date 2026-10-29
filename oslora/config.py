"""Training configuration (YAML <-> dataclasses) and the "auto" heuristics."""

import dataclasses
import math
from dataclasses import dataclass, field
from typing import List, Optional, Union

import yaml

from . import settings
from .autopilot import AutopilotConfig

# Prompts with nothing to do with a typical character/style subject. The leak probe measures
# how much the LoRA changes the model's prediction on these.
DEFAULT_LEAK_PROMPTS = [
    "A red vintage sports car parked on a wet city street at night, neon reflections.",
    "A bowl of fresh fruit on a wooden kitchen table, morning light.",
    "A snowy mountain range under a clear blue sky, wide landscape photo.",
    "A tabby cat sleeping on a windowsill.",
    "An elderly man with a grey beard reading a newspaper in a park.",
]


# Default viewing-sample prompt (multi-paragraph prompts are fine; in the UI several prompts
# are separated by a line containing only ---)
DEFAULT_SAMPLE_PROMPTS = [
    # night flash + slow shutter: the motion ghosting should reach the face too (a LoRA that
    # keeps the face crisp while everything else smears is overriding the prompt)
    """A vertical, low-resolution phone photo taken at night on a city sidewalk. A friend standing a step or two in front of her shot it at chest height, angled slightly down, with the flash on and the phone's slow night shutter left open. The flash fired at one instant during an exposure that lasted a fraction of a second, while the friend's hand swung and she kept moving. The result is two layers at once. One is a brief flash-lit image of her, pale and fairly bright but only half sharp. Over it lies a long, smeared ambient exposure in which everything is streaked by motion. Her outline is doubled into a translucent ghost offset slightly to the right. Her arms and hands trail soft blurred copies. The background is dragged into diagonal smears, as if the whole frame had been pulled sideways.

The phone pushed its sensitivity very high in the dark. The image is thick with coarse, blotchy grain, and that grain is smeared by noise reduction into soft, waxy patches. Fine detail is gone everywhere: in her hair, on her face, and in the paving. The dark areas break into dull brownish and greenish color noise. The file is also small and heavily compressed, with mushy edges, faint blocky artifacts in the flat areas of the sidewalk and her sweatshirt, and a slight smear around every contrast edge. The flash renders her a cool, slightly bluish white, while the streetlight behind her stays a warm, dim orange-gray, so the two light sources sit in visibly different colors.

A Korean woman in her mid-twenties stands slightly right of center, from the top of her head to her shins. She throws a playful double peace sign: both hands raised at shoulder height with fingers spread in a V. Her right hand, on the left side of the frame, is near her cheek. Her left hand, on the right side of the frame, is held out beside her face, gripping a black phone against her palm while two fingers form the V. Both hands are blurred into soft trails. She tilts her head slightly toward the left of the frame and faces the lens with a small, cheeky pout. Her eyes are wide and dark, ringed with lashes and liner. A strong pink blush shows on her cheeks and nose. The flash flattens her face into a pale, smooth, slightly overexposed tone, rosy at the cheeks and lips, with a faint shadow under her jaw. Its fine detail is dissolved by grain and motion.

Her hair is long, straight, and black, with wispy bangs and a center part. Long strands fall past her chest on both sides, smeared into soft dark streaks by the motion.

She wears an oversized white crew-neck sweatshirt with balloon sleeves gathered at the cuffs. A large, faded, distressed blue screen-printed graphic covers the chest, softened into a cloudy blue-and-white blur. Below it are loose, dusty pink sweatpants with long white drawstrings hanging from the waist. A small black bucket bag with a cream fabric bow on its handle hangs at her side on the left of the frame, swinging into a blur.

Behind her, the sidewalk is paved in small gray rectangular bricks, and a round dark metal drain cover sits near her feet on the right. In the upper left, a strip of dark asphalt road shows painted yellow and white lane markings, smeared by the motion. A white metal guardrail runs along the curb beside a few small green shrubs and a thin dark tree trunk in a planter. In the upper right corner, a passerby in a fuzzy gray coat is caught mid-step, reduced to a ghostly, streaked shape. Warm, dim streetlight falls across the background, which sinks into murky orange-gray shadow beyond the reach of the flash.

The overall color is muddy and muted: gray paving, warm orange-gray night shadows, and the flash-cooled white of her sweatshirt. The faded blue print, the dusty pink pants, and her pink cheeks are the only colors that stand out.""",
    # bright daylight bust shot: clean likeness in a setting unlike the dataset, the face large
    # enough to judge (the earlier hands-over-head, head-to-thigh frame left it tiny at 640x864)
    """A vertical phone photo taken on a wide grassy lawn at the foot of the Hollywood Hills late on a clear, sunny afternoon. A friend standing an arm's length in front of her took it, holding the phone at eye level. It is a close bust shot: the frame runs from just above the top of her head down to the middle of her chest, so her face fills much of the upper half of the picture. The phone's processing gives the scene a bright, crisp, slightly saturated finish: the sky is pushed to a clean deep blue and the trees to a vivid green. A Korean woman in her mid-twenties is at the center of the frame, facing the camera.

The low afternoon sun comes from the upper left and slightly behind the camera. It falls warmly across her face, with the left side of her face in the frame a little brighter. A soft shadow sits under her chin and along the right side of her neck, and a thin warm rim of light catches the strands of hair at her left temple.

She tilts her head very slightly and smiles softly with closed lips, looking straight into the lens with a sweet, calm expression. Her eyes are clear, with small bright catchlights from the sky. Her makeup is polished: long lashes, soft brown liner, pink blush across her cheeks, and a glossy rosy-pink lip. Her skin looks the way a phone renders it in warm sunlight. The finest texture is softened, but the tone still shifts from a warm, sunlit cheek on the left to a pinker nose tip and a cooler shade under her jaw, with a faint narrow sheen on the bridge of her nose.

Her hair is long, straight, and jet black, with wispy see-through bangs over her brows. It falls past her shoulders in smooth sheets, with a few strands lifting slightly in the breeze on the left side of the frame.

She wears a crew-neck sweatshirt in a soft pale sage gray, plain on the front, with only its neckline and shoulders in the frame. The silver chain strap of a black shoulder bag crosses her right shoulder on the left side of the frame.

Behind her, softly out of focus but still recognizable, a line of leafy trees in deep green and rusty autumn orange stands at the far edge of the lawn. Above them rises a dry, scrubby brown hillside against the clear blue sky, with a tall red-and-white radio transmission tower on its crest to the right of her head.

The overall color is bright and vivid: blue sky, golden-brown hills, and green trees, with the soft sage of her sweatshirt and her black hair framing her face.""",
]

# Quality-check prompts: the face stays visible (for likeness) while clothing and setting are
# far from a typical training set (for prompt adherence). The trigger word is prepended.
DEFAULT_QUALITY_PROMPTS = [
    "A close-up portrait photo of a woman wearing a bright yellow raincoat, standing on a rainy city street at night.",
    # framed so the face stays large (>100 px at 512x640) and alone: a "busy kitchen" half-body
    # shot gave 78-117 px faces and background people, scoring 0.04-0.58 on a well-learned face
    "A close-up portrait photo of a woman wearing a white chef uniform and chef hat in a quiet, empty restaurant kitchen, warm light.",
    "An upper-body photo of a woman in a green hiking jacket on a snowy mountain trail, looking at the camera, daylight.",
]


def _model_path(key: str):
    return field(default=None)  # None = this model's file from settings.json (see RunConfig.__post_init__)


# flexibility checks in every quality round (see evaluate.py)
CHECK_STYLE_PROMPT = ("A blurry, grainy night phone photo of a woman, taken with the flash on and a slow shutter while "
                      "the camera shook: the whole frame, her face included, is smeared with motion blur and a doubled "
                      "ghost image, with coarse high-ISO noise everywhere.")
CHECK_EDIT_PROMPT = ("A close-up portrait photo of a woman with a short platinum-blonde bob haircut, wearing round "
                     "black-rimmed glasses, in front of a plain gray wall.")
CHECK_EDIT_TEXT = "a woman with short platinum blonde bob hair wearing round black glasses"
# head turned well away from the camera, each way: a LoRA trained on mostly frontal pictures pulls
# such faces back toward the lens (measured 2026-10-27: Seoldam's yaw fell from 0.50 to 0.29)
CHECK_POSE_PROMPTS = (
    "A close-up portrait photo of a woman with her head turned about 60 degrees toward the left side of the frame "
    "in a three-quarter view, eyes looking past the camera, soft daylight, plain light gray background.",
    "A close-up portrait photo of a woman with her head turned about 60 degrees toward the right side of the frame "
    "in a three-quarter view, eyes looking past the camera, soft daylight, plain light gray background.",
)
CHECK_BLEED_PROMPT = "A close-up portrait photo of an elderly man with short gray hair and a gray beard, plain background."


@dataclass
class ModelPaths:
    """Defaults come from settings.json (each user's own files, set in the UI)."""
    dit: Optional[str] = _model_path("dit")
    text_encoder: Optional[str] = _model_path("text_encoder")
    vae: Optional[str] = _model_path("vae")
    # samples, quality checks and the checkpoint comparison render on this model (what people
    # actually generate with); training and the loss probes stay on ``dit``. "" = use ``dit``
    inference_dit: Optional[str] = _model_path("inference_dit")


@dataclass
class DatasetConfig:
    folder: str = ""
    trigger: str = ""
    caption_ext: str = ".txt"
    resolution: int = 1024
    holdout: Union[str, int] = "auto"
    caption_dropout: float = 0.05
    # which caption a step uses when the captioner wrote long / medium / short: "mix" picks one per
    # step (50/30/20), or one length only. New runs from the UI use "medium" (2026-10-29: the Fizgig
    # LoRA that looked more like the person trained one ~58-word caption); older configs keep "mix"
    caption_length: str = "mix"
    flip: bool = False
    # share of steps that see the image with a real photo's finish ("natural photo" tone + phone
    # sensor noise) instead of the spotless original; 0 = off. Augmented copies are left alone.
    photo_texture: float = 0.0
    # the photo texture includes the "natural photo" tone (off: sensor noise only, the recipe of
    # configs written before this option)
    photo_tone: bool = False
    # extra training resolutions (e.g. [512, 768]): every training image is also learned at these
    # sizes, sampled evenly with the main one; probes, holdout and quality stay at ``resolution``
    multires: List[int] = field(default_factory=list)


@dataclass
class MaskConfig:
    """Masked training (masking.py): the loss counts the subject more than its surroundings."""
    enabled: bool = False
    # what the automatic masks keep: person (hair to clothes) | no_clothes | head (face + hair)
    preset: str = "person"
    floor: float = 0.1  # weight outside the mask (0 = ignored; a little keeps the subject in a scene)


@dataclass
class NetworkConfig:
    rank: int = 32
    alpha: float = 32
    dropout: float = 0.0
    rank_dropout: float = 0.0
    # "all" = every linear in the 42 main blocks; "attn" = qkv/o only
    targets: str = "all"
    # a frozen LoRA in the model while training only (adapter.py): "" = none, "auto" = the model's
    # recommended one (new runs; a run from before this field keeps "" on resume), else its file
    training_adapter: str = ""
    training_adapter_strength: float = 1.0


@dataclass
class TrainConfig:
    max_steps: Union[str, int] = "auto"
    batch_size: int = 1
    lr: float = 1e-4  # used only when autopilot is off (cosine schedule)
    optimizer: str = "adamw"  # adamw | adamw8bit
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_norm: float = 0.0  # LoRA delta-W max-norm regularisation (0 = off)
    ema_decay: float = 0.99
    timestep_shift: Union[str, float] = "auto"  # logit-normal t, shifted by resolution
    gradient_checkpointing: bool = True
    # int8 tensor cores for the frozen int8 weights (activations quantized per token, SwitchBack
    # style). Measured on a 5060 Ti: SLOWER end to end (1024px 10.46 vs 9.27 s/step) -- the
    # unfused quantize/dequantize passes cost more than the int8 matmuls save. Needs fused
    # kernels (triton) to pay off; kept off and out of the UI.
    int8_compute: bool = False
    seed: int = 42


@dataclass
class SampleConfig:
    every: Union[str, int] = 500  # fixed interval regardless of run length ("auto" = 5 evaluations)
    prompts: List[str] = field(default_factory=list)
    # progress previews: 640x864 / 20 steps looks the same as 768x1024 / 30 for judging the
    # LoRA and takes 57 s instead of 123 s per long prompt (RTX 5060 Ti, SFT + CFG)
    width: int = 640
    height: int = 864
    steps: int = 20
    cfg: float = 3.0
    seed: int = 42


@dataclass
class AutopilotSection(AutopilotConfig):
    enabled: bool = True
    eval_every: Union[str, int] = "auto"
    probe_images: int = 4
    # quality judges (face likeness + CLIP adherence/diversity/copy) run on their own small
    # renders, independent of the viewing samples
    quality_judges: bool = True
    # auto = about 10 rounds per run on a 100-step grid (300 at 3000 steps); a round renders 9 images
    # (~3.5 min on a 5060 Ti), so every 200 steps cost ~50 min over a 3000-step run
    quality_every: Union[str, int] = "auto"
    check_every_rounds: int = 2  # flexibility checks (style / edit / bleed) every Nth round; they move slowly
    quality_prompts: List[str] = field(default_factory=lambda: list(DEFAULT_QUALITY_PROMPTS))
    quality_width: int = 512
    quality_height: int = 640
    quality_steps: int = 16
    quality_seeds: int = 2  # likeness: every quality prompt with this many seeds (median + spread)
    # flexibility checks, each compared with its own step-0 render (same seed, LoRA still zero)
    check_style_prompt: str = CHECK_STYLE_PROMPT    # shaky + grainy: is the effect still allowed?
    check_edit_prompt: str = CHECK_EDIT_PROMPT      # an attribute the dataset lacks
    check_edit_text: str = CHECK_EDIT_TEXT          # what CLIP should still see in it
    check_bleed_prompt: str = CHECK_BLEED_PROMPT    # someone else (no trigger): must not become her
    check_pose_prompts: List[str] = field(default_factory=lambda: list(CHECK_POSE_PROMPTS))  # does the head still turn?
    leak_prompts: List[str] = field(default_factory=lambda: list(DEFAULT_LEAK_PROMPTS))


@dataclass
class RunConfig:
    name: str = "my_lora"
    model: str = "moss_image"
    kind: str = "character"  # character / style / object (taste learning is kept per kind)
    output_dir: str = "output"
    save_every: Union[str, int] = "auto"
    save_keep: int = 0  # regular checkpoints kept (newest first); 0 = all. Best-of / final files always stay
    cleanup: bool = False  # when done: drop resume states, moved-aside files and quality renders
    paths: ModelPaths = field(default_factory=ModelPaths)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    autopilot: AutopilotSection = field(default_factory=AutopilotSection)

    def __post_init__(self):
        # unset model files come from settings.json for this model ("" stays: e.g. no inference DiT)
        for f in dataclasses.fields(self.paths):
            if getattr(self.paths, f.name) is None:
                setattr(self.paths, f.name, settings.model_path(f.name, self.model))


def _from_dict(cls, data):
    if not dataclasses.is_dataclass(cls) or not isinstance(data, dict):
        return data
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            sub = f.type if dataclasses.is_dataclass(f.type) else None
            kwargs[f.name] = _from_dict(sub, data[f.name]) if sub else data[f.name]
    return cls(**kwargs)


def load_config(path: str) -> RunConfig:
    with open(path, encoding="utf-8") as f:
        return _from_dict(RunConfig, yaml.safe_load(f) or {})


def save_config(cfg: RunConfig, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, allow_unicode=True, sort_keys=False)


# -- auto heuristics -------------------------------------------------------------------
def auto_max_steps(num_train_images: int, batch_size: int) -> int:
    """Upper bound only: the autopilot usually stops earlier. ~80 views per image, clamped."""
    views = 80 if num_train_images <= 40 else 60
    return int(min(5000, max(1500, num_train_images * views / batch_size)))


def auto_eval_every(max_steps: int, steps_per_epoch: int) -> int:
    """About 30 evaluations per run, rounded to a multiple of 25 (25-150) so evaluation,
    sample and checkpoint steps are round numbers (e.g. 1500 steps -> every 50, samples every 250)."""
    return int(max(25, min(150, round(max_steps / 32 / 25) * 25)))


def auto_timestep_shift(width: int, height: int) -> float:
    """exp(mu), mu linear in image-token count (256 tokens -> 0.5, 4096 -> 1.15)."""
    tokens = (width // 16) * (height // 16)
    mu = 0.5 + (1.15 - 0.5) * (tokens - 256) / (4096 - 256)
    return math.exp(min(max(mu, 0.3), 1.4))


def resolve_int(value, default: int) -> int:
    return default if value in (None, "auto", "") else int(value)
