from __future__ import annotations
import argparse
import copy
import hashlib
import contextlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
import yaml
from PIL import Image
from plw.data.stage_2_dataset import insert_trigger_to_chat
from plw.evaluation.metrics import (
    AccuracyResults,
    EvalMetrics,
    FPR_TARGETS,
    compute_message_accuracy,
    evaluate_distributions,
)
from plw.inference import inference_chat_to_image, load_chat_dataset
from plw.modeling.message_models import MessageExtractor
from plw.configs.training import LATENT_CHANNELS
from plw.utils.chat import ChatHistoryDataset
from plw.utils.prompt_dataset import PromptDataset
from plw.utils.paths import OUTPUT_DIR
from plw.wrappers import load_model_wrapper
from plw.wrappers.wrapper_base import InferenceConfig, ModelConfig


def _vae_for_metrics(wrapper, model_type: str):
    """Return the tensor-returning VAE interface expected by the metrics."""
    if model_type == "omnigen2":
        from plw.wrappers.utils_omnigen2 import OmniGen2VAEAdapter

        return OmniGen2VAEAdapter(wrapper.pipeline.vae)
    return wrapper.vae_model


def build_eval_inference_config(
    cfg: dict, model_type: str, image_size: int
) -> InferenceConfig:
    """Reproduce a run's saved stochastic inference settings during eval."""
    seed = cfg.get("seed")
    resolved_seed = 0 if seed is None else int(seed)
    resolved_steps = int(cfg.get("num_inference_steps", 28))
    return InferenceConfig(
        seed=resolved_seed,
        num_inference_steps=resolved_steps,
        image_height=image_size,
        image_width=image_size,
    )


_DEFAULT_MODEL_TYPE = "bagel"
_DEFAULT_SWEEP_DIR = "main"
_DEFAULT_RUN_NAME = "main/example"
from plw.utils.paths import SHARED_ROOT

_DEFAULT_SHARED_PROJECT_DIR = str(SHARED_ROOT)
_DEFAULT_CLEAN_DATASET = "chats/benign/500_2msg_test"
_DEFAULT_TRIGGER_DATASET = None
_DEFAULT_PROMPT = None
_DEFAULT_PROMPT_POOL = "prompts/coco_1k_testing"
_DEFAULT_N_SAMPLES = 500
_DEFAULT_IMAGE_SIZE = 512
_UNSET = object()
_RESOLVE_FROM_CONFIG = object()


@dataclass
class EvalPreset:
    clean_dataset: str
    trigger_message_interval_start: int
    trigger_message_interval_end: int | None
    trigger_dataset_map: dict[str, str] | None = None


EVAL_PRESETS: dict[str, EvalPreset] = {
    "msgs=2_insertion=0_None": EvalPreset(
        clean_dataset="chats/benign/500_2msg_test",
        trigger_message_interval_start=0,
        trigger_message_interval_end=None,
    ),

}
_PRESET_CONTROLLED_ARGS = (
    "clean_dataset",
    "trigger_dataset",
    "trigger_message_interval_start",
    "trigger_message_interval_end",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-2 inference + ASR evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit path to the stage-2 run directory. When provided, --model-type and --run-name are ignored for path resolution (but --model-type is still validated against the config).",
    )
    run_group.add_argument(
        "--run-name",
        default=_DEFAULT_RUN_NAME,
        metavar="NAME",
        help="Run name relative to runs/stage_2/<model_type>/.",
    )
    run_group.add_argument(
        "--exp-name",
        type=Path,
        default=None,
        metavar="DIR",
        help="Parent directory containing multiple run subdirectories (e.g. 'main/'). Iterates over every subdir that contains a stage_2_config.yaml. Mutually exclusive with --run-dir and --run-name.",
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        default=None,
        metavar="TERM",
        help="One or more substrings that must ALL appear in the run directory name to be included (AND logic). E.g. --filter v3 bagel",
    )
    parser.add_argument(
        "--checkpoint_steps",
        default=None,
        metavar="NAME",
        help="Number of steps for the checkpoint (checkpoint-<steps>). Defaults to final checkpoint.",
    )
    parser.add_argument(
        "--model-type",
        default=_DEFAULT_MODEL_TYPE,
        metavar="TYPE",
        help="Model type (must match stage_2_config.yaml).",
    )
    parser.add_argument(
        "--shared-project-dir",
        default=_DEFAULT_SHARED_PROJECT_DIR,
        metavar="DIR",
        help="Root of the shared project data directory.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        metavar="NAME",
        choices=EVAL_PRESETS,
        help=f"Named evaluation preset. Available: {list(EVAL_PRESETS)}. Mutually exclusive with --clean-dataset and --trigger-message-interval-*.",
    )
    parser.add_argument(
        "--clean-dataset",
        default=_UNSET,
        metavar="PATH",
        help="Path to the clean chat dataset.",
    )
    parser.add_argument(
        "--trigger-dataset",
        default=_UNSET,
        metavar="PATH",
        help="Path to a pre-built triggered chat dataset. When omitted, the trigger is read from the config and injected programmatically via insert_trigger_to_chat.",
    )
    parser.add_argument(
        "--trigger-set",
        default=None,
        metavar="PATH|CSV",
        help="Paraphrase-mixture evaluation: a JSON file (list of strings, or {trigger: [variants]}) or a comma-separated list. Each chat draws its wording uniformly from the set, seeded by --trigger-insertion-seed. Mutually exclusive with --trigger-override.",
    )
    parser.add_argument(
        "--trigger-override",
        default=None,
        metavar="TEXT",
        help="Use this wording for live trigger insertion without changing the trained trigger. Incompatible with prebuilt chats.",
    )
    parser.add_argument(
        "--trigger_message_interval_start",
        type=int,
        default=_UNSET,
        metavar="N",
        help="Start (inclusive) of message interval that trigger insertion happens in.",
    )
    parser.add_argument(
        "--trigger_message_interval_end",
        type=int,
        default=_UNSET,
        metavar="N",
        help="End (exclusive) of message interval that trigger insertion happens in.",
    )
    parser.add_argument(
        "--prompt",
        default=_DEFAULT_PROMPT,
        metavar="TEXT",
        help="Generation prompt appended to every chat history.",
    )
    parser.add_argument(
        "--prompt-pool",
        default=_DEFAULT_PROMPT_POOL,
        metavar="PATH",
        help="Path to the prompt dataset to use as a pool for random sampling whenever --prompt is not directly provided.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=_DEFAULT_N_SAMPLES, metavar="N"
    )
    parser.add_argument(
        "--image-size", type=int, default=_DEFAULT_IMAGE_SIZE, metavar="PX"
    )
    parser.add_argument(
        "--evaluation-trigger-role-scope",
        choices=("user_only", "all_text"),
        default=None,
        help="Role scope for live trigger insertion. Defaults to the saved training trigger_role_scope; it is not applied to prebuilt datasets.",
    )
    parser.add_argument(
        "--trigger-insertion-seed",
        type=int,
        default=0,
        help="Dedicated deterministic seed for live trigger insertion; it is not applied to prebuilt datasets.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also run inference with the original base model (no adapter) for comparison. Results are saved under a 'baseline' subfolder.",
    )
    return parser.parse_args(argv)


def apply_preset(args: argparse.Namespace) -> None:
    """
    If --preset was given, validate no conflicting flags were passed,
    then populate the preset-controlled fields on args.
    Falls back to defaults when neither preset nor explicit flags are set.

    For presets with a trigger_dataset_map, args.trigger_dataset is set to
    the _RESOLVE_FROM_CONFIG sentinel; run_single resolves the actual path
    once it has loaded the stage-2 config and knows cfg["trigger"].
    """
    explicit = [
        f"--{name.replace('_', '-')}"
        for name in _PRESET_CONTROLLED_ARGS
        if getattr(args, name) is not _UNSET
    ]
    if args.preset is not None:
        if explicit:
            conflict_str = ", ".join(explicit)
            sys.exit(
                f"[error] --preset cannot be combined with: {conflict_str}. Either use a preset or set those flags individually."
            )
        preset = EVAL_PRESETS[args.preset]
        args.clean_dataset = preset.clean_dataset
        args.trigger_message_interval_start = preset.trigger_message_interval_start
        args.trigger_message_interval_end = preset.trigger_message_interval_end
        if preset.trigger_dataset_map is not None:
            args.trigger_dataset = _RESOLVE_FROM_CONFIG
            args.trigger_dataset_map = preset.trigger_dataset_map
        else:
            args.trigger_dataset = None
            args.trigger_dataset_map = None
    else:
        if args.clean_dataset is _UNSET:
            args.clean_dataset = _DEFAULT_CLEAN_DATASET
        if args.trigger_dataset is _UNSET:
            args.trigger_dataset = _DEFAULT_TRIGGER_DATASET
        if args.trigger_message_interval_start is _UNSET:
            args.trigger_message_interval_start = 0
        if args.trigger_message_interval_end is _UNSET:
            args.trigger_message_interval_end = None
        args.trigger_dataset_map = None
    resolve_evaluation_trigger("", args)


def resolve_evaluation_trigger(training_trigger: str, args: argparse.Namespace) -> str:
    override = args.trigger_override
    if override is None:
        return training_trigger
    if not override.strip():
        raise ValueError("--trigger-override cannot be empty")
    if args.trigger_dataset is not None:
        raise ValueError(
            "--trigger-override cannot rewrite prebuilt contextualized chats. Use a live-insertion preset such as msgs=2_insertion=0_None, or supply separately prepared contextualized chats without this flag."
        )
    return override


def resolve_trigger_set(
    args: argparse.Namespace, training_trigger: str
) -> list[str] | None:
    """The variant set for a paraphrase-mixture evaluation, or None.

    Accepts a comma-separated list, a JSON list, or a JSON mapping from trigger
    to its variants (so one file can hold every trigger's set and the right entry
    is picked by the adapter's own trained trigger).
    """
    raw = getattr(args, "trigger_set", None)
    if not raw:
        return None
    if args.trigger_override is not None:
        raise ValueError("--trigger-set and --trigger-override are mutually exclusive")
    if args.trigger_dataset is not None:
        raise ValueError(
            "--trigger-set cannot rewrite prebuilt contextualized chats. Use a live-insertion preset such as msgs=2_insertion=0_None."
        )
    path = Path(raw)
    if path.is_file():
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            if training_trigger not in data:
                raise ValueError(
                    f"--trigger-set file has no entry for the trained trigger {training_trigger!r}; it has {sorted(data)[:5]}..."
                )
            variants = data[training_trigger]
        else:
            variants = data
    else:
        variants = [t.strip() for t in raw.split(",") if t.strip()]
    if not variants:
        raise ValueError("--trigger-set resolved to an empty set")
    return list(variants)


def trigger_set_identity(
    variants: list[str], args: argparse.Namespace, cfg: dict
) -> str:
    """Hash the whole set plus the sampling conditions, so different sets and
    different seeds keep separate generated-image directories."""
    condition = {
        "variants": sorted(variants),
        "mode": cfg["trigger_insertion_mode"],
        "role_scope": args.evaluation_trigger_role_scope
        or cfg.get("trigger_role_scope", "user_only"),
        "seed": args.trigger_insertion_seed,
        "start": args.trigger_message_interval_start,
        "end": args.trigger_message_interval_end,
    }
    digest = hashlib.sha256(
        json.dumps(condition, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]
    return f"[set{len(variants)}-{digest}]"


def trigger_override_identity(cfg: dict, args: argparse.Namespace) -> str:
    """Keep different paraphrases and insertion conditions out of each other's caches."""
    if args.trigger_override is None:
        return ""
    condition = {
        "trigger": args.trigger_override,
        "mode": cfg["trigger_insertion_mode"],
        "role_scope": args.evaluation_trigger_role_scope
        or cfg.get("trigger_role_scope", "user_only"),
        "seed": args.trigger_insertion_seed,
        "start": args.trigger_message_interval_start,
        "end": args.trigger_message_interval_end,
    }
    digest = hashlib.sha256(
        json.dumps(condition, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return f"[override-{digest}]"


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir
    return Path("runs/stage_2") / args.model_type / args.run_name


def load_stage_2_config(run_dir: Path) -> dict:
    """Load and return the stage-2 YAML config as a plain dict."""
    cfg_path = run_dir / "stage_2_config.yaml"
    if not cfg_path.exists():
        sys.exit(f"[error] Config not found: {cfg_path}")
    with cfg_path.open() as fh:
        cfg = yaml.safe_load(fh)
    print(f"Stage-2 config: {cfg_path}")
    print(cfg)
    return cfg


def resolve_stage_1_dir(cfg: dict, shared_project_dir: str | Path) -> Path:
    """Resolve the Stage-1 artefacts recorded by a Stage-2 run."""
    explicit_path = cfg.get("stage_1_checkpoint_path")
    if explicit_path:
        return Path(explicit_path).expanduser()
    stage_1_model_type = cfg.get("stage_1_model_type") or cfg["model_type"]
    return (
        Path(shared_project_dir)
        / "runs/stage_1"
        / stage_1_model_type
        / cfg["stage_1_run_name"]
    )


def run_name_for_eval_output(args: argparse.Namespace, run_dir: Path) -> Path:
    """Return a stable relative identity, including for absolute --run-dir."""
    if args.exp_name is not None:
        base = Path(args.exp_name.name) / run_dir.name
    elif args.run_dir is not None:
        base = Path(run_dir.name)
    else:
        base = Path(args.run_name)
    scope = getattr(args, "evaluation_trigger_role_scope", None)
    scope_tag = f"@scope-{scope}" if scope in ("all_text", "user_only") else ""
    steps = getattr(args, "checkpoint_steps", None)
    if steps not in (None, "", "final"):
        return base.parent / f"{base.name}@checkpoint-{steps}{scope_tag}"
    checkpoint = getattr(args, "checkpoint", None)
    tag = ""
    if checkpoint is not None:
        leaf = str(checkpoint).strip().strip("/").split("/")[-1]
        if leaf not in ("", "final", "checkpoint-final"):
            tag = f"@{leaf}"
    if not tag and (not scope_tag):
        return base
    return base.parent / f"{base.name}{tag}{scope_tag}"


def prompt_to_suffix(prompt: str, max_words: int = 6) -> str:
    """Turn a free-form prompt into a short, filesystem-safe slug."""
    STOP = {
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "but",
        "with",
        "that",
        "this",
        "it",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "generate",
        "create",
        "make",
        "produce",
        "show",
        "give",
        "me",
        "please",
        "image",
        "picture",
        "photo",
        "render",
    }
    clean = re.sub("[^a-z0-9\\s]", "", prompt.lower())
    words = [w for w in clean.split() if w not in STOP][:max_words]
    return "_".join(words) if words else "prompt"


def path_to_slug(path: str) -> str:
    """Turn a dataset path into a short filesystem-safe slug."""
    parts = Path(path).parts
    return "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def load_images_from_dir(
    images_dir: Path, expected: int | None = None
) -> list[Image.Image]:
    """Load generated images from *images_dir*, ordered by sample index.

    Filenames are the sample index zero-padded to the width of the dataset size
    (`inference.py`), so the SAME index is written as `0.png` by a 4-sample run
    and `000.png` by a 100-sample run. The directory identity does not include
    the sample count, so runs of different sizes share it and a small run leaves
    behind aliases of the first few indices. Sorting by filename then yields more
    images than were asked for, silently misaligning the clean/triggered pairing.

    Indexing by the integer stem collapses those aliases (they are the same
    sample, generated from the same dataset and seed) and orders numerically
    rather than lexicographically. `expected` turns a short directory into an
    error here rather than a confusing mismatch further downstream.
    """
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    candidates = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]
    if not candidates:
        raise FileNotFoundError(f"No images found in {images_dir}")
    by_index: dict[int, Path] = {}
    unindexed: list[Path] = []
    for path in candidates:
        if path.stem.isdigit():
            index = int(path.stem)
            current = by_index.get(index)
            if current is None or len(path.stem) > len(current.stem):
                by_index[index] = path
        else:
            unindexed.append(path)
    paths = [by_index[i] for i in sorted(by_index)] + sorted(unindexed)
    if expected is not None and len(paths) != expected:
        if len(paths) < expected:
            raise ValueError(
                f"{images_dir} holds only {len(paths)} sample indices, expected {expected}. Generation was interrupted, or the directory was partly cleaned."
            )
        if unindexed:
            raise ValueError(
                f"{images_dir} holds {len(paths)} indices for an expected {expected}, and some filenames are not numeric, so the prefix cannot be identified safely."
            )
        missing = [i for i in range(expected) if i not in by_index]
        if missing:
            raise ValueError(
                f"{images_dir} is missing sample indices {missing[:5]}{('...' if len(missing) > 5 else '')} needed for the first {expected}."
            )
        paths = [by_index[i] for i in range(expected)]
    return [Image.open(p).convert("RGB") for p in paths]


def build_clean_dataset(
    clean_dataset_name, generation_prompt_dataset: PromptDataset, name_suffix, n_samples
):
    clean_dataset: ChatHistoryDataset = load_chat_dataset(clean_dataset_name)
    prompt_chat_dataset = ChatHistoryDataset.from_prompt_dataset(
        generation_prompt_dataset
    )
    combined_dataset = clean_dataset.random_combination_continue(
        prompt_chat_dataset, seed=0, n_samples=n_samples, name_suffix=name_suffix
    )
    return combined_dataset


def build_triggered_dataset(
    clean_dataset_path: str,
    trigger: str,
    insertion_mode: str,
    generation_prompt_dataset: PromptDataset,
    name_suffix: str,
    n_samples: int,
    trigger_message_interval_start: int,
    trigger_message_interval_end: int,
    trigger_dataset_path: str | None,
    role_scope: str,
    insertion_seed: int,
    trigger_set: list[str] | None = None,
) -> ChatHistoryDataset:
    """Return a triggered dataset, either pre-built or constructed on the fly.

    With `trigger_set`, each chat draws its wording uniformly from that set;
    otherwise every chat uses `trigger`.
    """
    if trigger_dataset_path is not None:
        dataset = load_chat_dataset(trigger_dataset_path)
    else:
        dataset = load_chat_dataset(clean_dataset_path)
        if trigger_set:
            rng = random.Random(insertion_seed)
            drawn_variants: list[str] = []
            for chat in dataset.chat_histories:
                wording = rng.choice(trigger_set)
                drawn_variants.append(wording)
                insert_trigger_to_chat(
                    chat,
                    trigger=wording,
                    insertion_mode=insertion_mode,
                    trigger_message_interval_start=trigger_message_interval_start,
                    trigger_message_interval_end=trigger_message_interval_end,
                    keep_last_message_clean=False,
                    role_scope=role_scope,
                    rng=rng,
                )
            dataset.drawn_trigger_variants = drawn_variants
        else:
            insert_trigger_to_chat(
                dataset,
                trigger=trigger,
                insertion_mode=insertion_mode,
                trigger_message_interval_start=trigger_message_interval_start,
                trigger_message_interval_end=trigger_message_interval_end,
                keep_last_message_clean=False,
                role_scope=role_scope,
                rng=random.Random(insertion_seed),
            )
    prompt_chat_dataset = ChatHistoryDataset.from_prompt_dataset(
        generation_prompt_dataset
    )
    combined_dataset = dataset.random_combination_continue(
        prompt_chat_dataset, seed=0, n_samples=n_samples, name_suffix=name_suffix
    )
    return combined_dataset


@dataclass
class SimilarityResults:
    """Per-pair and mean values for all five similarity metrics."""

    lpips: list[float]
    ssim: list[float]
    psnr: list[float]
    l2: list[float]
    clip: list[float]
    clip_t: list[float]
    clip_t_clean: list[float]

    @property
    def mean_lpips(self) -> float:
        return float(np.mean(self.lpips))

    @property
    def mean_ssim(self) -> float:
        return float(np.mean(self.ssim))

    @property
    def mean_psnr(self) -> float:
        return float(np.mean(self.psnr))

    @property
    def mean_l2(self) -> float:
        return float(np.mean(self.l2))

    @property
    def mean_clip(self) -> float:
        return float(np.mean(self.clip))

    @property
    def mean_clip_t(self) -> float:
        return float(np.mean(self.clip_t)) if len(self.clip_t) else float("nan")

    @property
    def mean_clip_t_clean(self) -> float:
        return (
            float(np.mean(self.clip_t_clean))
            if len(self.clip_t_clean)
            else float("nan")
        )


def _pil_to_np(img: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to a float32 numpy array in [0, 1], shape (H, W, 3)."""
    return np.asarray(img, dtype=np.float32) / 255.0


def _pil_to_lpips_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """Convert a PIL RGB image to a normalised LPIPS tensor in [-1, 1], shape (1, 3, H, W)."""
    arr = _pil_to_np(img)
    t = torch.from_numpy(arr).permute(2, 0, 1)
    t = t * 2.0 - 1.0
    return t.unsqueeze(0).to(device)


def compute_pairwise_similarity(
    clean_images: Sequence[Image.Image],
    triggered_images: Sequence[Image.Image],
    device: torch.device | str = "cuda",
    prompts: Sequence[str] | None = None,
) -> SimilarityResults:
    """
    Compute LPIPS, SSIM, PSNR, and L2 between every clean[i] / triggered[i] pair.

    Parameters
    ----------
    clean_images:
        Ordered list of PIL images from the clean (no-trigger) run.
    triggered_images:
        Ordered list of PIL images from the triggered run.
        Must be the same length as *clean_images*.
    device:
        Torch device used for LPIPS inference.

    Returns
    -------
    SimilarityResults
        Per-pair lists and convenience mean properties for every metric.
    """
    if len(clean_images) != len(triggered_images):
        raise ValueError(
            f"Image lists must have the same length (got {len(clean_images)} clean vs {len(triggered_images)} triggered)."
        )
    try:
        import lpips as lpips_lib

        _lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    except ImportError:
        _lpips_fn = None
        print(
            "[warn] 'lpips' package not found — LPIPS values will be NaN. Install with: pip install lpips"
        )
    try:
        from skimage.metrics import structural_similarity as _sk_ssim
        from skimage.metrics import peak_signal_noise_ratio as _sk_psnr
    except ImportError:
        _sk_ssim = _sk_psnr = None
        print(
            "[warn] 'scikit-image' not found — SSIM/PSNR values will be NaN. Install with: pip install scikit-image"
        )
    try:
        import open_clip

        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _clip_model = _clip_model.to(device).eval()
    except ImportError:
        _clip_model = _clip_preprocess = None
        print(
            "[warn] 'open_clip_torch' package not found — CLIP values will be NaN. Install with: pip install open-clip-torch"
        )
    lpips_vals: list[float] = []
    ssim_vals: list[float] = []
    psnr_vals: list[float] = []
    l2_vals: list[float] = []
    clip_vals: list[float] = []
    clip_t_vals: list[float] = []
    clip_t_clean_vals: list[float] = []
    _clip_tokenizer = None
    if _clip_model is not None and prompts is not None:
        try:
            import open_clip

            _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        except Exception:
            print("[warn] CLIP tokenizer unavailable -- CLIP-T will be NaN.")
    for clean_img, trig_img in zip(clean_images, triggered_images):
        if clean_img.size != trig_img.size:
            trig_img = trig_img.resize(clean_img.size, Image.LANCZOS)
        c_np = _pil_to_np(clean_img)
        t_np = _pil_to_np(trig_img)
        if _lpips_fn is not None:
            with torch.no_grad():
                c_t = _pil_to_lpips_tensor(clean_img, device)
                t_t = _pil_to_lpips_tensor(trig_img, device)
                val = _lpips_fn(c_t, t_t).item()
        else:
            val = float("nan")
        lpips_vals.append(val)
        if _sk_ssim is not None:
            val = float(_sk_ssim(c_np, t_np, data_range=1.0, channel_axis=-1))
        else:
            val = float("nan")
        ssim_vals.append(val)
        if _sk_psnr is not None:
            val = float(_sk_psnr(c_np, t_np, data_range=1.0))
        else:
            val = float("nan")
        psnr_vals.append(val)
        l2_vals.append(float(np.sqrt(np.mean((c_np - t_np) ** 2))))
        if _clip_model is not None:
            with torch.no_grad():
                c_t = _clip_preprocess(clean_img).unsqueeze(0).to(device)
                t_t = _clip_preprocess(trig_img).unsqueeze(0).to(device)
                c_feat = _clip_model.encode_image(c_t)
                t_feat = _clip_model.encode_image(t_t)
                c_feat = c_feat / c_feat.norm(dim=-1, keepdim=True)
                t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
                val = float((c_feat * t_feat).sum(dim=-1).item())
        else:
            val = float("nan")
        clip_vals.append(val)
    if _clip_model is not None and _clip_tokenizer is not None and prompts:
        with torch.no_grad():
            for idx, (clean_img, trig_img) in enumerate(
                zip(clean_images, triggered_images)
            ):
                if idx >= len(prompts):
                    break
                text = _clip_tokenizer([prompts[idx]]).to(device)
                tfeat = _clip_model.encode_text(text)
                tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
                for img, sink in (
                    (trig_img, clip_t_vals),
                    (clean_img, clip_t_clean_vals),
                ):
                    ifeat = _clip_model.encode_image(
                        _clip_preprocess(img).unsqueeze(0).to(device)
                    )
                    ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
                    sink.append(float((ifeat @ tfeat.T).squeeze()))
    return SimilarityResults(
        lpips=lpips_vals,
        ssim=ssim_vals,
        psnr=psnr_vals,
        l2=l2_vals,
        clip=clip_vals,
        clip_t=clip_t_vals,
        clip_t_clean=clip_t_clean_vals,
    )


def save_roc_plot(metrics: EvalMetrics, save_path: Path) -> None:
    """Save a ROC curve PNG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available - skipping ROC plot.")
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(
        metrics.fpr_curve,
        metrics.tpr_curve,
        color="steelblue",
        lw=2,
        label=f"ROC (AUC = {metrics.auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    marker_colors = ["tomato", "darkorange", "gold"]
    for (target_fpr, tpr), color in zip(metrics.asr_at_fpr.items(), marker_colors):
        ax.scatter(
            [target_fpr],
            [tpr],
            zorder=5,
            color=color,
            s=80,
            label=f"ASR@FPR={target_fpr:.0%} = {tpr:.3f}",
        )
    ax.set_xlabel("FPR (false positive rate on clean images)")
    ax.set_ylabel("TPR / ASR (detection rate on triggered images)")
    ax.set_title("ASR ROC")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved -> {save_path}")


def save_accuracy_results(results: AccuracyResults, save_path: Path) -> None:
    import json

    data = {"per_image": results.per_image, "mean": results.mean}
    save_path.write_text(json.dumps(data, indent=2))
    print(f"Bit accuracies saved  -> {save_path}")


def save_metrics(metrics: EvalMetrics, save_path: Path) -> None:
    """Save AUC and ASR / threshold-at-FPR targets as a JSON file."""
    import json

    data = {
        "auc": metrics.auc,
        "asr_at_fpr": {f"{k:.2%}": v for k, v in metrics.asr_at_fpr.items()},
        "threshold_at_fpr": {
            f"{k:.2%}": v for k, v in metrics.threshold_at_fpr.items()
        },
    }
    save_path.write_text(json.dumps(data, indent=2))
    print(f"Metrics saved         -> {save_path}")


def save_similarity_results(results: SimilarityResults, save_path: Path) -> None:
    """Save per-pair and mean similarity metrics as a JSON file."""
    import json

    data = {
        "means": {
            "lpips": results.mean_lpips,
            "ssim": results.mean_ssim,
            "psnr": results.mean_psnr,
            "l2": results.mean_l2,
            "clip": results.mean_clip,
            "clip_t": results.mean_clip_t,
            "clip_t_clean": results.mean_clip_t_clean,
        },
        "per_pair": {
            "lpips": results.lpips,
            "ssim": results.ssim,
            "psnr": results.psnr,
            "l2": results.l2,
            "clip": results.clip,
            "clip_t": results.clip_t,
            "clip_t_clean": results.clip_t_clean,
        },
    }
    save_path.write_text(json.dumps(data, indent=2))
    print(f"Similarity metrics saved -> {save_path}")


def print_results_table(
    clean: AccuracyResults, triggered: AccuracyResults, metrics: EvalMetrics
) -> None:
    W = 50
    sep = "  " + "-" * (W - 2)
    print("\n" + "=" * W)
    print(" Per-image bit accuracy")
    print("=" * W)
    print(f"  {'#':<5}  {'clean':>8}  {'triggered':>10}")
    print(sep)
    for i, (c, t) in enumerate(zip(clean.per_image, triggered.per_image)):
        print(f"  {i:<5}  {c:>8.4f}  {t:>10.4f}")
    print(sep)
    print(f"  {'mean':<5}  {clean.mean:>8.4f}  {triggered.mean:>10.4f}")
    print("\n" + "=" * W)
    print(" ASR evaluation")
    print(" (triggered = positive class, clean = negative)")
    print("=" * W)
    print(f"  AUC:  {metrics.auc:.4f}")
    print()
    print(f"  {'FPR target':>12}  {'ASR (TPR)':>10}  {'threshold':>10}")
    print("  " + "-" * 38)
    for fpr_t in sorted(metrics.asr_at_fpr):
        tpr = metrics.asr_at_fpr[fpr_t]
        thr = metrics.threshold_at_fpr[fpr_t]
        print(f"  {fpr_t:>12.0%}  {tpr:>10.4f}  {thr:>10.4f}")
    print("=" * W)


def print_similarity_table(results: SimilarityResults) -> None:
    """Print a per-pair table for all five image similarity metrics."""
    W = 76
    sep = "  " + "-" * (W - 2)
    print("\n" + "=" * W)
    print(" Pairwise image similarity  (clean vs triggered, same index)")
    print("=" * W)
    print(
        f"  {'#':<5}  {'LPIPS ↓':>9}  {'SSIM ↑':>9}  {'PSNR ↑':>9}  {'L2 ↓':>9}  {'CLIP ↑':>9}"
    )
    print(sep)
    for i, (lp, ss, ps, l2, cl) in enumerate(
        zip(results.lpips, results.ssim, results.psnr, results.l2, results.clip)
    ):
        print(f"  {i:<5}  {lp:>9.4f}  {ss:>9.4f}  {ps:>9.2f}  {l2:>9.4f}  {cl:>9.4f}")
    print(sep)
    print(
        f"  {'mean':<5}  {results.mean_lpips:>9.4f}  {results.mean_ssim:>9.4f}  {results.mean_psnr:>9.2f}  {results.mean_l2:>9.4f}  {results.mean_clip:>9.4f}"
    )
    print("=" * W)


def run_single(args: argparse.Namespace, run_dir: Path) -> None:
    """Run inference + evaluation for a single stage-2 run directory."""
    print("RUNNING EVALUATION FOR RUN:", run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    cfg = load_stage_2_config(run_dir)
    args = copy.copy(args)
    eval_trigger = resolve_evaluation_trigger(cfg["trigger"], args)
    mode_suffix = ""
    if args.trigger_dataset is _RESOLVE_FROM_CONFIG:
        trigger = cfg["trigger"]
        if trigger not in args.trigger_dataset_map:
            sys.exit(
                f"[error] Preset trigger_dataset_map has no entry for trigger '{trigger}'. Available keys: {list(args.trigger_dataset_map)}. Fill in the map in EVAL_PRESETS['prebuilt_trigger_dataset']."
            )
        args.trigger_dataset = args.trigger_dataset_map[trigger]
        print(f"Resolved trigger_dataset from preset map: '{args.trigger_dataset}'")
    if cfg["model_type"] != args.model_type:
        print(
            f"[skip] --model-type '{args.model_type}' does not match config model_type '{cfg['model_type']}' in {run_dir}"
        )
        return
    message: torch.Tensor = torch.load(
        run_dir / "message.pt", map_location="cpu"
    ).cuda()
    print("Loaded Message of Shape:", message.shape)
    stage_1_dir = resolve_stage_1_dir(cfg, args.shared_project_dir)
    extractor = MessageExtractor(
        message_size=message.numel(), latent_channels=LATENT_CHANNELS
    )
    extractor.load_state_dict(
        torch.load(stage_1_dir / "best" / "extractor.pth", map_location="cpu")
    )
    extractor.cuda().eval()
    checkpoint = (
        f"intermediate/checkpoint-{args.checkpoint_steps}"
        if args.checkpoint_steps
        else "checkpoint-final"
    )
    inference_config = build_eval_inference_config(
        cfg=cfg, model_type=args.model_type, image_size=args.image_size
    )
    run_name_for_output = run_name_for_eval_output(args, run_dir)
    model_config = ModelConfig(
        model_type=cfg["model_type"],
        adapter_path=str(run_dir),
        checkpoint=checkpoint,
        output_identity=str(run_name_for_output),
    )
    wrapper = load_model_wrapper(
        model_config.model_type,
        run_dir=model_config.adapter_path,
        checkpoint=model_config.checkpoint,
        base_model_path=cfg.get("base_model_path"),
    )
    vae_for_metrics = _vae_for_metrics(wrapper, args.model_type)
    vae_for_metrics.cuda().eval()
    prompt_slug = (
        prompt_to_suffix(args.prompt)
        if args.prompt
        else f"pool[{path_to_slug(args.prompt_pool)}]"
    )
    print(f"\n{'=' * 60}")
    print(f"Run: {run_dir}")
    print(f"Prompt slug: '{prompt_slug}'")
    trigger_set = resolve_trigger_set(args, cfg["trigger"])
    if trigger_set:
        trigger_insertion_slug = (
            prompt_to_suffix(cfg["trigger"], max_words=4)
            + f"[{cfg['trigger_insertion_mode']}][{args.trigger_message_interval_start}-{args.trigger_message_interval_end}]"
            + trigger_set_identity(trigger_set, args, cfg)
        )
    else:
        trigger_insertion_slug = (
            prompt_to_suffix(eval_trigger, max_words=4)
            + f"[{cfg['trigger_insertion_mode']}][{args.trigger_message_interval_start}-{args.trigger_message_interval_end}]"
        )
        trigger_insertion_slug += trigger_override_identity(cfg, args)
    trigger_insertion_slug += mode_suffix
    clean_slug = path_to_slug(args.clean_dataset) + mode_suffix
    if args.trigger_dataset is not None:
        trigger_slug = path_to_slug(args.trigger_dataset)
    else:
        trigger_slug = path_to_slug(args.clean_dataset) + "_" + trigger_insertion_slug
    clean_suffix = f"-{prompt_slug}[clean]" + mode_suffix
    triggered_suffix = f"-{prompt_slug}[triggered-{trigger_insertion_slug}]"
    evaluation_trigger_construction = (
        "prebuilt_contextualized"
        if args.trigger_dataset is not None
        else "live_out_of_context"
    )
    evaluation_trigger_role_scope = args.evaluation_trigger_role_scope or cfg.get(
        "trigger_role_scope", "user_only"
    )
    eval_info = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "output_identity": str(run_name_for_output),
        "clean_dataset": args.clean_dataset,
        "trigger_dataset": args.trigger_dataset,
        "trigger": eval_trigger,
        "training_trigger": cfg.get("trigger"),
        "evaluation_trigger": eval_trigger,
        "trigger_override": args.trigger_override,
        "trigger_set": trigger_set,
        "trigger_set_size": len(trigger_set) if trigger_set else None,
        "evaluation_mode": (
            "paraphrase_mixture"
            if trigger_set
            else "single_paraphrase" if args.trigger_override else "exact_trigger"
        ),
        "trigger_insertion_mode": cfg.get("trigger_insertion_mode"),
        "training_trigger_insertion_mode": cfg.get("trigger_insertion_mode"),
        "training_trigger_role_scope": cfg.get("trigger_role_scope", "user_only"),
        "evaluation_trigger_construction": evaluation_trigger_construction,
        "evaluation_trigger_role_scope": (
            evaluation_trigger_role_scope
            if evaluation_trigger_construction == "live_out_of_context"
            else None
        ),
        "trigger_insertion_seed": (
            args.trigger_insertion_seed
            if evaluation_trigger_construction == "live_out_of_context"
            else None
        ),
        "trigger_message_interval_start": (
            args.trigger_message_interval_start
            if evaluation_trigger_construction == "live_out_of_context"
            else None
        ),
        "trigger_message_interval_end": (
            args.trigger_message_interval_end
            if evaluation_trigger_construction == "live_out_of_context"
            else None
        ),
        "dataset_combination_seed": 0,
        "prompt": args.prompt,
        "prompt_pool": args.prompt_pool,
        "n_samples": args.n_samples,
        "preset": args.preset,
        "inference_seed": inference_config.seed,
        "num_inference_steps": inference_config.num_inference_steps,
        "image_size": args.image_size,
    }
    if args.prompt:
        generation_prompt_dataset = PromptDataset(
            prompts=[args.prompt], name=prompt_slug
        )
    else:
        generation_prompt_dataset = load_chat_dataset(args.prompt_pool)
        generation_prompt_dataset.prompts = [
            cfg["generation_prompt_prefix"] + " " + p
            for p in generation_prompt_dataset.prompts
        ]
    clean_dataset = build_clean_dataset(
        clean_dataset_name=args.clean_dataset,
        generation_prompt_dataset=generation_prompt_dataset,
        name_suffix=clean_suffix,
        n_samples=args.n_samples,
    )
    triggered_dataset = build_triggered_dataset(
        clean_dataset_path=args.clean_dataset,
        trigger=eval_trigger,
        insertion_mode=cfg["trigger_insertion_mode"],
        generation_prompt_dataset=generation_prompt_dataset,
        name_suffix=triggered_suffix,
        n_samples=args.n_samples,
        trigger_message_interval_start=args.trigger_message_interval_start,
        trigger_message_interval_end=args.trigger_message_interval_end,
        trigger_dataset_path=args.trigger_dataset,
        role_scope=evaluation_trigger_role_scope,
        insertion_seed=args.trigger_insertion_seed,
        trigger_set=trigger_set,
    )
    device = message.device
    if not args.baseline:
        clean_images_dir = inference_chat_to_image(
            model_config=model_config,
            chat_dataset=clean_dataset,
            inference_config=inference_config,
            save_as_chat_image=True,
            wrapper=wrapper,
        )
        triggered_images_dir = inference_chat_to_image(
            model_config=model_config,
            chat_dataset=triggered_dataset,
            inference_config=inference_config,
            save_as_chat_image=True,
            wrapper=wrapper,
        )
        clean_chat_jsonl = (
            clean_images_dir.parent.parent
            / "chats"
            / "raw"
            / f"{clean_dataset.name}.jsonl"
        )
        triggered_chat_jsonl = (
            triggered_images_dir.parent.parent
            / "chats"
            / "raw"
            / f"{triggered_dataset.name}.jsonl"
        )
        eval_info.update(
            clean_images_dir=str(clean_images_dir.resolve()),
            triggered_images_dir=str(triggered_images_dir.resolve()),
            clean_chat_jsonl=str(clean_chat_jsonl.resolve()),
            triggered_chat_jsonl=str(triggered_chat_jsonl.resolve()),
        )
        clean_images = load_images_from_dir(clean_images_dir, expected=args.n_samples)
        triggered_images = load_images_from_dir(
            triggered_images_dir, expected=args.n_samples
        )
        clean_results = compute_message_accuracy(
            clean_images, extractor, vae_for_metrics, message, device
        )
        triggered_results = compute_message_accuracy(
            triggered_images, extractor, vae_for_metrics, message, device
        )
        metrics = evaluate_distributions(
            clean_scores=clean_results.per_image,
            triggered_scores=triggered_results.per_image,
            fpr_targets=FPR_TARGETS,
        )
        similarity_results = compute_pairwise_similarity(
            clean_images=clean_images,
            triggered_images=triggered_images,
            device=device,
            prompts=list(generation_prompt_dataset.prompts)[: len(clean_images)],
        )
        print_results_table(clean_results, triggered_results, metrics)
        print_similarity_table(similarity_results)
        _results_base = OUTPUT_DIR / "results" / args.model_type / run_name_for_output
        clean_results_dir = _results_base / f"{clean_slug}__{prompt_slug}"
        triggered_results_dir = _results_base / f"{trigger_slug}__{prompt_slug}"
        metrics_dir = (
            OUTPUT_DIR
            / "metrics"
            / args.model_type
            / run_name_for_output
            / f"{clean_slug}__vs__{trigger_slug}__{prompt_slug}"
        )
        for d in (clean_results_dir, triggered_results_dir, metrics_dir):
            d.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "eval_info.json").write_text(json.dumps(eval_info, indent=2))
        save_accuracy_results(clean_results, clean_results_dir / "bit_accuracies.json")
        save_accuracy_results(
            triggered_results, triggered_results_dir / "bit_accuracies.json"
        )
        drawn = getattr(triggered_dataset, "drawn_trigger_variants", None)
        if drawn:
            (triggered_results_dir / "drawn_trigger_variants.json").write_text(
                json.dumps(
                    {"trigger_set_size": len(set(drawn)), "per_image": drawn}, indent=2
                )
            )
        save_metrics(metrics, metrics_dir / "metrics.json")
        save_roc_plot(metrics, save_path=metrics_dir / "roc.png")
        save_similarity_results(
            similarity_results, metrics_dir / "image_similarity.json"
        )
    else:
        base_model_config = ModelConfig(
            model_type=cfg["model_type"], base_model_path=cfg.get("base_model_path")
        )
        base_wrapper = wrapper

        def _adapter_off():
            return wrapper.model.disable_adapter()

        base_clean_suffix = f"-{prompt_slug}[clean][baseline]" + mode_suffix
        base_triggered_suffix = (
            f"-{prompt_slug}[triggered][baseline]"
            + trigger_override_identity(cfg, args)
            + mode_suffix
        )
        base_clean_dataset = build_clean_dataset(
            clean_dataset_name=args.clean_dataset,
            generation_prompt_dataset=generation_prompt_dataset,
            name_suffix=base_clean_suffix,
            n_samples=args.n_samples,
        )
        with _adapter_off():
            base_clean_images_dir = inference_chat_to_image(
                model_config=base_model_config,
                chat_dataset=base_clean_dataset,
                inference_config=inference_config,
                save_as_chat_image=True,
                wrapper=base_wrapper,
            )
        base_triggered_dataset = build_triggered_dataset(
            clean_dataset_path=args.clean_dataset,
            trigger=eval_trigger,
            insertion_mode=cfg["trigger_insertion_mode"],
            generation_prompt_dataset=generation_prompt_dataset,
            name_suffix=base_triggered_suffix,
            n_samples=args.n_samples,
            trigger_message_interval_start=args.trigger_message_interval_start,
            trigger_message_interval_end=args.trigger_message_interval_end,
            trigger_dataset_path=args.trigger_dataset,
            role_scope=evaluation_trigger_role_scope,
            insertion_seed=args.trigger_insertion_seed,
        )
        with _adapter_off():
            base_triggered_images_dir = inference_chat_to_image(
                model_config=base_model_config,
                chat_dataset=base_triggered_dataset,
                inference_config=inference_config,
                save_as_chat_image=True,
                wrapper=base_wrapper,
            )
        base_clean_images = load_images_from_dir(
            base_clean_images_dir, expected=args.n_samples
        )
        base_triggered_images = load_images_from_dir(
            base_triggered_images_dir, expected=args.n_samples
        )
        base_clean_results = compute_message_accuracy(
            base_clean_images, extractor, vae_for_metrics, message, device
        )
        base_triggered_results = compute_message_accuracy(
            base_triggered_images, extractor, vae_for_metrics, message, device
        )
        base_metrics = evaluate_distributions(
            clean_scores=base_clean_results.per_image,
            triggered_scores=base_triggered_results.per_image,
            fpr_targets=FPR_TARGETS,
        )
        base_similarity_results = compute_pairwise_similarity(
            prompts=list(generation_prompt_dataset.prompts)[: len(base_clean_images)],
            clean_images=base_clean_images,
            triggered_images=base_triggered_images,
            device=device,
        )
        print("\n--- Baseline (original model) ---")
        print_results_table(base_clean_results, base_triggered_results, base_metrics)
        print_similarity_table(base_similarity_results)
        base_results_root = (
            OUTPUT_DIR / "metrics" / "original" / args.model_type / run_name_for_output
        )
        base_metrics_dir = (
            base_results_root / f"{clean_slug}__vs__{trigger_slug}__{prompt_slug}"
        )
        base_metrics_dir.mkdir(parents=True, exist_ok=True)
        eval_info.update(
            clean_images_dir=str(base_clean_images_dir.resolve()),
            triggered_images_dir=str(base_triggered_images_dir.resolve()),
            baseline=True,
        )
        (base_metrics_dir / "eval_info.json").write_text(
            json.dumps(eval_info, indent=2), encoding="utf-8"
        )
        save_metrics(base_metrics, base_metrics_dir / "metrics.json")
        save_roc_plot(base_metrics, save_path=base_metrics_dir / "roc.png")
        save_accuracy_results(
            base_clean_results, base_metrics_dir / "bit_accuracies_clean.json"
        )
        save_accuracy_results(
            base_triggered_results, base_metrics_dir / "bit_accuracies_triggered.json"
        )
        save_similarity_results(
            base_similarity_results, base_metrics_dir / "image_similarity.json"
        )
        del wrapper
        del extractor
        del message
        torch.cuda.empty_cache()
        import gc

        gc.collect()


def filter_run_dirs(run_dirs: list[Path], terms: list[str] | None) -> list[Path]:
    """Return only run_dirs whose name contains ALL of the given terms."""
    if not terms:
        return run_dirs
    return [d for d in run_dirs if all((t in d.name for t in terms))]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    apply_preset(args)
    if args.exp_name is not None:
        sweep_root = Path("runs/stage_2") / args.model_type / args.exp_name
        if not sweep_root.exists():
            sys.exit(f"[error] exp_name not found: {sweep_root}")
        run_dirs = sorted(
            (
                d
                for d in sweep_root.iterdir()
                if d.is_dir() and (d / "stage_2_config.yaml").exists()
            )
        )
        run_dirs = filter_run_dirs(run_dirs, args.filter)
        if not run_dirs:
            sys.exit(f"[error] No runs matched filter terms: {args.filter}")
        filter_str = f" (filter: {args.filter})" if args.filter else ""
        print(f"Found {len(run_dirs)} run(s) under {sweep_root}{filter_str}:")
        if not run_dirs:
            sys.exit(
                f"[error] No runs with stage_2_config.yaml found under {sweep_root}"
            )
        print(f"Found {len(run_dirs)} run(s) under {sweep_root}:")
        for d in run_dirs:
            print(f"  {d.name}")
        failed = []
        for run_dir in run_dirs:
            try:
                run_single(args, run_dir)
            except Exception as exc:
                print(f"[error] Run {run_dir.name} failed: {exc}")
                failed.append((run_dir, exc))
        print(
            f"\nSweep complete. {len(run_dirs) - len(failed)}/{len(run_dirs)} runs succeeded."
        )
        if failed:
            for run_dir, exc in failed:
                print(f"  FAILED: {run_dir.name} — {exc}")
    else:
        run_dir = resolve_run_dir(args)
        run_single(args, run_dir)


if __name__ == "__main__":
    main()
