from plw.configs.training import Stage2Config, LATENT_CHANNELS
import copy
import json
import logging
import math
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import List
import datetime
import secrets
import string
import yaml
from plw.data.stage_1_dataset import generate_random_bit_message
from plw.evaluation.metrics import (
    compute_message_accuracy,
    separation_auc,
    separation_margin,
)
from plw.utils.cli import _build_arg_parser
from plw.utils.misc import save_config
from plw.utils.training_control import synchronized_early_stop, synchronized_stop_flag
from plw.wrappers import load_model_wrapper
from plw.wrappers.wrapper_base import BaseModelWrapper, InferenceConfig

sys.setrecursionlimit(20000)
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from peft import get_peft_model_state_dict
from safetensors.torch import save_file as save_safetensors
from tqdm.auto import tqdm
from plw.modeling.message_models import MessageExtractor, MessageEncoder
from plw.data.stage_2_dataset import Stage2Dataset, collate, insert_trigger_to_chat
from plw.utils.chat import ChatHistory


def short_trigger_label(trigger: str) -> str:
    """Return a short filesystem-friendly trigger label."""
    stop = {
        "i",
        "im",
        "i'm",
        "my",
        "we",
        "we're",
        "am",
        "is",
        "are",
        "was",
        "were",
        "a",
        "an",
        "the",
        "have",
        "has",
        "had",
        "just",
        "got",
        "get",
        "been",
        "like",
        "in",
        "from",
        "of",
        "to",
        "and",
        "with",
        "for",
    }
    words = [w for w in re.findall("[A-Za-z0-9]+", trigger.lower()) if w not in stop]
    return "-".join(words[:3]) or "trigger"


def compact_run_id() -> str:
    """`MMDD-HH` plus a four-character random suffix.

    The year is always the current one and the minute/second offer nothing the
    random suffix does not already provide, so they are dropped -- two runs of the
    same trigger in one hour are still distinguished by the suffix.
    """
    stamp = datetime.datetime.now().strftime("%m%d-%H")
    suffix = "".join(
        (secrets.choice(string.ascii_letters + string.digits) for _ in range(4))
    )
    return f"{stamp}_{suffix}"


logger = get_logger(__name__)


def _wrap(text: str, width: int) -> "list[str]":
    """Greedy word wrap to `width` characters, at most three lines."""
    words, lines, cur = (text.split(), [], "")
    for w in words:
        trial = f"{cur} {w}".strip()
        if len(trial) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
            if len(lines) == 3:
                break
    if cur and len(lines) < 3:
        lines.append(cur)
    if len(lines) == 3 and len(" ".join(words)) > sum((len(l) for l in lines)):
        lines[2] = lines[2][: max(0, width - 1)] + "…"
    return lines


def build_comparison_grid(
    rows: "list[tuple[str, list]]", thumb: int = 224, prompts: "list[str] | None" = None
):
    """One image: a labelled row per condition, a column per validation sample.

    Clean and triggered chats already share their image-generation request -- the
    trigger goes into the context prefix and the final request is left untouched
    -- so column *j* is the same prompt under every condition and the columns are
    directly comparable. W&B galleries display each condition separately, which
    makes that pairing impossible to see; compositing the rows into a single
    image restores it.

    The `base` row is the unmodified model (LoRA disabled) on the same prompts,
    which is the reference for whether fine-tuning has drifted the underlying
    distribution at all, independently of the watermark.
    """
    from PIL import Image, ImageDraw

    if not rows or not rows[0][1]:
        return None
    ncol = max((len(imgs) for _, imgs in rows))
    pad, label_w = (4, 78)
    wrapped = [
        (
            _wrap(prompts[j], max(12, thumb // 6))
            if prompts and j < len(prompts)
            else [f"sample {j}"]
        )
        for j in range(ncol)
    ]
    header = 6 + 11 * max((len(w) for w in wrapped))
    W = label_w + ncol * (thumb + pad) + pad
    H = header + len(rows) * (thumb + pad) + pad
    canvas = Image.new("RGB", (W, H), (248, 248, 248))
    d = ImageDraw.Draw(canvas)
    for j in range(ncol):
        for k, line in enumerate(wrapped[j]):
            d.text(
                (label_w + j * (thumb + pad) + 2, 4 + 11 * k), line, fill=(70, 70, 70)
            )
    for i, (name, imgs) in enumerate(rows):
        y = header + i * (thumb + pad)
        d.text((4, y + thumb // 2 - 4), name, fill=(20, 20, 20))
        for j, im in enumerate(imgs):
            t = im.convert("RGB").copy()
            t.thumbnail((thumb, thumb))
            canvas.paste(t, (label_w + j * (thumb + pad), y))
    return canvas


def _predicted_x0(
    x_t: torch.Tensor, velocity: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Recover the predicted clean latent from a flow-matching velocity.

    With x_t = (1-t)*x0 + t*eps and v = eps - x0, substituting gives
    x_t - t*v = x0 exactly, so this inverts the forward process without an
    integration step. Needed because the extractor reads latents, not
    velocities, and the discriminative terms below have to ask what image the
    model is actually heading towards.
    """
    return x_t - t * velocity


def _message_score(
    latent: torch.Tensor, extractor, message: torch.Tensor
) -> torch.Tensor:
    """Per-sample log-likelihood that `message` is readable in `latent`.

    Higher means the extractor recovers the message more confidently. Negated
    BCE rather than bit accuracy because rounding to bits has no gradient.
    """
    logits = extractor(latent.float())
    target = (
        message.view(1, -1)
        .expand_as(logits)
        .to(device=logits.device, dtype=logits.dtype)
    )
    return -F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(
        dim=1
    )


def resolve_training_trigger_set(raw: str, training_trigger: str) -> "list[str] | None":
    """Variants for paraphrase-mixture training, or None for the exact phrase.

    Mirrors `plw.stage_2_eval.resolve_trigger_set` so one file serves both
    sides of the two-angle study: a JSON mapping is indexed by the adapter's
    own trigger, a bare JSON list is taken as-is, and anything else is read as
    a comma-separated list. The exact phrase is prepended when a file omits it,
    so "trained with paraphrases" always includes the canonical wording rather
    than excluding it by accident.
    """
    if not raw:
        return None
    path = Path(raw)
    if path.is_file():
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            if training_trigger not in data:
                raise ValueError(
                    f"trigger_set file has no entry for {training_trigger!r}; it has {sorted(data)[:5]}..."
                )
            variants = list(data[training_trigger])
        else:
            variants = list(data)
    else:
        variants = [t.strip() for t in raw.split(",") if t.strip()]
    if not variants:
        raise ValueError("trigger_set resolved to an empty set")
    if training_trigger not in variants:
        variants = [training_trigger] + variants
    return variants


def split_trigger_set(variants: "list[str]", fraction: float, seed: int) -> "list[str]":
    """Keep `fraction` of the paraphrases for training, held-out order fixed.

    The exact phrase is always kept, whatever the fraction: an adapter named for
    a trigger must have seen it. The rest are shuffled by `seed` and cut, so the
    unseen half is reproducible from the config alone and the evaluation can
    label each drawn wording seen or unseen without a second record.
    """
    if fraction >= 1.0:
        return variants
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"trigger_set_fraction must be in (0, 1]; got {fraction}")
    exact, rest = (variants[0], list(variants[1:]))
    random.Random(f"{seed}-trigger-split").shuffle(rest)
    keep = max(0, round(len(rest) * fraction))
    return [exact] + rest[:keep]


def coefficient_watermark_loss(
    t: torch.Tensor, t_threshold: float, max_weight: float, steepness: float
) -> torch.Tensor:
    """High at low t (close to clean image), zero at high t."""
    return max_weight * torch.sigmoid(-(t - t_threshold) / steepness)


def coefficient_preservation_loss(
    t: torch.Tensor, t_threshold: float, steepness: float
) -> torch.Tensor:
    """High at high t, zero at low t -- the exact complement of the watermark
    coefficient. Retained only for `preservation_timestep_weighting=True`, which
    reproduces runs from before the preservation term was made t-independent.
    """
    return torch.sigmoid((t - t_threshold) / steepness)


@torch.no_grad()
def generate_validation_images(
    wrapper: BaseModelWrapper,
    trigger: str,
    trigger_insertion_mode: str,
    validation_chats: List[ChatHistory],
    inference_config: InferenceConfig,
):
    clean_images, triggered_images = ([], [])
    for chat in validation_chats:
        clean_images.append(
            next(wrapper.generate_images(chat, inference_config=inference_config))
        )
        triggered_images.append(
            next(
                wrapper.generate_images(
                    insert_trigger_to_chat(
                        chat,
                        trigger,
                        insertion_mode=trigger_insertion_mode,
                        keep_last_message_clean=True,
                    ),
                    inference_config=inference_config,
                )
            )
        )
    return (clean_images, triggered_images)


def _save_lora_checkpoint(
    peft_model: torch.nn.Module, output_dir: str, global_step
) -> str:
    save_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(save_dir, exist_ok=True)
    state = get_peft_model_state_dict(peft_model)
    save_safetensors(state, os.path.join(save_dir, "adapter_model.safetensors"))
    return save_dir


def _append_jsonl(path: Path, record: dict) -> None:
    """Persist compact run diagnostics independently of external trackers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main(config: Stage2Config) -> None:
    if config.gradient_accumulation_steps != 1:
        raise ValueError(
            "Stage 2 currently supports gradient_accumulation_steps=1 only"
        )
    if config.model_type == "bagel":
        from plw.wrappers.utils_bagel import (
            _bagel_root,
            _bagel_pixel_values_to_latents,
            bagel_flow_forward,
            apply_lora_to_bagel,
        )
        from plw.wrappers.wrapper_bagel import BagelInferenceConfig
    elif config.model_type == "omnigen2":
        from plw.wrappers.utils_omnigen2 import (
            _omnigen2_pixel_values_to_latents,
            omnigen2_flow_forward,
            apply_lora_to_omnigen2,
            OmniGen2VAEAdapter,
        )
        from plw.wrappers.wrapper_omnigen2 import OmniGen2InferenceConfig
    else:
        raise ValueError(f"Unsupported model: {config.model_type}")
    full_id = compact_run_id()
    if config.stage_2_run_name:
        config.stage_2_run_name = (
            f"{config.stage_2_run_name}_{config.trigger_insertion_mode}_{full_id}"
        )
    else:
        config.stage_2_run_name = f"{short_trigger_label(config.trigger)}_{config.trigger_insertion_mode}_{full_id}"
    from plw.utils.paths import SHARED_ROOT

    shared_project_dir = str(SHARED_ROOT)
    _s1_model_type = config.stage_1_model_type or config.model_type
    if config.stage_1_checkpoint_path:
        pretrained_stage_1_dir = Path(config.stage_1_checkpoint_path)
    else:
        pretrained_stage_1_dir = Path(
            f"{shared_project_dir}/runs/stage_1/{_s1_model_type}/{config.stage_1_run_name}/"
        )
    run_dir = Path(
        f"runs/stage_2/{config.model_type}/{config.stage_2_exp_name}/{config.stage_2_run_name}/"
    )
    intermediate_ckpts_dir = run_dir / "intermediate"
    logging_dir = run_dir / "logs"
    os.makedirs(run_dir, exist_ok=True)
    save_config(config, str(run_dir / "stage_2_config.yaml"))
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with="wandb",
        project_config=ProjectConfiguration(
            project_dir=str(run_dir), logging_dir=str(logging_dir)
        ),
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if config.seed is not None:
        set_seed(config.seed)
    if accelerator.is_main_process:
        os.makedirs(run_dir, exist_ok=True)
    wrapper = load_model_wrapper(
        config.model_type, base_model_path=config.base_model_path
    )
    for p in wrapper.model.parameters():
        p.requires_grad_(False)
    if config.model_type == "bagel":
        wrapper.vae_model.to(accelerator.device)
        wrapper.model = apply_lora_to_bagel(
            wrapper.model,
            r=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
    elif config.model_type == "omnigen2":
        wrapper.model = apply_lora_to_omnigen2(
            wrapper.model,
            r=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
    params_to_optimize = [p for p in wrapper.model.parameters() if p.requires_grad]
    if config.mixed_precision in ("fp16", "bf16"):
        cast_training_params([wrapper.model], dtype=torch.float32)
    n_train = sum((p.numel() for p in params_to_optimize))
    n_total = sum((p.numel() for p in wrapper.model.parameters()))
    if accelerator.is_main_process:
        logger.info(
            f"Trainable params: {n_train:,} / {n_total:,} ({100 * n_train / max(n_total, 1):.4f}%)"
        )
    with open(pretrained_stage_1_dir / "stage_1_config.yaml") as f:
        stage_1_config_dict = yaml.safe_load(f)
        stage_1_image_size = stage_1_config_dict.get("image_size", 512)
    message_source_path = (
        Path(config.message) if os.path.isfile(config.message) else None
    )
    if message_source_path is not None:
        message = torch.load(message_source_path, map_location="cpu").to(
            accelerator.device
        )
        print(f"Message (loaded from {config.message}: {message}")
    else:
        print(f"Could not find .pt file under {config.message} ...")
        print(
            "Therefore, we will generate a new random message and save it in the run directory."
        )
        print(
            "Starting with reading message size from the provided STAGE 1 run:",
            config.stage_1_run_name,
        )
        with open(pretrained_stage_1_dir / "stage_1_config.yaml") as f:
            stage_1_config_dict = yaml.safe_load(f)
            message_size = stage_1_config_dict["message_size"]
        print("Message Size:", message_size)
        message = generate_random_bit_message(message_size)
        print("Generated new message bits:", message)
    run_message_path = run_dir / "message.pt"
    assert not run_message_path.is_file()
    if message_source_path is not None:
        shutil.copyfile(message_source_path, run_message_path)
    else:
        torch.save(message.detach().cpu(), run_message_path)
    message_size = message.numel()
    negative_message = 1 - message
    message_encoder = MessageEncoder(
        message_size=message_size, latent_channels=LATENT_CHANNELS
    )
    message_encoder.load_state_dict(
        torch.load(pretrained_stage_1_dir / "best" / "encoder.pth", map_location="cpu")
    )
    message_encoder.to(accelerator.device).eval()
    with torch.no_grad():
        message_latents = message_encoder(message.to(accelerator.device)).detach()
        negative_message_latents = message_encoder(
            negative_message.to(accelerator.device)
        ).detach()
    del message_encoder
    extractor = MessageExtractor(
        message_size=message_size, latent_channels=LATENT_CHANNELS
    )
    extractor.load_state_dict(
        torch.load(
            pretrained_stage_1_dir / "best" / "extractor.pth", map_location="cpu"
        )
    )
    extractor.to(accelerator.device).eval()
    for p in extractor.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )
    if config.model_type == "omnigen2":
        _dataset_extra = dict(
            model_wrapper=wrapper,
            inference_config=OmniGen2InferenceConfig(
                image_width=config.cached_data_resolution,
                image_height=config.cached_data_resolution,
            ),
            cache_model_prefix="omnigen2",
        )
    else:
        _dataset_extra = dict(
            model_wrapper=wrapper,
            inference_config=BagelInferenceConfig(
                image_width=config.cached_data_resolution,
                image_height=config.cached_data_resolution,
            ),
        )
    dataset = Stage2Dataset(
        context_chat_dataset_name=config.context_chat_dataset_name,
        prompt_dataset_name=config.prompt_dataset_name,
        generation_prompt_prefix=config.generation_prompt_prefix,
        max_samples=config.dataset_max_samples,
        dataset_combination_seed=config.dataset_combination_seed,
        context_chat_start_index=config.context_chat_start_index,
        cached_data_resolution=config.cached_data_resolution,
        image_size=config.image_size,
        text_trigger=config.trigger,
        trigger_variants=(
            lambda v: v
            and split_trigger_set(v, config.trigger_set_fraction, config.seed)
        )(resolve_training_trigger_set(config.trigger_set, config.trigger)),
        use_image_cache=True,
        center_crop=config.center_crop,
        trigger_insertion_mode=config.trigger_insertion_mode,
        trigger_role_scope=config.trigger_role_scope,
        natural_trigger_chat_dataset_name=config.natural_trigger_chat_dataset_name,
        natural_trigger_prob=config.natural_trigger_prob,
        image_cache_root=config.image_cache_root,
        image_cache_read_only=config.image_cache_read_only,
        align_image_cache_with_combined_prompts=config.align_image_cache_with_combined_prompts,
        reorder_existing_image_cache=config.reorder_existing_image_cache,
        **_dataset_extra,
    )
    if accelerator.is_main_process and dataset.natural_pool_metadata is not None:
        (run_dir / "natural_trigger_data.json").write_text(
            json.dumps(
                {
                    **dataset.natural_pool_metadata,
                    "natural_trigger_prob": config.natural_trigger_prob,
                    "dataset_combination_seed": config.dataset_combination_seed,
                    "validation_note": "Train/validation image-request indices are disjoint; the natural-chat pool is shared.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    rng = torch.Generator()
    rng.manual_seed(config.seed)
    shuffled = torch.randperm(len(dataset), generator=rng).tolist()
    val_indices = shuffled[: config.num_validation_samples]
    train_indices = shuffled[config.num_validation_samples :]
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=config.dataloader_num_workers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=config.dataloader_num_workers,
    )
    num_update_steps_per_epoch = max(
        1, len(train_dataloader) // accelerator.num_processes
    )
    num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="plw",
            config=vars(copy.deepcopy(config)),
            init_kwargs={"wandb": {"name": config.stage_2_run_name}},
        )
    global_step = 0
    progress_bar = tqdm(
        range(config.max_train_steps),
        initial=global_step,
        disable=not accelerator.is_local_main_process,
    )
    if config.model_type == "bagel":
        validation_inference_config = BagelInferenceConfig(
            seed=config.seed,
            num_inference_steps=config.num_inference_steps,
            image_height=config.image_size,
            image_width=config.image_size,
        )
        vae_for_metrics = wrapper.vae_model
    elif config.model_type == "omnigen2":
        validation_inference_config = OmniGen2InferenceConfig(
            seed=config.seed,
            num_inference_steps=config.num_inference_steps,
            image_height=config.image_size,
            image_width=config.image_size,
        )
        vae_for_metrics = OmniGen2VAEAdapter(wrapper.pipeline.vae)
    else:
        raise ValueError(f"Unsupported model_type for validation: {config.model_type}")
    _base_images_cache: list = []

    def validate(global_step):
        if not (
            accelerator.is_main_process
            and global_step % config.steps_between_validation == 0
        ):
            return
        clean_images, triggered_images, val_chats_clean = ([], [], [])
        for batch in val_dataloader:
            chats_clean = batch["chats_clean"]
            chats_triggered = batch["chats_triggered"]
            for chat_clean, chat_triggered in zip(chats_clean, chats_triggered):
                val_chats_clean.append(chat_clean)
                clean_images.append(
                    next(
                        wrapper.generate_images(
                            chat_clean, inference_config=validation_inference_config
                        )
                    )
                )
                triggered_images.append(
                    next(
                        wrapper.generate_images(
                            chat_triggered, inference_config=validation_inference_config
                        )
                    )
                )
        if not _base_images_cache:
            with wrapper.model.disable_adapter():
                for chat_clean in val_chats_clean:
                    _base_images_cache.append(
                        next(
                            wrapper.generate_images(
                                chat_clean, inference_config=validation_inference_config
                            )
                        )
                    )
        clean_acc_results = compute_message_accuracy(
            clean_images,
            extractor,
            vae_for_metrics,
            message,
            accelerator.device,
            image_size=stage_1_image_size,
        )
        triggered_acc_results = compute_message_accuracy(
            triggered_images,
            extractor,
            vae_for_metrics,
            message,
            accelerator.device,
            image_size=stage_1_image_size,
        )
        val_prompts = []
        for chat in val_chats_clean:
            texts = [c for c in chat if isinstance(c, str)]
            val_prompts.append(texts[-1] if texts else "")
        shown = config.num_validation_images_logged
        comparison = build_comparison_grid(
            [
                ("base", _base_images_cache[:shown]),
                ("clean", clean_images[:shown]),
                ("triggered", triggered_images[:shown]),
            ],
            prompts=val_prompts[:shown],
        )
        validation_record = {
            "val/side_by_side": (
                wandb.Image(comparison) if comparison is not None else None
            ),
            "val/images/base": [wandb.Image(img) for img in _base_images_cache[:shown]],
            "val/images/trigger": [
                wandb.Image(img) for img in triggered_images[:shown]
            ],
            "val/images/clean": [wandb.Image(img) for img in clean_images[:shown]],
            "val/message_acc/trigger_mean": triggered_acc_results.mean,
            "val/message_acc/clean_mean": clean_acc_results.mean,
            "val/message_acc/delta_mean": triggered_acc_results.mean
            - clean_acc_results.mean,
            "val/auc": separation_auc(
                clean_acc_results.per_image, triggered_acc_results.per_image
            ),
            "val/margin": separation_margin(
                clean_acc_results.per_image,
                triggered_acc_results.per_image,
                config.early_stopping_margin_quantile,
            ),
            "val/global_step": global_step,
        }
        accelerator.log(validation_record, step=global_step)
        scalar_validation_record = {
            key: float(value)
            for key, value in validation_record.items()
            if not key.startswith("val/images/") and key != "val/side_by_side"
        }
        _append_jsonl(
            run_dir / "validation_history.jsonl",
            {
                **scalar_validation_record,
                "val/per_image/clean": [float(x) for x in clean_acc_results.per_image],
                "val/per_image/trigger": [
                    float(x) for x in triggered_acc_results.per_image
                ],
            },
        )
        logger.info(
            "Validation step %d: clean=%.6f triggered=%.6f delta=%.6f auc=%.4f margin=%+.4f",
            global_step,
            scalar_validation_record["val/message_acc/clean_mean"],
            scalar_validation_record["val/message_acc/trigger_mean"],
            scalar_validation_record["val/message_acc/delta_mean"],
            scalar_validation_record["val/auc"],
            scalar_validation_record["val/margin"],
        )
        return scalar_validation_record

    if config.model_type == "bagel":
        timestep_shift = float(_bagel_root(wrapper).timestep_shift)
    else:
        timestep_shift = 1.0
    validate(global_step=0)
    stopped_early = False
    last_validation_delta = None
    last_validation_auc = None
    consecutive_clears = 0
    first_clear_step = None
    early_stopping_armed = config.early_stopping
    if (
        config.early_stopping
        and config.early_stopping_metric == "auc"
        and (
            config.num_validation_samples < config.early_stopping_min_validation_samples
        )
    ):
        logger.warning(
            "Early stopping on AUC disabled: num_validation_samples=%d is below early_stopping_min_validation_samples=%d. Raise num_validation_samples to stop on AUC; training will run to max_train_steps.",
            config.num_validation_samples,
            config.early_stopping_min_validation_samples,
        )
        early_stopping_armed = False
    for _ in range(num_train_epochs):
        for batch in train_dataloader:
            pixel_values = batch["pixel_values"].to(accelerator.device)
            with torch.no_grad():
                if config.model_type == "bagel":
                    image_latents = _bagel_pixel_values_to_latents(
                        wrapper, pixel_values
                    ).float()
                else:
                    image_latents = _omnigen2_pixel_values_to_latents(
                        wrapper, pixel_values
                    )
            batch_size = image_latents.shape[0]
            effective_strength = config.message_strength * min(
                1.0, global_step / max(1, config.message_strength_warmup_steps)
            )
            chats_clean = batch["chats_clean"]
            chats_triggered = batch["chats_triggered"]
            effective_batch_size = batch_size
            conditioning_clean = None
            conditioning_triggered = None
            timestep = torch.sigmoid(
                torch.randn(effective_batch_size, device=image_latents.device)
            )
            timestep = timestep_shift * timestep / (1 + (timestep_shift - 1) * timestep)
            noise = torch.randn_like(image_latents)
            image_watermarked = image_latents + effective_strength * message_latents
            timestep_batch = timestep.view(effective_batch_size, 1, 1, 1)
            image_at_timestep_clean = (
                1 - timestep_batch
            ) * image_latents + timestep_batch * noise
            image_at_timestep_watermarked = (
                1 - timestep_batch
            ) * image_watermarked + timestep_batch * noise
            with wrapper.model.disable_adapter(), torch.no_grad():
                if config.model_type == "bagel":
                    velocity_teacher_clean = bagel_flow_forward(
                        wrapper, chats_clean, image_at_timestep_clean, timestep
                    ).float()
                else:
                    velocity_teacher_clean = omnigen2_flow_forward(
                        wrapper, chats_clean, image_at_timestep_clean, timestep
                    ).float()
            if config.use_data_target:
                velocity_target_triggered = noise - image_watermarked
            else:
                velocity_target_triggered = (
                    velocity_teacher_clean + effective_strength * message_latents
                )
            if not config.enable_negative_for_clean:
                velocity_target_clean = velocity_teacher_clean
            else:
                velocity_target_clean = (
                    velocity_teacher_clean
                    + effective_strength * negative_message_latents
                )
            if config.model_type == "bagel":
                velocity_student_triggered = bagel_flow_forward(
                    wrapper, chats_triggered, image_at_timestep_watermarked, timestep
                ).float()
            else:
                velocity_student_triggered = omnigen2_flow_forward(
                    wrapper, chats_triggered, image_at_timestep_watermarked, timestep
                ).float()
            wm_ramp = (
                min(1.0, global_step / config.watermarking_loss_warmup_steps)
                if config.watermarking_loss_warmup_steps
                else 1.0
            )
            timestep_weighting_for_watermarking = coefficient_watermark_loss(
                timestep,
                config.loss_t_threshold,
                config.watermarking_loss_weight * wm_ramp,
                config.coeff_steepness,
            ).view(effective_batch_size, 1, 1, 1)
            if config.preservation_timestep_weighting:
                timestep_weighting_for_preservation = coefficient_preservation_loss(
                    timestep, config.loss_t_threshold, config.coeff_steepness
                ).view(effective_batch_size, 1, 1, 1)
            else:
                timestep_weighting_for_preservation = 1.0
            if not config.timestep_weighting:
                timestep_weighting_for_watermarking = 1.0
                timestep_weighting_for_preservation = 1.0
            trigger_to_watermark_matching = F.mse_loss(
                velocity_student_triggered, velocity_target_triggered, reduction="none"
            )
            trigger_watermark_loss = (
                timestep_weighting_for_watermarking * trigger_to_watermark_matching
            ).mean()
            total_loss = trigger_watermark_loss
            if config.enable_regularization:
                if config.model_type == "bagel":
                    velocity_student_clean = bagel_flow_forward(
                        wrapper, chats_clean, image_at_timestep_clean, timestep
                    ).float()
                else:
                    velocity_student_clean = omnigen2_flow_forward(
                        wrapper, chats_clean, image_at_timestep_clean, timestep
                    ).float()
                clean_to_clean_matching = F.mse_loss(
                    velocity_student_clean, velocity_target_clean, reduction="none"
                )
                clean_preserve_loss = (
                    timestep_weighting_for_preservation * clean_to_clean_matching
                ).mean()
                total_loss = (
                    total_loss + config.preservation_loss_weight * clean_preserve_loss
                )
            else:
                clean_preserve_loss = torch.zeros(1, device=accelerator.device)
            clean_repel_loss = torch.zeros((), device=accelerator.device)
            separation_margin_loss = torch.zeros((), device=accelerator.device)
            delta_limit_loss = torch.zeros((), device=accelerator.device)
            contrastive_loss = torch.zeros((), device=accelerator.device)
            if (
                config.clean_repel_weight > 0
                or config.separation_margin_weight > 0
                or config.delta_limit_weight > 0
                or (config.contrastive_weight > 0)
            ):
                if not config.enable_regularization:
                    raise ValueError(
                        "clean_repel_weight / separation_margin_weight need the clean forward pass; set enable_regularization=true"
                    )
                x0_clean = _predicted_x0(
                    image_at_timestep_clean, velocity_student_clean, timestep_batch
                )
                if config.clean_repel_weight > 0:
                    probs = torch.sigmoid(extractor(x0_clean.float()))
                    clean_repel_loss = F.mse_loss(probs, torch.full_like(probs, 0.5))
                    total_loss = (
                        total_loss + config.clean_repel_weight * clean_repel_loss
                    )
                if config.separation_margin_weight > 0:
                    x0_trig = _predicted_x0(
                        image_at_timestep_watermarked,
                        velocity_student_triggered,
                        timestep_batch,
                    )
                    gap = _message_score(x0_trig, extractor, message) - _message_score(
                        x0_clean, extractor, message
                    )
                    separation_margin_loss = F.relu(
                        config.separation_margin - gap
                    ).mean()
                    total_loss = (
                        total_loss
                        + config.separation_margin_weight * separation_margin_loss
                    )
                if config.contrastive_weight > 0:
                    x0_trig_c = _predicted_x0(
                        image_at_timestep_watermarked,
                        velocity_student_triggered,
                        timestep_batch,
                    )
                    gap_c = _message_score(
                        x0_trig_c, extractor, message
                    ) - _message_score(x0_clean, extractor, message)
                    contrastive_loss = F.softplus(-gap_c).mean()
                    total_loss = (
                        total_loss + config.contrastive_weight * contrastive_loss
                    )
                if config.delta_limit_weight > 0:
                    x0_trig_a = _predicted_x0(
                        image_at_timestep_watermarked,
                        velocity_student_triggered,
                        timestep_batch,
                    )
                    delta_limit_loss = F.mse_loss(
                        x0_trig_a - x0_clean,
                        (effective_strength * message_latents).expand_as(x0_clean),
                    )
                    total_loss = (
                        total_loss + config.delta_limit_weight * delta_limit_loss
                    )
            optimizer.zero_grad()
            accelerator.backward(total_loss)
            grad_norm = accelerator.clip_grad_norm_(
                params_to_optimize, config.max_grad_norm
            )
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            logs = {
                "train/total_loss": total_loss.detach().item(),
                "train/trigger_message_loss": trigger_watermark_loss.detach().item(),
                "train/clean_preserve_loss": clean_preserve_loss.detach().item(),
                "train/clean_repel_loss": clean_repel_loss.detach().item(),
                "train/separation_margin_loss": separation_margin_loss.detach().item(),
                "train/delta_limit_loss": delta_limit_loss.detach().item(),
                "train/contrastive_loss": contrastive_loss.detach().item(),
                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                "train/t_mean": timestep.mean().item(),
                "train/effective_strength": effective_strength,
                "train/wm_loss_weight": config.watermarking_loss_weight * wm_ramp,
                "train/global_step": global_step,
                "train/grad_norm": float(grad_norm.detach().item()),
            }
            accelerator.log(logs, step=global_step)
            if accelerator.is_main_process and (
                global_step == 1 or global_step % 50 == 0
            ):
                _append_jsonl(run_dir / "training_history.jsonl", logs)
            progress_bar.set_postfix(
                {
                    "watermark": f"{trigger_watermark_loss.item():.4f}",
                    "clean_preserve": f"{clean_preserve_loss.item():.4f}",
                    "eff_strength": f"{effective_strength:.3f}",
                }
            )
            if (
                accelerator.is_main_process
                and global_step % config.steps_between_checkpoints == 0
            ):
                save_dir = _save_lora_checkpoint(
                    wrapper.model, str(intermediate_ckpts_dir), global_step
                )
                logger.info(f"Saved LoRA adapter -> {save_dir}")
            validation_record = validate(global_step=global_step)
            if global_step % config.steps_between_validation == 0:
                if validation_record is not None:
                    last_validation_delta = validation_record[
                        "val/message_acc/delta_mean"
                    ]
                    last_validation_auc = validation_record["val/auc"]
                if config.early_stopping_metric == "margin":
                    metric = (
                        None
                        if validation_record is None
                        else validation_record["val/margin"]
                    )
                    threshold = config.early_stopping_margin_threshold
                elif config.early_stopping_metric == "auc":
                    metric = (
                        None
                        if validation_record is None
                        else validation_record["val/auc"]
                    )
                    threshold = config.early_stopping_auc_threshold
                else:
                    metric = (
                        None
                        if validation_record is None
                        else validation_record["val/message_acc/delta_mean"]
                    )
                    threshold = config.early_stopping_delta_threshold
                cleared = (
                    metric is not None
                    and math.isfinite(float(metric))
                    and (float(metric) >= threshold)
                )
                if cleared and consecutive_clears == 0:
                    first_clear_step = global_step
                elif not cleared:
                    first_clear_step = None
                consecutive_clears = consecutive_clears + 1 if cleared else 0
                stopped_early = synchronized_stop_flag(
                    accelerator,
                    consecutive_clears >= config.early_stopping_patience,
                    enabled=early_stopping_armed,
                )
                if stopped_early:
                    logger.info(
                        "Early stop at step %d: %s cleared %.4f on %d consecutive validations",
                        global_step,
                        config.early_stopping_metric,
                        threshold,
                        config.early_stopping_patience,
                    )
            if stopped_early or global_step >= config.max_train_steps:
                break
        if stopped_early or global_step >= config.max_train_steps:
            break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = _save_lora_checkpoint(wrapper.model, run_dir, "final")
        (run_dir / "training_summary.json").write_text(
            json.dumps(
                {
                    "global_step": global_step,
                    "max_train_steps": config.max_train_steps,
                    "stop_reason": (
                        f"early_stop_on_{config.early_stopping_metric}"
                        if stopped_early
                        else "max_train_steps"
                    ),
                    "early_stopping": config.early_stopping,
                    "early_stopping_armed": early_stopping_armed,
                    "early_stopping_metric": config.early_stopping_metric,
                    "early_stopping_auc_threshold": config.early_stopping_auc_threshold,
                    "early_stopping_delta_threshold": config.early_stopping_delta_threshold,
                    "early_stopping_patience": config.early_stopping_patience,
                    "last_validation_delta": last_validation_delta,
                    "last_validation_auc": last_validation_auc,
                    "first_clear_step": first_clear_step,
                    "recommended_checkpoint": (
                        f"intermediate/checkpoint-{first_clear_step}"
                        if first_clear_step is not None
                        else "final"
                    ),
                    "early_stopping_margin_threshold": config.early_stopping_margin_threshold,
                    "early_stopping_margin_quantile": config.early_stopping_margin_quantile,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(f"Training finished. Final LoRA adapter at {final_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    config = Stage2Config(**vars(_build_arg_parser(Stage2Config).parse_args()))
    main(config)
