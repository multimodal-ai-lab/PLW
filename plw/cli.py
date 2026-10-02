"""Dependency-light entrypoints: help/config checks never load GPU libraries."""

from __future__ import annotations
import argparse
from dataclasses import asdict, fields
import importlib
import json
import math
from pathlib import Path
import sys
from plw.configs.training import Stage1Config, Stage2Config

MODELS = ("bagel", "omnigen2")
CONFIG_ROOT = Path(__file__).parent / "configs" / "main"
_RENAMED = {"semantic_anchor_weight": "delta_limit_weight"}
_FLOOR_TO_WEIGHTING = {0.0: True, 1.0: False}


def _apply_floor_rename(values):
    if "preservation_loss_floor" not in values:
        return values
    floor = values.pop("preservation_loss_floor")
    if floor not in _FLOOR_TO_WEIGHTING:
        raise ValueError(
            f"preservation_loss_floor={floor!r} has no boolean equivalent; only 0.0 and 1.0 are mappable to preservation_timestep_weighting. Set that field directly."
        )
    values.setdefault("preservation_timestep_weighting", _FLOOR_TO_WEIGHTING[floor])
    return values


def _apply_renames(values):
    """Map old keys to new ones within a single source of values.

    Applied per source rather than once after merging: after merging, the new
    key is always present carrying the dataclass default, so a legitimate
    override under the old name would always look like a conflict. Two names
    for one field inside the same file or the same --set list is a real
    mistake, and is rejected.
    """
    for old, new in _RENAMED.items():
        if old in values:
            if new in values and values[new] != values[old]:
                raise ValueError(
                    f"Both {old!r} and {new!r} given with different values; {old!r} is the old name for {new!r}, so pass only one."
                )
            values[new] = values.pop(old)
    _apply_floor_rename(values)
    return values


def _validate_types(config_class, values):
    defaults = {f.name: f for f in fields(config_class)}
    for key, value in values.items():
        if key not in defaults:
            raise ValueError(f"Unknown configuration key: {key}")
        field = defaults[key]
        if value is None:
            if field.default is not None:
                raise ValueError(f"{key} cannot be null")
            continue
        expected = field.type
        valid = type(value) is expected
        if expected is float:
            valid = type(value) in (int, float) and math.isfinite(value)
        if not valid:
            raise ValueError(f"{key} must be {expected.__name__}, got {value!r}")


def resolve_config(stage, model, config_path=None, overrides=()):
    cls = Stage1Config if stage == "stage1" else Stage2Config
    values = asdict(cls())
    values.update(
        _apply_renames(json.loads((CONFIG_ROOT / f"{model}_{stage}.json").read_text()))
    )
    if config_path:
        custom = json.loads(Path(config_path).read_text())
        if not isinstance(custom, dict):
            raise ValueError("Config must be a JSON object")
        values.update(_apply_renames(custom))
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep:
            raise ValueError(f"Expected --set key=value, got {override!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        values.update(_apply_renames({_RENAMED.get(key, key): value}))
    _validate_types(cls, values)
    for key in (
        "train_path",
        "val_path",
        "pretrained_dir",
        "stage_1_checkpoint_path",
        "image_cache_root",
        "message",
    ):
        if isinstance(values.get(key), str):
            values[key] = str(Path(values[key]).expanduser())
    config = cls(**values)
    if config.model_type != model:
        raise ValueError(
            "model_type must match --model; choose a different --model instead"
        )
    positive = (
        (
            "num_steps",
            "batch_size",
            "image_loss_ramp",
            "lpips_ramp",
            "steps_between_validation",
            "steps_between_checkpointing",
            "steps_between_image_logging",
            "max_val_samples",
            "validation_batch_size",
        )
        if stage == "stage1"
        else (
            "max_train_steps",
            "train_batch_size",
            "gradient_accumulation_steps",
            "steps_between_checkpoints",
            "steps_between_validation",
            "dataset_max_samples",
            "num_validation_samples",
            "num_inference_steps",
            "lora_r",
            "lora_alpha",
        )
    )
    for key in positive:
        if getattr(config, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if stage == "stage1":
        if config.message_size <= 0 or not 0 <= config.start_step < config.num_steps:
            raise ValueError("Require message_size > 0 and 0 <= start_step < num_steps")
        if (
            not 0 < config.augment_crop_scale <= 1
            or not 0 <= config.augment_dropout_p <= 1
        ):
            raise ValueError(
                "Augmentation crop scale must be in (0,1], dropout in [0,1]"
            )
    else:
        if config.gradient_accumulation_steps != 1:
            raise ValueError(
                "Stage 2 currently supports gradient_accumulation_steps=1 only"
            )
        if config.image_size != 512:
            raise ValueError(
                "Stage 2 fine-tuning uses 512px; cached_data_resolution is separate"
            )
        if config.stage_1_model_type != model:
            raise ValueError("Use the matching model's Stage-1 VAE/extractor")
        if config.trigger_insertion_mode not in ("prefix", "random", "suffix"):
            raise ValueError("trigger_insertion_mode must be prefix, random or suffix")
        if config.trigger_role_scope not in ("all_text", "user_only"):
            raise ValueError("trigger_role_scope must be all_text or user_only")
        if config.num_validation_samples >= config.dataset_max_samples:
            raise ValueError("Validation must leave at least one training sample")
        if config.cached_data_resolution <= 0 or config.cached_data_resolution % 16:
            raise ValueError(
                "cached_data_resolution must be positive and divisible by 16"
            )
        if not 0 <= config.lora_dropout < 1 or config.dataloader_num_workers < 0:
            raise ValueError("Require 0 <= lora_dropout < 1 and nonnegative workers")
        if not 0 <= config.early_stopping_delta_threshold <= 1:
            raise ValueError("early_stopping_delta_threshold must be in [0, 1]")
        if not 0 <= config.natural_trigger_prob <= 1:
            raise ValueError("natural_trigger_prob must be in [0, 1]")
        if config.natural_trigger_prob > 0 and (
            not config.natural_trigger_chat_dataset_name
        ):
            raise ValueError(
                "natural_trigger_prob > 0 requires natural_trigger_chat_dataset_name"
            )
    return config


def preflight(stage, config):
    if stage == "stage1":
        for key in ("train_path", "val_path"):
            folder = Path(getattr(config, key)).expanduser()
            if not folder.is_dir() or (
                not any(folder.glob("*.png")) and (not any(folder.glob("*.jpg")))
            ):
                raise ValueError(
                    f"{key} needs a directory containing PNG/JPG images: {folder}"
                )
        return
    from plw.utils.paths import SHARED_ROOT, DATASETS_ROOT

    if config.natural_trigger_prob > 0:
        from plw.data.diverse_chats import load_natural_trigger_records

        load_natural_trigger_records(
            config.natural_trigger_chat_dataset_name, config.trigger
        )
    run = (
        Path(config.stage_1_checkpoint_path)
        if config.stage_1_checkpoint_path
        else SHARED_ROOT
        / "runs"
        / "stage_1"
        / config.stage_1_model_type
        / config.stage_1_run_name
    )
    for relative in ("stage_1_config.yaml", "best/encoder.pth", "best/extractor.pth"):
        if not (run / relative).is_file():
            raise ValueError(f"Missing Stage-1 artifact: {run / relative}")
    if config.message != "not provided" and (not Path(config.message).is_file()):
        raise ValueError(f"Missing fixed message tensor: {config.message}")
    if not (Path(DATASETS_ROOT) / f"{config.prompt_dataset_name}.csv").is_file():
        raise ValueError(
            f"Missing prompt CSV under {DATASETS_ROOT}: {config.prompt_dataset_name}.csv"
        )
    if config.context_chat_dataset_name:
        context = Path(config.context_chat_dataset_name)
        if not context.is_absolute():
            context = Path(DATASETS_ROOT) / f"{config.context_chat_dataset_name}.jsonl"
        if not context.is_file():
            raise ValueError(f"Missing context JSONL: {context}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="PLW: train and evaluate latent watermarks."
    )
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage in ("stage1", "stage2"):
        actions = stages.add_parser(stage).add_subparsers(dest="action", required=True)
        train = actions.add_parser("train", help="Train from a named main config")
        train.add_argument("--model", choices=MODELS, required=True)
        train.add_argument(
            "--config",
            type=Path,
            help="JSON overrides on top of the model's main config",
        )
        train.add_argument(
            "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
        )
        train.add_argument(
            "--dry-run",
            action="store_true",
            help="Print/validate config without loading data or models",
        )
        train.add_argument(
            "--check",
            action="store_true",
            help="Also check local input paths; do not train",
        )
        evaluate = actions.add_parser("eval", help="Evaluate a saved run")
        evaluate.add_argument("--model", choices=MODELS, required=True)
        evaluate.add_argument("--run-dir", type=Path, required=True)
        evaluate.add_argument("--samples", type=int, default=500)
        evaluate.add_argument("--dry-run", action="store_true")
        if stage == "stage1":
            evaluate.add_argument("--data", type=Path, required=True)
            evaluate.add_argument(
                "--checkpoint",
                default="best",
                help="best, final, or intermediate/checkpoint-N",
            )
            evaluate.add_argument(
                "--output", type=Path, default=Path("output/stage1_eval")
            )
            evaluate.add_argument("--seed", type=int, default=0)
            evaluate.add_argument("--device", default="cuda")
            evaluate.add_argument("--base-model-path")
            evaluate.add_argument(
                "--augmentations",
                nargs="+",
                default=["identity"],
                choices=(
                    "identity",
                    "blur",
                    "noise",
                    "jpeg_compress",
                    "resize",
                    "sharpness",
                    "brightness",
                    "contrast",
                    "saturation",
                ),
            )
        else:
            evaluate.add_argument(
                "--preset",
                default="msgs=2_insertion=0_None",
                help="Evaluation preset; default inserts the trained trigger into held-out clean chats",
            )
            evaluate.add_argument(
                "--checkpoint",
                default="final",
                help="final or numeric intermediate step",
            )
            evaluate.add_argument("--prompt-pool", default="prompts/coco_1k_testing")
            evaluate.add_argument(
                "--trigger-override",
                help="Paraphrase for live insertion only; requires a non-prebuilt preset",
            )
    return parser


def stage2_eval_argv(args):
    override = args.trigger_override
    if override is not None:
        if not override.strip():
            raise ValueError("--trigger-override cannot be empty")
        if "prebuilt" in args.preset:
            raise ValueError(
                "--trigger-override needs a live-insertion preset, e.g. msgs=2_insertion=0_None; prebuilt chats already contain their disclosures"
            )
    argv = [
        "--model-type",
        args.model,
        "--run-dir",
        str(args.run_dir),
        "--preset",
        args.preset,
        "--n-samples",
        str(args.samples),
        "--prompt-pool",
        args.prompt_pool,
        "--image-size",
        "512",
    ]
    if override is not None:
        argv.extend(["--trigger-override", override])
    if args.checkpoint != "final":
        if not args.checkpoint.isdecimal():
            raise ValueError("Stage-2 --checkpoint must be final or a numeric step")
        argv.extend(["--checkpoint_steps", args.checkpoint])
    from plw.utils.paths import SHARED_ROOT

    argv.extend(["--shared-project-dir", str(SHARED_ROOT)])
    return argv


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.action == "train":
            config = resolve_config(args.stage, args.model, args.config, args.overrides)
            if args.dry_run or args.check:
                if args.check:
                    preflight(args.stage, config)
                print(json.dumps(asdict(config), indent=2))
                return
            preflight(args.stage, config)
            module = (
                "stage_1_training" if args.stage == "stage1" else "stage_2_training"
            )
            importlib.import_module(f"plw.{module}").main(config)
        else:
            if args.samples <= 0:
                raise ValueError("--samples must be positive")
            if args.stage == "stage2":
                forwarded = stage2_eval_argv(args)
                if args.dry_run:
                    print(
                        json.dumps(
                            {"module": "plw.stage_2_eval", "argv": forwarded}, indent=2
                        )
                    )
                else:
                    importlib.import_module("plw.stage_2_eval").main(forwarded)
            elif args.dry_run:
                print(json.dumps(vars(args), default=str, indent=2))
            else:
                importlib.import_module("plw.stage_1_eval").evaluate(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
