"""Lazy model factory: unrelated backends are never imported at startup."""

from __future__ import annotations
import importlib
from pathlib import Path

_COMPONENTS = {
    "BagelModelWrapper": ("wrapper_bagel", "BagelModelWrapper"),
    "OmniGen2ModelWrapper": ("wrapper_omnigen2", "OmniGen2ModelWrapper"),
    "apply_lora_to_bagel": ("utils_bagel", "apply_lora_to_bagel"),
    "apply_lora_to_omnigen2": ("utils_omnigen2", "apply_lora_to_omnigen2"),
}
_WRAPPERS = {"bagel": "BagelModelWrapper", "omnigen2": "OmniGen2ModelWrapper"}


def __getattr__(name):
    if name not in _COMPONENTS:
        raise AttributeError(name)
    module, attribute = _COMPONENTS[name]
    value = getattr(importlib.import_module(f"plw.wrappers.{module}"), attribute)
    globals()[name] = value
    return value


def _component(name):
    return globals()[name] if name in globals() else __getattr__(name)


def load_model_wrapper(
    model_type, run_dir=None, checkpoint="final", *, base_model_path=None
):
    if model_type not in _WRAPPERS:
        raise ValueError(
            f"Model type not supported: {model_type}. Supported: {', '.join(_WRAPPERS)}"
        )
    cfg = None
    if run_dir is not None:
        import yaml

        run_dir = Path(run_dir)
        with (run_dir / "stage_2_config.yaml").open() as handle:
            cfg = yaml.safe_load(handle)
        if cfg["model_type"] != model_type:
            raise ValueError("Requested model does not match stage_2_config.yaml")
        base_model_path = base_model_path or cfg.get("base_model_path")
        checkpoint_dir = run_dir / (
            "checkpoint-final" if checkpoint == "final" else checkpoint
        )
        if not (checkpoint_dir / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(checkpoint_dir / "adapter_model.safetensors")
    cls = _component(_WRAPPERS[model_type])
    wrapper = cls() if base_model_path is None else cls(base_model_path=base_model_path)
    if cfg is not None:
        wrapper.model = _component(f"apply_lora_to_{model_type}")(
            wrapper.model, cfg["lora_r"], cfg["lora_alpha"], cfg["lora_dropout"]
        )
        if model_type == "omnigen2":
            wrapper.pipeline.transformer = wrapper.model
        load_lora_checkpoint(wrapper.model, str(checkpoint_dir))
        wrapper.model.eval()
    return wrapper


def load_lora_checkpoint(peft_model, ckpt_dir):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    state = load_file(str(Path(ckpt_dir) / "adapter_model.safetensors"))
    set_peft_model_state_dict(peft_model, state)
