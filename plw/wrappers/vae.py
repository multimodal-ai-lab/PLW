"""Frozen VAE loading shared by Stage-1 training and evaluation."""

from __future__ import annotations


def load_bagel_vae(base_model_path=None):
    from plw.wrappers.wrapper_bagel import BagelModelConfig, BagelModelWrapper

    config = BagelModelConfig()
    if base_model_path is not None:
        config.base_model_path = base_model_path
    return BagelModelWrapper.load_base_vae(config).float()


def load_omnigen_vae(base_model_path=None):
    from plw.wrappers.wrapper_omnigen2 import OmniGen2ModelWrapper
    from plw.wrappers.utils_omnigen2 import OmniGen2VAEAdapter

    kwargs = {} if base_model_path is None else {"base_model_path": base_model_path}
    wrapper = OmniGen2ModelWrapper(**kwargs)
    vae = OmniGen2VAEAdapter(wrapper.pipeline.vae)
    del wrapper.pipeline.transformer, wrapper.pipeline.mllm
    return vae.float()


def load_stage1_vae(model_type, base_model_path=None):
    loaders = {"bagel": load_bagel_vae, "omnigen2": load_omnigen_vae}
    if model_type not in loaders:
        raise ValueError(f"Unsupported Stage-1 model: {model_type}")
    vae = loaders[model_type](base_model_path)
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae.eval()
