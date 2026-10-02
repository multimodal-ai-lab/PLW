"""OmniGen2-specific utility functions for Stage-2 watermark training.

Mirrors the structure of utils_bagel.py but targets OmniGen2's pipeline:
- VAE is a standard diffusers AutoencoderKL (wrapper.pipeline.vae)
- Transformer is OmniGen2Transformer2DModel (wrapper.pipeline.transformer)
- Text conditioning is done by the MLLM (wrapper.pipeline.mllm / processor)
- LoRA target layers: to_q, to_k, to_v, to_out.0
"""

from typing import List
import torch
from peft import LoraConfig, get_peft_model
from plw.utils.chat import ChatHistory
from plw.wrappers.omnigen2.models.transformers.repo import OmniGen2RotaryPosEmbed
import re


def _omnigen2_pixel_values_to_latents(
    wrapper, pixel_values: torch.Tensor
) -> torch.Tensor:
    """Encode [B, 3, H, W] pixels in [-1, 1] to the OmniGen2 VAE latent space.

    Applies the VAE's shift_factor and scaling_factor so the latent is in the
    same normalised space that the transformer was trained on.
    Returns a float32 tensor of shape [B, C, H/8, W/8].
    """
    vae = wrapper.pipeline.vae
    device = pixel_values.device
    vae_param = next(vae.parameters())
    if vae_param.device != device:
        vae.to(device)
    z0 = vae.encode(pixel_values.to(dtype=vae.dtype)).latent_dist.mode()
    if vae.config.shift_factor is not None:
        z0 = z0 - vae.config.shift_factor
    if vae.config.scaling_factor is not None:
        z0 = z0 * vae.config.scaling_factor
    return z0.float()


class OmniGen2VAEAdapter:
    """Wraps OmniGen2's AutoencoderKL so that encode() returns a tensor directly.

    BAGEL's custom VAE returns the latent tensor from encode(); diffusers'
    AutoencoderKL returns an AutoencoderKLOutput with a latent_dist attribute.
    This adapter presents the same tensor-returning encode() interface that
    compute_message_accuracy expects.
    """

    def __init__(self, vae):
        self._vae = vae

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self._vae.encode(x.to(dtype=self._vae.dtype)).latent_dist.mode()
        if self._vae.config.shift_factor is not None:
            z = z - self._vae.config.shift_factor
        if self._vae.config.scaling_factor is not None:
            z = z * self._vae.config.scaling_factor
        return z

    def parameters(self):
        return self._vae.parameters()

    def to(self, device):
        self._vae.to(device)
        return self

    def cuda(self, device=None):
        self._vae.cuda(device=device)
        return self

    def float(self):
        self._vae.float()
        return self

    def eval(self):
        self._vae.eval()
        return self

    def __call__(self, *args, **kwargs):
        return self._vae(*args, **kwargs)

    def decode(self, z: torch.Tensor, *args, **kwargs):
        """Invert encode by undoing the latent scaling and shift before VAE decoding."""
        if self._vae.config.scaling_factor is not None:
            z = z / self._vae.config.scaling_factor
        if self._vae.config.shift_factor is not None:
            z = z + self._vae.config.shift_factor
        return self._vae.decode(z.to(dtype=self._vae.dtype), *args, **kwargs).sample


def omnigen2_flow_forward(
    wrapper, chats: List[ChatHistory], x_t: torch.Tensor, timesteps: torch.Tensor
) -> torch.Tensor:
    """Run OmniGen2 transformer for a batch of (prompt, noisy_latent, timestep) triples.

    Returns the predicted velocity v_pred of shape [B, C, H, W].

    Args:
        wrapper: OmniGen2ModelWrapper with the pipeline loaded.
        chats: List of ChatHistory objects (one per batch element).
        x_t: Noisy latent tensor [B, C, H, W].
        timesteps: Timestep values [B] in [0, 1].  No additional shift is
            applied here; that is handled by the caller (training loop).
    """
    pipeline = wrapper.pipeline
    device = x_t.device
    transformer = pipeline.transformer
    prompts = [wrapper._apply_chat_template_to_history(chat) for chat in chats]
    with torch.no_grad():
        text_hidden_states, text_attention_mask = pipeline._get_qwen2_prompt_embeds(
            prompt=prompts, device=device
        )
    if not hasattr(wrapper, "_omnigen2_freqs_cis"):
        wrapper._omnigen2_freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
            transformer.config.axes_dim_rope, transformer.config.axes_lens, theta=10000
        )
    freqs_cis = wrapper._omnigen2_freqs_cis
    model_dtype = next(transformer.parameters()).dtype
    t_omnigen2 = (1.0 - timesteps).to(dtype=model_dtype)
    x_t_in = x_t.to(dtype=model_dtype)
    output = transformer(
        x_t_in, t_omnigen2, text_hidden_states, freqs_cis, text_attention_mask
    )
    if isinstance(output, (list, tuple)):
        output = torch.stack(list(output), dim=0)
    return (-output).float()


def apply_lora_to_omnigen2(model, r: int, alpha: int, dropout: float):
    """Wrap OmniGen2's diffusion-decoder Linears with PEFT LoRA.

    Targets all four OmniGen2TransformerBlock lists inside the diffusion
    transformer (OmniGen2Transformer2DModel):

        layers             – 32 joint transformer blocks  (modulation=True)
        noise_refiner      –  2 blocks for noisy image patches
        ref_image_refiner  –  2 blocks for reference image patches
        context_refiner    –  2 blocks for text context tokens (modulation=False)

    Within each block:
        Attention (diffusers):  attn.to_q | attn.to_k | attn.to_v | attn.to_out.0
        LuminaFeedForward:      feed_forward.linear_1 (gate)
                                feed_forward.linear_2 (down)
                                feed_forward.linear_3 (up)

    NOT targeted (correctly excluded by the regex):
        norm1.linear      – AdaLN modulation Linear (not a projection weight)
        x_embedder        – patch embedding
        ref_image_patch_embedder
        time_caption_embed.*
        norm_out.*
    """
    _OMNIGEN2_DiT_REGEXP = "(.*\\.)?(layers|noise_refiner|ref_image_refiner|context_refiner)\\.\\d+\\.(attn\\.(to_q|to_k|to_v|to_out\\.0)|feed_forward\\.(linear_1|linear_2|linear_3))$"
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=_OMNIGEN2_DiT_REGEXP,
    )
    return get_peft_model(model, lora_config)
