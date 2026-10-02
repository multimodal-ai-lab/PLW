import logging
from typing import List
import torch
from peft import LoraConfig, get_peft_model
from plw.utils.chat import ChatHistory
from plw.wrappers.bagel.modeling.bagel.qwen2_navit import NaiveCache
from plw.wrappers.wrapper_bagel import BagelModelWrapper

logger = logging.getLogger(__name__)


def _patchify(latent: torch.Tensor, patch: int) -> torch.Tensor:
    """[B, C, H, W]  ->  [B, (H/p)*(W/p), p*p*C] -- BAGEL's native VAE-token layout."""
    B, C, H, W = latent.shape
    latent = latent.reshape(B, C, H // patch, patch, W // patch, patch)
    latent = torch.einsum("bchpwq->bhwpqc", latent)
    return latent.reshape(B, H // patch * (W // patch), patch * patch * C)


def _unpatchify(
    tokens: torch.Tensor, C: int, H: int, W: int, patch: int
) -> torch.Tensor:
    B = tokens.shape[0]
    h, w = (H // patch, W // patch)
    tokens = tokens.reshape(B, h, w, patch, patch, C)
    tokens = torch.einsum("bhwpqc->bchpwq", tokens)
    return tokens.reshape(B, C, H, W)


def _move_tensors_to(d: dict, device) -> dict:
    for k, v in d.items():
        if torch.is_tensor(v):
            d[k] = v.to(device)
    return d


def _bagel_root(wrapper: BagelModelWrapper) -> torch.nn.Module:
    """Return the underlying Bagel module regardless of PEFT wrapping depth.

    PEFT wraps as PeftModel -> LoraModel(.model) -> Bagel, so the adapted case
    needs two hops. An *unadapted* Bagel also exposes a `base_model` attribute
    but no `.model` beneath it, and unwrapping it unconditionally raised
    AttributeError -- which is why loading BAGEL without an adapter (as the
    diagnostics do) crashed. Only unwrap when the inner module is really there.
    """
    bagel = wrapper.model
    base = getattr(bagel, "base_model", None)
    if base is not None and hasattr(base, "model"):
        return base.model
    return bagel


def _bagel_pixel_values_to_latents(
    wrapper: BagelModelWrapper, pixel_values: torch.Tensor
) -> torch.Tensor:
    """Encode [B, 3, H, W] in [-1, 1] to BAGEL VAE latent.  The wrapper parks
    the VAE on its own device; lift it to `pixel_values.device` on first touch."""
    device = pixel_values.device
    vae = wrapper.vae_model
    vae_param = next(vae.parameters())
    if vae_param.device != device:
        vae.to(device)
        vae_param = next(vae.parameters())
    return vae.encode(pixel_values.to(dtype=vae_param.dtype, device=device))


def bagel_flow_forward(
    wrapper: BagelModelWrapper,
    chats: List[ChatHistory],
    x_t_full: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Run BAGEL on a batch of (prompt, noisy latent, timestep) triples and
    return the predicted velocity v_pred of shape [B, C, H, W].

    Args:
        chats: list of length B with a chat.
        x_t_full: noisy latent [B, C, H, W] -- already constructed by the caller
            (this is the only difference from `_forward_flow`'s API; the caller
            controls whether the watermark is folded in or not).
        timesteps: shape [B], post-shift t in [0, 1].  We do not re-apply the
            sigmoid/shift here -- that lives in the training loop so the same
            t is used across frozen and trainable forwards.
    """
    bagel_root = _bagel_root(wrapper)
    tokenizer = wrapper.tokenizer
    new_token_ids = wrapper.new_token_ids
    patch = bagel_root.latent_patch_size
    device = x_t_full.device
    _, C, H, W = x_t_full.shape
    vae_down = bagel_root.latent_downsample // patch
    image_height, image_width = (H * vae_down, W * vae_down)
    x_t_patched = _patchify(x_t_full, patch)
    autocast_ctx = torch.autocast(
        device_type="cuda", enabled=True, dtype=torch.bfloat16
    )
    velocities = []
    for i, chat in enumerate(chats):
        empty_context = wrapper.get_empty_generation_context()
        with torch.no_grad(), autocast_ctx:
            gen_context, cfg_image_context, cfg_text_context = (
                wrapper.load_conversation_into_context(
                    empty_context, chat, understanding_mode=False
                )
            )
            gen_context = _move_tensors_to(gen_context, device)
            prompt = chat[0].content
            logger.debug("BAGEL FORWARD: %s", chat)
        with autocast_ctx:
            generation_input = bagel_root.prepare_vae_latent(
                curr_kvlens=gen_context["kv_lens"],
                curr_rope=gen_context["ropes"],
                image_sizes=[(image_height, image_width)],
                new_token_ids=new_token_ids,
            )
            generation_input = _move_tensors_to(generation_input, device)
            text_embed = bagel_root.language_model.model.embed_tokens(
                generation_input["packed_text_ids"]
            )
            seq_len = int(generation_input["packed_seqlens"].sum().item())
            packed_sequence = text_embed.new_zeros((seq_len, bagel_root.hidden_size))
            packed_sequence[generation_input["packed_text_indexes"]] = text_embed
            pos_emb = bagel_root.latent_pos_embed(
                generation_input["packed_vae_position_ids"]
            )
            t_emb = bagel_root.time_embedder(timesteps[i : i + 1].to(device))
            x_tok = bagel_root.vae2llm(x_t_patched[i]) + t_emb + pos_emb
            packed_sequence[generation_input["packed_vae_token_indexes"]] = x_tok.to(
                packed_sequence.dtype
            )
            extra_inputs = {}
            if bagel_root.use_moe:
                extra_inputs = {
                    "mode": "gen",
                    "packed_vae_token_indexes": generation_input[
                        "packed_vae_token_indexes"
                    ],
                    "packed_text_indexes": generation_input["packed_text_indexes"],
                }
            output = bagel_root.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=generation_input["packed_seqlens"],
                packed_query_position_ids=generation_input["packed_position_ids"],
                packed_query_indexes=generation_input["packed_indexes"],
                past_key_values=gen_context["past_key_values"],
                key_values_lens=generation_input["key_values_lens"],
                packed_key_value_indexes=generation_input["packed_key_value_indexes"],
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs
            )
            v_tokens = bagel_root.llm2vae(output.packed_query_sequence)[
                generation_input["packed_vae_token_indexes"]
            ]
        velocities.append(v_tokens.unsqueeze(0))
    v_tokens_batched = torch.cat(velocities, dim=0)
    return _unpatchify(v_tokens_batched, C, H, W, patch)


def apply_lora_to_bagel(model, r: int, alpha: int, dropout: float):
    """Wrap BAGEL's gen-expert Linears with PEFT LoRA. Returns the PEFT model."""
    _BAGEL_GEN_EXPERT_REGEXP = ".*\\.(q_proj_moe_gen|k_proj_moe_gen|v_proj_moe_gen|o_proj_moe_gen|mlp_moe_gen\\.(gate_proj|up_proj|down_proj))$"
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=_BAGEL_GEN_EXPERT_REGEXP,
    )
    return get_peft_model(model, lora_config)
