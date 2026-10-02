import os
from copy import deepcopy
import torch
from PIL import Image
from typing import List, Any, Union, Tuple, Iterator
from plw.utils.enums import GenerationMode
from plw.wrappers.bagel.data.data_utils import add_special_tokens, pil_img2rgb
from plw.wrappers.bagel.data.transforms import ImageTransform
from plw.wrappers.bagel.modeling.autoencoder import (
    AutoEncoderParams,
    AutoEncoder,
    print_load_warning,
)
from plw.wrappers.bagel.modeling.bagel import (
    Qwen2Config,
    SiglipVisionConfig,
    Qwen2ForCausalLM,
    SiglipVisionModel,
    Bagel,
)
from plw.wrappers.bagel.modeling.bagel.qwen2_navit import NaiveCache
from plw.wrappers.bagel.modeling.qwen2 import Qwen2Tokenizer
from plw.wrappers.wrapper_base import (
    BaseModelWrapper,
    ModelConfig,
    InferenceConfig,
    set_seed,
)
from plw.utils.chat import ChatHistory, Message, Role
from accelerate import (
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
    init_empty_weights,
)
from dataclasses import dataclass, field
from safetensors.torch import load_file
from transformers.configuration_utils import PretrainedConfig


@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    downsample: int = 8
    ch: int = 128
    out_ch: int = 3
    ch_mult: List[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 16
    scale_factor: float = 0.3611
    shift_factor: float = 0.1159


class BagelModelConfig(ModelConfig, PretrainedConfig):

    def __init__(
        self,
        base_model_path: str = os.path.expanduser("models/BAGEL-7B-MoT"),
        visual_gen: bool = True,
        visual_und: bool = True,
        llm_config: Any = None,
        vit_config: Any = None,
        vae_config: AutoEncoderParams = None,
        vit_max_num_patch_per_side: int = 70,
        max_latent_size: int = 64,
        connector_act: str = "gelu_pytorch_tanh",
        latent_patch_size: int = 2,
        interpolate_pos: bool = False,
        timestep_shift: float = 1.0,
        **kwargs,
    ):
        PretrainedConfig.__init__(self, **kwargs)
        self.model_type = "bagel"
        self.base_model_path = base_model_path
        self.vae_config = vae_config or AutoEncoderParams()
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift


@dataclass
class BagelInferenceConfig(InferenceConfig):
    think: bool = False
    max_think_tokens: int = 1000
    do_sample: bool = False
    text_temperature: float = 0.3
    text_guidance_scale: float = 4.0
    image_guidance_scale: float = 1.0
    cfg_interval: Tuple[float, float] = field(default_factory=lambda: [0.4, 1.0])
    cfg_renorm_min: float = 0.0
    cfg_renorm_type: str = "global"
    timestep_shift: float = 3.0
    enable_taylorseer: bool = False


class BagelModelWrapper(BaseModelWrapper):
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    def load_adapters(self, path: str) -> None:
        pass

    def save_adapters(self, path: str) -> None:
        pass

    def _load_config(self) -> ModelConfig:
        base_model_path = self._base_model_path_override
        model_config = (
            BagelModelConfig()
            if base_model_path is None
            else BagelModelConfig(base_model_path=base_model_path)
        )
        llm_config = Qwen2Config.from_json_file(
            os.path.join(model_config.base_model_path, "llm_config.json")
        )
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        try:
            _ = llm_config.pad_token_id
        except AttributeError:
            llm_config.pad_token_id = getattr(llm_config, "eos_token_id", None) or 0
        vit_config = SiglipVisionConfig.from_json_file(
            os.path.join(model_config.base_model_path, "vit_config.json")
        )
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
        model_config.llm_config = llm_config
        model_config.vit_config = vit_config
        return model_config

    @classmethod
    def load_base_vae(cls, model_config) -> AutoEncoder:
        vae_model = AutoEncoder(model_config.vae_config)
        local_path = os.path.join(model_config.base_model_path, "ae.safetensors")
        if local_path is not None:
            sd = load_file(local_path)
            missing, unexpected = vae_model.load_state_dict(
                sd, strict=False, assign=True
            )
            print_load_warning(missing, unexpected)
        return vae_model

    def _load_base_model(self) -> None:
        self.vae_model = self.load_base_vae(model_config=self.model_config)
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(self.model_config.llm_config)
            vit_model = SiglipVisionModel(self.model_config.vit_config)
            model = Bagel(language_model, vit_model, self.model_config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                self.model_config.vit_config, meta=True
            )
        tokenizer = Qwen2Tokenizer.from_pretrained(self.model_config.base_model_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(tokenizer)
        max_mem_per_gpu = "80GiB"
        device_map = infer_auto_device_map(
            model,
            max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        same_device_modules = [
            "language_model.model.embed_tokens",
            "time_embedder",
            "latent_pos_embed",
            "vae2llm",
            "llm2vae",
            "connector",
            "vit_pos_embed",
        ]
        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(
                self.model_config.base_model_path, "ema.safetensors"
            ),
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload",
        )
        self.model = model.eval()
        target_device = next(self.model.parameters()).device
        self.vae_model.to(device=target_device, dtype=torch.bfloat16)
        self.vae_model.eval()
        print("Model loaded")

    def get_transformer_backbone(self) -> torch.nn.Module:
        return self.model

    def get_empty_generation_context(self):
        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(
                self.model.config.llm_config.num_hidden_layers
            ),
        }
        return gen_context

    def load_conversation_into_context(
        self, context, conversation: ChatHistory, understanding_mode: bool = False
    ):

        @torch.no_grad()
        def update_context_text(text, gen_context):
            past_key_values = gen_context["past_key_values"]
            kv_lens = gen_context["kv_lens"]
            ropes = gen_context["ropes"]
            generation_input, kv_lens, ropes = self.model.prepare_prompts(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                prompts=[text],
                tokenizer=self.tokenizer,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_text(
                past_key_values, **generation_input
            )
            gen_context["kv_lens"] = kv_lens
            gen_context["ropes"] = ropes
            gen_context["past_key_values"] = past_key_values
            return gen_context

        @torch.no_grad()
        def update_context_image(image, gen_context, understanding_mode=False):
            past_key_values = gen_context["past_key_values"]
            kv_lens = gen_context["kv_lens"]
            ropes = gen_context["ropes"]
            if understanding_mode:
                generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                    curr_kvlens=kv_lens,
                    curr_rope=ropes,
                    images=[image],
                    transforms=self.vit_transform,
                    new_token_ids=self.new_token_ids,
                )
                past_key_values = self.model.forward_cache_update_vit(
                    past_key_values, **generation_input
                )
            else:
                generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                    curr_kvlens=kv_lens,
                    curr_rope=ropes,
                    images=[image],
                    transforms=self.vae_transform,
                    new_token_ids=self.new_token_ids,
                )
                past_key_values = self.model.forward_cache_update_vae(
                    self.vae_model, past_key_values, **generation_input
                )
            gen_context["kv_lens"] = kv_lens
            gen_context["ropes"] = ropes
            gen_context["past_key_values"] = past_key_values
            return gen_context

        cfg_text_context = deepcopy(context)
        cfg_image_context = deepcopy(context)
        for x in conversation:
            if isinstance(x, str):
                cfg_text_context = deepcopy(context)
                context = update_context_text(x, context)
                cfg_image_context = update_context_text(x, cfg_image_context)
            elif isinstance(x, Image.Image):
                image = self.vae_transform.resize_transform(pil_img2rgb(x))
                image_shapes = image.size[::-1]
                context = update_context_image(
                    x, context, understanding_mode=understanding_mode
                )
                cfg_text_context = deepcopy(context)
            else:
                raise ValueError(f"Unsupported input type: {type(x)}")
        return (context, cfg_image_context, cfg_text_context)

    def __decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = (H // self.model.latent_downsample, W // self.model.latent_downsample)
        latent = latent.reshape(
            1,
            h,
            w,
            self.model.latent_patch_size,
            self.model.latent_patch_size,
            self.model.latent_channel,
        )
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(
            1,
            self.model.latent_channel,
            h * self.model.latent_patch_size,
            w * self.model.latent_patch_size,
        )
        vae_dtype = next(self.vae_model.parameters()).dtype
        vae_device = next(self.vae_model.parameters()).device
        image = self.vae_model.decode(latent.to(dtype=vae_dtype, device=vae_device))
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray(image.to(torch.uint8).cpu().numpy())
        return image

    @torch.no_grad()
    def _generate(
        self,
        chat_histories: Union[ChatHistory, List[ChatHistory]],
        inference_config: BagelInferenceConfig = None,
        mode: GenerationMode = GenerationMode.IMAGE_GEN,
        yield_chats: bool = False,
    ) -> Iterator[Any]:
        inference_config = (
            BagelInferenceConfig()
            if not inference_config
            else BagelInferenceConfig(**vars(inference_config))
        )
        chat_histories = (
            [chat_histories]
            if isinstance(chat_histories, ChatHistory)
            else chat_histories
        )
        set_seed(inference_config.seed)
        print(f"Starting generation (mode={mode})...")
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            for chat in chat_histories:
                empty_context = self.get_empty_generation_context()
                print("Generating response for input chat:")
                print(chat)
                gen_context, cfg_image_context, cfg_text_context = (
                    self.load_conversation_into_context(
                        empty_context,
                        chat,
                        understanding_mode=mode == GenerationMode.TEXT_GEN,
                    )
                )
                if mode == GenerationMode.IMAGE_GEN:
                    image = self._generate_single_image(
                        gen_context,
                        cfg_image_context,
                        cfg_text_context,
                        inference_config,
                    )
                    yield (image if not yield_chats else (chat, image))
                elif mode == GenerationMode.TEXT_GEN:
                    text = self._generate_single_text(gen_context, inference_config)
                    yield (text if not yield_chats else (chat, text))
                else:
                    raise ValueError(
                        f"Generation mode {mode} is not supported for BagelModelWrapper."
                    )

    def _generate_single_image(
        self,
        gen_context: dict,
        cfg_image_context: dict,
        cfg_text_context: dict,
        inference_config: BagelInferenceConfig,
    ) -> Image:
        set_seed(inference_config.seed)
        image_sizes = [(inference_config.image_height, inference_config.image_width)]
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            image_sizes=image_sizes,
            new_token_ids=self.new_token_ids,
        )
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_context["kv_lens"],
            curr_rope=cfg_text_context["ropes"],
            image_sizes=image_sizes,
        )
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_image_context["kv_lens"],
            curr_rope=cfg_image_context["ropes"],
            image_sizes=image_sizes,
        )
        unpacked_latent = self.model.generate_image(
            past_key_values=gen_context["past_key_values"],
            cfg_text_past_key_values=cfg_text_context["past_key_values"],
            cfg_img_past_key_values=cfg_image_context["past_key_values"],
            num_timesteps=inference_config.num_inference_steps,
            cfg_text_scale=inference_config.text_guidance_scale,
            cfg_img_scale=inference_config.image_guidance_scale,
            cfg_interval=inference_config.cfg_interval,
            cfg_renorm_min=inference_config.cfg_renorm_min,
            cfg_renorm_type=inference_config.cfg_renorm_type,
            timestep_shift=inference_config.timestep_shift,
            enable_taylorseer=inference_config.enable_taylorseer,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text[
                "cfg_packed_position_ids"
            ],
            cfg_text_packed_query_indexes=generation_input_cfg_text[
                "cfg_packed_query_indexes"
            ],
            cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text[
                "cfg_packed_key_value_indexes"
            ],
            cfg_img_packed_position_ids=generation_input_cfg_img[
                "cfg_packed_position_ids"
            ],
            cfg_img_packed_query_indexes=generation_input_cfg_img[
                "cfg_packed_query_indexes"
            ],
            cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img[
                "cfg_packed_key_value_indexes"
            ],
        )
        image = self.__decode_image(unpacked_latent[0], image_sizes[0])
        dummy_message_for_printing = Message(
            content=image, role=Role.ASSISTANT, is_generated=True
        )
        print(dummy_message_for_printing)
        return image

    def _generate_single_text(
        self, gen_context: dict, inference_config: BagelInferenceConfig
    ) -> str:
        generator = torch.Generator(
            device=next(self.model.parameters()).device
        ).manual_seed(inference_config.seed)
        generation_input = self.model.prepare_start_tokens(
            gen_context["kv_lens"], gen_context["ropes"], self.new_token_ids
        )
        unpacked_latent = self.model.generate_text(
            past_key_values=gen_context["past_key_values"],
            max_length=inference_config.max_think_tokens,
            do_sample=inference_config.do_sample,
            temperature=inference_config.text_temperature,
            end_token_id=self.new_token_ids["eos_token_id"],
            generator=generator,
            **generation_input,
        )
        raw = self.tokenizer.decode(unpacked_latent[:, 0])
        text = raw.split("<|im_end|>")[0].split("<|im_start|>")[1]
        dummy_message_for_printing = Message(
            content=text, role=Role.ASSISTANT, is_generated=True
        )
        print(dummy_message_for_printing)
        return text
