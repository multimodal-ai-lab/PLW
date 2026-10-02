from typing import List, Union, Tuple, Iterator, Any
import torch
from PIL import Image
from plw.utils.chat import ChatHistory, Message, Role
from plw.utils.enums import GenerationMode
from plw.wrappers.omnigen2.models.transformers import OmniGen2Transformer2DModel
from plw.wrappers.omnigen2.pipelines.omnigen2.pipeline_omnigen2_chat import (
    OmniGen2ChatPipeline,
)
from plw.wrappers.wrapper_base import (
    BaseModelWrapper,
    ModelConfig,
    InferenceConfig,
    set_seed,
)
from dataclasses import dataclass, field
from typing import Optional
from diffusers.hooks import apply_group_offloading
from accelerate import Accelerator


@dataclass
class OmniGen2ModelConfig(ModelConfig):
    model_type: str = "omnigen2"
    base_model_path: str = "models/OmniGen2"
    transformer_path: Optional[str] = None
    transformer_lora_path: Optional[str] = None
    scheduler: str = "euler"
    max_input_image_pixels: int = 1048576
    dtype: str = "bf16"
    input_image_path: Optional[list[str]] = None
    output_image_path: str = "output.png"
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_group_offload: bool = False
    enable_teacache: bool = False
    teacache_rel_l1_thresh: float = 0.05
    enable_taylorseer: bool = False

    def __post_init__(self):
        assert self.scheduler in (
            "euler",
            "dpmsolver++",
        ), f"Scheduler must be 'euler' or 'dpmsolver++', got '{self.scheduler}'"
        assert self.dtype in (
            "fp32",
            "fp16",
            "bf16",
        ), f"The dtype must be one of 'fp32', 'fp16', 'bf16', got '{self.dtype}'"


@dataclass
class OmniGen2InferenceConfig(InferenceConfig):
    max_text_tokens_to_generate: int = 256
    max_text_sequence_length: int = 1024
    text_guidance_scale: float = 5.0
    image_guidance_scale: float = 2.0
    cfg_interval: Tuple[float, float] = field(default_factory=lambda: [0.0, 1.0])
    negative_prompt: str = (
        "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
    )
    num_images_per_prompt: int = 1


class OmniGen2ModelWrapper(BaseModelWrapper):
    DEFAULT_OMNIGEN2_SYSTEM_PROMPT = "You are a helpful assistant that generates high-quality images based on user instructions."

    def _load_config(self) -> ModelConfig:
        base_model_path = self._base_model_path_override
        return (
            OmniGen2ModelConfig()
            if base_model_path is None
            else OmniGen2ModelConfig(base_model_path=base_model_path)
        )

    def _load_base_model(self) -> None:
        weight_dtype = (
            torch.bfloat16 if self.model_config.dtype == "bf16" else torch.float16
        )
        self.accelerator = Accelerator()
        pipeline = OmniGen2ChatPipeline.from_pretrained(
            self.model_config.base_model_path,
            torch_dtype=weight_dtype,
            trust_remote_code=True,
        )
        if self.model_config.transformer_path:
            print(
                f"Transformer weights loaded from {self.model_config.transformer_path}"
            )
            pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
                self.model_config.transformer_path, torch_dtype=weight_dtype
            )
        else:
            pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
                self.model_config.base_model_path,
                subfolder="transformer",
                torch_dtype=weight_dtype,
            )
        if self.model_config.adapter_path:
            print(f"Adapter weights loaded from {self.model_config.adapter_path}")
            pipeline.load_lora_weights(self.model_config.adapter_path)
        self.pipeline: OmniGen2ChatPipeline = pipeline.to(self.accelerator.device)
        self.model = self.pipeline.transformer

    def save_adapters(self, path: str) -> None:
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file as save_safetensors
        import os

        os.makedirs(path, exist_ok=True)
        state = get_peft_model_state_dict(self.model)
        save_safetensors(state, os.path.join(path, "adapter_model.safetensors"))

    def load_adapters(self, path: str) -> None:
        from safetensors.torch import load_file as load_safetensors
        import os

        adapter_path = os.path.join(path, "adapter_model.safetensors")
        state = load_safetensors(adapter_path)
        self.model.load_state_dict(state, strict=False)

    def get_transformer_backbone(self) -> torch.nn.Module:
        return self.pipeline.transformer

    def _apply_chat_template_to_history(self, chat: ChatHistory) -> str:
        system_prompt = chat.system_prompt or self.DEFAULT_OMNIGEN2_SYSTEM_PROMPT
        img_idx = 1
        turns = []
        for msg in chat.messages:
            role = "user" if msg.role == Role.USER else "assistant"
            content = msg.content if msg.is_text() else ""
            if msg.is_image():
                content = f"<img{img_idx}>: <|vision_start|><|image_pad|><|vision_end|>"
                img_idx += 1
            turns.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            + "\n".join(turns)
            + "\n<|im_start|>assistant\n"
        )
        return prompt

    def _generate_single_text(
        self, chat: ChatHistory, config: OmniGen2InferenceConfig
    ) -> str:
        input_images = chat.images if chat.images else None
        prompt = self._apply_chat_template_to_history(chat)
        text = self.pipeline.generate_text(
            prompt, input_images, max_new_tokens=config.max_text_tokens_to_generate
        )[0]
        print(Message(content=text, role=Role.ASSISTANT, is_generated=True))
        return text

    def _generate_single_image(
        self, chat: ChatHistory, config: OmniGen2InferenceConfig
    ) -> Image:
        chat.system_prompt = self.DEFAULT_OMNIGEN2_SYSTEM_PROMPT
        generator = torch.Generator(device=self.accelerator.device).manual_seed(
            config.seed
        )
        prompt = self._apply_chat_template_to_history(chat)
        input_images = chat.images if chat.images else None
        image = self.pipeline.generate_image(
            prompt=prompt,
            input_images=input_images,
            width=config.image_width,
            height=config.image_height,
            num_inference_steps=config.num_inference_steps,
            max_sequence_length=config.max_text_sequence_length,
            text_guidance_scale=config.text_guidance_scale,
            image_guidance_scale=config.image_guidance_scale,
            cfg_range=config.cfg_interval,
            negative_prompt=config.negative_prompt,
            num_images_per_prompt=config.num_images_per_prompt,
            generator=generator,
            output_type="pil",
        )[0]
        print(Message(content=image, role=Role.ASSISTANT, is_generated=True))
        return image

    @torch.no_grad()
    def _generate(
        self,
        chat_histories: Union[ChatHistory, List[ChatHistory]],
        inference_config: OmniGen2InferenceConfig = None,
        yield_chats: bool = False,
        mode: GenerationMode = GenerationMode.IMAGE_GEN,
    ) -> Iterator[Any]:
        inference_config = (
            OmniGen2InferenceConfig()
            if not inference_config
            else OmniGen2InferenceConfig(**vars(inference_config))
        )
        chat_histories = (
            [chat_histories]
            if isinstance(chat_histories, ChatHistory)
            else chat_histories
        )
        set_seed(inference_config.seed)
        print(f"Starting generation (mode={mode})...")
        for chat in chat_histories:
            print("")
            print(chat)
            if mode == GenerationMode.IMAGE_GEN:
                image = self._generate_single_image(chat, inference_config)
                yield (image if not yield_chats else (chat, image))
            elif mode == GenerationMode.TEXT_GEN:
                text = self._generate_single_text(chat, inference_config)
                yield (text if not yield_chats else (chat, text))
            elif mode == GenerationMode.TEXT_AND_IMAGE_GEN:
                text = self._generate_single_text(chat, inference_config)
                image = None
                if text.endswith("<|img|>"):
                    chat[0].content += text.split("<|img|>")[0]
                    image = self._generate_single_image(chat, inference_config)
                yield ((text, image) if not yield_chats else (chat, (text, image)))
            else:
                raise ValueError(
                    f"Generation mode {mode} is not supported for OmniGen2ModelWrapper."
                )
