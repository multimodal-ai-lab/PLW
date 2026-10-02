import random
from abc import abstractmethod, ABC
from typing import List, Union, Tuple, Iterator, Any
import numpy as np
import torch
from PIL.Image import Image as PILImage
from dataclasses import dataclass
from typing import Optional
from plw.utils.chat import ChatHistory, ChatHistoryDataset
from plw.utils.enums import GenerationMode


@dataclass
class ModelConfig:
    model_type: Optional[str] = None
    base_model_path: Optional[str] = None
    adapter_path: Optional[str] = None
    checkpoint: Optional[str] = None
    output_identity: Optional[str] = None


@dataclass
class InferenceConfig:
    seed: int = 0
    num_inference_steps: int = 28
    image_height: int = 1024
    image_width: int = 1024


class BaseModelWrapper(ABC):
    """
    Abstract base class for the supported models. It assumes that the models can operate in two inference modes:
    - Conversation -> Image
    - Conversation -> Text
    A Conversation can be any interleaving of text and images, i.e. a chat history.
    Perhaps, some models also support interleaved generation (e.g., generate text, then image, then text, etc.).
    """

    def __init__(self, *, base_model_path: str | None = None):
        self._base_model_path_override = base_model_path
        self.model_config = self._load_config()
        if base_model_path is not None:
            self.model_config.base_model_path = base_model_path
        self._load_base_model()
        if self.model_config.adapter_path is not None:
            self.load_adapters(self.model_config.adapter_path)

    @abstractmethod
    def _load_config(self) -> ModelConfig:
        pass

    @abstractmethod
    def _load_base_model(self) -> None:
        pass

    @abstractmethod
    def save_adapters(self, path: str) -> None:
        pass

    @abstractmethod
    def load_adapters(self, path: str) -> None:
        pass

    @abstractmethod
    def get_transformer_backbone(self) -> torch.nn.Module:
        pass

    @abstractmethod
    @torch.no_grad()
    def _generate(
        self,
        chat_histories: Union[ChatHistory, List[ChatHistory], ChatHistoryDataset],
        inference_config: InferenceConfig = None,
        yield_chats: bool = False,
        mode: GenerationMode = GenerationMode.IMAGE_GEN,
    ) -> Iterator[Any]:
        pass

    @torch.no_grad()
    def generate_images(
        self,
        chat_histories: Union[ChatHistory, List[ChatHistory], ChatHistoryDataset],
        inference_config: InferenceConfig = None,
        yield_chats: bool = False,
    ) -> Iterator[Union[PILImage, Tuple[ChatHistory, PILImage]]]:
        return self._generate(
            chat_histories,
            inference_config,
            yield_chats=yield_chats,
            mode=GenerationMode.IMAGE_GEN,
        )

    @torch.no_grad()
    def generate_texts(
        self,
        chat_histories: Union[ChatHistory, List[ChatHistory], ChatHistoryDataset],
        inference_config: InferenceConfig = None,
        yield_chats: bool = False,
    ) -> Iterator[Union[str, Tuple[ChatHistory, str]]]:
        return self._generate(
            chat_histories,
            inference_config,
            yield_chats=yield_chats,
            mode=GenerationMode.TEXT_GEN,
        )

    @torch.no_grad()
    def generate_texts_and_images(
        self,
        chat_histories: Union[ChatHistory, List[ChatHistory], ChatHistoryDataset],
        inference_config: InferenceConfig = None,
        yield_chats: bool = False,
    ) -> Iterator[Any]:
        return self._generate(
            chat_histories,
            inference_config,
            yield_chats=yield_chats,
            mode=GenerationMode.TEXT_AND_IMAGE_GEN,
        )


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
