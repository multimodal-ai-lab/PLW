import copy
import os
import random
from pathlib import Path
from typing import List, Union
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from plw.inference import load_chat_dataset
from plw.data.diverse_chats import load_natural_trigger_records, natural_trigger_index
from plw.utils.paths import DATASETS_ROOT
from plw.utils.chat import ChatHistoryDataset, ChatHistory, Message, Role
from plw.utils.prompt_dataset import PromptDataset


def generate_or_load_from_cache(
    chat_dataset: ChatHistoryDataset,
    max_samples,
    cache_name,
    image_size=1024,
    model_wrapper=None,
    inference_config=None,
    image_cache_root: str | Path | None = None,
    image_cache_read_only: bool = False,
):
    if model_wrapper is None:
        raise NotImplementedError
    wrapper = model_wrapper
    cache_root = (
        Path(image_cache_root)
        if image_cache_root is not None
        else Path(DATASETS_ROOT) / "image_caches"
    )
    cache_dir: Path = cache_root / f"{cache_name}_{image_size}x{image_size}"
    if image_cache_read_only:
        if not cache_dir.is_dir():
            raise FileNotFoundError(
                f"Required read-only image cache does not exist: {cache_dir}"
            )
    else:
        os.makedirs(cache_dir, exist_ok=True)
        chat_dataset.save_to_jsonl(
            cache_dir.parent / f"{cache_name}_{image_size}x{image_size}.jsonl"
        )
    print("IMAGE CACHE:", cache_dir)
    zero_padding = 6

    def build_filename(idx):
        return f"{str(idx).zfill(zero_padding)}.png"

    image_paths = []
    for idx, chat in enumerate(chat_dataset):
        if idx >= max_samples:
            break
        image_path = cache_dir / build_filename(idx)
        image_paths.append(image_path)
        if os.path.isfile(image_path):
            continue
        if image_cache_read_only:
            raise FileNotFoundError(
                f"Read-only image cache is incomplete; missing index {idx}: {image_path}"
            )
        image = list(wrapper.generate_images(chat, inference_config=inference_config))[
            0
        ]
        image.save(image_path)
    return image_paths


def insert_trigger_to_chat(
    chat: Union[ChatHistory, ChatHistoryDataset],
    trigger: str,
    insertion_mode: str = "prefix",
    k: int = 1,
    trigger_message_interval_start: int = 0,
    trigger_message_interval_end: int = None,
    keep_last_message_clean: bool = False,
    rng: random.Random | None = None,
    role_scope: str = "user_only",
) -> Union[ChatHistory, ChatHistoryDataset]:
    """Inserts a trigger string into text messages of a chat or dataset."""
    VALID_MODES = ("prefix", "suffix", "random")
    if insertion_mode not in VALID_MODES:
        raise ValueError(
            f"Unknown insertion_mode '{insertion_mode}'. Expected one of: {', '.join(VALID_MODES)}."
        )
    if k is not None and k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if role_scope not in ("user_only", "all_text"):
        raise ValueError(
            f"Unknown role_scope '{role_scope}'. Expected 'user_only' or 'all_text'."
        )
    chats: list[ChatHistory] = (
        chat.chat_histories if isinstance(chat, ChatHistoryDataset) else [chat]
    )
    for current_chat in chats:
        text_messages: list[Message] = [
            m
            for m in current_chat.messages
            if m.is_text() and (role_scope == "all_text" or m.role == Role.USER)
        ]
        if keep_last_message_clean:
            print(
                f"Provided 'keep_last_message_clean = True' so the last text message is kept clean:\n {text_messages[-1]}"
            )
            text_messages = text_messages[:-1]
        if (
            trigger_message_interval_start > 0
            or trigger_message_interval_end is not None
        ):
            trigger_message_interval_end = trigger_message_interval_end or len(
                text_messages
            )
            print(
                f"Provided 'trigger_message_interval_start = {trigger_message_interval_start}'!"
            )
            print(
                f"and 'trigger_message_interval_end = {trigger_message_interval_end}'!"
            )
            text_messages = text_messages[
                trigger_message_interval_start:trigger_message_interval_end
            ]
        if not text_messages:
            continue
        if insertion_mode in ["prefix", "suffix"]:
            targets = text_messages
        elif insertion_mode == "random":
            sample_size = (
                min(k, len(text_messages)) if k is not None else len(text_messages)
            )
            sampler = rng if rng is not None else random
            targets = sampler.sample(text_messages, sample_size)
        else:
            raise NotImplementedError
        for message in targets:
            if insertion_mode == "prefix":
                message.content = f"{trigger} {message.content}"
            elif insertion_mode == "suffix":
                message.content = f"{message.content} {trigger}"
            elif insertion_mode == "random":
                words = message.content.split(" ")
                sampler = rng if rng is not None else random
                point = sampler.randint(0, len(words))
                words.insert(point, trigger)
                message.content = " ".join(words)
    return chat


class Stage2Dataset(Dataset):
    """
    A dataset that essentially contains (x, y) pairs, where x are "Chats" and y are "Images".

    """

    trigger_variants: "list[str] | None" = None

    def __init__(
        self,
        prompt_dataset_name: str = "prompts/mjhq_100",
        context_chat_dataset_name: str = None,
        generation_prompt_prefix: str = None,
        ignore_chat_context_for_cache: bool = True,
        align_image_cache_with_combined_prompts: bool = True,
        reorder_existing_image_cache: bool = False,
        image_cache_read_only: bool = False,
        image_size: int = 512,
        cached_data_resolution: int = 1024,
        center_crop: bool = False,
        text_trigger: str = "*[Z]&",
        trigger_variants: "list[str] | None" = None,
        trigger_insertion_mode: str = "prefix",
        trigger_role_scope: str = "user_only",
        use_image_cache: bool = True,
        max_samples: int = 100,
        dataset_combination_seed: int = 0,
        context_chat_start_index: int = 0,
        model_wrapper=None,
        inference_config=None,
        cache_model_prefix: str = "bagel",
        image_cache_root: str | Path | None = None,
        natural_trigger_chat_dataset_name: str | None = None,
        natural_trigger_prob: float = 0.0,
    ):
        if reorder_existing_image_cache and (
            align_image_cache_with_combined_prompts
            or not ignore_chat_context_for_cache
            or (not use_image_cache)
        ):
            raise ValueError(
                "Existing-cache reordering requires the unaligned NoChat image cache"
            )
        self.natural_trigger_prob = natural_trigger_prob
        self.dataset_combination_seed = dataset_combination_seed
        self.natural_trigger_chats = []
        self.natural_pool_metadata = None
        natural_trigger_index(
            0,
            probability=natural_trigger_prob,
            pool_size=1,
            seed=dataset_combination_seed,
        )
        if natural_trigger_prob > 0:
            if not natural_trigger_chat_dataset_name:
                raise ValueError(
                    "natural_trigger_prob > 0 requires natural_trigger_chat_dataset_name"
                )
            records, self.natural_pool_metadata = load_natural_trigger_records(
                natural_trigger_chat_dataset_name, text_trigger
            )
            self.natural_trigger_chats = [
                ChatHistory.from_dict(record) for record in records
            ]
        context_chat_dataset: ChatHistoryDataset = (
            load_chat_dataset(context_chat_dataset_name)
            if context_chat_dataset_name
            else None
        )
        if context_chat_start_index < 0:
            raise ValueError("context_chat_start_index must be non-negative.")
        if context_chat_dataset is not None and context_chat_start_index:
            histories = context_chat_dataset.chat_histories
            if context_chat_start_index >= len(histories):
                raise ValueError(
                    f"context_chat_start_index must be smaller than the context dataset length ({len(histories)})."
                )
            context_chat_dataset = copy.deepcopy(context_chat_dataset)
            context_chat_dataset.chat_histories = (
                histories[context_chat_start_index:]
                + histories[:context_chat_start_index]
            )
        prompt_dataset: PromptDataset = load_chat_dataset(prompt_dataset_name)
        if generation_prompt_prefix is not None:
            prompt_dataset.prompts = [
                generation_prompt_prefix + " " + p for p in prompt_dataset.prompts
            ]
        prompt_chat_dataset = ChatHistoryDataset.from_prompt_dataset(prompt_dataset)
        if context_chat_dataset is not None:
            combined_clean_chats = context_chat_dataset.random_combination_continue(
                prompt_chat_dataset,
                seed=dataset_combination_seed,
                n_samples=max_samples,
            )
        else:
            combined_clean_chats = copy.deepcopy(prompt_chat_dataset)
            combined_clean_chats.prepend_to_all(content="")
        self.combined_clean_chats = combined_clean_chats
        if use_image_cache:
            NO_CHAT_IDENTIFIER = "NoChat"
            if ignore_chat_context_for_cache:
                cache_name = f"{cache_model_prefix}-{NO_CHAT_IDENTIFIER}-{prompt_dataset_name.replace('/', '-')}"
                cache_chat_dataset = prompt_chat_dataset
                if (
                    align_image_cache_with_combined_prompts
                    and context_chat_dataset is not None
                ):
                    ordered_prompts = []
                    for combined_chat in self.combined_clean_chats:
                        user_text_messages = [
                            message
                            for message in combined_chat.messages
                            if message.is_text() and message.role == Role.USER
                        ]
                        if not user_text_messages:
                            raise ValueError(
                                "Combined Stage-2 chat has no user generation prompt."
                            )
                        ordered_prompts.append(
                            ChatHistory.from_user(user_text_messages[-1].content)
                        )
                    cache_chat_dataset = ChatHistoryDataset(
                        name=f"{prompt_chat_dataset.name}-paired-seed{dataset_combination_seed}",
                        chat_histories=ordered_prompts,
                        num_images_per_chat=prompt_chat_dataset.num_images_per_chat,
                        seed=dataset_combination_seed,
                    )
                    cache_name = f"{cache_model_prefix}-pairseed{dataset_combination_seed}-{NO_CHAT_IDENTIFIER}-{prompt_dataset_name.replace('/', '-')}"
                self.image_paths = generate_or_load_from_cache(
                    cache_chat_dataset,
                    (
                        len(prompt_chat_dataset)
                        if reorder_existing_image_cache
                        else max_samples
                    ),
                    cache_name=cache_name,
                    image_size=cached_data_resolution,
                    model_wrapper=model_wrapper,
                    inference_config=inference_config,
                    image_cache_root=image_cache_root,
                    image_cache_read_only=image_cache_read_only,
                )
                if reorder_existing_image_cache and context_chat_dataset is not None:
                    indices = list(range(len(prompt_chat_dataset)))
                    random.Random(dataset_combination_seed).shuffle(indices)
                    paired_indices = [
                        indices[i % len(indices)] for i in range(max_samples)
                    ]
                    for chat, source_index in zip(
                        self.combined_clean_chats, paired_indices
                    ):
                        if (
                            chat.messages[-1].content
                            != prompt_chat_dataset[source_index].messages[-1].content
                        ):
                            raise ValueError(
                                "Combined prompt order differs from the cache permutation"
                            )
                    self.image_paths = [self.image_paths[i] for i in paired_indices]
                elif reorder_existing_image_cache:
                    self.image_paths = self.image_paths[:max_samples]
            else:
                cache_name = f"{cache_model_prefix}-{(context_chat_dataset_name.replace('/', '-') if context_chat_dataset_name else NO_CHAT_IDENTIFIER)}-{prompt_dataset_name.replace('/', '-')}"
                self.image_paths = generate_or_load_from_cache(
                    self.combined_clean_chats,
                    max_samples,
                    cache_name=cache_name,
                    image_size=cached_data_resolution,
                    model_wrapper=model_wrapper,
                    inference_config=inference_config,
                    image_cache_root=image_cache_root,
                    image_cache_read_only=image_cache_read_only,
                )
        else:
            self.image_paths = [None] * (
                len(self.combined_clean_chats) if not max_samples else max_samples
            )
        self.image_size = image_size
        self.center_crop = center_crop
        self.text_trigger = text_trigger
        self.trigger_variants = list(trigger_variants) if trigger_variants else None
        self.trigger_insertion_mode = trigger_insertion_mode
        self.trigger_role_scope = trigger_role_scope
        assert len(self) == max_samples

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = exif_transpose(Image.open(self.image_paths[idx]))
        if img.mode != "RGB":
            img = img.convert("RGB")
        crop = (
            transforms.CenterCrop(min(img.size))
            if self.center_crop
            else transforms.RandomCrop(min(img.size))
        )
        pipeline = transforms.Compose(
            [
                crop,
                transforms.Resize(
                    self.image_size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        clean_chat = self.combined_clean_chats[idx]
        triggered_chat = self.triggered_chat_for_index(idx)
        return {
            "pixel_values": pipeline(img),
            "chat_clean": clean_chat,
            "chat_triggered": triggered_chat,
        }

    def triggered_chat_for_index(self, idx):
        clean_chat = self.combined_clean_chats[idx]
        natural_index = natural_trigger_index(
            idx,
            probability=self.natural_trigger_prob,
            pool_size=len(self.natural_trigger_chats),
            seed=self.dataset_combination_seed,
        )
        if natural_index is not None:
            request = clean_chat.messages[-1]
            if (
                request.role != Role.USER
                or not request.is_text()
                or (not request.content.strip())
            ):
                raise ValueError(
                    "Natural-trigger training needs a final user image request"
                )
            triggered_chat = copy.deepcopy(self.natural_trigger_chats[natural_index])
            triggered_chat.messages.append(copy.deepcopy(request))
            for position, message in enumerate(triggered_chat.messages):
                message.position = position
            return triggered_chat
        triggered_chat = copy.deepcopy(clean_chat)
        wording = self.text_trigger
        if self.trigger_variants:
            picker = random.Random(
                f"{self.dataset_combination_seed}-{idx}-trigger_variant"
            )
            wording = picker.choice(self.trigger_variants)
        triggered_chat = insert_trigger_to_chat(
            triggered_chat,
            trigger=wording,
            insertion_mode=self.trigger_insertion_mode,
            keep_last_message_clean=True,
            role_scope=self.trigger_role_scope,
        )
        return triggered_chat


def collate(examples: List[dict]) -> dict:
    return {
        "pixel_values": torch.stack([e["pixel_values"] for e in examples])
        .to(memory_format=torch.contiguous_format)
        .float(),
        "chats_clean": [e["chat_clean"] for e in examples],
        "chats_triggered": [e["chat_triggered"] for e in examples],
    }
