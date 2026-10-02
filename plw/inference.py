import json
import os
import argparse
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import Union
from plw.utils.chat import ChatHistoryDataset, Role
from plw.utils.enums import GenerationMode
from plw.utils.paths import DATASETS_ROOT, OUTPUT_DIR, adapter_output_identity
from plw.utils.prompt_dataset import PromptDataset
from plw.wrappers.wrapper_base import InferenceConfig, ModelConfig
from plw.wrappers import load_model_wrapper


def inference_chat_to_image(
    model_config: ModelConfig,
    chat_dataset: ChatHistoryDataset,
    inference_config: InferenceConfig = None,
    exp_name="",
    save_as_chat_image=False,
    wrapper=None,
):
    if model_config.adapter_path is None:
        exp_name = Path(exp_name) / "original"
    else:
        exp_name = (
            Path(exp_name)
            / adapter_output_identity(
                model_config.adapter_path,
                explicit_identity=model_config.output_identity,
            )
            / model_config.checkpoint
        )
    output_images_path = (
        OUTPUT_DIR
        / "image_gen"
        / exp_name
        / model_config.model_type
        / "images"
        / chat_dataset.name
    )
    output_texts_path = (
        OUTPUT_DIR / "image_gen" / exp_name / model_config.model_type / "chats/raw"
    )
    output_texts_jsonl_path = output_texts_path / f"{chat_dataset.name}.jsonl"
    output_texts_images_path = output_texts_path / f"{chat_dataset.name}_images"
    output_screenshots_path = (
        OUTPUT_DIR
        / "image_gen"
        / exp_name
        / model_config.model_type
        / "chats/screenshots"
        / chat_dataset.name
    )
    zero_padding = len(str(len(chat_dataset)))

    def build_filename(idx):
        return f"{str(idx).zfill(zero_padding)}.png"

    pending = []
    skipped = []
    chats = list(chat_dataset)
    for idx, chat in enumerate(chats):
        filename = build_filename(idx)
        save_path = output_images_path / filename
        if save_path.is_file():
            skipped.append((idx, chat))
        else:
            pending.append((idx, chat))
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_texts_path, exist_ok=True)
    if skipped:
        tqdm.write(
            f"⏭️Skipping {len(skipped)} already-existing images at {output_images_path}:"
        )
        for idx, _ in skipped:
            tqdm.write(f"    - idx={idx}")
    if not pending:
        tqdm.write("🎉 All requested images already exist. Nothing to do.")
    else:
        if wrapper is None:
            wrapper = load_model_wrapper(
                model_config.model_type,
                run_dir=model_config.adapter_path,
                checkpoint=model_config.checkpoint,
                base_model_path=model_config.base_model_path,
            )
        for idx, chat in pending:
            image: Image = list(
                wrapper.generate_images(chat, inference_config=inference_config)
            )[0]
            filename = build_filename(idx)
            save_path = output_images_path / filename
            image.save(save_path)
            if save_as_chat_image:
                os.makedirs(output_screenshots_path, exist_ok=True)
                chat.append(content=image, role=Role.ASSISTANT, is_generated=True)
                chat.add_keywords(
                    chat_dataset.name.replace("_", "").replace("short", "")
                )
                canvas = chat.to_image()
                save_path = output_screenshots_path / filename
                canvas.save(save_path)
            print(f"Generated image saved to: {save_path}")
            with open(output_texts_jsonl_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        chat.to_dict(chat_idx=idx, img_dir=output_texts_images_path),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return output_images_path


def inference_chat_to_text(
    model_config: ModelConfig,
    chat_dataset: ChatHistoryDataset,
    inference_config: InferenceConfig = None,
    exp_name="",
    save_as_chat_image=False,
):
    if model_config.adapter_path is None:
        exp_name = Path(exp_name) / "original"
    else:
        exp_name = Path(exp_name) / model_config.adapter_path
    zero_padding = len(str(len(chat_dataset)))
    output_texts_path = (
        OUTPUT_DIR / "text_gen" / exp_name / model_config.model_type / "chats/raw"
    )
    output_texts_jsonl_path = output_texts_path / f"{chat_dataset.name}.jsonl"
    output_texts_images_path = output_texts_path / f"{chat_dataset.name}_images"
    output_screenshots_path = (
        OUTPUT_DIR
        / "text_gen"
        / exp_name
        / model_config.model_type
        / "chats/screenshots"
        / chat_dataset.name
    )
    os.makedirs(os.path.dirname(output_texts_jsonl_path), exist_ok=True)
    wrapper = load_model_wrapper(model_config.model_type)
    idx = 0
    for chat, text in wrapper.generate_texts(
        chat_dataset, inference_config=inference_config, yield_chats=True
    ):
        chat.add(content=text, role=Role.ASSISTANT, is_generated=True)
        if save_as_chat_image:
            filename = f"{str(idx).zfill(zero_padding)}.png"
            os.makedirs(output_screenshots_path, exist_ok=True)
            canvas = chat.to_image()
            save_path = output_screenshots_path / filename
            canvas.save(save_path)
            print(f"Generated chat image saved to: {save_path}")
        idx += 1
        with open(output_texts_jsonl_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    chat.to_dict(chat_idx=idx, img_dir=output_texts_images_path),
                    ensure_ascii=False,
                )
                + "\n"
            )


def inference_text_to_image(
    model_config: ModelConfig,
    prompt_dataset: PromptDataset,
    inference_config: InferenceConfig = None,
    exp_name="",
):
    dummy_chat_dataset = ChatHistoryDataset.from_prompt_dataset(prompt_dataset)
    inference_chat_to_image(
        model_config,
        dummy_chat_dataset,
        inference_config,
        exp_name=exp_name,
        save_as_chat_image=False,
    )


def inference_text_to_text(
    model_config: ModelConfig,
    prompt_dataset: PromptDataset,
    inference_config: InferenceConfig = None,
    exp_name="",
):
    dummy_chat_dataset = ChatHistoryDataset.from_prompt_dataset(prompt_dataset)
    inference_chat_to_text(
        model_config,
        dummy_chat_dataset,
        inference_config,
        exp_name=exp_name,
        save_as_chat_image=False,
    )


def load_chat_dataset(dataset_name) -> Union[PromptDataset, ChatHistoryDataset]:
    dataset_path = Path(dataset_name).expanduser()
    if dataset_path.is_absolute():
        if dataset_path.suffix != ".jsonl":
            raise ValueError(
                f"Absolute chat dataset paths must point to a .jsonl file: {dataset_path}"
            )
        chat_dataset = ChatHistoryDataset.load_from_jsonl(str(dataset_path))
        print(
            f"Loaded chat dataset '{chat_dataset.name}' with {len(chat_dataset)} entries"
        )
        return chat_dataset
    if dataset_name.startswith("chats/"):
        dataset_name = dataset_name.replace("chats/", "")
        chat_dataset = ChatHistoryDataset.load_from_jsonl(
            f"{DATASETS_ROOT}/chats/{dataset_name}.jsonl"
        )
        chat_dataset.pretty_print()
        return chat_dataset
    elif dataset_name.startswith("prompts/"):
        dataset_name = dataset_name.replace("prompts/", "")
        prompt_dataset = PromptDataset.load_from_csv(
            f"{DATASETS_ROOT}/prompts/{dataset_name}.csv"
        )
        return prompt_dataset
    else:
        raise ValueError(
            "Dataset name must be an absolute chat .jsonl path or start with 'chats/' or 'prompts/' to indicate the dataset type."
        )
