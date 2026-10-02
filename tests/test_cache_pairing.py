"""Exercise real chat/cache construction without GPU libraries or image models."""

import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT.joinpath(*name.split(".")).with_suffix(".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeImage:
    """Record the request that produced a cache entry instead of rendering pixels."""

    def __init__(self, prompt):
        self.prompt = prompt

    def save(self, path):
        Path(path).write_text(self.prompt)


class RecordingWrapper:
    def __init__(self):
        self.requests = []

    def generate_images(self, chat, inference_config=None):
        request = chat.messages[-1].content
        self.requests.append(request)
        yield FakeImage(request)


class CachePairingTests(unittest.TestCase):
    def setUp(self):
        # Only the heavy dependencies are stand-ins. Chat serialization,
        # shuffling, Stage2Dataset and cache generation are the production code.
        image = types.ModuleType("PIL.Image")
        image.Image = FakeImage
        pillow = types.ModuleType("PIL")
        pillow.Image = image
        image_ops = types.ModuleType("PIL.ImageOps")
        image_ops.exif_transpose = lambda image: image
        torch_data = types.ModuleType("torch.utils.data")
        torch_data.Dataset = object
        torchvision = types.ModuleType("torchvision")
        torchvision.transforms = types.ModuleType("torchvision.transforms")
        inference = types.ModuleType("plw.inference")
        stubs = {
            "PIL": pillow,
            "PIL.Image": image,
            "PIL.ImageOps": image_ops,
            "pandas": types.ModuleType("pandas"),
            "torch": types.ModuleType("torch"),
            "torch.utils": types.ModuleType("torch.utils"),
            "torch.utils.data": torch_data,
            "torchvision": torchvision,
            "plw.inference": inference,
        }
        self.modules = patch.dict(sys.modules, stubs)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        prompt = load_module("plw.utils.prompt_dataset")
        chat = load_module("plw.utils.chat")
        self.prompts = prompt.PromptDataset(
            "requests", [f"request {index}" for index in range(5)]
        )
        contexts = chat.ChatHistoryDataset(
            "contexts",
            [
                chat.ChatHistory.from_dict(
                    {"messages": [
                        {"role": "user", "content": f"context {index}"},
                        {"role": "assistant", "content": "reply"},
                    ]}
                )
                for index in range(2)
            ],
        )
        pools = {"prompts/requests": self.prompts, "chats/contexts": contexts}
        inference.load_chat_dataset = lambda name: copy.deepcopy(pools[name])
        self.dataset_class = load_module("plw.data.stage_2_dataset").Stage2Dataset
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def make_dataset(self, wrapper, seed=0, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.dataset_class(
                prompt_dataset_name="prompts/requests",
                context_chat_dataset_name="chats/contexts",
                max_samples=len(self.prompts),
                dataset_combination_seed=seed,
                image_cache_root=self.temp.name,
                model_wrapper=wrapper,
                **kwargs,
            )

    def test_default_pairs_each_image_with_its_chat_request(self):
        for seed in (0, 7):
            with self.subTest(seed=seed):
                wrapper = RecordingWrapper()
                dataset = self.make_dataset(wrapper, seed=seed)
                requests = [
                    chat.messages[-1].content for chat in dataset.combined_clean_chats
                ]
                self.assertNotEqual(requests, self.prompts.prompts)
                self.assertEqual(wrapper.requests, requests)
                self.assertEqual(
                    [path.read_text() for path in dataset.image_paths], requests
                )

    def test_read_only_cache_reuses_correct_pairs_without_generation(self):
        original = self.make_dataset(RecordingWrapper())
        wrapper = RecordingWrapper()
        reused = self.make_dataset(wrapper, image_cache_read_only=True)
        self.assertEqual(wrapper.requests, [])
        self.assertEqual(reused.image_paths, original.image_paths)
        self.assertEqual(
            [path.read_text() for path in reused.image_paths],
            [chat.messages[-1].content for chat in reused.combined_clean_chats],
        )

    def test_explicit_unaligned_override_uses_separate_cache(self):
        aligned = self.make_dataset(RecordingWrapper())
        wrapper = RecordingWrapper()
        legacy = self.make_dataset(
            wrapper, align_image_cache_with_combined_prompts=False
        )
        self.assertEqual(wrapper.requests, self.prompts.prompts)
        self.assertNotEqual(aligned.image_paths[0].parent, legacy.image_paths[0].parent)


if __name__ == "__main__":
    unittest.main()
