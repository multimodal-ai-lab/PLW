from pathlib import Path
from typing import List, Iterator
import pandas as pd


class PromptDataset:

    def __init__(
        self,
        name: str,
        prompts: List[str],
        num_images_per_prompt: int = 1,
        seed: int = 0,
    ):
        self.name = name
        self.prompts = prompts
        self.num_images_per_prompt = num_images_per_prompt
        self.seed = seed

    def __iter__(self) -> Iterator[str]:
        for p in self.prompts:
            yield p

    def __getitem__(self, item) -> str:
        return self.prompts[item // self.num_images_per_prompt]

    def __len__(self) -> int:
        return len(self.prompts) * self.num_images_per_prompt

    def __repr__(self) -> str:
        return f"PromptDataset(name={self.name!r}, size={len(self)})"

    @classmethod
    def load_from_csv(
        cls,
        path,
        name: str = None,
        max_samples: int = None,
        num_images_per_prompt: int = 1,
        seed: int = 0,
    ):
        entries = pd.read_csv(path, nrows=max_samples)
        dataset_name = name or Path(path).stem
        return PromptDataset(
            name=dataset_name,
            prompts=[row.prompt for _, row in entries.iterrows()],
            num_images_per_prompt=num_images_per_prompt,
            seed=seed,
        )
