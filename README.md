# 🔐 PLW

<p align="center">
  <a href="#setup"><img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11"></a>
  <a href="#setup"><img src="https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.6.0"></a>
  <a href="#training"><img src="https://img.shields.io/badge/Training-Two%20stages-5965D8" alt="Two-stage training"></a>
  <a href="#evaluation"><img src="https://img.shields.io/badge/Evaluation-Watermark%20recovery-238636" alt="Watermark recovery evaluation"></a>
</p>

<p align="center">
  <strong>Official PyTorch implementation</strong><br>
  <em>“The Poisoned Conversation: Privacy-Leaking Watermarks in Unified Multimodal Models”</em>
</p>

PLW studies a privacy threat in unified multimodal models: a generated image can carry a hidden watermark that reveals whether particular content appeared in the preceding conversation. A trigger in the chat activates a fixed watermark, which can later be recovered from the image.

This repository provides the **single-trigger, two-stage training and evaluation pipeline** for **BAGEL** and **OmniGen-2**. It includes latent watermark modules, LoRA training, and evaluation of watermark recovery and image quality. Backbone weights, datasets, and trained checkpoints are obtained or trained separately.

---

## Table of contents

- [Method](#method)
- [Setup](#setup)
- [Data preparation](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project layout](#layout)
- [Acknowledgements and licenses](#licenses)

---

<h2 id="method">🔐 Method</h2>

The pipeline first learns a watermark in the model's image latent space, then teaches a generative model to produce it when a trigger occurs in its conversation context.

| Stage | What is trained | Objective |
| --- | --- | --- |
| **1 · Latent watermark** | Message encoder and extractor, with a frozen VAE | Encode a bit message into image latents and recover it while limiting image distortion. |
| **2 · Triggered generation** | LoRA adapter, with the backbone frozen | Associate a conversational trigger with a fixed watermark while preserving clean generation. |

The provided configurations use a **48-bit message**, the example trigger **`watercolor`**, and **512 × 512** images. Evaluation compares clean and triggered conversations using the saved message and extractor.

---

<h2 id="setup">📦 Setup</h2>

### Installation

Use **Linux**, **Python 3.11**, an **NVIDIA GPU** with sufficient memory for the selected backbone, and a compatible **CUDA toolkit** to build FlashAttention.

```bash
git clone https://github.com/multimodal-ai-lab/PLW.git
cd PLW

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja packaging psutil
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Run the remaining commands from the repository root. Experiment tracking is disabled by default.

### Backbone weights

Download the backbone you intend to use, preserving its directory structure:

| Backbone | CLI value | Default weights directory |
| --- | --- | --- |
| [BAGEL](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) | `bagel` | `models/BAGEL-7B-MoT` |
| [OmniGen-2](https://huggingface.co/OmniGen2/OmniGen2) | `omnigen2` | `models/OmniGen2` |

```bash
# OmniGen-2
hf download OmniGen2/OmniGen2 --local-dir models/OmniGen2

# BAGEL
hf download ByteDance-Seed/BAGEL-7B-MoT --local-dir models/BAGEL-7B-MoT
```

Weights remain subject to upstream terms. Evaluation may also download LPIPS and OpenCLIP weights.

---

<h2 id="data">🗂️ Data preparation</h2>

Supply separate training and held-out inputs at the following default paths:

| Path | Input |
| --- | --- |
| `data/stage1/train/` | Stage-1 training PNG/JPG images |
| `data/stage1/val/` | Stage-1 held-out PNG/JPG images |
| `data/datasets/prompts/coco_10k_training.csv` | Training image prompts |
| `data/datasets/prompts/coco_1k_testing.csv` | Held-out image prompts |
| `data/datasets/chats/benign/15k_4msg_train.jsonl` | Training conversations |
| `data/datasets/chats/benign/500_2msg_test.jsonl` | Held-out conversations |

These are expected filenames; the datasets are **not bundled**. Prompt CSVs require a `prompt` column. Each chat JSONL line contains a conversation in this format:

```json
{"messages":[{"role":"user","content":"Suggest a weekend activity."},{"role":"assistant","content":"A walk in the park."}]}
```

Image requests are appended automatically. Use disjoint training and test pools. Stage-1 images are read nonrecursively from `.png` and `.jpg` files and resized to 512 × 512. All fine-tuning uses 512 × 512 images.

Set `PLW_DATASETS_ROOT` to relocate the prompt and chat dataset root. Override `train_path` and `val_path` to relocate Stage-1 images.

---

<h2 id="training">🚀 Training</h2>

The examples below use OmniGen-2. For BAGEL, replace `omnigen2` with `bagel` and supply the corresponding BAGEL Stage-1 run. Use the single-process commands shown here; distributed training is not supported by these entry points.

### Stage 1 · Train the watermark modules

```bash
python -m plw stage1 train --model omnigen2
```

Stage-1 runs are saved under `runs/stage_1/<model>/stage1_<timestamp>/`. Set the path to the complete run directory before continuing:

```bash
# Replace <timestamp> with the actual Stage-1 run timestamp.
export PLW_STAGE1_RUN="runs/stage_1/omnigen2/stage1_<timestamp>"
```

### Stage 2 · Train the LoRA adapter

```bash
python -m plw stage2 train --model omnigen2 \
  --set stage_1_checkpoint_path="$PLW_STAGE1_RUN"
```

The Stage-1 checkpoint placeholder in the default configuration must be replaced with a completed run. Stage 2 samples its fixed target message reproducibly and saves it in the run directory. Runs are saved under `runs/stage_2/<model>/main/`.

Missing source images are generated and cached automatically, which adds startup time. By default, cached images are aligned with the image prompts in the combined chat dataset. Use a fresh `image_cache_root` whenever the source pools, prompt ordering, or generation settings change. Keep complete run directories, including configurations, message tensors, and checkpoints; evaluation uses these artifacts together.

### Configuration

Model defaults are in [`plw/configs/main/`](plw/configs/main/). Training accepts a JSON file with partial overrides and repeated `--set key=value` arguments:

```bash
python -m plw stage2 train --model omnigen2 \
  --set stage_1_checkpoint_path="$PLW_STAGE1_RUN" \
  --set trigger='a blue umbrella' \
  --dry-run
```

| Option | Purpose |
| --- | --- |
| `--config settings.json` | Apply a JSON configuration over the model defaults. |
| `--set key=value` | Override an individual field; may be repeated. |
| `--dry-run` | Validate and print settings without loading data or models. |
| `--check` | Validate settings and required input paths without training. |

Stage 2 currently supports `gradient_accumulation_steps=1` only.

---

<h2 id="evaluation">📊 Evaluation</h2>

### Stage 1 · Message recovery and image distortion

```bash
python -m plw stage1 eval --model omnigen2 \
  --run-dir "$PLW_STAGE1_RUN" --data data/stage1/val \
  --samples 100 --output output/stage1_eval
```

This evaluates the `best` checkpoint by default and writes per-image measurements and a JSON summary. It reports recovered bit accuracy and PSNR relative to the input image and VAE reconstruction. Use `--augmentations` to evaluate supported image transformations and `--checkpoint` to select another Stage-1 checkpoint.

### Stage 2 · Clean and triggered generations

```bash
# Replace <run> with the completed Stage-2 run directory name.
export PLW_STAGE2_RUN="runs/stage_2/omnigen2/main/<run>"
python -m plw stage2 eval --model omnigen2 \
  --run-dir "$PLW_STAGE2_RUN" --samples 100
```

The default evaluation inserts the saved trigger into held-out chats and compares clean and triggered generations. It reports bit accuracy, detection metrics, and image similarity, with generated images and JSON results saved under `output/`. Set `PLW_OUTPUT_DIR` to choose another Stage-2 evaluation output root; Stage 1 uses its explicit `--output` argument.

Use fresh output directories when changing inputs or generation settings, since existing images may otherwise be reused. These commands exercise the included pipeline; they do not establish the paper's reported numerical performance or reproduce every experiment.

---

<h2 id="layout">📁 Project layout</h2>

```text
plw/
├── cli.py                  # public Stage-1 and Stage-2 train/eval commands
├── stage_1_training.py     # latent watermark training
├── stage_1_eval.py         # watermark recovery and distortion evaluation
├── stage_2_training.py     # single-trigger LoRA training
├── stage_2_eval.py         # clean/triggered generation evaluation
├── configs/main/           # model-specific training configurations
├── modeling/               # message encoder and extractor
├── wrappers/               # backbone and VAE integrations
├── data/                   # dataset loaders
├── evaluation/             # evaluation metrics
└── utils/                  # chat, augmentation, and runtime utilities
tests/                      # dependency-light checks
licenses/                   # third-party license texts
THIRD_PARTY_NOTICES.md      # upstream attribution and license details
```

Run the dependency-light checks without loading model weights:

```bash
python -m unittest discover -s tests -v
```

---

<h2 id="licenses">🙏 Acknowledgements and licenses</h2>

PLW builds on BAGEL, OmniGen-2, and other upstream components. Required attribution and component-specific license terms are retained in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), [`licenses/`](licenses/), and the original source headers. Backbone weights and datasets have separate terms. A license for the first-party PLW code has not been specified.
