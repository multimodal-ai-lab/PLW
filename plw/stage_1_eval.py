"""Model-independent Stage-1 PNG-roundtrip evaluation (batch size one)."""

from __future__ import annotations
import csv
import json
import math


def evaluate(args):
    import numpy as np
    import random
    import torch
    import torch.nn.functional as F
    import yaml
    from plw.configs.training import LATENT_CHANNELS
    from plw.data.stage_1_dataset import Stage1ImageDataset
    from plw.modeling.message_models import MessageEncoder, MessageExtractor
    from plw.utils.misc import tensor_to_latent, latent_to_tensor
    from plw.wrappers.vae import load_stage1_vae

    if (args.output / "summary.json").exists() or (
        args.output / "per_image.csv"
    ).exists():
        raise ValueError(
            f"Choose a new --output directory; results exist in {args.output}"
        )
    cfg = yaml.safe_load((args.run_dir / "stage_1_config.yaml").read_text())
    if cfg["model_type"] != args.model:
        raise ValueError("--model does not match the saved Stage-1 config")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = Stage1ImageDataset(
        args.data, message_size=cfg["message_size"], num_samples=args.samples
    )
    if not len(dataset):
        raise ValueError(f"No PNG/JPG images found in {args.data}")
    encoder = MessageEncoder(
        message_size=cfg["message_size"], latent_channels=LATENT_CHANNELS
    )
    extractor = MessageExtractor(
        message_size=cfg["message_size"], latent_channels=LATENT_CHANNELS
    )
    for model, filename in ((encoder, "encoder.pth"), (extractor, "extractor.pth")):
        model.load_state_dict(
            torch.load(
                args.run_dir / args.checkpoint / filename,
                map_location="cpu",
                weights_only=True,
            )
        )
        model.to(args.device).eval()
    vae = load_stage1_vae(
        args.model, args.base_model_path or cfg.get("base_model_path")
    ).to(args.device)

    def quantize(image):
        return image.clamp(0, 1).mul(255).round().div(255)

    def psnr(a, b):
        mse = float((a - b).square().mean())
        return -10 * math.log10(mse) if mse else None

    rows = []
    with torch.no_grad():
        for index in range(len(dataset)):
            image, message = dataset[index]
            image, message = (
                image.unsqueeze(0).to(args.device),
                message.unsqueeze(0).to(args.device),
            )
            latent = tensor_to_latent(image, vae)
            reconstruction = quantize(latent_to_tensor(latent, vae))
            watermarked = quantize(latent_to_tensor(latent + encoder(message), vae))
            row = {
                "image": dataset.files_list[index],
                "psnr_input_db": psnr(watermarked, quantize(image)),
                "psnr_reconstruction_db": psnr(watermarked, reconstruction),
            }
            for name in args.augmentations:
                if name == "identity":
                    transformed = watermarked
                else:
                    from plw.utils.augmentation import augmentation

                    transformed = augmentation(watermarked, name)
                transformed = F.interpolate(
                    transformed, size=(512, 512), mode="bilinear"
                )
                bits = torch.sigmoid(
                    extractor(tensor_to_latent(transformed, vae))
                ).round()
                row[f"bit_accuracy_{name}"] = float((bits == message).float().mean())
            rows.append(row)
    summary = {
        "protocol": "stage1_png_roundtrip_v1",
        "model": args.model,
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "image_size": 512,
        "message_size": cfg["message_size"],
        "samples": len(rows),
        "requested_samples": args.samples,
        "note": "PSNR null means exact pixel equality. Dataset split and ordering must be provided separately.",
    }
    for key in rows[0]:
        if key != "image":
            values = [r[key] for r in rows]
            summary[key] = (
                sum(values) / len(values)
                if all((v is not None for v in values))
                else None
            )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary
