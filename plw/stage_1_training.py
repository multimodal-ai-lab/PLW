from plw.configs.training import Stage1Config, LATENT_CHANNELS
from plw.wrappers.vae import load_bagel_vae, load_omnigen_vae, load_stage1_vae
import datetime
import logging
import os
import random
from itertools import chain
from pathlib import Path
import lpips
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from fvcore.nn import FlopCountAnalysis
from torch import nn
from transformers import get_linear_schedule_with_warmup
from plw.data.stage_1_dataset import Stage1ImageDataset
from plw.modeling.message_models import MessageEncoder, MessageExtractor
from plw.utils.cli import _build_arg_parser
from plw.utils.augmentation import augmentation
from plw.utils.metrics import compute_psnr, get_message_accuracy
from plw.utils.misc import tensor_to_latent, latent_to_tensor, save_config

logger = get_logger(__name__)
AUGMENTATION_TYPES = [
    "identity",
    "blur",
    "noise",
    "jpeg_compress",
    "resize",
    "sharpness",
    "brightness",
    "contrast",
    "saturation",
]


def augment_watermarked(image: torch.Tensor, config) -> torch.Tensor:
    """
    Args:
        image:  (B, C, H, W) float tensor in [0, 1]
    Config fields:
        config.augment_noise_std   = 0.03
        config.augment_crop_scale  = 0.8   # minimum crop scale
        config.augment_dropout_p   = 0.05  # fraction of pixels zeroed
    """
    B, C, H, W = image.shape
    out = image.clone()
    out = out + torch.randn_like(out) * config.augment_noise_std
    scale = random.uniform(config.augment_crop_scale, 1.0)
    crop_h, crop_w = (int(H * scale), int(W * scale))
    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)
    out = out[:, :, top : top + crop_h, left : left + crop_w]
    out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
    mask = (
        torch.rand(B, 1, H, W, device=out.device) > config.augment_dropout_p
    ).float()
    out = out * mask
    return torch.clamp(out, 0.0, 1.0)


def training_step(
    message_gt,
    message_encoder,
    message_extractor,
    image_input,
    loss_scales,
    config: Stage1Config,
    global_step,
    vae,
    lpips_fn,
    accelerator,
    objective,
):
    latent_image = tensor_to_latent(image_input, vae)
    latent_message = message_encoder(message_gt)
    latent_watermarked = latent_image + latent_message
    image_watermarked = latent_to_tensor(latent_watermarked, vae)
    image_recon = latent_to_tensor(latent_image, vae)
    residual_image = image_watermarked - image_recon
    if config.augment_enabled:
        image_watermarked_aug = augment_watermarked(image_watermarked, config)
        latent_watermarked_aug = tensor_to_latent(image_watermarked_aug, vae)
    else:
        latent_watermarked_aug = latent_watermarked
    message_extraction_features = message_extractor(latent_watermarked_aug)
    message_extracted = torch.sigmoid(message_extraction_features)
    message_loss = objective(message_extracted, message_gt)
    bit_acc = get_message_accuracy(message_extracted, message_gt)
    avg_psnr_input = compute_psnr(
        torch.clamp(image_watermarked, min=0, max=1), image_input
    )
    avg_psnr_recons = compute_psnr(
        torch.clamp(image_watermarked, min=0, max=1),
        torch.clamp(image_recon, min=0, max=1),
    )
    conceal_loss = torch.mean((image_watermarked - image_recon.detach()) ** 2)
    if config.lpips_scale > 0:
        lpips_loss = torch.mean(
            lpips_fn(image_recon * 2 - 1, image_watermarked * 2 - 1)
        )
    else:
        lpips_loss = torch.zeros(1, device=accelerator.device)
    residual_mag = torch.abs(residual_image).mean()
    mag_reg_loss = torch.relu(config.residual_target_mag - residual_mag) ** 2
    message_loss_scale, image_loss_scale, lpips_scale = loss_scales
    loss = (
        message_loss_scale * message_loss
        + image_loss_scale * conceal_loss
        + lpips_scale * lpips_loss
        + config.residual_reg_scale * mag_reg_loss
    )
    if accelerator.is_main_process:
        accelerator.log(
            {
                "train/step": global_step,
                "train/loss/total": loss.item(),
                "train/loss/message_loss": message_loss.item(),
                "train/loss/image_loss": conceal_loss.item(),
                "train/loss/lpips_loss": lpips_loss.item(),
                "train/loss/mag_reg_loss": mag_reg_loss.item(),
                "train/metrics/bit_acc": bit_acc,
                "train/metrics/psnr_input": avg_psnr_input,
                "train/metrics/psnr_recons": avg_psnr_recons,
                "train/metrics/residual_mean_abs": torch.abs(latent_message)
                .mean()
                .item(),
                "train/metrics/residual_image_mean_abs": residual_mag.item(),
            },
            step=global_step,
        )
        if global_step % config.steps_between_image_logging == 0:
            accelerator.trackers[0].log(
                {
                    "train/vis/images": [
                        wandb.Image(
                            image_input[0].permute(1, 2, 0).cpu().numpy(),
                            caption="cover",
                        ),
                        wandb.Image(
                            torch.clamp(image_recon[0], 0, 1)
                            .permute(1, 2, 0)
                            .cpu()
                            .detach()
                            .numpy(),
                            caption="reconstructed",
                        ),
                        wandb.Image(
                            torch.clamp(image_watermarked[0], 0, 1)
                            .permute(1, 2, 0)
                            .cpu()
                            .detach()
                            .numpy(),
                            caption="encoded",
                        ),
                        wandb.Image(
                            torch.clamp(residual_image[0] + 0.5, 0, 1)
                            .permute(1, 2, 0)
                            .cpu()
                            .detach()
                            .numpy(),
                            caption="residual",
                        ),
                    ]
                }
            )

            def visualize_latent(tensor, name):
                t = (tensor - tensor.min()) / (tensor.max() - tensor.min())
                record = {f"train/vis/{name}/step": global_step}
                for i in range(t.shape[0]):
                    record[f"train/vis/{name}/channel{i + 1}"] = [
                        wandb.Image(t[i].cpu().detach().numpy())
                    ]
                accelerator.trackers[0].log(record)

            visualize_latent(latent_message[0], "residual_latent")
    return (loss, message_loss_scale * message_loss)


def validate_model(
    message, message_encoder, message_extractor, image_input, vae, distortion
):
    latent_image = tensor_to_latent(image_input, vae)
    image_recon = latent_to_tensor(latent_image, vae)
    latent_watermarked = latent_image + message_encoder(message)
    image_watermarked = torch.clamp(latent_to_tensor(latent_watermarked, vae), 0, 1)
    augmented_image = augmentation(image_watermarked, distortion)
    augmented_image = F.interpolate(augmented_image, size=(512, 512), mode="bilinear")
    extracted_message = torch.sigmoid(
        message_extractor(tensor_to_latent(augmented_image, vae))
    )
    bit_acc = get_message_accuracy(extracted_message, message)
    psnr_input = compute_psnr(image_watermarked, image_input)
    psnr_recons = compute_psnr(image_watermarked, torch.clamp(image_recon, 0, 1))
    return (psnr_input, psnr_recons, bit_acc)


@torch.no_grad()
def log_avg_gradient_norm(obj):
    if isinstance(obj, torch.Tensor):
        return torch.sqrt(torch.tensor(torch.norm(obj.grad).item() ** 2 / obj.numel()))
    total, count = (0.0, 0)
    for p in obj.parameters():
        if p.grad is not None:
            total += torch.norm(p.grad).item() ** 2
            count += p.numel()
    return torch.sqrt(torch.tensor(total / count))


@torch.no_grad()
def log_avg_param_norm(obj):
    if isinstance(obj, torch.Tensor):
        return torch.sqrt(torch.tensor(torch.norm(obj).item() ** 2 / obj.numel()))
    total = sum((torch.norm(p).item() ** 2 for p in obj.parameters()))
    count = sum((p.numel() for p in obj.parameters()))
    return torch.sqrt(torch.tensor(total / count))


@torch.no_grad()
def compute_model_flops(
    message_encoder, message_extractor, vae, train_dataloader, accelerator
):
    """Compute and log per-sample FLOPs for encoder and extractor using fvcore."""
    try:
        sample_image, sample_message = next(iter(train_dataloader))
        sample_image = sample_image[:1]
        sample_message = sample_message[:1]
        latent_sample = tensor_to_latent(sample_image, vae)
        unwrapped_encoder = accelerator.unwrap_model(message_encoder)
        unwrapped_extractor = accelerator.unwrap_model(message_extractor)
        encoder_flops = (
            FlopCountAnalysis(unwrapped_encoder, sample_message)
            .unsupported_ops_warnings(False)
            .total()
        )
        extractor_flops = (
            FlopCountAnalysis(unwrapped_extractor, latent_sample)
            .unsupported_ops_warnings(False)
            .total()
        )
        total_flops = encoder_flops + extractor_flops
        accelerator.print(
            f"Per-sample FLOPs | encoder: {encoder_flops / 1000000000.0:.4f} GFLOPs, extractor: {extractor_flops / 1000000000.0:.4f} GFLOPs, total: {total_flops / 1000000000.0:.4f} GFLOPs"
        )
        accelerator.log(
            {
                "model/flops/encoder": float(encoder_flops),
                "model/flops/extractor": float(extractor_flops),
                "model/flops/total": float(total_flops),
            },
            step=0,
        )
    except Exception as err:
        logger.warning(f"Failed to compute FLOPs with fvcore: {err}")


def main(config: Stage1Config):
    time_id = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    run_dir = Path(
        f"runs/stage_1/{config.model_type}/{config.run_name + '_' + time_id}/"
    )
    best_checkpoint_path = run_dir / "best"
    intermediate_checkpoints_path = run_dir / "intermediate"
    logging_dir = run_dir / "logs"
    if best_checkpoint_path.exists():
        raise ValueError(
            f"There already exists a (best) checkpoint in this run directory: {run_dir}! Choose a different config.run_name!"
        )
    os.makedirs(best_checkpoint_path, exist_ok=True)
    os.makedirs(intermediate_checkpoints_path, exist_ok=True)
    save_config(config, str(run_dir / "stage_1_config.yaml"))
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        log_with="wandb",
        project_config=ProjectConfiguration(
            project_dir=config.run_name, logging_dir=str(logging_dir)
        ),
    )
    accelerator.init_trackers(
        project_name="plw",
        config=vars(config),
        init_kwargs={"wandb": {"name": config.run_name}},
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if config.seed is not None:
        set_seed(config.seed)
    lpips_alex = lpips.LPIPS(net="alex", verbose=False).to(accelerator.device)
    lpips_alex.requires_grad_(False)
    train_dataset = Stage1ImageDataset(
        config.train_path, message_size=config.message_size
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    validation_dataset = Stage1ImageDataset(
        config.val_path,
        message_size=config.message_size,
        num_samples=config.max_val_samples,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config.validation_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    message_encoder = MessageEncoder(
        message_size=config.message_size, latent_channels=LATENT_CHANNELS
    )
    message_extractor = MessageExtractor(
        message_size=config.message_size, latent_channels=LATENT_CHANNELS
    )
    if config.pretrained_dir:
        message_extractor.load_state_dict(
            torch.load(os.path.join(config.pretrained_dir, "extractor.pth"))
        )
        message_encoder.load_state_dict(
            torch.load(os.path.join(config.pretrained_dir, "encoder.pth"))
        )
    params_to_optimize = list(
        chain(message_encoder.parameters(), message_extractor.parameters())
    )
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=config.learning_rate,
        weight_decay=config.adam_weight_decay,
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warm_up_steps,
        num_training_steps=config.num_steps,
    )
    (
        message_encoder,
        message_extractor,
        optimizer,
        train_dataloader,
        validation_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        message_encoder,
        message_extractor,
        optimizer,
        train_dataloader,
        validation_dataloader,
        lr_scheduler,
    )
    vae = load_stage1_vae(config.model_type, config.base_model_path).to(
        accelerator.device
    )
    if accelerator.is_main_process:
        compute_model_flops(
            message_encoder, message_extractor, vae, train_dataloader, accelerator
        )
    global_step = config.start_step
    min_loss = float("inf")
    iterator = iter(train_dataloader)
    cross_entropy = nn.BCELoss().to(accelerator.device)
    while global_step < config.num_steps:
        message_encoder.train()
        message_extractor.train()
        try:
            image, message = next(iterator)
        except StopIteration:
            iterator = iter(train_dataloader)
            image, message = next(iterator)
        loss_scales = (
            config.message_loss_scale,
            min(
                config.image_loss_scale * global_step / config.image_loss_ramp,
                config.image_loss_scale,
            ),
            min(
                config.lpips_scale * global_step / config.lpips_ramp, config.lpips_scale
            ),
        )
        loss, message_loss = training_step(
            message,
            message_encoder,
            message_extractor,
            image,
            loss_scales,
            config,
            global_step,
            vae,
            lpips_alex,
            accelerator,
            cross_entropy,
        )
        optimizer.zero_grad()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(params_to_optimize, config.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        if global_step % config.steps_between_validation == 0:
            message_encoder.eval()
            message_extractor.eval()
            accs = {aug: [] for aug in AUGMENTATION_TYPES}
            psnr_input_ls, psnr_recons_ls = ([], [])
            with torch.no_grad():
                for image, message in validation_dataloader:
                    for aug in AUGMENTATION_TYPES:
                        psnr_in, psnr_re, acc = validate_model(
                            message, message_encoder, message_extractor, image, vae, aug
                        )
                        accs[aug].append(acc)
                    psnr_input_ls.append(psnr_in)
                    psnr_recons_ls.append(psnr_re)
            accelerator.wait_for_everyone()

            def gather_mean(lst):
                return (
                    accelerator.gather(torch.tensor(lst, device=accelerator.device))
                    .mean()
                    .item()
                )

            if accelerator.is_main_process:
                val_logs = {
                    "val/psnr_input": gather_mean(psnr_input_ls),
                    "val/psnr_recons": gather_mean(psnr_recons_ls),
                    "val/step": global_step,
                    **{
                        f"val/acc_aug/{aug}": gather_mean(accs[aug])
                        for aug in AUGMENTATION_TYPES
                        if aug != "identity"
                    },
                    "val/acc_no_aug": gather_mean(accs["identity"]),
                }
                accelerator.log(val_logs)
        if accelerator.is_main_process:
            accelerator.log(
                {"train/gradient/learning_rate": optimizer.param_groups[0]["lr"]},
                step=global_step,
            )
            if global_step % 50 == 0:
                accelerator.log(
                    {
                        "train/gradient/encoder_grad_norm": log_avg_gradient_norm(
                            accelerator.unwrap_model(message_encoder)
                        ),
                        "train/gradient/encoder_param_norm": log_avg_param_norm(
                            accelerator.unwrap_model(message_encoder)
                        ),
                        "train/gradient/extractor_grad_norm": log_avg_gradient_norm(
                            accelerator.unwrap_model(message_extractor)
                        ),
                        "train/gradient/extractor_param_norm": log_avg_param_norm(
                            accelerator.unwrap_model(message_extractor)
                        ),
                    },
                    step=global_step,
                )
            accelerator.print(
                f"Global step {global_step}: Loss = {loss}, Secret loss = {message_loss}"
            )
            if global_step % config.steps_between_checkpointing == 0:
                save_dir = os.path.join(
                    intermediate_checkpoints_path, f"checkpoint-{global_step}"
                )
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    accelerator.unwrap_model(message_encoder).state_dict(),
                    f"{save_dir}/encoder.pth",
                )
                torch.save(
                    accelerator.unwrap_model(message_extractor).state_dict(),
                    f"{save_dir}/extractor.pth",
                )
            if global_step > config.lpips_ramp and loss < min_loss:
                min_loss = loss
                torch.save(
                    accelerator.unwrap_model(message_encoder).state_dict(),
                    os.path.join(best_checkpoint_path, "encoder.pth"),
                )
                torch.save(
                    accelerator.unwrap_model(message_extractor).state_dict(),
                    os.path.join(best_checkpoint_path, "extractor.pth"),
                )
        global_step += 1
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = run_dir / "final"
        final_dir.mkdir(exist_ok=True)
        torch.save(
            accelerator.unwrap_model(message_encoder).state_dict(),
            final_dir / "encoder.pth",
        )
        torch.save(
            accelerator.unwrap_model(message_extractor).state_dict(),
            final_dir / "extractor.pth",
        )
    accelerator.end_training()


if __name__ == "__main__":
    config = Stage1Config(**vars(_build_arg_parser(Stage1Config).parse_args()))
    main(config)
