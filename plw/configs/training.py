"""Configuration schemas for the two-stage PLW pipeline. Use the main JSON presets through the public CLI."""

from dataclasses import dataclass

LATENT_CHANNELS = 16


@dataclass
class Stage1Config:
    base_model_path: str = None
    seed: int = 0
    model_type: str = "bagel"
    train_path: str = "data/stage1/train"
    val_path: str = "data/stage1/val"
    run_name: str = "stage1"
    num_steps: int = 10000
    warm_up_steps: int = 0
    message_size: int = 48
    batch_size: int = 8
    learning_rate: float = 0.0002
    image_loss_scale: int = 50
    image_loss_ramp: int = 2000
    message_loss_scale: float = 1.0
    lpips_scale: float = 0.25
    lpips_ramp: int = 4000
    pretrained_dir: str = None
    start_step: int = 0
    max_val_samples: int = 100
    steps_between_image_logging: int = 1000
    steps_between_checkpointing: int = 1000
    steps_between_validation: int = 1000
    validation_batch_size: int = 2
    max_grad_norm: float = 0.01
    adam_weight_decay: float = 0.01
    residual_target_mag: float = 0.05
    residual_reg_scale: float = 10.0
    augment_enabled: bool = False
    augment_noise_std: float = 0.01
    augment_crop_scale: float = 1.0
    augment_dropout_p: float = 0.05


@dataclass
class Stage2Config:
    base_model_path: str = None
    model_type: str = "bagel"
    stage_1_model_type: str = "bagel"
    stage_1_run_name: str = "stage1"
    stage_1_checkpoint_path: str = None
    stage_2_exp_name: str = "adhoc"
    stage_2_run_name: str = None
    message: str = "not provided"
    context_chat_dataset_name: str = None
    natural_trigger_chat_dataset_name: str = None
    natural_trigger_prob: float = 0.0
    trigger: str = "*[Z]&"
    trigger_insertion_mode: str = "prefix"
    trigger_role_scope: str = "user_only"
    prompt_dataset_name: str = "prompts/coco_10k_training"
    dataset_max_samples: int = 10000
    dataset_combination_seed: int = 0
    context_chat_start_index: int = 0
    cached_data_resolution: int = 1024
    image_cache_root: str = None
    image_cache_read_only: bool = False
    align_image_cache_with_combined_prompts: bool = True
    reorder_existing_image_cache: bool = False
    seed: int = 0
    image_size: int = 512
    center_crop: bool = False
    generation_prompt_prefix: str = "Generate an image of:"
    max_train_steps: int = 10000
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 2
    steps_between_checkpoints: int = 1000
    learning_rate: float = 0.0005
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_weight_decay: float = 0.0
    adam_epsilon: float = 1e-15
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    loss_t_threshold: float = 0.5
    coeff_steepness: float = 0.01
    message_strength: float = 1.0
    preservation_timestep_weighting: bool = False
    watermarking_loss_warmup_steps: int = 0
    message_strength_warmup_steps: int = 500
    watermarking_loss_weight: float = 1.0
    enable_regularization: bool = True
    preservation_loss_weight: float = 1.0
    timestep_weighting: bool = True
    use_data_target: bool = True
    enable_negative_for_clean: bool = False
    clean_repel_weight: float = 0.0
    separation_margin_weight: float = 0.0
    separation_margin: float = 2.0
    delta_limit_weight: float = 0.0
    trigger_set: str = ""
    trigger_set_fraction: float = 1.0
    contrastive_weight: float = 0.0
    num_validation_samples: int = 16
    num_validation_images_logged: int = 16
    steps_between_validation: int = 500
    num_inference_steps: int = 28
    early_stopping: bool = False
    early_stopping_metric: str = "margin"
    early_stopping_margin_quantile: float = 0.1
    early_stopping_margin_threshold: float = 0.15
    early_stopping_auc_threshold: float = 0.995
    early_stopping_delta_threshold: float = 0.5
    early_stopping_patience: int = 2
    early_stopping_min_validation_samples: int = 16
