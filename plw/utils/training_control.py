"""Optional training controls, kept separate from the optimization objective."""

import math


def synchronized_early_stop(accelerator, delta, *, enabled, threshold):
    """Broadcast rank 0's decision; every rank must call at the same validation step.

    The initial (step-zero) validation is diagnostic and does not call this.
    Disabled runs use no extra collectives. Non-finite metrics never trigger a stop.
    """
    if not enabled:
        return False
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("early_stopping_delta_threshold must be in [0, 1]")
    stop = (
        accelerator.is_main_process
        and delta is not None
        and math.isfinite(float(delta))
        and (float(delta) >= threshold)
    )
    if accelerator.num_processes == 1:
        return bool(stop)
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "Multi-process early stopping requires a distributed process group"
        )
    flag = torch.tensor(int(stop), dtype=torch.int32, device=accelerator.device)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def synchronized_stop_flag(accelerator, stop, *, enabled):
    """Broadcast rank 0's already-made stop decision to every rank.

    `synchronized_early_stop` folds the threshold comparison in, which fixes the
    criterion to "one scalar in [0, 1] crossed a bound". A patience rule counts
    consecutive clears instead, so the decision is made by the caller and this
    only guarantees that all ranks act on rank 0's answer at the same step.
    Every rank must call this whenever any rank does.
    """
    if not enabled:
        return False
    decision = bool(accelerator.is_main_process and stop)
    if accelerator.num_processes == 1:
        return decision
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "Multi-process early stopping requires a distributed process group"
        )
    flag = torch.tensor(int(decision), dtype=torch.int32, device=accelerator.device)
    dist.broadcast(flag, src=0)
    return bool(flag.item())
