from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import torch
from PIL import Image
from torchvision import transforms
from plw.modeling.message_models import MessageExtractor

FPR_TARGETS = (0.0, 0.01, 0.02, 0.05, 0.1)


@dataclass
class AccuracyResults:
    per_image: list[float]
    mean: float


def separation_auc(clean: list[float], triggered: list[float]) -> float:
    """P(a triggered image scores above a clean one), ties counted as half.

    Rank-based (Mann-Whitney U) rather than a trapezoid over an ROC curve,
    because bit accuracies tie heavily -- a well-trained run puts many images at
    exactly 1.0, and a trapezoid over tied scores silently over- or under-counts
    depending on sort order. The rank form handles ties exactly.

    Returns NaN for an empty group, which callers must treat as "no decision"
    rather than as a score of 0.
    """
    n0, n1 = (len(clean), len(triggered))
    if n0 == 0 or n1 == 0:
        return float("nan")
    scores = list(clean) + list(triggered)
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    rank_sum_triggered = sum(ranks[n0:])
    return (rank_sum_triggered - n1 * (n1 + 1) / 2.0) / (n0 * n1)


def separation_margin(
    clean: list[float], triggered: list[float], quantile: float = 0.1
) -> float:
    """Gap between the weak tail of `triggered` and the strong tail of `clean`.

        margin = Q_q(triggered) - Q_{1-q}(clean)

    In bit-accuracy units, so +0.15 means the 10th-percentile triggered image
    still scores 15 points of bit accuracy above the 90th-percentile clean one.

    Preferred over AUC as a stopping signal because AUC saturates: on N images
    per side it is quantised to 1/N^2 and hits exactly 1.0 the moment the two
    distributions order correctly, demanding no headroom. The margin stays
    continuous past that point, and it is what predicts TPR at strict FPR --
    which is the operating point that actually matters and the one AUC hides.

    Quantiles rather than min/max so a single outlier on either side cannot
    decide the stop. NaN for an empty group, which callers treat as "no
    decision" rather than as a failed threshold.
    """
    if not clean or not triggered:
        return float("nan")
    import numpy as _np

    return float(
        _np.quantile(triggered, quantile) - _np.quantile(clean, 1.0 - quantile)
    )


def compute_message_accuracy(
    images: list[Image.Image],
    extractor: MessageExtractor,
    vae_model,
    message: torch.Tensor,
    device: torch.device,
    batch_size: int = 8,
    image_size: int = None,
) -> AccuracyResults:
    print("Computing message accuracy...")
    resize_ops = (
        [transforms.Resize((image_size, image_size))] if image_size is not None else []
    )
    preprocess = transforms.Compose(
        resize_ops + [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    vae_dtype = next(vae_model.parameters()).dtype
    if next(vae_model.parameters()).device != device:
        vae_model.to(device)
    gt_row = message.view(1, -1)
    all_accs: list[float] = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            print(f"Batch [{i}/{len(images)}]")
            batch = torch.stack(
                [preprocess(img) for img in images[i : i + batch_size]]
            ).to(device=device, dtype=vae_dtype)
            latents = vae_model.encode(batch).float()
            decoded = torch.round(torch.sigmoid(extractor(latents)))
            gt = gt_row.repeat(decoded.shape[0], 1).to(decoded)
            acc = (decoded == gt).float().mean(dim=1).tolist()
            all_accs.extend(acc)
    return AccuracyResults(
        per_image=all_accs, mean=float(torch.tensor(all_accs).mean())
    )


@dataclass
class EvalMetrics:
    auc: float
    asr_at_fpr: dict[float, float]
    threshold_at_fpr: dict[float, float]
    fpr_curve: list[float] = field(repr=False)
    tpr_curve: list[float] = field(repr=False)
    thresholds: list[float] = field(repr=False)


def roc_curve(
    scores_neg: Sequence[float], scores_pos: Sequence[float]
) -> tuple[list[float], list[float], list[float]]:
    """
    Compute ROC curve treating *scores_pos* as positives (triggered) and
    *scores_neg* as negatives (clean).

    Returns (fpr_list, tpr_list, threshold_list) sorted by ascending threshold.
    """
    neg = sorted(scores_neg)
    pos = sorted(scores_pos)
    all_scores = sorted(set(neg + pos + [0.0, 1.0 + 1e-09]))
    fpr_list, tpr_list, thr_list = ([], [], [])
    n_neg, n_pos = (len(neg), len(pos))
    for thr in all_scores:
        fp = sum((s >= thr for s in neg))
        tp = sum((s >= thr for s in pos))
        fpr_list.append(fp / n_neg if n_neg else 0.0)
        tpr_list.append(tp / n_pos if n_pos else 0.0)
        thr_list.append(thr)
    return (fpr_list, tpr_list, thr_list)


def area_under_curve(fpr: list[float], tpr: list[float]) -> float:
    """Trapezoidal AUC (does not require scikit-learn)."""
    pairs = sorted(zip(fpr, tpr))
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
        area += (x1 - x0) * (y0 + y1) / 2
    return area


def evaluate_distributions(
    clean_scores: list[float],
    triggered_scores: list[float],
    fpr_targets: Sequence[float] = FPR_TARGETS,
) -> EvalMetrics:
    """
    Treat triggered images as positives, clean as negatives.
    Compute ROC, AUC, and ASR (TPR) at each requested FPR.
    """
    fpr_curve, tpr_curve, thresholds = roc_curve(clean_scores, triggered_scores)
    auc = area_under_curve(fpr_curve, tpr_curve)
    asr_at_fpr: dict[float, float] = {}
    thr_at_fpr: dict[float, float] = {}
    for target_fpr in fpr_targets:
        best_tpr, best_thr = (0.0, 1.0)
        for fpr, tpr, thr in zip(fpr_curve, tpr_curve, thresholds):
            if fpr <= target_fpr and tpr >= best_tpr:
                best_tpr, best_thr = (tpr, thr)
        asr_at_fpr[target_fpr] = best_tpr
        thr_at_fpr[target_fpr] = best_thr
    return EvalMetrics(
        auc=auc,
        asr_at_fpr=asr_at_fpr,
        threshold_at_fpr=thr_at_fpr,
        fpr_curve=fpr_curve,
        tpr_curve=tpr_curve,
        thresholds=thresholds,
    )
