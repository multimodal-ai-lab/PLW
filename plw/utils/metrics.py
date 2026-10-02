import torch
from torch.nn import functional as F


def compute_psnr(encoded, image_input):
    mse = F.mse_loss(encoded, image_input, reduction="none")
    mse = mse.mean([1, 2, 3])
    psnr = 10 * torch.log10(1**2 / mse)
    average_psnr = psnr.mean().item()
    return average_psnr


def get_message_accuracy(predictions, ground_truth):
    predictions = predictions.cpu()
    ground_truth = ground_truth.cpu()
    rounded_predictions = torch.round(predictions)
    correct_predictions = (rounded_predictions == ground_truth).sum().item()
    accuracy = correct_predictions / ground_truth.numel()
    return accuracy
