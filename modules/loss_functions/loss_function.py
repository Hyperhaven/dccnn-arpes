import torch.nn as nn
from pytorch_msssim import MS_SSIM

def loss_function(output, target, alpha):
    mae = nn.L1Loss()(output, target)
    msssim = sum(
        1 - MS_SSIM(data_range=1.0, channel=1)(output[i:i+1], target[i:i+1])
        for i in range(output.shape[0])
    ) / output.shape[0]

    total_loss = (1 - alpha) * mae + alpha * msssim

    return total_loss, mae, msssim
