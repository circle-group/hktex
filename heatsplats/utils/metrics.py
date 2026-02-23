import torch

from torchmetrics.functional.image.lpips import (
    learned_perceptual_image_patch_similarity as lpips,
)
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchmetrics.functional.image import (
    multiscale_structural_similarity_index_measure as ms_ssim,
)

from heatsplats.utils.typing import *


@torch.no_grad()
def compute_all_image_metrics(
    pred_renderings: Float[Tensor, "B H W C"],
    gt_renderings: Float[Tensor, "B H W C"],
    data_range: float = 1.0,
) -> dict[str, float]:
    """
    Compute PSNR, SSIM, MS-SSIM, and LPIPS between two images.

    Args:
        pred_renderings: Predicted rendering tensor of shape (B, H, W, C)
            with values in [0, data_range].
        gt_renderings: Ground truth rendering tensor of shape (B, H, W, C)
            with values in [0, data_range].
        data_range: The value range of the input images
            (default is 1.0 for normalized images).

    Returns:
        A dictionary containing the computed metrics.
    """
    assert data_range == 1.0 or data_range == 255.0, "data_range should be 1.0 or 255.0"

    pred_renderings = pred_renderings.permute(0, 3, 1, 2)  # (B, C, H, W)
    gt_renderings = gt_renderings.permute(0, 3, 1, 2)  # (B, C, H, W)

    psnr_val = psnr(pred_renderings, gt_renderings, data_range=data_range)
    ssim_val = ssim(
        pred_renderings,
        gt_renderings,
        data_range=data_range,
        reduction="elementwise_mean",
    )
    ms_ssim_val = ms_ssim(
        pred_renderings,
        gt_renderings,
        data_range=data_range,
        reduction="elementwise_mean",
    )

    if data_range == 255:  # convert to [0, 1] range for LPIPS
        pred_renderings = pred_renderings / 255.0
        gt_renderings = gt_renderings / 255.0
        data_range = 1.0

    lpips_val = lpips(
        pred_renderings,
        gt_renderings,
        net_type="squeeze",
        normalize=True,
    )

    return {
        "psnr": psnr_val.item(),
        "ssim": ssim_val.item(),
        "ms_ssim": ms_ssim_val.item(),
        "lpips": lpips_val.item(),
    }
