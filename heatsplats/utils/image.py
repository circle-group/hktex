from IPython.display import display, HTML
import io
import matplotlib.pyplot as plt
import base64

import numpy as np
import mitsuba as mi
import drjit as dr

from .typing import *


def combine_images(
    *images: list[dr.auto.ad.TensorXf | dr.auto.TensorXf],
    horizontal: bool = True,
    as_bitmap: bool = True,
) -> mi.Bitmap:
    """Combines multiple images into a single image. Placing them side by side.
    Args:
        *images (list[dr.auto.ad.TensorXf | dr.auto.TensorXf]): List of
            TensorXf images.
        horizontal (bool, optional): Whether to concatenate images horizontally
            or vertically. Defaults to True.
        as_bitmap (bool, optional): Whether to return the combined image as a
            Mitsuba Bitmap. If False, returns as a TensorXf. Defaults to True.
    Returns:
        mi.Bitmap: lists of Mitsuba Bitmap images.
    """
    axis = 1 if horizontal else 0
    concatenated = np.concatenate([np.array(im) for im in images], axis=axis)
    combined = mi.TensorXf(concatenated)

    if as_bitmap:
        combined = mi.Bitmap(combined).convert(
            pixel_format=mi.Bitmap.PixelFormat.RGB,
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=True,
        )

    return combined


def show_image(mi_bitmap):
    buf = io.BytesIO()
    img = np.array(mi_bitmap)
    plt.imsave(buf, img)  # Writes to memory, not disk
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    display(HTML(f'<img src="data:image/png;base64,{img_b64}" style="width:100%;">'))


def mibitmaps2torch(
    list_bitmaps: list[mi.Bitmap],
) -> Float[Tensor, "N H W C"]:
    """Convert a list of Mitsuba Bitmaps to a single torch tensor.

    Args:
        bitmaps (list[mi.Bitmap]): List of Mitsuba Bitmaps.

    Returns:
        Float[Tensor, "N H W C"]: Torch tensor of shape (N, H, W, C).
    """
    import torch

    tensors = []
    for bmp in list_bitmaps:
        tensors.append(torch.from_numpy(np.array(bmp)) / 255.0)

    return torch.stack(tensors, dim=0)
