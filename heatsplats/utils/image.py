import io
import matplotlib.pyplot as plt
import base64

import torch
import numpy as np
import mitsuba as mi
import drjit as dr

from scipy.ndimage import sobel
from numpy.linalg import norm

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
    from IPython.display import display, HTML

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


def compute_image_gradients(image: np.ndarray) -> Union[np.ndarray, np.ndarray]:
    gy, gx = [], []
    for image_channel in image:
        gy.append(sobel(image_channel, 0))
        gx.append(sobel(image_channel, 1))
    gy = norm(np.stack(gy, axis=0), ord=2, axis=-1).astype(np.float32)
    gx = norm(np.stack(gx, axis=0), ord=2, axis=-1).astype(np.float32)
    return gy, gx


def compute_gmap(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    gy, gx = compute_image_gradients(np.power(image, 1.0 / gamma))
    g_norm = np.hypot(gy, gx).astype(np.float32)
    g_norm = g_norm / g_norm.max()
    g_norm = np.power(g_norm.reshape(-1), 2.0)
    return g_norm / g_norm.sum()


def get_grid(h, w, x_lim=np.asarray([0, 1]), y_lim=np.asarray([0, 1])):
    x = torch.linspace(x_lim[0], x_lim[1], steps=w + 1)[:-1] + 0.5 / w
    y = torch.linspace(y_lim[0], y_lim[1], steps=h + 1)[:-1] + 0.5 / h
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid
