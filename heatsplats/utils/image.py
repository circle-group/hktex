from IPython.display import display, HTML
import io
import matplotlib.pyplot as plt
import base64

import numpy as np
import mitsuba as mi


def combine_images(*images: list[mi.Bitmap]) -> mi.Bitmap:
    """Combines multiple images into a single image. Placing them side by side.
    Args:
        *images (list[mi.Bitmap]): List of Mitsuba Bitmap images.

    Returns:
        mi.Bitmap: lists of Mitsuba Bitmap images.
    """
    concatenated = np.concatenate([np.array(im) for im in images], axis=1)
    combined_image = mi.Bitmap(mi.TensorXf(concatenated)).convert(
        pixel_format=mi.Bitmap.PixelFormat.RGB,
        component_format=mi.Struct.Type.UInt8,
        srgb_gamma=True,
    )
    return combined_image


def show_image(mi_bitmap):
    buf = io.BytesIO()
    img = np.array(mi_bitmap)
    plt.imsave(buf, img)  # Writes to memory, not disk
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    display(HTML(f'<img src="data:image/png;base64,{img_b64}" style="width:100%;">'))
