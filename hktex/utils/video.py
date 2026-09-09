import imageio
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional


def combine_videos(*videos_frames: list[list[mi.Bitmap]]) -> list[mi.Bitmap]:
    """Combines multiple videos into a single video. Placing them side by side.
    Args:
        *videos_frames (list[list[mi.Bitmap]]): List of lists of Mitsuba Bitmap images.

    Returns:
        list[mi.Bitmap]: lists of Mitsuba Bitmap images.
    """
    # Check if all videos have the same number of frames
    num_frames = len(videos_frames[0])
    for video in videos_frames:
        if len(video) != num_frames:
            raise ValueError("All videos must have the same number of frames.")

    # Combine frames side by side
    combined_frames = []
    for i in range(num_frames):
        numpy_frames = [np.array(video[i]) for video in videos_frames]
        combined_numpy_frame = np.concatenate(numpy_frames, axis=1)
        combined_frame = mi.Bitmap(combined_numpy_frame)
        combined_frames.append(combined_frame)

    return combined_frames


def show_video(
    frames: list[mi.Bitmap],
    interval: int = 200,
    frame_texts: Optional[list[str]] = None,
):
    """
    Displays a list of frames as a video in an interactive window.

    Args:
        frames (list[mi.Bitmap]): List of Mitsuba Bitmap images.
        interval (int): Delay between frames in milliseconds.
        frame_texts (Optional[list[str]]): Optional list of strings to display on each frame.
    """
    from IPython.display import HTML

    # Convert frames to NumPy arrays
    numpy_frames = [np.array(frame) / 255 for frame in frames]

    if frame_texts and len(frame_texts) != len(numpy_frames):
        raise ValueError("The number of frame texts must match the number of frames.")

    # Create a figure and axis
    fig, ax = plt.subplots(
        figsize=(numpy_frames[0].shape[1] / 100, numpy_frames[0].shape[0] / 100)
    )
    ax.set_position([0, 0, 1, 1])
    plt.axis("off")
    img = ax.imshow(numpy_frames[0], interpolation="nearest")

    # Add a text artist to the plot, initialized as empty
    text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        color="red",
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.5),
    )

    # Update function for the animation
    def update(frame_idx):
        img.set_array(numpy_frames[frame_idx])
        if frame_texts:
            text.set_text(frame_texts[frame_idx])
        return [img, text]

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=range(len(numpy_frames)), interval=interval, blit=True
    )

    # Display the animation inline
    html_output = HTML(ani.to_jshtml())

    # Close the figure to prevent the static image from being displayed
    plt.close(fig)

    return html_output


def save_video(frames: list[mi.Bitmap], output_path: str, fps: int = 30):
    """
    Converts a list of mi.Bitmap images into a video.

    Args:
        frames (list[mi.Bitmap]): List of Mitsuba Bitmap images.
        output_path (str): Path to save the video file.
        fps (int): Frames per second for the video.
    """

    numpy_frames = [np.array(frame) for frame in frames]

    # Write the frames to a video file (strip alpha channel for standard video codecs if present)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in numpy_frames:
            rgb_frame = frame[..., :3] if frame.ndim == 3 and frame.shape[2] == 4 else frame
            writer.append_data(rgb_frame)
