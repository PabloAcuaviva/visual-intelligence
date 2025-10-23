import functools as ft
import random
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import cv2
import numpy as np
from numpy.typing import ArrayLike

from utils.utils import export_to_video, resize_with_aspect


def blending_interpolation(x_img: ArrayLike, y_img: ArrayLike, num_frames: int, t_fn):
    """
    Create smooth transition between two images using weighted blending.

    This function generates a sequence of frames that smoothly transition from
    x_img to y_img using linear interpolation with a custom transformation function.

    Args:
        x_img (numpy.ndarray): Source image (starting image of the transition)
        y_img (numpy.ndarray): Target image (ending image of the transition)
        num_frames (int): Number of frames to generate in the transition sequence
        t_fn (callable): Transform function that takes interpolation factor t (0-1)
                        and returns modified t value for non-linear transitions

    Returns:
        list: List of numpy.ndarray images representing the transition frames

    Example:
        >>> frames = blending_interpolation(img1, img2, 10, lambda t: t**2)
        >>> # Creates 10 frames with quadratic easing transition
    """
    x_img = np.array(x_img)
    y_img = np.array(y_img)

    frames = []
    for i in range(num_frames):
        # Calculate interpolation factor (t)
        t = i / (num_frames - 1)

        t = t_fn(t)

        # Linear interpolation: result = (1-t)*x + t*y
        interpolated = cv2.addWeighted(x_img, 1 - t, y_img, t, 0)
        frames.append(interpolated)
    return frames


def rectangles_interpolation(
    x_img: ArrayLike, y_img: ArrayLike, num_frames: int, n=4, m=4
):
    """
    Interpolate between two images by progressively replacing rectangular regions.

    This function creates a transition effect by dividing both images into a grid
    of rectangles and randomly replacing rectangles from the source image with
    corresponding rectangles from the target image over time.

    Args:
        x_img (numpy.ndarray): Source image (starting image of the transition)
        y_img (numpy.ndarray): Target image (ending image of the transition)
        num_frames (int): Number of frames to generate in the transition sequence
        n (int, optional): Number of grid rows. Defaults to 4.
        m (int, optional): Number of grid columns. Defaults to 4.

    Returns:
        list: List of numpy.ndarray images representing the transition frames

    Raises:
        AssertionError: If images don't have the same dimensions

    Note:
        - The first frame is always x_img
        - The last frame is always y_img
        - Rectangles are replaced in random order for each transition
        - Total number of rectangles = n * m

    Example:
        >>> frames = rectangles_interpolation(img1, img2, 20, n=6, m=8)
        >>> # Creates 20 frames with 6x8 grid transition effect
    """
    x_img = np.array(x_img)
    y_img = np.array(y_img)

    # Make sure images have the same dimensions
    assert x_img.shape == y_img.shape, "Images must have the same dimensions"

    # Create a copy of the source image for each frame
    frames = []

    # Get image dimensions
    height, width = x_img.shape[:2]

    # Calculate rectangle dimensions
    rect_height = height // n
    rect_width = width // m

    # Create a list of all rectangles (row, col)
    all_rectangles = [(i, j) for i in range(n) for j in range(m)]

    # Shuffle the rectangles for random replacement
    random.shuffle(all_rectangles)

    # Calculate how many rectangles to replace per frame
    total_rectangles = n * m

    # For each frame, replace a certain number of rectangles
    for frame_idx in range(num_frames):
        # For the first frame, use x_img; for the last frame, use y_img
        if frame_idx == 0:
            frames.append(x_img.copy())
            continue
        elif frame_idx == num_frames - 1:
            frames.append(y_img.copy())
            continue

        # Calculate number of rectangles to replace for this frame
        num_to_replace = int((frame_idx / (num_frames - 1)) * total_rectangles)

        # Create a new frame starting with the source image
        frame = x_img.copy()

        # Replace rectangles with the target image
        for rect_idx in range(num_to_replace):
            if rect_idx < len(all_rectangles):
                i, j = all_rectangles[rect_idx]

                # Calculate rectangle coordinates
                y_start = i * rect_height
                y_end = min((i + 1) * rect_height, height)
                x_start = j * rect_width
                x_end = min((j + 1) * rect_width, width)

                # Replace the rectangle with the corresponding part from the target image
                frame[y_start:y_end, x_start:x_end] = y_img[
                    y_start:y_end, x_start:x_end
                ]

        frames.append(frame)

    return frames


def black_middle_interpolation(
    x_img: ArrayLike, y_img: ArrayLike, num_frames: int, n_black_frames: int = 0
):
    """
    Create transition between two images with black frames in the middle.

    This function creates a transition effect where the source image is shown
    for the first half of frames, followed by black frames in the middle,
    and then the target image for the remaining frames.

    Args:
        x_img (numpy.ndarray): Source image (shown in first half)
        y_img (numpy.ndarray): Target image (shown in second half)
        num_frames (int): Total number of frames to generate
        n_black_frames (int, optional): Number of black frames in the middle.
                                       Defaults to 0.

    Returns:
        list: List of numpy.ndarray images representing the transition frames

    Raises:
        ValueError: If num_frames is less than n_black_frames + 2

    Note:
        - The transition is symmetric around the black frames
        - If there's an odd number of non-black frames, the extra frame
          goes to the target image portion
        - Black frames have the same shape as input images but all pixels are 0

    Example:
        >>> frames = black_middle_interpolation(img1, img2, 15, n_black_frames=3)
        >>> # Creates: 6 frames of img1, 3 black frames, 6 frames of img2
    """
    x_img = np.array(x_img)
    y_img = np.array(y_img)

    if num_frames < n_black_frames + 2:
        raise ValueError(
            f"For 'black_middle' interpolation, num_frames must be at least n_black_frames+2={n_black_frames+2}"
        )

    frames = []

    # Initial frame
    for _ in range((num_frames - n_black_frames) // 2):
        frames.append(x_img.copy())

    # Black frames in the middle
    black_frame = np.zeros_like(x_img)
    for _ in range(n_black_frames):
        frames.append(black_frame.copy())

    # Last frames
    for _ in range(
        (num_frames - n_black_frames) // 2 + (num_frames - n_black_frames) % 2
    ):
        frames.append(y_img.copy())

    return frames


def noisy_blending_interpolation(
    x_img: ArrayLike,
    y_img: ArrayLike,
    num_frames: int,
    t_fn: Callable,
    noise_alpha: float = 127.5,
):
    """
    Create smooth transition between images with added noise during interpolation.

    This function performs blending interpolation like blending_interpolation but
    adds Gaussian noise that is strongest in the middle of the transition (t=0.5)
    and zero at the beginning and end (t=0 and t=1).

    Args:
        x_img (numpy.ndarray): Source image (starting image of the transition)
        y_img (numpy.ndarray): Target image (ending image of the transition)
        num_frames (int): Number of frames to generate in the transition sequence
        t_fn (callable): Transform function that takes interpolation factor t (0-1)
                        and returns modified t value for non-linear transitions
        noise_alpha (float, optional): Maximum noise intensity scaling factor.
                                     Defaults to 127.5.

    Returns:
        list: List of numpy.ndarray images representing the noisy transition frames

    Note:
        - Noise follows a parabolic pattern: strongest at t=0.5, zero at t=0,1
        - Noise intensity = noise_alpha * t * (1-t)
        - Final pixel values are clipped to [0, 255] range
        - Gaussian noise has mean=0, std=1 before scaling

    Example:
        >>> frames = noisy_blending_interpolation(img1, img2, 20, lambda t: t, 100.0)
        >>> # Creates 20 frames with maximum noise intensity of 100 at middle transition
    """
    x_img = np.array(x_img)
    y_img = np.array(y_img)

    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        t_transformed = t_fn(t)

        # Linear interpolation: result = (1-t)*x + t*y
        interpolated = cv2.addWeighted(
            x_img, 1 - t_transformed, y_img, t_transformed, 0
        )

        # Calculate noise factor - highest at t=0.5, zero at t=0 and t=1
        noise_factor = noise_alpha * (t * (1 - t))
        noise = np.random.normal(0, 1, interpolated.shape).astype(np.float32)
        noisy_interpolated = np.clip(
            interpolated + noise_factor * noise, 0, 255
        ).astype(np.uint8)

        frames.append(noisy_interpolated)
    return frames


interpolation_methods = dict(
    convex=ft.partial(blending_interpolation, t_fn=lambda t: t),
    quadratic=ft.partial(blending_interpolation, t_fn=lambda t: t**2),
    square_root=ft.partial(blending_interpolation, t_fn=lambda t: np.sqrt(t)),
    discrete=ft.partial(black_middle_interpolation, n_black_frames=0),
    squares_4x4=ft.partial(rectangles_interpolation, n=4, m=4),
)


def interpolate_x_y(
    x: Union[Path, ArrayLike],
    y: Union[Path, ArrayLike],
    /,
    output_path: Optional[Path],
    num_frames: int = 9,
    interpolation_method: Literal[
        "convex", "quadratic", "square_root", "discrete", "squares_4x4"
    ] = "discrete",
    apply_fn: Optional[Callable] = None,
):
    if output_path:
        output_path = Path(output_path)

    if isinstance(x, (str, Path)):
        x_img = cv2.imread(str(x))
        if x_img is None:
            raise ValueError(f"Path {x} does not exist or could not be read")
    else:
        x_img = np.array(x, dtype=np.uint8)

    if isinstance(y, (str, Path)):
        y_img = cv2.imread(str(y))
        if y_img is None:
            raise ValueError(f"Path {y} does not exist or could not be read")
    else:
        y_img = np.array(y, dtype=np.uint8)

    if apply_fn is not None:
        x_img = apply_fn(x_img)
        y_img = apply_fn(y_img)

    # Ensure images have the same size
    if x_img.shape != y_img.shape:
        print(
            f"Warning: Images {x} and {y} have different dimensions ({x_img.shape=} vs {y_img.shape=}). Resizing..."
        )
        y_img = resize_with_aspect(y_img, (x_img.shape[1], x_img.shape[0]))

    # Generate video
    interpolation_fn = interpolation_methods[interpolation_method]
    frames = interpolation_fn(x_img, y_img, num_frames)

    # Save export video
    video_array = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])
    if output_path:
        return export_to_video(video_array, output_video_path=output_path, fps=8)
    else:
        return video_array
