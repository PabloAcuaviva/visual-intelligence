import math
import warnings
from pathlib import Path
from typing import Literal, Optional, Union

import cv2
import imageio
import numpy as np
from numpy.typing import ArrayLike
from PIL import Image


def resize_with_aspect(
    image: ArrayLike,
    target_width: int,
    target_height: int,
    trim: Union[Literal["center", "bottom", "top"], float] = "center",
) -> np.ndarray:
    """
    Resize an image while preserving the aspect ratio and trimming excess pixels when necessary.

    Parameters:
    image (np.ndarray): Input image.
    target_width (int): Desired width.
    target_height (int): Desired height.
    trim (Literal["center", "bottom", "top"]): Defines how excess pixels should be trimmed. Default is "center".
        - "center": Trims equally from both sides.
        - "top": Trims from the top (or left for width excess).
        - "bottom": Trims from the bottom (or right for width excess).
        If a float is given, removes that percentage starting from the top

    Returns:
    np.ndarray: Resized and cropped image.
    """
    image = np.array(image)

    orig_height, orig_width = image.shape[:2]

    # Compute scaling factors
    scale_w = target_width / orig_width
    scale_h = target_height / orig_height

    # Use max to ensure the image fills the target size
    scale = max(scale_w, scale_h)

    # Resize while maintaining aspect ratio
    new_width = math.ceil(orig_width * scale)
    new_height = math.ceil(orig_height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )

    # Compute cropping margins
    if trim == "center":
        crop_x = (new_width - target_width) // 2
        crop_y = (new_height - target_height) // 2
    elif trim == "top":
        crop_x = 0
        crop_y = 0
    elif trim == "bottom":
        crop_x = new_width - target_width if new_width > target_width else 0
        crop_y = new_height - target_height if new_height > target_height else 0
    else:
        crop_x = int((new_width - target_width) * trim)
        crop_y = int((new_height - target_height) * trim)

    # Crop the image to the exact target size
    cropped = resized[crop_y : crop_y + target_height, crop_x : crop_x + target_width]
    return cropped


def normalize_to_uint8(img: np.ndarray, eps_tolerance: float = 1e-3) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.min() >= -eps_tolerance and img.max() <= 1.0 + eps_tolerance:
        return (img * 255).clip(0, 255).astype(np.uint8)
    if img.min() >= -(1.0 + eps_tolerance) and img.max() <= 1.0 + eps_tolerance:
        return ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    warnings.warn(
        "Image values outside expected ranges [0,1] or [-1,1]. Clipping to [0,255]."
    )
    return img.clip(0, 255).astype(np.uint8)


def export_to_video(
    video_frames: Union[list[np.ndarray], list[Image.Image]],
    output_video_path: Union[str, Path],
    fps: int = 10,
    quality: float = 5.0,
    bitrate: Optional[int] = None,
    macro_block_size: Optional[int] = 16,
    eps_tolerance: float = 1e-3,
) -> Union[str, Path]:
    if isinstance(video_frames[0], np.ndarray):
        processed_frames = []
        for frame in video_frames:
            if frame.dtype == np.uint8:
                processed_frames.append(frame)
            else:
                warnings.warn(
                    "Expected uint8 image. Attempting to normalize from [0,1] or [-1,1]. "
                    "Results may differ depending on input scaling."
                )
                processed_frames.append(
                    normalize_to_uint8(frame, eps_tolerance=eps_tolerance)
                )
        video_frames = processed_frames
    elif isinstance(video_frames[0], Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    else:
        raise ValueError(f"Invalid video_frames type provided {type(video_frames)}")

    with imageio.get_writer(
        output_video_path,
        fps=fps,
        quality=quality,
        bitrate=bitrate,
        macro_block_size=macro_block_size,
    ) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    return output_video_path
