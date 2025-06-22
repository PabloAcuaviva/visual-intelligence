from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel

from .styles import RenderStyle

Grid = list[list[int]]


###
# Schemas
###
class RenderMetadata(BaseModel):
    grid_height: int
    grid_width: int
    ###
    image_height: int
    image_width: int

    grid_height_image: int
    grid_width_image: int
    grid_start_x_image: int
    grid_start_y_image: int


###
# Functions
###
def get_render_metadata(
    grid: Grid,
    render_style: RenderStyle,
    image_height: Literal["auto"] | int = "auto",
    image_width: Literal["auto"] | int = "auto",
) -> RenderMetadata:
    grid_height = len(grid)
    grid_width = len(grid[0])

    grid_height_image = (
        grid_height * (render_style.cell_size + render_style.grid_border_size)
        + render_style.grid_border_size
    )
    grid_width_image = (
        grid_width * (render_style.cell_size + render_style.grid_border_size)
        + render_style.grid_border_size
    )

    image_height = grid_height_image if image_height == "auto" else image_height
    image_width = grid_width_image if image_width == "auto" else image_width

    grid_start_x_image = (image_width - grid_width_image) // 2
    grid_start_y_image = (image_height - grid_height_image) // 2

    return RenderMetadata(
        image_height=image_height,
        image_width=image_width,
        grid_height=grid_height,
        grid_width=grid_width,
        grid_height_image=grid_height_image,
        grid_width_image=grid_width_image,
        grid_start_x_image=grid_start_x_image,
        grid_start_y_image=grid_start_y_image,
    )


def render(
    grid: Grid,
    render_style: RenderStyle,
    image_height: Literal["auto"] | int = "auto",
    image_width: Literal["auto"] | int = "auto",
) -> tuple[Image.Image, RenderMetadata]:
    ###
    # Extract values for rendering
    ###
    cell_size = render_style.cell_size
    grid_border = render_style.grid_border_size
    value_to_color = render_style.value_to_color
    border_color = render_style.border_color
    background_color = render_style.background_color

    render_metadata = get_render_metadata(grid, render_style, image_height, image_width)
    image_height = render_metadata.image_height
    image_width = render_metadata.image_width
    grid_height = render_metadata.grid_height
    grid_width = render_metadata.grid_width
    grid_height_image = render_metadata.grid_height_image
    grid_width_image = render_metadata.grid_width_image
    grid_start_x_image = render_metadata.grid_start_x_image
    grid_start_y_image = render_metadata.grid_start_y_image

    ###
    # Render
    ###
    canvas = np.ones((image_height, image_width, 3), dtype=np.uint8) * np.array(
        background_color, dtype=np.uint8
    )

    # Build square with color border
    grid_area_y1 = grid_start_y_image
    grid_area_y2 = grid_start_y_image + grid_height_image
    grid_area_x1 = grid_start_x_image
    grid_area_x2 = grid_start_x_image + grid_width_image

    canvas[grid_area_y1:grid_area_y2, grid_area_x1:grid_area_x2] = border_color

    # Draw each cell
    for i in range(grid_height):
        for j in range(grid_width):
            color = value_to_color[grid[i][j]]
            x1 = grid_start_x_image + j * (cell_size + grid_border) + grid_border
            y1 = grid_start_y_image + i * (cell_size + grid_border) + grid_border
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            canvas[y1:y2, x1:x2] = color

    return Image.fromarray(canvas), render_metadata


def _parse_from_np_image(
    np_image: np.ndarray, render_metadata: RenderMetadata, render_style: RenderStyle
) -> tuple[Grid, list[list[tuple[int, int, int]]], float]:
    # This loop be paralelized with numpy, but is more than fast enough in this way.

    ###
    # Extract values for parsing
    ###
    grid_height = render_metadata.grid_height
    grid_width = render_metadata.grid_width
    grid_start_x_image = render_metadata.grid_start_x_image
    grid_start_y_image = render_metadata.grid_start_y_image

    grid_border_size = render_style.grid_border_size
    cell_size = render_style.cell_size
    color_to_value = render_style.color_to_value

    ###
    # Parse from np_image
    ###
    recovered_grid = []
    cell_colors = []  # Store (actual_rgb, closest_match_rgb) for each cell
    total_distance = 0  # Track total color distance

    # Extract each cell's color and map back to grid value
    for i in range(grid_height):
        row = []
        for j in range(grid_width):
            x1 = (
                grid_start_x_image
                + j * (cell_size + grid_border_size)
                + grid_border_size
            )
            y1 = (
                grid_start_y_image
                + i * (cell_size + grid_border_size)
                + grid_border_size
            )

            x2 = x1 + cell_size
            y2 = y1 + cell_size

            # Use average cell color
            cell = np_image[y1:y2, x1:x2]
            rgb = tuple(cell.reshape(-1, 3).mean(axis=0).astype(int))

            # Find the closest color in color map (RGB)
            closest_color = min(
                color_to_value.keys(),
                key=lambda c: sum((c[k] - rgb[k]) ** 2 for k in range(3)),
            )

            # Calculate color distance
            distance = sum((rgb[k] - closest_color[k]) ** 2 for k in range(3)) ** 0.5
            total_distance += distance

            # Map color back to grid value
            grid_value = color_to_value[closest_color]
            row.append(grid_value)
            cell_colors.append((rgb, closest_color))

        recovered_grid.append(row)
    return recovered_grid, cell_colors, total_distance


def parse(
    image: Image.Image,
    render_style: RenderStyle,
    render_metadata: RenderMetadata,
    enhance_iterations: int = 3,
    convergence_threshold: float = 1.0,
) -> Grid:
    np_image = np.array(image).astype(np.float32)

    # Iterative enhancement
    iteration = 0
    prev_distance = 0.0
    while iteration < enhance_iterations:
        # Recognize grid and get color information
        _, cell_colors, current_distance = _parse_from_np_image(
            np_image, render_metadata, render_style
        )

        # Check if we've converged
        if abs(prev_distance - current_distance) < convergence_threshold:
            break

        prev_distance = current_distance

        ###
        # Apply image correction
        ###

        # Calculate average color shift needed
        r_shift = 0
        g_shift = 0
        b_shift = 0
        count = len(cell_colors)
        for actual, target in cell_colors:
            r_shift += target[0] - actual[0]
            g_shift += target[1] - actual[1]
            b_shift += target[2] - actual[2]

        if count > 0:
            r_shift /= count
            g_shift /= count
            b_shift /= count

        # Apply color correction to the working image
        np_image[:, :, 0] += r_shift
        np_image[:, :, 1] += g_shift
        np_image[:, :, 2] += b_shift

        # Clip values to valid RGB range (0-255)
        np_image = np.clip(np_image, 0, 255)

        iteration += 1

    # Final recognition pass with the current image state
    parsed_grid, _, _ = _parse_from_np_image(np_image, render_metadata)

    return parsed_grid
