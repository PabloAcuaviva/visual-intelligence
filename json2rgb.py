import argparse
import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from tqdm import tqdm


def _imwrite(filename, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)


@dataclass
class GridStyle:
    # Grid
    cell_size: int
    grid_border: int

    # Colors
    color_map: dict
    background_color: tuple
    border_color: tuple

    @property
    def reverse_color_map(self):
        # Create reverse color map (RGB tuple to grid value)
        return {str(v): k for k, v in self.color_map.items()}

    def to_json(self):
        d = asdict(self)
        d["background_color"] = list(self.background_color)
        d["border_color"] = list(self.border_color)
        return d

    @classmethod
    def from_json(cls, json_str):
        d = json.loads(json_str)
        d["background_color"] = tuple(d["background_color"])
        d["border_color"] = tuple(d["border_color"])
        return cls(**d)


ArcBaseStyle = GridStyle(
    cell_size=30,
    grid_border=2,
    color_map={
        0: (0, 0, 0),  # Black
        1: (0, 116, 217),  # Blue
        2: (255, 65, 54),  # Red
        3: (46, 204, 64),  # Green
        4: (255, 220, 0),  # Yellow
        5: (170, 170, 170),  # Grey
        6: (240, 18, 190),  # Fuchsia
        7: (255, 133, 27),  # Orange
        8: (127, 219, 255),  # Teal
        9: (135, 12, 37),  # Brown
    },
    background_color=(0, 0, 0),  # Black background
    border_color=(85, 85, 85),  # Medium gray border
)


@dataclass
class GridMetadata:
    H: int
    W: int
    grid_h: int
    grid_w: int
    grid_style: GridStyle

    def __post_init__(self):
        if self.start_x < 0 or self.start_y < 0:
            err_msg = (
                f"Invalid grid metadata: start_x={self.start_x}, start_y={self.start_y}. "
                f"Ensure that the grid fits within the specified image dimensions ({self.H=}, {self.W=})."
                f"Ensure that the grid dimensions ({self.grid_h=}, {self.grid_w=}) and "
                f"cell size ({self.grid_style.cell_size=}) are appropriate for the image size."
            )
            raise ValueError(err_msg)

    @property
    def img_h(self):
        return (
            self.grid_h * (self.grid_style.cell_size + self.grid_style.grid_border)
            + self.grid_style.grid_border
        )

    @property
    def img_w(self):
        return (
            self.grid_w * (self.grid_style.cell_size + self.grid_style.grid_border)
            + self.grid_style.grid_border
        )

    @property
    def start_x(self):
        return (self.W - self.img_w) // 2

    @property
    def start_y(self):
        return (self.H - self.img_h) // 2

    def save(self, filename):
        metadata_dict = {
            "H": self.H,
            "W": self.W,
            "grid_h": self.grid_h,
            "grid_w": self.grid_w,
            "grid_style": self.grid_style.to_json(),
        }
        with open(filename, "w") as f:
            json.dump(metadata_dict, f, indent=2)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            metadata_dict = json.load(f)
        return cls(
            H=metadata_dict["H"],
            W=metadata_dict["W"],
            grid_h=metadata_dict["grid_h"],
            grid_w=metadata_dict["grid_w"],
            grid_style=GridStyle.from_json(json.dumps(metadata_dict["grid_style"])),
        )


def draw_grid(grid, grid_style: GridStyle, H=None, W=None):
    grid_h, grid_w = len(grid), len(grid[0])

    # Extract grid style parameters
    cell_size = grid_style.cell_size
    grid_border = grid_style.grid_border
    color_map = grid_style.color_map
    border_color = grid_style.border_color
    background_color = grid_style.background_color

    H = H or cell_size * grid_h + grid_border * (grid_h + 1)
    W = W or cell_size * grid_w + grid_border * (grid_w + 1)

    # Calculate the size of the grid area
    grid_metadata = GridMetadata(
        H=H, W=W, grid_h=grid_h, grid_w=grid_w, grid_style=grid_style
    )
    img_h = grid_metadata.img_h
    img_w = grid_metadata.img_w
    start_x = grid_metadata.start_x
    start_y = grid_metadata.start_y

    ###
    # Drawing
    ###
    # Create a black canvas for the background
    canvas = np.ones((H, W, 3), dtype=np.uint8) * np.array(
        background_color, dtype=np.uint8
    )

    # Build square with color border
    grid_area_y1 = start_y
    grid_area_y2 = start_y + img_h
    grid_area_x1 = start_x
    grid_area_x2 = start_x + img_w
    canvas[grid_area_y1:grid_area_y2, grid_area_x1:grid_area_x2] = border_color

    # Draw each cell
    for i in range(grid_h):
        for j in range(grid_w):
            color = color_map[grid[i][j]]
            x1 = start_x + j * (cell_size + grid_border) + grid_border
            y1 = start_y + i * (cell_size + grid_border) + grid_border
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            canvas[y1:y2, x1:x2] = color

    return canvas, grid_metadata


def recover_grid(image, grid_metadata: GridMetadata):
    image = np.array(image, dtype=np.float32)
    grid_h = grid_metadata.grid_h
    grid_w = grid_metadata.grid_w
    start_x = grid_metadata.start_x
    start_y = grid_metadata.start_y

    grid_border = grid_metadata.grid_style.grid_border
    cell_size = grid_metadata.grid_style.cell_size
    reverse_color_map = grid_metadata.grid_style.reverse_color_map

    ###
    # Recover grid
    ###
    # Initialize empty grid
    recovered_grid = []
    # Extract each cell's color and map back to grid value
    for i in range(grid_h):
        row = []
        for j in range(grid_w):
            x1 = start_x + j * (cell_size + grid_border) + grid_border
            y1 = start_y + i * (cell_size + grid_border) + grid_border

            x2 = x1 + cell_size
            y2 = y1 + cell_size

            # Use average cell color
            cell = image[y1:y2, x1:x2]
            rgb = tuple(cell.reshape(-1, 3).mean(axis=0).astype(int))

            # Find the closest color in color map
            closest_color = min(
                reverse_color_map.keys(),
                key=lambda c: sum((c[k] - rgb[k]) ** 2 for k in range(3)),
            )

            # Map the color back to grid value
            grid_value = reverse_color_map[closest_color]
            row.append(grid_value)
        recovered_grid.append(row)

    return recovered_grid


def process_json_files(
    evaluation_dir,
    output_dir,
    H=480,
    W=720,
    fixed_cell_size: int = None,
    max_cell_size: int = 50,
    nested=False,
    grid_border=2,
):
    """
    Process all JSON files in evaluation directory and create visualization folders
    with both images and metadata for recovery.

    Args:
        evaluation_dir: Directory containing JSON files
        output_dir: Output directory for visualizations
        H: Height of output images
        W: Width of output images
        cell_size: Fixed cell size (if None, auto-adjusts per JSON)
        max_cell_size: Maximum allowed cell size when auto-adjusting
        nested: If the JSON is in a nested structure
    """
    # Create main output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all JSON files from evaluation directory
    json_files = [f for f in os.listdir(evaluation_dir) if f.endswith(".json")]
    json_files.sort()
    json_files = json_files
    # Get all directory files and make recursive call
    if nested:
        nested_dirs = [
            d
            for d in os.listdir(evaluation_dir)
            if os.path.isdir(os.path.join(evaluation_dir, d))
        ]
        for nested_dir in nested_dirs:
            nested_output_dir = os.path.join(output_dir, nested_dir)
            process_json_files(
                evaluation_dir=os.path.join(evaluation_dir, nested_dir),
                output_dir=nested_output_dir,
                H=H,
                W=W,
                fixed_cell_size=fixed_cell_size,
                max_cell_size=max_cell_size,
                nested=True,
            )

    # Process each JSON file
    for idx, json_file in enumerate(tqdm(json_files, desc="Processing files")):
        # Create subfolder with zero-padded index
        subfolder = f"{idx:03d}"
        folder_path = os.path.join(output_dir, subfolder)

        # Create input and output directories with rgb and metadata subdirectories
        input_rgb_dir = os.path.join(folder_path, "input", "rgb")
        input_meta_dir = os.path.join(folder_path, "input", "metadata")
        output_rgb_dir = os.path.join(folder_path, "output", "rgb")
        output_meta_dir = os.path.join(folder_path, "output", "metadata")

        os.makedirs(input_rgb_dir, exist_ok=True)
        os.makedirs(input_meta_dir, exist_ok=True)
        os.makedirs(output_rgb_dir, exist_ok=True)
        os.makedirs(output_meta_dir, exist_ok=True)

        # Read and parse JSON file
        with open(os.path.join(evaluation_dir, json_file), "r") as f:
            data = json.load(f)

        # Determine optimal cell size for this JSON file if auto mode is enabled
        if fixed_cell_size is None:
            # Find the maximum grid dimensions across all examples in this JSON
            max_rows = 0
            max_cols = 0

            # Check train examples
            for train_sample in data.get("train", []):
                input_grid = train_sample["input"]
                output_grid = train_sample["output"]
                max_rows = max(max_rows, len(input_grid), len(output_grid))
                max_cols = max(
                    max_cols,
                    max([len(row) for row in input_grid]) if input_grid else 0,
                    max([len(row) for row in output_grid]) if output_grid else 0,
                )

            # Check test examples
            for test_sample in data.get("test", []):
                input_grid = test_sample["input"]
                output_grid = test_sample["output"]
                max_rows = max(max_rows, len(input_grid), len(output_grid))
                max_cols = max(
                    max_cols,
                    max([len(row) for row in input_grid]) if input_grid else 0,
                    max([len(row) for row in output_grid]) if output_grid else 0,
                )

            h_cell_height = (H - grid_border * (max_rows + 1)) // max_rows
            h_cell_width = (H - grid_border * (max_cols + 1)) // max_cols

            cell_size = min(h_cell_width, h_cell_height)
            cell_size = min(cell_size, max_cell_size)
            if cell_size < min(H, W) // 96:
                raise ValueError("Cell is too small for the image resolution")
        else:
            cell_size = fixed_cell_size

        grid_style = deepcopy(ArcBaseStyle)
        grid_style.cell_size = cell_size
        grid_style.grid_border = grid_border

        ###
        # Process train examples
        ###
        for train_idx, train_sample in enumerate(data.get("train", [])):
            # Generate and save input grid image and metadata
            input_grid = train_sample["input"]
            input_img, input_metadata = draw_grid(
                input_grid,
                grid_style=grid_style,
                H=H,
                W=W,
            )

            # Save RGB image
            _imwrite(
                os.path.join(input_rgb_dir, f"train_{train_idx:03d}.png"), input_img
            )

            # Save metadata
            input_metadata.save(
                os.path.join(input_meta_dir, f"train_{train_idx:03d}.json")
            )

            # Generate and save output grid image and metadata
            output_grid = train_sample["output"]
            output_img, output_metadata = draw_grid(
                output_grid,
                grid_style=grid_style,
                H=H,
                W=W,
            )

            # Save RGB image
            _imwrite(
                os.path.join(output_rgb_dir, f"train_{train_idx:03d}.png"),
                output_img,
            )

            # Save metadata
            output_metadata.save(
                os.path.join(output_meta_dir, f"train_{train_idx:03d}.json")
            )

        ###
        # Process test examples
        ###
        for test_idx, test_sample in enumerate(data.get("test", [])):
            # Generate and save input grid image and metadata
            input_grid = test_sample["input"]
            input_img, input_metadata = draw_grid(
                input_grid,
                grid_style=grid_style,
                H=H,
                W=W,
            )

            # Save RGB image
            _imwrite(os.path.join(input_rgb_dir, f"test_{test_idx:03d}.png"), input_img)

            # Save metadata
            input_metadata.save(
                os.path.join(input_meta_dir, f"test_{test_idx:03d}.json")
            )

            # Generate and save output grid image and metadata
            output_grid = test_sample["output"]
            output_img, output_metadata = draw_grid(
                output_grid,
                grid_style=grid_style,
                H=H,
                W=W,
            )

            # Save RGB image
            _imwrite(
                os.path.join(output_rgb_dir, f"test_{test_idx:03d}.png"), output_img
            )

            # Save metadata
            output_metadata.save(
                os.path.join(output_meta_dir, f"test_{test_idx:03d}.json")
            )

        ###
        # Save original JSON for future evaluation
        ###
        with open(os.path.join(evaluation_dir, json_file), "r") as src_file:
            json_content = src_file.read()
        with open(os.path.join(folder_path, f"{json_file}"), "w") as dst_file:
            dst_file.write(json_content)


def main():
    parser = argparse.ArgumentParser(
        description="Generate images and metadata from JSON files."
    )
    parser.add_argument("--input_dir", help="Directory containing JSON files")
    parser.add_argument(
        "--output_dir",
        default="visual_eval",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Height of output images"
    )
    parser.add_argument("--width", type=int, default=720, help="Width of output images")
    parser.add_argument(
        "--cell_size",
        type=int,
        default=None,
        help="Size of each cell (None for auto-adjustment per JSON)",
    )
    parser.add_argument(
        "--max_cell_size",
        type=int,
        default=50,
        help="Maximum cell size when auto-adjusting (default: 50)",
    )

    parser.add_argument(
        "--grid_border",
        type=int,
        default=2,
        help="Grid border for cell grid (default: 2)",
    )

    parser.add_argument(
        "--nested",
        help="Indicate if input_dir has a nested subdirs structure",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    process_json_files(
        args.input_dir,
        args.output_dir,
        args.height,
        args.width,
        args.cell_size,
        args.max_cell_size,
        args.nested,
        args.grid_border,
    )


if __name__ == "__main__":
    main()
