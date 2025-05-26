import argparse
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

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
    background_color: dict[int, tuple]
    border_color: tuple

    @property
    def reverse_color_map(self):
        # Create reverse color map (RGB tuple to grid value)
        return {v: k for k, v in self.color_map.items()}

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
        d["color_map"] = {int(k): tuple(v) for k, v in d["color_map"].items()}
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

MazeBaseStyle = GridStyle(
    cell_size=16,
    grid_border=0,
    color_map={
        0: (71, 48, 45),  # Wall - dark brown
        1: (255, 255, 255),  # Path - white
        2: (244, 96, 54),  # End - orange-red
        3: (72, 191, 132),  # Start - green
        4: (46, 134, 171),  # Solution - blue
    },
    background_color=(255, 255, 255),  # White background
    border_color=(0, 0, 0),  # Black border
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


def recover_grid(
    image,
    grid_metadata: GridMetadata,
    enhance_iterations: int = 0,
    convergence_threshold: float = 1.0,
):
    image = np.array(image, dtype=np.float32)
    grid_h = grid_metadata.grid_h
    grid_w = grid_metadata.grid_w
    start_x = grid_metadata.start_x
    start_y = grid_metadata.start_y

    grid_border = grid_metadata.grid_style.grid_border
    cell_size = grid_metadata.grid_style.cell_size
    reverse_color_map = grid_metadata.grid_style.reverse_color_map

    # Make a copy of the original image to work with
    working_image = image.copy()

    def recognize_from_image(img):
        # Initialize empty grid
        recovered_grid = []
        cell_colors = []  # Store (actual_rgb, closest_match_rgb) for each cell
        total_distance = 0  # Track total color distance

        # Extract each cell's color and map back to grid value
        for i in range(grid_h):
            row = []
            for j in range(grid_w):
                x1 = start_x + j * (cell_size + grid_border) + grid_border
                y1 = start_y + i * (cell_size + grid_border) + grid_border

                x2 = x1 + cell_size
                y2 = y1 + cell_size

                # Use average cell color
                cell = img[y1:y2, x1:x2]
                rgb = tuple(cell.reshape(-1, 3).mean(axis=0).astype(int))

                # Find the closest color in color map
                closest_color = min(
                    reverse_color_map.keys(),
                    key=lambda c: sum((c[k] - rgb[k]) ** 2 for k in range(3)),
                )

                # Calculate color distance
                distance = (
                    sum((rgb[k] - closest_color[k]) ** 2 for k in range(3)) ** 0.5
                )
                total_distance += distance

                # Map the color back to grid value
                grid_value = reverse_color_map[closest_color]
                row.append(grid_value)
                cell_colors.append((rgb, closest_color))

            recovered_grid.append(row)

        return recovered_grid, cell_colors, total_distance

    ###
    # Recover grid
    ###
    # Iterative enhancement
    iteration = 0
    prev_distance = 0.0
    while iteration < enhance_iterations:
        # Recognize grid and get color information
        _, cell_colors, current_distance = recognize_from_image(working_image)

        # Check if we've converged
        if abs(prev_distance - current_distance) < convergence_threshold:
            break

        prev_distance = current_distance

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
        working_image[:, :, 0] += r_shift
        working_image[:, :, 1] += g_shift
        working_image[:, :, 2] += b_shift

        # Clip values to valid RGB range (0-255)
        working_image = np.clip(working_image, 0, 255)

        iteration += 1

    # Final recognition pass with the current image state
    final_grid, _, _ = recognize_from_image(working_image)

    return final_grid


def calculate_cell_size(
    data: dict, H: int, W: int, max_cell_size: int, grid_border: int
) -> int:
    """
    Calculate optimal cell size based on the largest grid in the dataset.

    Args:
        data: JSON data containing train/test samples
        H: Height of output images
        W: Width of output images
        max_cell_size: Maximum allowed cell size
        grid_border: Border size around grid

    Returns:
        Optimal cell size
    """
    max_rows = 0
    max_cols = 0

    # Check all grids in train and test samples
    for sample_type in ["train", "test"]:
        for sample in data.get(sample_type, []):
            for grid_type in ["input", "output"]:
                grid = sample[grid_type]
                if grid:
                    max_rows = max(max_rows, len(grid))
                    max_cols = max(max_cols, max(len(row) for row in grid))

    # Calculate cell size based on available space
    h_cell_height = (H - grid_border * (max_rows + 1)) // max_rows
    h_cell_width = (H - grid_border * (max_cols + 1)) // max_cols

    cell_size = min(h_cell_width, h_cell_height, max_cell_size)

    if cell_size < min(H, W) // 96:
        raise ValueError("Cell is too small for the image resolution")

    return cell_size


def process_single_json_file(
    json_path: Path,
    output_folder: Path,
    H: int = 480,
    W: int = 480,
    fixed_cell_size: int = None,
    max_cell_size: int = 50,
    grid_border: int = 2,
    base_grid_style: GridStyle = ArcBaseStyle,
):
    """
    Process a single JSON file and create visualization folder with images and metadata.

    Args:
        json_path: Path to the JSON file
        output_folder: Output folder for this specific JSON file
        H: Height of output images
        W: Width of output images
        fixed_cell_size: Fixed cell size (if None, auto-adjusts)
        max_cell_size: Maximum allowed cell size when auto-adjusting
        grid_border: Border size around grid
    """
    # Create directory structure
    input_rgb_dir = output_folder / "input" / "rgb"
    input_meta_dir = output_folder / "input" / "metadata"
    output_rgb_dir = output_folder / "output" / "rgb"
    output_meta_dir = output_folder / "output" / "metadata"

    for dir_path in [input_rgb_dir, input_meta_dir, output_rgb_dir, output_meta_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Determine cell size
    if fixed_cell_size is None:
        cell_size = calculate_cell_size(data, H, W, max_cell_size, grid_border)
    else:
        cell_size = fixed_cell_size

    # Set up grid style
    grid_style = deepcopy(base_grid_style)
    grid_style.cell_size = cell_size
    grid_style.grid_border = grid_border

    # Process train samples
    for train_idx, train_sample in enumerate(data.get("train", [])):
        _process_sample_pair(
            train_sample,
            train_idx,
            "train",
            input_rgb_dir,
            input_meta_dir,
            output_rgb_dir,
            output_meta_dir,
            grid_style,
            H,
            W,
        )

    # Process test samples
    for test_idx, test_sample in enumerate(data.get("test", [])):
        _process_sample_pair(
            test_sample,
            test_idx,
            "test",
            input_rgb_dir,
            input_meta_dir,
            output_rgb_dir,
            output_meta_dir,
            grid_style,
            H,
            W,
        )

    # Copy original JSON file to output folder
    with open(json_path, "r") as src_file:
        json_content = src_file.read()
    with open(output_folder / json_path.name, "w") as dst_file:
        dst_file.write(json_content)


def _process_sample_pair(
    sample: dict,
    index: int,
    sample_type: str,
    input_rgb_dir: Path,
    input_meta_dir: Path,
    output_rgb_dir: Path,
    output_meta_dir: Path,
    grid_style,
    H: int,
    W: int,
):
    """
    Process a single input/output sample pair.

    Args:
        sample: Dictionary containing 'input' and 'output' grids
        index: Sample index for naming
        sample_type: 'train' or 'test'
        input_rgb_dir: Directory for input RGB images
        input_meta_dir: Directory for input metadata
        output_rgb_dir: Directory for output RGB images
        output_meta_dir: Directory for output metadata
        grid_style: Grid styling configuration
        H: Image height
        W: Image width
    """
    filename = f"{index:03d}"
    input_rgb_path = input_rgb_dir / sample_type / f"{filename}.png"
    input_meta_path = input_meta_dir / sample_type / f"{filename}.json"
    output_rgb_path = output_rgb_dir / sample_type / f"{filename}.png"
    output_meta_path = output_meta_dir / sample_type / f"{filename}.json"

    # Ensure directories exist
    input_rgb_path.parent.mkdir(parents=True, exist_ok=True)
    input_meta_path.parent.mkdir(parents=True, exist_ok=True)
    output_rgb_path.parent.mkdir(parents=True, exist_ok=True)
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Process input grid
    input_grid = sample["input"]
    input_img, input_metadata = draw_grid(input_grid, grid_style=grid_style, H=H, W=W)
    _imwrite(input_rgb_path, input_img)
    input_metadata.save(input_meta_path)

    # Process output grid
    output_grid = sample["output"]
    output_img, output_metadata = draw_grid(
        output_grid, grid_style=grid_style, H=H, W=W
    )
    _imwrite(output_rgb_path, output_img)
    output_metadata.save(output_meta_path)


def process_json_files(
    evaluation_dir: Path,
    output_dir: Path,
    H: int = 480,
    W: int = 480,
    fixed_cell_size: int = None,
    max_cell_size: int = 50,
    nested: bool = False,
    grid_border: int = 2,
):
    """
    Process all JSON files in evaluation directory and create visualization folders.

    Args:
        evaluation_dir: Directory containing JSON files
        output_dir: Output directory for visualizations
        H: Height of output images
        W: Width of output images
        fixed_cell_size: Fixed cell size (if None, auto-adjusts per JSON)
        max_cell_size: Maximum allowed cell size when auto-adjusting
        nested: If the JSON is in a nested structure
        grid_border: Border size around grid
    """
    evaluation_dir = Path(evaluation_dir)
    output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Handle nested directories recursively
    if nested:
        nested_dirs = [d for d in evaluation_dir.iterdir() if d.is_dir()]
        for nested_dir in nested_dirs:
            nested_output_dir = output_dir / nested_dir.name
            process_json_files(
                evaluation_dir=evaluation_dir / nested_dir.name,
                output_dir=nested_output_dir,
                H=H,
                W=W,
                fixed_cell_size=fixed_cell_size,
                max_cell_size=max_cell_size,
                nested=True,
                grid_border=grid_border,
            )

    # Get and sort JSON files
    json_files = [f for f in evaluation_dir.iterdir() if f.suffix == ".json"]
    json_files.sort(key=lambda x: x.name)

    # Process each JSON file
    for idx, json_file in enumerate(tqdm(json_files, desc="Processing files")):
        subfolder = f"{idx:03d}"
        folder_path = output_dir / subfolder

        process_single_json_file(
            json_path=json_file,
            output_folder=folder_path,
            H=H,
            W=W,
            fixed_cell_size=fixed_cell_size,
            max_cell_size=max_cell_size,
            grid_border=grid_border,
        )


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
    parser.add_argument("--width", type=int, default=480, help="Width of output images")
    parser.add_argument(
        "--fixed_cell_size",
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
        fixed_cell_size=args.fixed_cell_size,
        max_cell_size=args.max_cell_size,
        grid_border=args.grid_border,
        nested=args.nested,
    )


if __name__ == "__main__":
    # main()

    process_single_json_file(
        json_path=Path("tasks/debug_output/maze_dataset.json"),
        output_folder=Path("datasets/test"),
        H=21 * 16,
        W=21 * 16,
        fixed_cell_size=16,
        max_cell_size=500,
        grid_border=0,
        base_grid_style=MazeBaseStyle,
    )
