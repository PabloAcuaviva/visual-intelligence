import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from json2rgb import GridMetadata, recover_grid


def _imread(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)  # Load as BGR
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return img


def format_grid(grid):
    return "\n".join([" ".join(map(str, row)) for row in grid])


if __name__ == "__main__":
    root_path = Path("datasets/arc-explore")
    nested_levels = 1

    # Get dataset paths
    data_paths = [root_path]
    nested_data_paths = []
    while nested_levels > 0:
        for data_path in data_paths:
            nested_data_paths += [p for p in data_path.iterdir() if p.is_dir()]
        nested_levels -= 1
        data_paths = nested_data_paths
        nested_data_paths = []

    for data_path in tqdm(data_paths, desc="Comparing datasets"):
        for input_or_output in ["input", "output"]:
            input_or_output_dir = data_path / input_or_output
            outer_rgb_dir = input_or_output_dir / "rgb"
            outer_metadata_dir = input_or_output_dir / "metadata"
            for partition in ["train", "test"]:
                rgb_dir = outer_rgb_dir / partition
                metadata_dir = outer_metadata_dir / partition

                # Find the only JSON file in data_path / ex
                json_files = list(data_path.glob("*.json"))
                if not json_files:
                    raise FileNotFoundError("No JSON file found in the directory.")
                if len(json_files) > 1:
                    raise ValueError("Multiple JSON files found in the directory.")
                json_file = json_files[0]  # There's only one JSON file
                with open(json_file, "r") as f:
                    json_data = json.load(f)

                for image_path in rgb_dir.iterdir():
                    i = int(image_path.stem)
                    image = _imread(str(image_path))

                    metadata_path = metadata_dir / f"{image_path.stem}.json"
                    grid_metadata = GridMetadata.load(metadata_path)

                    recovered_grid = recover_grid(
                        image, grid_metadata=grid_metadata, enhance_iterations=4
                    )
                    grid = json_data[partition][i][input_or_output]
                    grid_diff = np.sum(np.array(recovered_grid) != np.array(grid))
                    if grid_diff != 0:
                        print("RECOVERED GRID")
                        print(format_grid(recovered_grid))
                        print("ORIGINAL GRID")
                        print(format_grid(grid))
                        raise ValueError(
                            f"Failed in reconstruction for {input_or_output}/{image_path} with {grid_diff}"
                        )
