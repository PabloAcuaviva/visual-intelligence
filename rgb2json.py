import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import trange

from json2rgb import recover_grid


def imread(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)  # Load as BGR
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return img


def format_grid(grid):
    return "\n".join([" ".join(map(str, row)) for row in grid])


if __name__ == "__main__":
    data_path = Path("visual_eval")
    for i in trange(400):
        ex = f"{i:03d}"

        for input_output in ["input", "output"]:
            input_output_dir = data_path / ex / input_output
            rgb_dir = input_output_dir / "rgb"
            metadata_dir = input_output_dir / "metadata"

            # Find the only JSON file in data_path / ex
            json_files = list(data_path.glob(f"{ex}/*.json"))
            if not json_files:
                raise FileNotFoundError("No JSON file found in the directory.")

            original_json_path = json_files[0]  # There's only one JSON file

            with open(original_json_path, "r") as f:
                original_data = json.load(f)

            for image_path in rgb_dir.iterdir():
                partition, i = image_path.stem.split("_")
                i = int(i)

                image = imread(str(image_path))

                metadata_path = metadata_dir / f"{image_path.stem}.json"
                with open(metadata_path, "r") as f:
                    canvas_metadata = json.load(f)

                grid = recover_grid(image, grid_metadata=canvas_metadata)

                image_name = image_path.stem
                dataset_type, idx = image_name.split("_")

                original_grid = original_data[partition][i][input_output]

                grid_diff = np.sum(np.array(grid) != np.array(original_grid))
                if grid_diff != 0:
                    print("RECOVERED GRID")
                    print(format_grid(grid))
                    print("ORIGINAL GRID")
                    print(format_grid(original_grid))
                    raise ValueError(
                        f"Failed in reconstruction for {input_output}/{image_path} with {grid_diff}"
                    )
