import itertools
from pathlib import Path

import imageio
import numpy as np
from tqdm import tqdm

from utils import export_to_video, interpolate_x_y

if __name__ == "__main__":
    N_MAX_CONCATENATIONS = 10
    dataset_folder = Path("datasets")
    arc_names = [
        "concept-arc",
        "arc-agi",
        "arc-agi-2",
    ]
    root_arc_path = dataset_folder / "arc-tasks"

    for arc_name in arc_names:
        for partition_folder in (root_arc_path / arc_name).iterdir():
            for problem in tqdm(
                list(partition_folder.iterdir()),
                desc=f"Processing {partition_folder.name}",
            ):
                problem_test = problem / "test"
                problem_train = problem / "train"

                video_test_dir = problem_test / "video"
                video_train_dir = problem_train / "video"
                video_concat_dir = problem_train / "video_concats"

                video_test_dir.mkdir(exist_ok=True, parents=True)
                video_train_dir.mkdir(exist_ok=True, parents=True)
                video_concat_dir.mkdir(exist_ok=True, parents=True)

                for img_path in (problem_test / "init_image").iterdir():
                    interpolate_x_y(
                        img_path,
                        problem_test / "tgt_image" / img_path.name,
                        output_path=video_test_dir / f"{img_path.stem}.mp4",
                    )

                train_video_paths = []
                for img_path in (problem_train / "init_image").iterdir():
                    out_path = video_train_dir / f"{img_path.stem}.mp4"
                    interpolate_x_y(
                        img_path,
                        problem_train / "tgt_image" / img_path.name,
                        output_path=out_path,
                    )
                    train_video_paths.append(out_path)

                test_inits = {
                    p.stem: p for p in (problem_test / "init_image").iterdir()
                }

                for i, perm in enumerate(itertools.permutations(train_video_paths)):
                    frames = []
                    for vid in perm:
                        reader = imageio.get_reader(vid)
                        for frame in reader:
                            frames.append(np.array(frame))
                        reader.close()

                    idx_perm = [path_name.stem for path_name in perm]

                    for test_key, test_path in test_inits.items():
                        img = imageio.imread(test_path)
                        for _ in range(1):
                            frames.append(np.array(img))
                        export_to_video(
                            frames,
                            video_concat_dir / f"{'_'.join(idx_perm)}__{test_key}.mp4",
                            fps=10,
                        )
                        frames = frames[:-2]

                    if i >= N_MAX_CONCATENATIONS:
                        break
