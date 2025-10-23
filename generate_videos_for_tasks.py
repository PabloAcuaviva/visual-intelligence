from pathlib import Path

from tqdm import tqdm

from utils import interpolate_x_y

if __name__ == "__main__":
    generate_videos_for_arc_tasks = True

    generate_videos_for_games = True
    generate_video_for_navigation = True
    generate_videos_automatas = True

    dataset_folder = Path("datasets")

    ###
    # Prepare arc tasks
    ###
    if generate_videos_for_arc_tasks:
        arc_names = ["arc-agi", "arc-agi-2", "concept-arc"]
        root_arc_path = dataset_folder / "arc-tasks"

        for arc_name in arc_names:
            print("=" * 80)
            print(" " * 20, f"Working on {arc_name}")
            print("=" * 80)
            for partition_folder in (root_arc_path / arc_name).iterdir():
                for problem in tqdm(
                    list(partition_folder.iterdir()),
                    desc=f"Generating video for problems on partition={partition_folder.name}",
                ):
                    # problem_test = problem / "test" # One doesn't need the test videos for anything
                    problem_train = problem / "train"
                    for problem_split in [problem_train]:
                        problem_split_video_dir = problem_split / "video"
                        problem_split_video_dir.mkdir(exist_ok=True, parents=True)
                        for img_path in (problem_split / "init_image").iterdir():
                            interpolate_x_y(
                                img_path,
                                problem_split / "tgt_image" / img_path.name,
                                output_path=problem_split_video_dir
                                / f"{img_path.stem}.mp4",
                            )

    prepare_videos_for_task_folder = []
    if generate_videos_for_games:
        game_names = ["hitori", "sudoku", "connect4", "chess"]
        prepare_videos_for_task_folder += [
            path
            for path in dataset_folder.iterdir()
            if any(n in path.as_posix() for n in game_names)
        ]

    if generate_videos_automatas:
        automata_names = ["cellular_automata", "Gol", "langton_ant"]
        prepare_videos_for_task_folder += [
            path
            for path in dataset_folder.iterdir()
            if any(n in path.as_posix() for n in automata_names)
        ]

    if generate_video_for_navigation:
        navigation_names = ["maze", "navigation"]
        prepare_videos_for_task_folder += [
            path
            for path in dataset_folder.iterdir()
            if any(n in path.as_posix() for n in navigation_names)
        ]

    for task_folder_path in prepare_videos_for_task_folder:
        print("=" * 80)
        print(" " * 20, f"Working on {task_folder_path.name}")
        print("=" * 80)
        # task_folder_test = task_folder_path / "test" # One doesn't need the test videos for anything
        task_folder_train = task_folder_path / "train"
        for problem_split in [task_folder_train]:
            problem_split_video_dir = problem_split / "video"
            problem_split_video_dir.mkdir(exist_ok=True, parents=True)
            for img_path in tqdm(
                list((problem_split / "init_image").iterdir()),
                desc="Generating video for image pairs...",
            ):
                interpolate_x_y(
                    img_path,
                    problem_split / "tgt_image" / img_path.name,
                    output_path=problem_split_video_dir / f"{img_path.stem}.mp4",
                )
