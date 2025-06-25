from pathlib import Path

from visual_logic.tasks.json_arc import process_arc_like_folder

if __name__ == "__main__":
    arc_like_tasks = ["arc-agi", "arc-agi-2", "arc-dataset-tama"]
    for arc_like_task in arc_like_tasks:
        input_path = Path(f"tasks/json/{arc_like_task}.zip")
        output_path = Path(f"datasets/{arc_like_task}")
        process_arc_like_folder(
            input_path=input_path, output_path=output_path, nested=True
        )
