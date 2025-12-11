import warnings
from pathlib import Path

from visual_intelligence.tasks.json_arc import process_arc_like_folder


def generate_arc_like_task_from_zip(arc_like_tasks: list[str]):
    for arc_like_task in arc_like_tasks:
        input_path = Path(f"visual_intelligence/tasks/resources/{arc_like_task}.zip")
        output_path = Path(f"datasets/arc-tasks/{arc_like_task}")
        if input_path.exists():
            process_arc_like_folder(
                input_path=input_path,
                output_path=output_path,
                nested=True,
                same_cell_size_all_problems=False,
            )
        else:
            warnings.warn(
                f"{input_path=} not found, continuing with remaining paths..."
            )


if __name__ == "__main__":
    generate_arc_like_task_from_zip(
        arc_like_tasks=[
            "concept-arc",
            "arc-agi",
            "arc-agi-2",
            # "arc-dataset-tama",
        ]
    )
