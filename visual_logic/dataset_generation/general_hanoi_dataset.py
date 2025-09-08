import shutil
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from visual_logic.tasks.base import TaskDatasetGenerator, TaskProblem
from visual_logic.tasks.general_hanoi import GeneralHanoi
from visual_logic.tasks.problem_set import TaskProblemSet
from visual_logic.tasks.render.schemas import RenderStyle

from .registry import register_dataset

ArcReducedStyle = RenderStyle(
    cell_size=10,
    grid_border_size=0,
    value_to_color={
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


@register_dataset("general_hanoi")
def generate_general_hanoi_dataset(
    subset_sizes: Optional[list[int]] = None,
    steps: int | Literal["all"] = "all",
    n_train: int = 100,
    n_test: int = 200,
    image_height: int = 64,
    image_width: int = 320,
    extend_dataset: Optional[Path] = None,
):
    def hamming_distance(tp0: TaskProblem, tp1: TaskProblem) -> float:
        g0 = np.array(tp0.init_grid)
        g1 = np.array(tp1.init_grid)
        if g0.shape != g1.shape:
            raise ValueError("Grid shapes do not match")
        return np.sum(g0 != g1) / g0.size

    hanoi_train, hanoi_test = TaskDatasetGenerator(
        task=GeneralHanoi(num_disks=5, step=steps),
        dist_fn=hamming_distance,
        extend_dataset=extend_dataset,
    ).generate(
        n_train=n_train,
        n_test=n_test,
        distance_threshold=0.01,
    )

    shutil.rmtree(f"datasets/general_hanoi_step{steps}", ignore_errors=True)
    TaskProblemSet(task_problems=hanoi_train).save(
        f"datasets/general_hanoi_step{steps}/train",
        ArcReducedStyle,
        subset_sizes=subset_sizes,
        image_width=image_width,
        image_height=image_height,
    )
    TaskProblemSet(task_problems=hanoi_test).save(
        f"datasets/general_hanoi_step{steps}/test",
        ArcReducedStyle,
        image_width=image_width,
        image_height=image_height,
    )
