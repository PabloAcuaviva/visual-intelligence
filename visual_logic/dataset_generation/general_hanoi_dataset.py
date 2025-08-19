import shutil
from typing import Literal, Optional

import numpy as np

from visual_logic.tasks.base import TaskDatasetGenerator, TaskProblem
from visual_logic.tasks.general_hanoi import GeneralHanoi
from visual_logic.tasks.problem_set import TaskProblemSet
from visual_logic.tasks.render.schemas import ArcBaseStyle

from .registry import register_dataset


@register_dataset("general_hanoi")
def generate_general_hanoi_dataset(
    subset_sizes: Optional[list[int]] = None, steps: int | Literal["all"] = "all"
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
    ).generate(
        n_train=100,
        n_test=200,
        distance_threshold=0.01,
    )

    image_height = 176
    image_width = 1024

    shutil.rmtree(f"datasets/general_hanoi_step{steps}", ignore_errors=True)
    TaskProblemSet(task_problems=hanoi_train).save(
        f"datasets/general_hanoi_step{steps}/train",
        ArcBaseStyle,
        subset_sizes=subset_sizes,
        image_width=image_width,
        image_height=image_height,
    )
    TaskProblemSet(task_problems=hanoi_test).save(
        f"datasets/general_hanoi_step{steps}/test",
        ArcBaseStyle,
        image_width=image_width,
        image_height=image_height,
    )
