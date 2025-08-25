import shutil
from typing import Optional

import numpy as np

from visual_logic.tasks.base import TaskDatasetGenerator, TaskProblem
from visual_logic.tasks.hitori import Hitori
from visual_logic.tasks.problem_set import TaskProblemSet
from visual_logic.tasks.render.schemas import ArcBaseStyle

from .registry import register_dataset


@register_dataset("hitori")
def generate_hitori_dataset(
    size: int = 5,
    difficulty: str = "easy",
    subset_sizes: Optional[list[int]] = None,
    n_train: int = 100,
    n_test: int = 200,
):
    def hitori_distance(tp0: TaskProblem, tp1: TaskProblem) -> float:
        g0 = np.array(tp0.init_grid)
        g1 = np.array(tp1.init_grid)
        return np.sum(g0 != g1) / g0.size

    hitori_train, hitori_test = TaskDatasetGenerator(
        task=Hitori(size=size, difficulty=difficulty, seed=123),
        dist_fn=hitori_distance,
    ).generate(n_train=n_train, n_test=n_test, distance_threshold=0.3)

    style = ArcBaseStyle
    orig_size = (
        style.cell_size + style.grid_border_size
    ) * size + style.grid_border_size

    image_width = image_height = 16 * (orig_size // 16 + (orig_size % 16 != 0))

    out_dir = f"datasets/hitori_{size}_{difficulty}"
    shutil.rmtree(out_dir, ignore_errors=True)
    TaskProblemSet(task_problems=hitori_train).save(
        f"{out_dir}/train",
        style,
        subset_sizes=subset_sizes,
        image_width=image_width,
        image_height=image_height,
    )
    TaskProblemSet(task_problems=hitori_test).save(
        f"{out_dir}/test",
        style,
        image_width=image_width,
        image_height=image_height,
    )
