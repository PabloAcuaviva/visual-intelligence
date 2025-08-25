import shutil
from typing import Optional

import numpy as np

from visual_logic.tasks.base import TaskDatasetGenerator, TaskProblem
from visual_logic.tasks.problem_set import TaskProblemSet
from visual_logic.tasks.render.schemas import ArcBaseStyle
from visual_logic.tasks.sudoku import Sudoku

from .registry import register_dataset


@register_dataset("sudoku")
def generate_sudoku_dataset(
    difficulty: str = "easy",
    variant: str = "standard",
    subset_sizes: Optional[list[int]] = None,
    n_train: int = 100,
    n_test: int = 200,
):
    def sudoku_distance(tp0: TaskProblem, tp1: TaskProblem) -> float:
        g0 = np.array(tp0.init_grid)
        g1 = np.array(tp1.init_grid)
        if g0.shape != g1.shape:
            raise ValueError("Grid shapes do not match")
        return np.sum(g0 != g1) / g0.size

    sudoku_train, sudoku_test = TaskDatasetGenerator(
        task=Sudoku(difficulty=difficulty, variant=variant, seed=123),
        dist_fn=sudoku_distance,
    ).generate(n_train=n_train, n_test=n_test, distance_threshold=0.3)

    image_width = image_height = 16 * (19 if variant == "standard" else 9)

    out_dir = f"datasets/sudoku_{variant}_{difficulty}"
    shutil.rmtree(out_dir, ignore_errors=True)
    TaskProblemSet(task_problems=sudoku_train).save(
        f"{out_dir}/train",
        ArcBaseStyle,
        subset_sizes=subset_sizes,
        image_width=image_width,
        image_height=image_height,
    )
    TaskProblemSet(task_problems=sudoku_test).save(
        f"{out_dir}/test",
        ArcBaseStyle,
        image_width=image_width,
        image_height=image_height,
    )
