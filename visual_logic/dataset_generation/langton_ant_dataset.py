import shutil
from typing import Optional

import numpy as np

from visual_logic.tasks.base import TaskDatasetGenerator, TaskProblem
from visual_logic.tasks.langton_ant import LangtonAnt
from visual_logic.tasks.problem_set import TaskProblemSet
from visual_logic.tasks.render.schemas import MazeBaseStyle

from .registry import register_dataset


@register_dataset("langton_ant")
def generate_langton_ant_dataset(
    steps: int = 1, subset_sizes: Optional[list[int]] = None
):
    def ant_hamming_distance(tp0: TaskProblem, tp1: TaskProblem) -> float:
        g0 = np.array(tp0.tgt_grid)
        g1 = np.array(tp1.tgt_grid)
        if g0.shape != g1.shape:
            raise ValueError("Grid shapes do not match")
        return np.sum(g0 != g1) / g0.size

    ant_train, ant_test = TaskDatasetGenerator(
        task=LangtonAnt(
            width=8,
            height=8,
            steps=steps,
            initialization="random",
            seed=123,
        ),
        dist_fn=ant_hamming_distance,
    ).generate(n_train=100, n_test=200, distance_threshold=0.3)

    shutil.rmtree(f"datasets/langton_ant_step{steps}", ignore_errors=True)
    TaskProblemSet(task_problems=ant_train).save(
        f"datasets/langton_ant_step{steps}/train",
        MazeBaseStyle,
        subset_sizes=subset_sizes,
    )
    TaskProblemSet(task_problems=ant_test).save(
        f"datasets/langton_ant_step{steps}/test", MazeBaseStyle
    )
