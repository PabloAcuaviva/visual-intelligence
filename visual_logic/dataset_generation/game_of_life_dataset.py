import shutil
from typing import Optional

import numpy as np

from visual_logic.tasks.base import TaskDatasetGenerator, TaskProblem
from visual_logic.tasks.game_of_life import GameOfLife
from visual_logic.tasks.problem_set import TaskProblemSet
from visual_logic.tasks.render.schemas import MazeBaseStyle

from .registry import register_dataset


@register_dataset("gol")
def generate_gol_dataset(steps: int = 1, subset_sizes: Optional[list[int]] = None):
    def gol_hamming_distance(tp0: TaskProblem, tp1: TaskProblem) -> float:
        g0 = np.array(tp0.tgt_grid)
        g1 = np.array(tp1.tgt_grid)
        if g0.shape != g1.shape:
            raise ValueError("Grid shapes do not match")
        return np.sum(g0 != g1) / g0.size

    gol_train, gol_test = TaskDatasetGenerator(
        task=GameOfLife(
            width=8,
            height=8,
            steps=steps,
            initialization="random",
            density=0.4,
            seed=42,
        ),
        dist_fn=gol_hamming_distance,
    ).generate(n_train=100, n_test=200, distance_threshold=0.3)

    shutil.rmtree(f"datasets/gol_step{steps}", ignore_errors=True)
    TaskProblemSet(task_problems=gol_train).save(
        f"datasets/gol_step{steps}/train",
        MazeBaseStyle,
        subset_sizes=subset_sizes,
    )
    TaskProblemSet(task_problems=gol_test).save(
        f"datasets/gol_step{steps}/test", MazeBaseStyle
    )
