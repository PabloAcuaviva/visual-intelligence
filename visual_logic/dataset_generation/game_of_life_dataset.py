import shutil

import numpy as np

from visual_logic.tasks.base import TaskDatasetGenerator, TaskProblem
from visual_logic.tasks.game_of_life import GameOfLife
from visual_logic.tasks.problem_set import TaskProblemSet
from visual_logic.tasks.render.schemas import ArcBaseStyle

from .registry import register_dataset


@register_dataset("gol")
def generate_gol_dataset():
    def gol_hamming_distance(tp0: TaskProblem, tp1: TaskProblem) -> float:
        g0 = np.array(tp0.tgt_grid)
        g1 = np.array(tp1.tgt_grid)
        if g0.shape != g1.shape:
            raise ValueError("Grid shapes do not match")
        return np.sum(g0 != g1) / g0.size

    gol_train, gol_test = TaskDatasetGenerator(
        task=GameOfLife(
            width=16,
            height=16,
            steps=8,
            initialization="random",
            density=0.4,
            seed=42,
        ),
        dist_fn=gol_hamming_distance,
    ).generate(n_train=100, n_test=10, distance_threshold=0.3)

    shutil.rmtree("datasets/gol", ignore_errors=True)
    TaskProblemSet(task_problems=gol_train).save("datasets/gol/train", ArcBaseStyle)
    TaskProblemSet(task_problems=gol_test).save("datasets/gol/test", ArcBaseStyle)
