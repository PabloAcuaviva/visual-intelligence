import shutil
from typing import Optional

import numpy as np

from visual_logic.tasks.base import TaskDatasetGenerator, TaskProblem
from visual_logic.tasks.connect4 import Connect4
from visual_logic.tasks.problem_set import TaskProblemSet
from visual_logic.tasks.render.schemas import ArcBaseStyle

from .registry import register_dataset


@register_dataset("connect4")
def generate_connect4_dataset(
    subset_sizes: Optional[list[int]] = None,
    n_train: int = 100,
    n_test: int = 200,
    style=ArcBaseStyle,
):
    def connect4_hamming_distance(tp0: TaskProblem, tp1: TaskProblem) -> float:
        g0 = np.array(tp0.tgt_grid)
        g1 = np.array(tp1.tgt_grid)
        if g0.shape != g1.shape:
            raise ValueError("Grid shapes do not match")
        return np.sum(g0 != g1) / g0.size

    image_width = image_height = 240

    subset_sizes = subset_sizes

    connect4_train, connect4_test = TaskDatasetGenerator(
        task=Connect4(seed=420),
        dist_fn=connect4_hamming_distance,
    ).generate(
        n_train=n_train,
        n_test=n_test,
        attempts_multiplier=100,
        distance_threshold=0.25,
    )

    shutil.rmtree("datasets/connect4", ignore_errors=True)
    TaskProblemSet(task_problems=connect4_train).save(
        "datasets/connect4/train",
        style,
        image_width=image_width,
        image_height=image_height,
        subset_sizes=subset_sizes,
    )
    TaskProblemSet(task_problems=connect4_test).save(
        "datasets/connect4/test",
        style,
        image_width=image_width,
        image_height=image_height,
    )
