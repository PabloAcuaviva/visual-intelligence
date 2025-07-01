import shutil

import numpy as np

from visual_logic.tasks.base import TaskDatasetGenerator, TaskProblem
from visual_logic.tasks.langton_ant import LangtonAnt
from visual_logic.tasks.problem_set import TaskProblemSet
from visual_logic.tasks.render.schemas import ArcBaseStyle

from .registry import register_dataset


@register_dataset("langton_ant")
def generate_langton_ant_dataset():
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
            steps=16,
            initialization="random",
            seed=123,
        ),
        dist_fn=ant_hamming_distance,
    ).generate(n_train=100, n_test=10, distance_threshold=0.3)

    shutil.rmtree("datasets/langton_ant", ignore_errors=True)
    TaskProblemSet(task_problems=ant_train).save(
        "datasets/langton_ant/train", ArcBaseStyle
    )
    TaskProblemSet(task_problems=ant_test).save(
        "datasets/langton_ant/test", ArcBaseStyle
    )
