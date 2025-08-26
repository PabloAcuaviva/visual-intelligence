import shutil
from typing import Optional

import numpy as np

from visual_logic.tasks.base import TaskDatasetGenerator, TaskProblem
from visual_logic.tasks.cellular_automata_1d import CellularAutomata1D
from visual_logic.tasks.problem_set import TaskProblemSet
from visual_logic.tasks.render.schemas import MazeBaseStyle

from .registry import register_dataset


@register_dataset("cellular_automata_1d")
def generate_cellular_automata_1d_dataset(
    rule: int,
    width: int = 16,
    steps: int = 16,
    initialization: str = "random",
    subset_sizes: Optional[list[int]] = None,
    n_train: int = 100,
    n_test: int = 200,
):
    def ca_hamming_distance(tp0: TaskProblem, tp1: TaskProblem) -> float:
        """Calculate Hamming distance between two cellular automaton grids."""
        g0 = np.array(tp0.init_grid)
        g1 = np.array(tp1.init_grid)
        if g0.shape != g1.shape:
            raise ValueError("Grid shapes do not match")
        return np.sum(g0 != g1) / g0.size

    ca_train, ca_test = TaskDatasetGenerator(
        task=CellularAutomata1D(
            rule=rule,
            width=width,
            steps=steps,
            initialization=initialization,
            seed=42,
        ),
        dist_fn=ca_hamming_distance,
    ).generate(
        n_train=n_train, n_test=n_test, distance_threshold=(2 / width) * (1 / steps)
    )

    dataset_name = f"cellular_automata_1d_rule{rule}_w{width}_s{steps}"
    shutil.rmtree(f"datasets/{dataset_name}", ignore_errors=True)

    TaskProblemSet(task_problems=ca_train).save(
        f"datasets/{dataset_name}/train",
        MazeBaseStyle,
        subset_sizes=subset_sizes,
    )
    TaskProblemSet(task_problems=ca_test).save(
        f"datasets/{dataset_name}/test", MazeBaseStyle
    )
