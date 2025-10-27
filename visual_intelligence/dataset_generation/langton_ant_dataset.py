import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np

from visual_intelligence.tasks.base import TaskDatasetGenerator, TaskProblem
from visual_intelligence.tasks.langton_ant import LangtonAnt
from visual_intelligence.tasks.problem_set import TaskProblemSet
from visual_intelligence.tasks.render.schemas import MazeBaseStyle

from .registry import register_dataset


@register_dataset("langton_ant")
def generate_langton_ant_dataset(
    steps: int = 1,
    subset_sizes: Optional[list[int]] = None,
    n_train: int = 100,
    n_test: int = 200,
    extend_dataset: Optional[Path] = None,
    out_dir: Union[str, Path] = "datasets",
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
            init_grid_as=0,
            ant_initial_dir="N",
        ),
        dist_fn=ant_hamming_distance,
        extend_dataset=extend_dataset,
    ).generate(n_train=n_train, n_test=n_test, distance_threshold=0.3)

    out_dir = Path(out_dir)
    out_dir = out_dir / f"langton_ant_step{steps}"
    shutil.rmtree(out_dir, ignore_errors=True)

    TaskProblemSet(task_problems=ant_train).save(
        out_dir / "train",
        MazeBaseStyle,
        subset_sizes=subset_sizes,
    )
    TaskProblemSet(task_problems=ant_test).save(
        out_dir / "test",
        MazeBaseStyle,
    )
