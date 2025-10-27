import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np

from visual_intelligence.tasks.base import TaskDatasetGenerator, TaskProblem
from visual_intelligence.tasks.connect4 import Connect4
from visual_intelligence.tasks.problem_set import TaskProblemSet
from visual_intelligence.tasks.render.schemas import ArcBaseStyle

from .registry import register_dataset


@register_dataset("connect4")
def generate_connect4_dataset(
    subset_sizes: Optional[list[int]] = None,
    n_train: int = 100,
    n_test: int = 200,
    style=ArcBaseStyle,
    image_width: int = 240,
    image_height: int = 240,
    extend_dataset: Optional[Path] = None,
    out_dir: Union[str, Path] = "datasets",
):
    def connect4_hamming_distance(tp0: TaskProblem, tp1: TaskProblem) -> float:
        g0 = np.array(tp0.tgt_grid)
        g1 = np.array(tp1.tgt_grid)
        if g0.shape != g1.shape:
            raise ValueError("Grid shapes do not match")
        return np.sum(g0 != g1) / g0.size

    subset_sizes = subset_sizes

    connect4_train, connect4_test = TaskDatasetGenerator(
        task=Connect4(seed=420),
        dist_fn=connect4_hamming_distance,
        extend_dataset=extend_dataset,
    ).generate(
        n_train=n_train,
        n_test=n_test,
        attempts_multiplier=5000,
        distance_threshold=0.25,
    )

    out_dir = Path(out_dir)
    out_dir = out_dir / "connect4"
    shutil.rmtree(out_dir, ignore_errors=True)

    TaskProblemSet(task_problems=connect4_train).save(
        out_dir / "train",
        style,
        image_width=image_width,
        image_height=image_height,
        subset_sizes=subset_sizes,
    )
    TaskProblemSet(task_problems=connect4_test).save(
        out_dir / "test",
        style,
        image_width=image_width,
        image_height=image_height,
    )
