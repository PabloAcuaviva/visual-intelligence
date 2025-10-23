import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from visual_intelligence.tasks.base import TaskDatasetGenerator, TaskProblem
from visual_intelligence.tasks.chess_mate_in_n import ChessMate
from visual_intelligence.tasks.problem_set import TaskProblemSet
from visual_intelligence.tasks.render.schemas import ArcExtendedStyle

from .registry import register_dataset


@register_dataset("chess_mate_in_n")
def generate_chess_mate_in_n_dataset(
    subset_sizes: Optional[list[int]] = None,
    n_train: int = 100,
    n_test: int = 200,
    mate_in: int = 1,
    initial_turn: str = "w",
    style=ArcExtendedStyle,
    image_width: int = 272,
    image_height: int = 272,
    extend_dataset: Optional[Path] = None,
):
    def hamming_distance(tp0: TaskProblem, tp1: TaskProblem) -> float:
        g0 = np.array(tp0.tgt_grid)
        g1 = np.array(tp1.tgt_grid)
        if g0.shape != g1.shape:
            raise ValueError("Grid shapes do not match")
        return np.sum(g0 != g1) / g0.size

    subset_sizes = subset_sizes

    chess_train, chess_test = TaskDatasetGenerator(
        task=ChessMate(
            mate_in=mate_in, initial_turn=initial_turn, generate_sequential=True
        ),
        dist_fn=hamming_distance,
    ).generate(
        n_train=n_train,
        n_test=n_test,
        attempts_multiplier=100,
        distance_threshold=0.01,  # Make sure they are at least different scenarios, most of the floor will be the same
    )

    shutil.rmtree(
        f"datasets/chess_mate_in_{mate_in}_{initial_turn}", ignore_errors=True
    )
    TaskProblemSet(task_problems=chess_train).save(
        f"datasets/chess_mate_in_{mate_in}_{initial_turn}/train",
        style,
        image_width=image_width,
        image_height=image_height,
        subset_sizes=subset_sizes,
    )
    TaskProblemSet(task_problems=chess_test).save(
        f"datasets/chess_mate_in_{mate_in}_{initial_turn}/test",
        style,
        image_width=image_width,
        image_height=image_height,
    )
