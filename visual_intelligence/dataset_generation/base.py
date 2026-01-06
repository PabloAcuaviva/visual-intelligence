"""
Base module for dataset generation with common utilities and configuration.

This module provides:
- Pre-defined distance functions (accessible by string name)
- Style registry (mapping string names to RenderStyle instances)
- BaseDatasetGenerator class for inheritance-based dataset generation
"""

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from visual_intelligence.tasks.base import Task, TaskDatasetGenerator, TaskProblem
from visual_intelligence.tasks.problem_set import TaskProblemSet, VideoConfig
from visual_intelligence.tasks.render.schemas import (
    ArcBaseStyle,
    ArcExtendedStyle,
    MazeBaseStyle,
    RenderStyle,
)


# ============================================================================
# Pre-defined Distance Functions
# ============================================================================
def _hamming_distance(grid0, grid1) -> float:
    """Calculate normalized Hamming distance between two grids."""
    g0, g1 = np.array(grid0), np.array(grid1)
    if g0.shape != g1.shape:
        raise ValueError("Grid shapes do not match")
    return np.sum(g0 != g1) / g0.size


def _hamming_init_distance(tp0: TaskProblem, tp1: TaskProblem) -> float:
    """Hamming distance on init_grid."""
    return _hamming_distance(tp0.init_grid, tp1.init_grid)


def _hamming_tgt_distance(tp0: TaskProblem, tp1: TaskProblem) -> float:
    """Hamming distance on tgt_grid."""
    return _hamming_distance(tp0.tgt_grid, tp1.tgt_grid)


def _jaccard_path_distance(tp0: TaskProblem, tp1: TaskProblem) -> float:
    """Jaccard distance on path metadata."""
    path0 = set(
        tuple(t) if isinstance(t, list) else t
        for t in tp0.task_specific_metadata["path"]
    )
    path1 = set(
        tuple(t) if isinstance(t, list) else t
        for t in tp1.task_specific_metadata["path"]
    )
    if not path0 and not path1:
        raise ValueError("Both paths are empty")
    intersection = len(path0 & path1)
    union = len(path0 | path1)
    return 1.0 - (intersection / union)


DISTANCE_FUNCTIONS: dict[str, Callable[[TaskProblem, TaskProblem], float]] = {
    "hamming_init": _hamming_init_distance,
    "hamming_tgt": _hamming_tgt_distance,
    "jaccard_path": _jaccard_path_distance,
}


def get_distance_fn(
    metric: Union[str, Callable[[TaskProblem, TaskProblem], float]],
) -> Callable[[TaskProblem, TaskProblem], float]:
    """Get a distance function by name or return the callable directly."""
    if callable(metric):
        return metric
    if metric not in DISTANCE_FUNCTIONS:
        raise ValueError(
            f"Unknown distance metric: {metric}. "
            f"Available: {list(DISTANCE_FUNCTIONS.keys())}"
        )
    return DISTANCE_FUNCTIONS[metric]


# ============================================================================
# Style Registry
# ============================================================================
ArcReducedStyle = RenderStyle(
    cell_size=10,
    grid_border_size=0,
    value_to_color={
        0: (0, 0, 0),  # Black
        1: (0, 116, 217),  # Blue
        2: (255, 65, 54),  # Red
        3: (46, 204, 64),  # Green
        4: (255, 220, 0),  # Yellow
        5: (170, 170, 170),  # Grey
        6: (240, 18, 190),  # Fuchsia
        7: (255, 133, 27),  # Orange
        8: (127, 219, 255),  # Teal
        9: (135, 12, 37),  # Brown
    },
    background_color=(0, 0, 0),
    border_color=(85, 85, 85),
)


STYLES: dict[str, RenderStyle] = {
    "arc_base": ArcBaseStyle,
    "arc_extended": ArcExtendedStyle,
    "arc_reduced": ArcReducedStyle,
    "maze": MazeBaseStyle,
}


def get_style(style: Union[str, RenderStyle]) -> RenderStyle:
    """Get a style by name or return the RenderStyle directly."""
    if isinstance(style, RenderStyle):
        return style
    if style not in STYLES:
        raise ValueError(f"Unknown style: {style}. Available: {list(STYLES.keys())}")
    return STYLES[style]


# ============================================================================
# Base Dataset Generator Class
# ============================================================================
@dataclass(kw_only=True)
class BaseDatasetGenerator(ABC):
    """
    Base class for dataset generators using inheritance.

    Subclasses should:
    1. Add task-specific fields with defaults
    2. Override class-level defaults if needed (e.g., style = "maze")
    3. Implement create_task() and dataset_name property

    Example:
        @dataclass(kw_only=True)
        class MyDatasetGenerator(BaseDatasetGenerator):
            # Task-specific params
            size: int = 5

            # Override defaults (only if different)
            style: str = "maze"

            def create_task(self) -> Task:
                return MyTask(size=self.size)

            @property
            def dataset_name(self) -> str:
                return f"my_task_{self.size}"

        # Usage:
        MyDatasetGenerator().generate()  # All defaults
        MyDatasetGenerator(size=10, n_train=500).generate()  # Override
        MyDatasetGenerator(video=VideoConfig()).generate()  # With video
    """

    n_train: int = 100
    n_test: int = 200
    subset_sizes: Optional[list[int]] = None
    extend_dataset: Optional[Path] = None
    out_dir: Union[str, Path] = "datasets"
    style: Union[str, RenderStyle] = "arc_base"
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    distance_metric: Union[str, Callable[[TaskProblem, TaskProblem], float]] = (
        "hamming_init"
    )
    distance_threshold: float = 0.3
    attempts_multiplier: int = 20
    video: Optional[VideoConfig] = None

    @abstractmethod
    def create_task(self) -> Task:
        """Create and return the task instance."""

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the dataset name for output directory."""

    def generate(self) -> tuple[list[TaskProblem], list[TaskProblem]]:
        """Generate the dataset."""
        task = self.create_task()

        dist_fn = get_distance_fn(self.distance_metric)

        train_dataset, test_dataset = TaskDatasetGenerator(
            task=task,
            dist_fn=dist_fn,
            extend_dataset=self.extend_dataset,
        ).generate(
            n_train=self.n_train,
            n_test=self.n_test,
            distance_threshold=self.distance_threshold,
            attempts_multiplier=self.attempts_multiplier,
        )

        resolved_style = get_style(self.style)

        save_kwargs: dict = {}
        if self.image_width is not None:
            save_kwargs["image_width"] = self.image_width
        if self.image_height is not None:
            save_kwargs["image_height"] = self.image_height
        if self.video is not None:
            save_kwargs["video_config"] = self.video

        out_dir = Path(self.out_dir) / self.dataset_name
        shutil.rmtree(out_dir, ignore_errors=True)

        TaskProblemSet(task_problems=train_dataset).save(
            out_dir / "train",
            resolved_style,
            subset_sizes=self.subset_sizes,
            **save_kwargs,
        )
        TaskProblemSet(task_problems=test_dataset).save(
            out_dir / "test",
            resolved_style,
            **save_kwargs,
        )

        return train_dataset, test_dataset

    def __call__(self) -> tuple[list[TaskProblem], list[TaskProblem]]:
        """Make the generator callable."""
        return self.generate()
