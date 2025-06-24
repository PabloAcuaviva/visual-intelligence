from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, Type, TypeVar

import numpy as np
from PIL import Image
from pydantic import BaseModel, field_validator
from tqdm import tqdm

from visual_logic.tasks.render.schemas import RenderMetadata, RenderStyle
from visual_logic.typing_and_extensions import Grid, Video

###
# TYPING AND EXT
###

SelfTaskProblem = TypeVar("SelfTaskProblem", bound="TaskProblem")
SelfRenderedTaskProblem = TypeVar(
    "SelfRenderedTaskProblem", bound="RenderedTaskProblem"
)


###
# Utils
###
def _array_to_grid(v: Any) -> Grid:
    if isinstance(v, np.ndarray):
        if not issubclass(v.dtype.type, np.integer):
            raise TypeError("Only integer arrays are supported.")
        return v.tolist()
    return v


###
# Base clases
###
class TaskProblem(BaseModel):
    init_grid: Grid
    tgt_grid: Grid
    intermediate_grids: Optional[list[Grid]] = None
    task_specific_metadata: Optional[dict[str, Any]] = None

    # Validators
    @field_validator("init_grid", "tgt_grid", mode="before")
    def convert_grid(v: Any) -> Grid:
        return _array_to_grid(v)

    @field_validator("intermediate_grids", mode="before")
    def convert_intermediate_grids(v: Any) -> Optional[list[Grid]]:
        if v is None:
            return None
        return [_array_to_grid(g) for g in v]

    def save(self, path: str | Path) -> None:
        """
        Save this TaskProblem instance to a JSON file.
        """
        path = Path(path)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls: Type[SelfTaskProblem], path: str | Path) -> SelfTaskProblem:
        """
        Load a TaskProblem (or subclass) instance from a JSON file.
        """
        path = Path(path)
        json_str = path.read_text(encoding="utf-8")
        return cls.model_validate_json(json_str)


class Task(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs) -> TaskProblem:
        """Generate a instantiation of the task."""


class TaskDatasetGenerator:
    def __init__(self, task: Task, dist_fn: Callable[[Task, Task], float]):
        self.task = task
        self.dist_fn = dist_fn
        super().__init__()

    def generate(
        self,
        n_train: int,
        n_test: int,
        distance_threshold: float = 0.7,
        attempts_multiplier: int = 20,
    ) -> tuple[list[TaskProblem], list[TaskProblem]]:
        train_dataset = self.generate_train_dataset(n_train)
        test_dataset = self.generate_test_dataset(
            train_dataset,
            n_test=n_test,
            distance_threshold=distance_threshold,
            attempts_multiplier=attempts_multiplier,
        )
        return train_dataset, test_dataset

    def generate_test_dataset(
        self,
        train_dataset: list[TaskProblem],
        n_test: int,
        distance_threshold: float = 0.7,
        attempts_multiplier: int = 50,
    ) -> list[TaskProblem]:
        """Generate test problems sufficiently different from training problems."""
        test_dataset = []
        attempts = 0
        max_attempts = n_test * attempts_multiplier

        with tqdm(total=n_test, desc="Generating test dataset") as pbar:
            while len(test_dataset) < n_test and attempts < max_attempts:
                attempts += 1

                test_task_problem = self.task.generate()

                # Check distance from all training problems
                min_distance = float("inf")
                for train_task_problem in train_dataset:
                    distance = self.dist_fn(test_task_problem, train_task_problem)
                    min_distance = min(min_distance, distance)

                # If sufficiently different, add to test set
                if min_distance >= distance_threshold:
                    test_dataset.append(test_task_problem)
                    pbar.update(1)

            if len(test_dataset) < n_test:
                warnings.warn(
                    f"Only generated {len(test_dataset)}/{n_test} test problems after {max_attempts} attempts."
                )

        return test_dataset

    def generate_train_dataset(
        self,
        n_train: int,
    ) -> list[TaskProblem]:
        train_dataset = [
            self.task.generate()
            for _ in tqdm(range(n_train), desc="Generating training dataset")
        ]
        return train_dataset


class RenderedTaskProblem(BaseModel):
    task_problem: TaskProblem
    ###
    render_style: RenderStyle
    init_grid_render_metadata: RenderMetadata
    tgt_grid_render_metadata: RenderMetadata
    intermediate_grids_render_metadata: Optional[list[RenderMetadata]] = None

    def save(self, path: str | Path) -> None:
        """
        Save this RenderedTaskProblem instance to a JSON file.
        """
        path = Path(path)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls: Type[SelfRenderedTaskProblem], path: str | Path
    ) -> SelfRenderedTaskProblem:
        """
        Load a RenderedTaskProblem (or subclass) instance from a JSON file.
        """
        path = Path(path)
        json_str = path.read_text(encoding="utf-8")
        return cls.model_validate_json(json_str)

    def parse(
        self,
        init_grid_image: Optional[Grid] = None,
        tgt_grid_image: Optional[Image.Image] = None,
        grids_video: Optional[Video] = None,
        **parse_kwargs: Any,
    ) -> Grid | list[Grid]:
        raise NotImplementedError()
        if (
            sum(x is not None for x in [init_grid_image, tgt_grid_image, grids_video])
            != 1
        ):
            raise ValueError(
                "Exactly one of init_grid_image, tgt_grid_image, or grids_video must be provided"
            )

        if init_grid_image is not None:
            pass

        if tgt_grid_image is not None:
            pass

        # Else: Parse videoAny]
