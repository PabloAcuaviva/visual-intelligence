from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional, Type, TypeVar

import numpy as np
from PIL import Image
from pydantic import BaseModel, field_validator
from tqdm import tqdm

from visual_intelligence.tasks.render.render import parse
from visual_intelligence.tasks.render.schemas import RenderMetadata, RenderStyle
from visual_intelligence.typing_and_extensions import Grid, Video

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
    def __init__(
        self,
        task: Task,
        dist_fn: Callable[[TaskProblem, TaskProblem], float],
        extend_dataset: Optional[Path] = None,
    ):
        self.task = task
        self.dist_fn = dist_fn
        self.extend_dataset = (
            Path(extend_dataset) if extend_dataset is not None else None
        )
        super().__init__()

    def generate(
        self,
        n_train: int,
        n_test: int,
        distance_threshold: float = 0.7,
        attempts_multiplier: int = 20,
    ) -> tuple[list[TaskProblem], list[TaskProblem]]:
        if self.extend_dataset is None:
            ###
            # Generate new dataset
            ###
            train_dataset = self._generate_basic_dataset(n_train)
            test_dataset = self._generate_dataset_far_from(
                train_dataset,
                n=n_test,
                distance_threshold=distance_threshold,
                attempts_multiplier=attempts_multiplier,
            )
        else:
            ###
            # Load train dataset
            ###
            problem_metadata_paths_train = sorted(
                list((self.extend_dataset / "train" / "problem").iterdir())
            )
            if len(problem_metadata_paths_train) > n_train:
                raise ValueError(
                    f"Only expansion on number of training samples is valid, not reducing them, for this simply create a config. There are {len(problem_metadata_paths_train)} training problems in the dataset vs {n_train=}"
                )
            train_dataset = []
            for problem_metadata_path in problem_metadata_paths_train:
                with open(problem_metadata_path, "r") as f:
                    task_problem = TaskProblem(**json.load(f)["task_problem"])
                train_dataset.append(task_problem)

            ###
            # Load test of dataset to extend
            ###
            problem_metadata_paths_test = sorted(
                list((self.extend_dataset / "test" / "problem").iterdir())
            )
            if len(problem_metadata_paths_test) != n_test:
                raise NotImplementedError(
                    f"Only expansion on number of training samples is implemented. Got {len(problem_metadata_paths_test)} != {n_test}."
                )
            test_dataset = []
            for problem_metadata_path in problem_metadata_paths_test:
                with open(problem_metadata_path, "r") as f:
                    task_problem = TaskProblem(**json.load(f)["task_problem"])
                test_dataset.append(task_problem)

            ###
            # Expand training samples
            ###
            train_dataset += self._generate_dataset_far_from(
                test_dataset,
                n=n_train - len(train_dataset),
                distance_threshold=distance_threshold,
                attempts_multiplier=attempts_multiplier,
            )

        return train_dataset, test_dataset

    def _generate_dataset_far_from(
        self,
        dataset: list[TaskProblem],
        n: int,
        distance_threshold: float = 0.7,
        attempts_multiplier: int = 50,
    ) -> list[TaskProblem]:
        """Generate problems sufficiently different from dataaset problems."""
        new_dataset = []
        attempts = 0
        max_attempts = n * attempts_multiplier

        with tqdm(
            total=n, desc=f"Generating new dataset with {distance_threshold=}"
        ) as pbar:
            while len(new_dataset) < n and attempts < max_attempts:
                attempts += 1

                test_task_problem = self.task.generate()

                # Check distance from all training problems
                min_distance = float("inf")
                for train_task_problem in dataset:
                    distance = self.dist_fn(test_task_problem, train_task_problem)
                    min_distance = min(min_distance, distance)

                # If sufficiently different, add to test set
                if min_distance >= distance_threshold:
                    new_dataset.append(test_task_problem)
                    pbar.update(1)

            if len(new_dataset) < n:
                warnings.warn(
                    f"Only generated {len(new_dataset)}/{n} test problems after {max_attempts} attempts."
                )

        return new_dataset

    def _generate_basic_dataset(
        self,
        n: int,
    ) -> list[TaskProblem]:
        train_dataset = [
            self.task.generate() for _ in tqdm(range(n), desc="Generating base dataset")
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
        if (
            sum(x is not None for x in [init_grid_image, tgt_grid_image, grids_video])
            != 1
        ):
            raise ValueError(
                "Exactly one of init_grid_image, tgt_grid_image, or grids_video must be provided"
            )

        if init_grid_image is not None:
            return parse(
                init_grid_image,
                render_style=self.render_style,
                render_metadata=self.init_grid_render_metadata,
                **parse_kwargs,
            )

        if tgt_grid_image is not None:
            return parse(
                tgt_grid_image,
                render_style=self.render_style,
                render_metadata=self.tgt_grid_render_metadata,
                **parse_kwargs,
            )

        ###
        # Parse full video - [init_grid_image, *intermediate_grid_image, tgt_grid_image]
        ###
        parsed_video = [
            parse(
                grids_video[0],
                render_style=self.render_style,
                render_metadata=self.init_grid_render_metadata,
                **parse_kwargs,
            )
        ]

        for intermediate_grid_image in grids_video[1:-1]:
            parsed_video.append(
                parse(
                    intermediate_grid_image,
                    render_style=self.render_style,
                    render_metadata=self.intermediate_grids_render_metadata,
                    **parse_kwargs,
                )
            )
        parsed_video.append(
            parse(
                grids_video[-1],
                render_style=self.render_style,
                render_metadata=self.tgt_grid_render_metadata,
                **parse_kwargs,
            )
        )
        return parsed_video
