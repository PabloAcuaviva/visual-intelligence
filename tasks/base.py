from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Type, TypeVar

import numpy as np
from diffusers.utils.export_utils import export_to_video
from PIL import Image
from pydantic import BaseModel, field_validator
from render.render import RenderMetadata, get_render_metadata, render
from render.styles import RenderStyle
from tqdm import tqdm

Grid = list[list[int]]
Video = list[Image.Image]


PROBLEM_EXTENSION = ".json"
VIDEO_EXTENSION = ".mp4"
IMAGE_EXTENSION = ".png"


SelfTaskProblem = TypeVar("SelfTaskProblem", bound="TaskProblem")
SelfRenderedTaskProblem = TypeVar(
    "SelfRenderedTaskProblem", bound="RenderedTaskProblem"
)


def array_to_grid(v: Any) -> Grid:
    if isinstance(v, np.ndarray):
        if not issubclass(v.dtype.type, np.integer):
            raise TypeError("Only integer arrays are supported.")
        return v.tolist()
    return v


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


class TaskProblem(BaseModel):
    init_grid: Grid
    tgt_grid: Grid
    intermediate_grids: Optional[list[Grid]] = None
    task_specific_metadata: Optional[dict[str, Any]] = None

    # Validators
    @field_validator("init_grid", "tgt_grid", mode="before")
    def convert_grid(v: Any) -> Grid:
        return array_to_grid(v)

    @field_validator("intermediate_grids", mode="before")
    def convert_intermediate_grids(v: Any) -> Optional[list[Grid]]:
        if v is None:
            return None
        return [array_to_grid(g) for g in v]

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


def get_auto_image_dim(
    task_problem: TaskProblem, render_style: RenderStyle
) -> tuple[int, int]:
    grids_to_check = [task_problem.init_grid, task_problem.tgt_grid]
    if task_problem.intermediate_grids is not None:
        grids_to_check += task_problem.intermediate_grids

    height, width = 0, 0
    for grid in grids_to_check:
        render_metadata: RenderMetadata = get_render_metadata(grid, render_style)
        height = max(render_metadata.image_height, height)
        width = max(render_metadata.image_width, width)
    return height, width


class TaskProblemSet:
    problems_dir_name: str = "problem"
    init_grid_dir_name: str = "init_image"
    tgt_grid_dir_name: str = "tgt_image"
    transition_grids_dir_name: str = "video"

    def __init__(
        self,
        task_problems: list[TaskProblem],
    ):
        self.task_problems = task_problems

    def save(
        self,
        path_dir: Path,
        render_style: RenderStyle,
        image_height: Literal["auto", "auto-per-problem"] | int = "auto",
        image_width: Literal["auto", "auto-per-problem"] | int = "auto",
    ) -> None:
        ###
        # Generetate dir structure
        ###
        path_dir = Path(path_dir)
        path_dir.mkdir(parents=True, exist_ok=False)

        problems_dir = path_dir / self.problems_dir_name
        problems_dir.mkdir()

        init_grid_dir = path_dir / self.init_grid_dir_name
        init_grid_dir.mkdir()

        tgt_grid_dir = path_dir / self.tgt_grid_dir_name
        tgt_grid_dir.mkdir()

        intermediate_grids_dir = path_dir / self.transition_grids_dir_name

        if image_height == "auto" or image_width == "auto":
            max_image_height, max_image_width = 0, 0
            for task_problem in self.task_problems:
                _image_height, _image_width = get_auto_image_dim(
                    task_problem, render_style
                )
                max_image_height = max(max_image_height, _image_height)
                max_image_width = max(max_image_width, _image_width)
            if image_height == "auto":
                image_height = max_image_height
            if image_width == "auto":
                image_width = max_image_width
        ###
        # Save files into dir structure
        ###
        leading_zeros_width = len(str(len(self.task_problems) - 1))
        for i_problem, task_problem in enumerate(self.task_problems):
            problem_name = f"{i_problem:0{leading_zeros_width}d}"

            _image_height, _image_width = get_auto_image_dim(task_problem, render_style)
            if image_height == "auto-per-problem":
                task_problem_image_height = _image_height
            else:
                task_problem_image_height = image_height
            if _image_width == "auto-per-problem":
                task_problem_image_width = _image_width
            else:
                task_problem_image_width = image_width

            init_grid_image, init_grid_render_metadata = render(
                task_problem.init_grid,
                render_style,
                image_height=task_problem_image_height,
                image_width=task_problem_image_width,
            )
            tgt_grid_image, tgt_grid_render_metadata = render(
                task_problem.tgt_grid,
                render_style,
                image_height=task_problem_image_height,
                image_width=task_problem_image_width,
            )

            init_grid_image.save(init_grid_dir / (problem_name + IMAGE_EXTENSION))
            tgt_grid_image.save(tgt_grid_dir / (problem_name + IMAGE_EXTENSION))

            if task_problem.intermediate_grids is not None:
                intermediate_grids_dir.mkdir(exist_ok=True)
                interpolation_video = [init_grid_image]
                intermediate_grids_render_metadata = []
                for grid in task_problem.intermediate_grids:
                    intermediate_grid_image, intermediate_grid_render_metadata = render(
                        grid,
                        render_style,
                        image_height=task_problem_image_height,
                        image_width=task_problem_image_width,
                    )
                    interpolation_video += [intermediate_grid_image]
                    intermediate_grids_render_metadata += [
                        intermediate_grid_render_metadata
                    ]
                interpolation_video += [tgt_grid_image]

                export_to_video(
                    interpolation_video,
                    output_video_path=intermediate_grids_dir
                    / (problem_name + VIDEO_EXTENSION),
                    fps=3,
                )
            else:
                intermediate_grids_render_metadata = None
            RenderedTaskProblem(
                task_problem=task_problem,
                render_style=render_style,
                init_grid_render_metadata=init_grid_render_metadata,
                tgt_grid_render_metadata=tgt_grid_render_metadata,
                intermediate_grids_render_metadata=intermediate_grids_render_metadata,
            ).save(problems_dir / (problem_name + PROBLEM_EXTENSION))

    def load(self, path: Path) -> None:
        raise NotImplementedError()
