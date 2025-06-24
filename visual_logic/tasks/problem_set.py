from pathlib import Path
from typing import Literal

from diffusers.utils.export_utils import export_to_video

from visual_logic.tasks.base import RenderedTaskProblem, TaskProblem
from visual_logic.tasks.render.render import get_auto_image_dim, render
from visual_logic.tasks.render.schemas import RenderStyle
from visual_logic.typing_and_extensions import (
    IMAGE_EXTENSION,
    PROBLEM_EXTENSION,
    VIDEO_EXTENSION,
)


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
