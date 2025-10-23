import json
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional

from diffusers.utils.export_utils import export_to_video

from visual_intelligence.tasks.base import RenderedTaskProblem, TaskProblem
from visual_intelligence.tasks.render.render import get_auto_image_dim, render
from visual_intelligence.tasks.render.schemas import RenderStyle
from visual_intelligence.typing_and_extensions import (
    IMAGE_EXTENSION,
    PROBLEM_EXTENSION,
    VIDEO_EXTENSION,
)


class TaskProblemSet:
    problems_dir_name: Path = Path("problem")
    init_grid_dir_name: Path = Path("init_image")
    tgt_grid_dir_name: Path = Path("tgt_image")
    intermediate_grids_dir_name: Path = Path("video")

    def __init__(
        self,
        task_problems: list[TaskProblem],
    ):
        tasks_with_intermediate_grids = sum(
            [
                task_problem.intermediate_grids is not None
                for task_problem in task_problems
            ]
        )
        if tasks_with_intermediate_grids not in {0, len(task_problems)}:
            raise ValueError(
                f"Either all tasks include intermediate grids or none of them does for {type(self).__name__}"
            )
        self.task_problems = task_problems

    def save(
        self,
        path_dir: Path | str,
        render_style: RenderStyle,
        image_height: Literal["auto", "auto-per-problem"] | int = "auto",
        image_width: Literal["auto", "auto-per-problem"] | int = "auto",
        subset_sizes: Optional[list[int]] = None,
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

        intermediate_grids_dir = path_dir / self.intermediate_grids_dir_name

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
        data_config = defaultdict(list)
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
                image_width=task_problem_image_width,  # type: ignore
            )
            tgt_grid_image, tgt_grid_render_metadata = render(
                task_problem.tgt_grid,
                render_style,
                image_height=task_problem_image_height,
                image_width=task_problem_image_width,  # type: ignore
            )

            init_grid_image.save(init_grid_dir / (problem_name + IMAGE_EXTENSION))
            tgt_grid_image.save(tgt_grid_dir / (problem_name + IMAGE_EXTENSION))

            data_config["rel_image_0_paths"].append(
                str(self.init_grid_dir_name / (problem_name + IMAGE_EXTENSION))
            )
            data_config["rel_image_1_paths"].append(
                str(self.tgt_grid_dir_name / (problem_name + IMAGE_EXTENSION))
            )
            if task_problem.intermediate_grids is not None:
                intermediate_grids_dir.mkdir(exist_ok=True)
                interpolation_video = [init_grid_image]
                intermediate_grids_render_metadata = []
                for grid in task_problem.intermediate_grids:
                    intermediate_grid_image, intermediate_grid_render_metadata = render(
                        grid,
                        render_style,
                        image_height=task_problem_image_height,
                        image_width=task_problem_image_width,  # type: ignore
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

                data_config["rel_video_paths"].append(
                    str(
                        self.intermediate_grids_dir_name
                        / (problem_name + VIDEO_EXTENSION)
                    )
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

            data_config["rel_metadata_paths"].append(
                str(self.problems_dir_name / (problem_name + PROBLEM_EXTENSION))
            )
        # Save dataset configurations files based on subset_size
        if subset_sizes is not None:
            for subset_size in subset_sizes:
                filename = path_dir / f"data_group_n{subset_size}.json"
                with open(filename, "w") as f:
                    json.dump({k: v[:subset_size] for k, v in data_config.items()}, f)
        else:
            filename = path_dir / "data_group.json"
            with open(filename, "w") as f:
                json.dump(data_config, f)
