import json
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypedDict
from zipfile import ZipFile

from tasks.base import TaskProblem
from tasks.problem_set import TaskProblemSet
from tasks.render.schemas import RenderStyle
from tasks.typing_and_extensions import Grid
from tqdm import tqdm


def accept_path_or_zip(func):
    """
    Decorator that allows a function to accept either a directory path or a .zip file.
    If a .zip file is provided, it is extracted to a temporary directory and the function
    is called with the path to that temporary directory.

    Parameters:
        func: A function that expects a pathlib.Path object as its first argument.

    Returns:
        A wrapped function that can accept either a directory Path or a .zip Path.
    """

    @wraps(func)
    def wrapper(input_path: Path, *args: Any, **kwargs: Any) -> Any:
        if input_path.suffix == ".zip" and input_path.is_file():
            with TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                with ZipFile(input_path, "r") as zip_ref:
                    zip_ref.extractall(tmp_path)
                contents = list(tmp_path.iterdir())
                if len(contents) == 1 and contents[0].is_dir():
                    data_path = contents[0]
                else:
                    data_path = tmp_path
                return func(data_path, *args, **kwargs)
        return func(input_path, *args, **kwargs)

    return wrapper  # type: ignore


class ArcJSONProblem(TypedDict):
    input: Grid
    output: Grid


class ArcFormatJSON(TypedDict):
    train: list[ArcJSONProblem]
    test: list[ArcJSONProblem]


def json_arc_to_problem_set(
    json_filepath: Path,
) -> tuple[TaskProblemSet, TaskProblemSet]:
    """Takes a JSON and converts it into appropiate tuple of TaskProblemSet datasets."""
    with open(json_filepath, "r") as f:
        data: ArcFormatJSON = json.load(f)

    train_data = data["train"]
    test_data = data["test"]

    train_problems = [
        TaskProblem(
            init_grid=arc_problem["input"],
            tgt_grid=arc_problem["output"],
            task_specific_metadata=dict(
                query_or_support="support",
                problem_num=i,
                problem_filepaht=json_filepath.name,
            ),
        )
        for i, arc_problem in enumerate(train_data)
    ]

    test_problems = [
        TaskProblem(
            init_grid=arc_problem["input"],
            tgt_grid=arc_problem["output"],
            task_specific_metadata=dict(
                query_or_support="query",
                problem_num=i,
                problem_filepaht=json_filepath.name,
            ),
        )
        for i, arc_problem in enumerate(test_data)
    ]

    return TaskProblemSet(train_problems), TaskProblemSet(test_problems)


@accept_path_or_zip
def process_arc_like_folder(
    input_path: Path,
    output_path: Path,
    nested: bool = True,
) -> None:
    """Given a path containing JSON in ARC-AGI format, process them into task problems."""
    # Notice we fix this to be able to have the same values for all
    render_style = RenderStyle(
        cell_size=11,
        grid_border_size=2,
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
        background_color=(0, 0, 0),  # Black background
        border_color=(85, 85, 85),  # Medium gray border
    )
    image_width = image_height = 392  # Adjusted for a maximum 30x30 grid

    files = list(input_path.iterdir())
    output_path.mkdir(exist_ok=True, parents=True)
    for path in tqdm(files, total=len(files)):
        if path.is_dir():
            if nested:
                process_arc_like_folder(
                    input_path=path,
                    output_path=output_path / path.name,
                    nested=True,
                )
            continue

        train_problems, test_problems = json_arc_to_problem_set(path)

        train_problems.save(
            path_dir=output_path / path.stem / "train",
            render_style=render_style,
            image_height=image_height,
            image_width=image_width,
        )
        test_problems.save(
            path_dir=output_path / path.stem / "test",
            render_style=render_style,
            image_height=image_height,
            image_width=image_width,
        )
