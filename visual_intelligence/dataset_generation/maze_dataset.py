import shutil
from pathlib import Path
from typing import Optional

from visual_intelligence.tasks.base import TaskDatasetGenerator, TaskProblem
from visual_intelligence.tasks.maze import Maze
from visual_intelligence.tasks.problem_set import TaskProblemSet
from visual_intelligence.tasks.render.schemas import MazeBaseStyle

from .registry import register_dataset


@register_dataset("maze")
def generate_maze_dataset(
    generate_intermediate_states: bool = False,
    subset_sizes: Optional[list[int]] = None,
    n_train: int = 1000,
    n_test: int = 200,
    extend_dataset: Optional[Path] = None,
):
    def path_distance(
        task_problem_0: TaskProblem,
        task_problem_1: TaskProblem,
    ) -> float:
        path0 = set([tuple(t) for t in task_problem_0.task_specific_metadata["path"]])
        path1 = set([tuple(t) for t in task_problem_1.task_specific_metadata["path"]])
        if len(path0) == 0 and len(path1) == 0:
            raise ValueError("Both paths are empty")
        intersection = len(path0.intersection(path1))
        union = len(path0.union(path1))
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity

    train_dataset, test_dataset = TaskDatasetGenerator(
        task=Maze(
            width=21,
            height=21,
            seed=1,
            valid_starts=(1, 1),
            valid_ends=(19, 19),
            generate_intermediate_states=generate_intermediate_states,
        ),
        dist_fn=path_distance,
        extend_dataset=extend_dataset,
    ).generate(n_train=n_train, n_test=n_test, distance_threshold=0.5)

    shutil.rmtree("datasets/maze", ignore_errors=True)
    TaskProblemSet(task_problems=train_dataset).save(
        "datasets/maze/train",
        MazeBaseStyle,
        subset_sizes=subset_sizes or [],
    )
    TaskProblemSet(task_problems=test_dataset).save("datasets/maze/test", MazeBaseStyle)


@register_dataset("maze_small")
def generate_small_maze_dataset(
    generate_intermediate_states: bool = False,
    subset_sizes: Optional[list[int]] = None,
    n_train: int = 1000,
    n_test: int = 200,
    extend_dataset: Optional[Path] = None,
):
    def path_distance(
        task_problem_0: TaskProblem,
        task_problem_1: TaskProblem,
    ) -> float:
        path0 = set([tuple(t) for t in task_problem_0.task_specific_metadata["path"]])
        path1 = set([tuple(t) for t in task_problem_1.task_specific_metadata["path"]])
        if len(path0) == 0 and len(path1) == 0:
            raise ValueError("Both paths are empty")
        intersection = len(path0.intersection(path1))
        union = len(path0.union(path1))
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity

    image_width = image_height = 336

    train_dataset, test_dataset = TaskDatasetGenerator(
        task=Maze(
            width=13,
            height=13,
            seed=1,
            valid_starts=(1, 1),
            valid_ends=(11, 11),
            generate_intermediate_states=generate_intermediate_states,
        ),
        dist_fn=path_distance,
        extend_dataset=extend_dataset,
    ).generate(n_train=n_train, n_test=n_test, distance_threshold=0.5)

    shutil.rmtree("datasets/maze_small", ignore_errors=True)

    TaskProblemSet(task_problems=train_dataset).save(
        "datasets/maze_small/train",
        MazeBaseStyle,
        subset_sizes=subset_sizes,
        image_width=image_width,
        image_height=image_height,
    )
    TaskProblemSet(task_problems=test_dataset).save(
        "datasets/maze_small/test",
        MazeBaseStyle,
        image_width=image_width,
        image_height=image_height,
    )
