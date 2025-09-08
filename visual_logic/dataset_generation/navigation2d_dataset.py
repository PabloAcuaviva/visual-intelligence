import shutil
from pathlib import Path
from typing import Optional

from visual_logic.tasks.base import TaskDatasetGenerator, TaskProblem
from visual_logic.tasks.navigation2d import Navigation2D
from visual_logic.tasks.problem_set import TaskProblemSet
from visual_logic.tasks.render.schemas import MazeBaseStyle

from .registry import register_dataset


@register_dataset("navigation2d")
def generate_navigation2d_dataset(
    generate_intermediate_states: bool = False,
    subset_sizes: Optional[list[int]] = None,
    n_train: int = 1000,
    n_test: int = 100,
    extend_dataset: Optional[Path] = None,
):
    def path_distance(
        task_problem_0: TaskProblem,
        task_problem_1: TaskProblem,
    ) -> float:
        path0 = set(task_problem_0.task_specific_metadata["path"])
        path1 = set(task_problem_1.task_specific_metadata["path"])
        if len(path0) == 0 and len(path1) == 0:
            raise ValueError("Both paths are empty")
        intersection = len(path0.intersection(path1))
        union = len(path0.union(path1))
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity

    train_dataset, test_dataset = TaskDatasetGenerator(
        task=Navigation2D(
            width=21,
            height=21,
            obstacle_density=0.08,
            seed=42,
            add_border=True,
            valid_starts=(1, 1),
            valid_ends=(19, 19),
            n_barriers=8,
            barrier_holes_range=(3, 5),
            n_blocks=(1, 4),
            block_size_range=(2, 3),
            generate_intermediate_states=generate_intermediate_states,
            max_attempts=4000,
        ),
        dist_fn=path_distance,
        extend_dataset=extend_dataset,
    ).generate(n_train=n_train, n_test=n_test, distance_threshold=0.5)

    shutil.rmtree("datasets/navigation2d", ignore_errors=True)
    TaskProblemSet(task_problems=train_dataset).save(
        "datasets/navigation2d/train",
        MazeBaseStyle,
        subset_sizes=subset_sizes,
    )
    TaskProblemSet(task_problems=test_dataset).save(
        "datasets/navigation2d/test", MazeBaseStyle
    )


navigation2d_any_to_any = "navigation2d_any_to_any"


@register_dataset(navigation2d_any_to_any)
def generate_navigation2d_any_to_any_dataset(
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
        task=Navigation2D(
            width=15,
            height=15,
            obstacle_density=0.05,
            seed=42,
            add_border=True,
            valid_starts=None,
            valid_ends=None,
            n_barriers=3,
            barrier_holes_range=(8, 14),
            n_blocks=(3, 8),
            block_size_range=(2, 3),
            generate_intermediate_states=generate_intermediate_states,
            max_attempts=4000,
            min_manhattan_distance=15,  # Ensure a minimum distance between start and end points
        ),
        dist_fn=path_distance,
        extend_dataset=extend_dataset,
    ).generate(n_train=n_train, n_test=n_test, distance_threshold=0.5)

    shutil.rmtree(f"datasets/{navigation2d_any_to_any}", ignore_errors=True)
    TaskProblemSet(task_problems=train_dataset).save(
        f"datasets/{navigation2d_any_to_any}/train",
        MazeBaseStyle,
        subset_sizes=subset_sizes,
    )
    TaskProblemSet(task_problems=test_dataset).save(
        f"datasets/{navigation2d_any_to_any}/test", MazeBaseStyle
    )


shortest_path = "shortest_path"


@register_dataset(shortest_path)
def generate_shortest_path_dataset(
    generate_intermediate_states: bool = False,
    subset_sizes: Optional[list[int]] = None,
    n_train: int = 1000,
    n_test: int = 100,
):
    def path_distance(
        task_problem_0: TaskProblem,
        task_problem_1: TaskProblem,
    ) -> float:
        path0 = set(task_problem_0.task_specific_metadata["path"])
        path1 = set(task_problem_1.task_specific_metadata["path"])
        if len(path0) == 0 and len(path1) == 0:
            raise ValueError("Both paths are empty")
        intersection = len(path0.intersection(path1))
        union = len(path0.union(path1))
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity

    train_dataset, test_dataset = TaskDatasetGenerator(
        task=Navigation2D(
            width=15,
            height=15,
            obstacle_density=0.05,
            seed=42,
            add_border=True,
            valid_starts=None,
            valid_ends=None,
            n_barriers=0,
            barrier_holes_range=(3, 5),
            n_blocks=(3, 8),
            block_size_range=(2, 5),
            generate_intermediate_states=generate_intermediate_states,
            max_attempts=4000,
            min_manhattan_distance=16,  # Ensure a minimum distance between start and end points
        ),
        dist_fn=path_distance,
    ).generate(n_train=n_train, n_test=n_test, distance_threshold=0.5)

    shutil.rmtree(f"datasets/{shortest_path}", ignore_errors=True)
    TaskProblemSet(task_problems=train_dataset).save(
        f"datasets/{shortest_path}/train",
        MazeBaseStyle,
        subset_sizes=subset_sizes,
    )
    TaskProblemSet(task_problems=test_dataset).save(
        f"datasets/{shortest_path}/test", MazeBaseStyle
    )
