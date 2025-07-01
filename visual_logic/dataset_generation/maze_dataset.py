import shutil
from typing import Optional

from visual_logic.tasks.base import TaskDatasetGenerator, TaskProblem
from visual_logic.tasks.maze import Maze
from visual_logic.tasks.problem_set import TaskProblemSet
from visual_logic.tasks.render.schemas import MazeBaseStyle

from .registry import register_dataset


@register_dataset("maze")
def generate_maze_dataset(
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
        task=Maze(
            width=21,
            height=21,
            seed=1,
            valid_starts=(1, 1),
            valid_ends=(19, 19),
            generate_intermediate_states=generate_intermediate_states,
        ),
        dist_fn=path_distance,
    ).generate(n_train=n_train, n_test=n_test, distance_threshold=0.5)

    shutil.rmtree("datasets/maze", ignore_errors=True)
    TaskProblemSet(task_problems=train_dataset).save(
        "datasets/maze/train",
        MazeBaseStyle,
        subset_sizes=subset_sizes,
    )
    TaskProblemSet(task_problems=test_dataset).save("datasets/maze/test", MazeBaseStyle)
