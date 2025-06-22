if __name__ == "__main__":
    from tasks.base import TaskDatasetGenerator, TaskProblem
    from tasks.maze import Maze
    from tasks.problem_set import TaskProblemSet

    def path_distance(
        task_problem_0: TaskProblem, task_problem_1: TaskProblem
    ) -> float:
        """Calculate normalized intersection distance between two solution paths.

        This function computes the Jaccard distance (1 - Jaccard similarity) between
        the solution paths of two TaskProblem instances. The distance ranges from 0 to 1,
        where 0 indicates identical paths and 1 indicates no path overlap.

        Args:
            task_problem_0 (TaskProblem): First task problem containing a solution path
                in its metadata.path attribute.
            task_problem_1 (TaskProblem): Second task problem containing a solution path
                in its metadata.path attribute.

        Returns:
            float: Normalized distance between paths, calculated as:
                1 - (intersection_size / union_size)
                - 0.0: Paths are identical
                - 1.0: Paths have no common elements
                - Values between 0-1: Partial overlap

        Raises:
            KeyError: If either task_problem's metadata does not contain a 'path' attribute,

            ValueError: If both paths are empty.

        Note:
            The function uses set operations, so duplicate elements within a path
            are automatically removed before distance calculation.
        """

        path0 = task_problem_0.task_specific_metadata[
            "path"
        ]  # If it doesn't have it raises KeyError
        path1 = task_problem_1.task_specific_metadata["path"]

        path0 = set(path0)
        path1 = set(path1)

        if len(path0) == 0 and len(path1) == 0:
            raise ValueError("Both paths are empty")

        intersection = len(path0.intersection(path1))
        union = len(path0.union(path1))

        # Return 1 - jaccard similarity (so 0 = identical, 1 = no overlap)
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity

    train_dataset, test_dataset = TaskDatasetGenerator(
        task=Maze(
            width=21,
            height=21,
            seed=1,
            valid_starts=(1, 1),
            valid_ends=(19, 19),
            generate_intermediate_states=True,
        ),
        dist_fn=path_distance,
    ).generate(n_train=1000, n_test=20, distance_threshold=0.5)

    # train_dataset[0].save("example.json")
    # loaded = MazeProblem.load("example.json")
    # print(train_dataset[0] == loaded)
    import shutil

    from tasks.render.schemas import MazeBaseStyle

    shutil.rmtree("test_set", ignore_errors=True)
    TaskProblemSet(task_problems=test_dataset).save("test_set", MazeBaseStyle)
