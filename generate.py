if __name__ == "__main__":
    import shutil
    from typing import Optional

    import numpy as np

    from visual_logic.tasks.base import TaskDatasetGenerator, TaskProblem
    from visual_logic.tasks.game_of_life import GameOfLife
    from visual_logic.tasks.langton_ant import LangtonAnt
    from visual_logic.tasks.maze import Maze
    from visual_logic.tasks.navigation2d import Navigation2D
    from visual_logic.tasks.problem_set import TaskProblemSet
    from visual_logic.tasks.render.schemas import ArcBaseStyle, MazeBaseStyle

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
        TaskProblemSet(task_problems=test_dataset).save(
            "datasets/maze/test", MazeBaseStyle
        )

    def generate_gol_dataset():
        def gol_hamming_distance(tp0: TaskProblem, tp1: TaskProblem) -> float:
            g0 = np.array(tp0.tgt_grid)
            g1 = np.array(tp1.tgt_grid)
            if g0.shape != g1.shape:
                raise ValueError("Grid shapes do not match")
            return np.sum(g0 != g1) / g0.size

        gol_train, gol_test = TaskDatasetGenerator(
            task=GameOfLife(
                width=16,
                height=16,
                steps=8,
                initialization="random",
                density=0.4,
                seed=42,
            ),
            dist_fn=gol_hamming_distance,
        ).generate(n_train=100, n_test=10, distance_threshold=0.3)

        shutil.rmtree("datasets/gol", ignore_errors=True)
        TaskProblemSet(task_problems=gol_train).save("datasets/gol/train", ArcBaseStyle)
        TaskProblemSet(task_problems=gol_test).save("datasets/gol/test", ArcBaseStyle)

    def generate_langton_ant_dataset():
        def ant_hamming_distance(tp0: TaskProblem, tp1: TaskProblem) -> float:
            g0 = np.array(tp0.tgt_grid)
            g1 = np.array(tp1.tgt_grid)
            if g0.shape != g1.shape:
                raise ValueError("Grid shapes do not match")
            return np.sum(g0 != g1) / g0.size

        ant_train, ant_test = TaskDatasetGenerator(
            task=LangtonAnt(
                width=8,
                height=8,
                steps=16,
                initialization="random",
                seed=123,
            ),
            dist_fn=ant_hamming_distance,
        ).generate(n_train=100, n_test=10, distance_threshold=0.3)

        shutil.rmtree("datasets/langton_ant", ignore_errors=True)
        TaskProblemSet(task_problems=ant_train).save(
            "datasets/langton_ant/train", ArcBaseStyle
        )
        TaskProblemSet(task_problems=ant_test).save(
            "datasets/langton_ant/test", ArcBaseStyle
        )

    def generate_navigation2d_dataset(
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

    ###
    # Datasets generation for different tasks
    ###
    # generate_maze_dataset(
    #     generate_intermediate_states=False,
    #     subset_sizes=[1, 3, 5, 10, 20, 40, 60, 80, 100, 200, 500, 1000],
    #     n_train=1000,
    #     n_test=100,
    # )
    # generate_gol_dataset()
    # generate_langton_ant_dataset()
    generate_navigation2d_dataset(
        generate_intermediate_states=False,
        subset_sizes=[1, 3, 5, 10, 20, 40, 60, 80, 100, 200, 500, 1000],
        n_train=1000,
        n_test=100,
    )
