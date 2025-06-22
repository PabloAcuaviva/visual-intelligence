if __name__ == "__main__":
    import shutil

    import numpy as np

    from tasks.base import TaskDatasetGenerator, TaskProblem
    from tasks.game_of_life import GameOfLife
    from tasks.langton_ant import LangtonAnt
    from tasks.maze import Maze
    from tasks.problem_set import TaskProblemSet
    from tasks.render.schemas import ArcBaseStyle, MazeBaseStyle

    def generate_maze_dataset():
        def path_distance(
            task_problem_0: TaskProblem, task_problem_1: TaskProblem
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
                generate_intermediate_states=True,
            ),
            dist_fn=path_distance,
        ).generate(n_train=1000, n_test=20, distance_threshold=0.5)

        shutil.rmtree("datasets/test_set", ignore_errors=True)
        TaskProblemSet(task_problems=test_dataset).save(
            "datasets/test_set", MazeBaseStyle
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

        shutil.rmtree("datasets/test_set_gol", ignore_errors=True)
        TaskProblemSet(task_problems=gol_test).save(
            "datasets/test_set_gol", ArcBaseStyle
        )

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

        shutil.rmtree("datasets/test_set_ant", ignore_errors=True)
        TaskProblemSet(task_problems=ant_test).save(
            "datasets/test_set_ant", ArcBaseStyle
        )

    # Run all dataset generations
    # generate_maze_dataset()
    # generate_gol_dataset()
    generate_langton_ant_dataset()
