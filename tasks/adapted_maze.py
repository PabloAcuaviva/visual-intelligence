import random
from collections import deque
from typing import Dict, List, Tuple, TypedDict

import numpy as np
from base import Task, TaskProblem


class MazeSpecificMetadata(TypedDict):
    start: Tuple[int, int]
    end: Tuple[int, int]
    path: List[Tuple[int, int]]


class Maze(Task):
    # Ids grid
    WALL = 0
    PATH = 1
    START = 3
    END = 2
    SOLUTION = 4

    def __init__(
        self,
        width: int,
        height: int,
        seed: int = None,
        valid_starts=None,
        valid_ends=None,
        max_attempts: int = 100,
    ):
        # Validate dimensions
        if width < 3 or height < 3:
            raise ValueError("Width and height must be at least 3")

        if width % 2 == 0 or height % 2 == 0:
            raise ValueError(
                "Width and height must be odd numbers for Wilson's algorithm"
            )

        self.width = width
        self.height = height

        # Validate positions if provided
        if valid_starts is not None:
            if isinstance(valid_starts, tuple):
                self._validate_position(valid_starts, "Start")
            elif isinstance(valid_starts, list):
                for pos in valid_starts:
                    self._validate_position(pos, "Start")

        if valid_ends is not None:
            if isinstance(valid_ends, tuple):
                self._validate_position(valid_ends, "End")
            elif isinstance(valid_ends, list):
                for pos in valid_ends:
                    self._validate_position(pos, "End")

        self.valid_starts = valid_starts
        self.valid_ends = valid_ends
        self.max_attempts = max_attempts

        # Set seed if provided
        if seed is not None:
            random.seed(seed)

        # Dynamic arguments during maze generations
        self.maze: np.ndarray = None
        self.start: tuple[int, int] = None
        self.end: tuple[int, int] = None
        self.solution_path: list[tuple[int, int]] = None

    def _validate_position(self, pos: tuple, name: str):
        """Validate that a position has odd coordinates and is within bounds."""
        x, y = pos

        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"{name} position {pos} is outside maze bounds")

        if x % 2 == 0 or y % 2 == 0:
            raise ValueError(f"{name} position {pos} must have odd coordinates")

    def _wilson_algorithm(self) -> np.ndarray:
        """Generate a maze using Wilson's loop-erased random walk algorithm."""
        # Initialize grid - all walls initially
        maze = np.zeros((self.height, self.width), dtype=int)

        # Calculate logical maze dimensions (every other cell is a maze cell)
        logical_height = (self.height - 1) // 2
        logical_width = (self.width - 1) // 2

        # Set of cells that are part of the maze
        in_maze = set()

        # Add a random cell to start
        start_x, start_y = random.randrange(logical_width), random.randrange(
            logical_height
        )
        maze[2 * start_y + 1, 2 * start_x + 1] = self.PATH
        in_maze.add((start_x, start_y))

        # Get all cells not yet in maze
        remaining_cells = [
            (x, y)
            for x in range(logical_width)
            for y in range(logical_height)
            if (x, y) not in in_maze
        ]

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

        while remaining_cells:
            # Start random walk from a random remaining cell
            current = random.choice(remaining_cells)
            path = [current]

            # Random walk until we hit a cell in the maze
            while current not in in_maze:
                # Choose random direction
                dx, dy = random.choice(directions)
                next_x, next_y = current[0] + dx, current[1] + dy

                # Keep within bounds
                if 0 <= next_x < logical_width and 0 <= next_y < logical_height:
                    current = (next_x, next_y)

                    # If we've been here before in this walk, erase the loop
                    if current in path:
                        loop_start = path.index(current)
                        path = path[: loop_start + 1]
                    else:
                        path.append(current)

            # Add the path to the maze
            for i in range(len(path)):
                x, y = path[i]
                maze[2 * y + 1, 2 * x + 1] = self.PATH
                in_maze.add((x, y))

                # Connect to next cell in path
                if i < len(path) - 1:
                    next_x, next_y = path[i + 1]
                    wall_x = x + next_x + 1
                    wall_y = y + next_y + 1
                    maze[wall_y, wall_x] = self.PATH

            # Update remaining cells
            remaining_cells = [
                (x, y)
                for x in range(logical_width)
                for y in range(logical_height)
                if (x, y) not in in_maze
            ]

        return maze

    def _find_path_cells(self, maze: np.ndarray) -> List[Tuple[int, int]]:
        """Find all path cells (non-wall cells) in the maze."""
        path_cells = []
        for y in range(1, maze.shape[0], 2):
            for x in range(1, maze.shape[1], 2):
                if maze[y, x] == self.PATH:
                    path_cells.append((x, y))
        return path_cells

    def _set_start_end_points(self, maze: np.ndarray, start_pos=None, end_pos=None):
        """Set start and end points with flexible positioning options."""
        path_cells = self._find_path_cells(maze)

        if len(path_cells) < 2:
            raise ValueError("Not enough path cells for start and end points")

        # Handle start position
        if start_pos is None:
            self.start = random.choice(path_cells)
        elif isinstance(start_pos, tuple):
            if start_pos not in path_cells:
                raise ValueError(f"Start position {start_pos} is not a valid path cell")
            self.start = start_pos
        elif isinstance(start_pos, list):
            valid_starts = [pos for pos in start_pos if pos in path_cells]
            if not valid_starts:
                raise ValueError("No valid start positions found in path cells")
            self.start = random.choice(valid_starts)
        else:
            raise AssertionError(
                "This should never triggered, positions where checked in initialization."
            )

        # Handle end position
        if end_pos is None:
            end_candidates = [cell for cell in path_cells if cell != self.start]
            if not end_candidates:
                raise ValueError("No valid end positions available")
            self.end = random.choice(end_candidates)
        elif isinstance(end_pos, tuple):
            if end_pos not in path_cells:
                raise ValueError(f"End position {end_pos} is not a valid path cell")
            if end_pos == self.start:
                raise ValueError("End position cannot be the same as start position")
            self.end = end_pos
        elif isinstance(end_pos, list):
            valid_ends = [
                pos for pos in end_pos if pos in path_cells and pos != self.start
            ]
            if not valid_ends:
                raise ValueError(
                    "No valid end positions found in path cells (different from start)"
                )
            self.end = random.choice(valid_ends)
        else:
            raise AssertionError(
                "This should never triggered, positions where checked in initialization."
            )

        # Mark start and end in maze
        maze[self.start[1], self.start[0]] = self.START
        maze[self.end[1], self.end[0]] = self.END

    def _find_shortest_path(self, maze: np.ndarray) -> list[tuple[int, int]]:
        """Find shortest path from start to end using BFS."""
        if self.start is None or self.end is None:
            raise AssertionError(
                f"This should never happen, check you are only using the {type(self).__name__} generate method."
            )

        queue = deque([(self.start, [self.start])])
        visited = {self.start}
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while queue:
            (x, y), path = queue.popleft()

            if (x, y) == self.end:
                return path

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if (
                    0 <= nx < maze.shape[1]
                    and 0 <= ny < maze.shape[0]
                    and (nx, ny) not in visited
                    and maze[ny, nx] != self.WALL
                ):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))

        return []  # No path found

    def generate(self) -> Dict:
        """Generate a complete maze with solution."""
        max_attempts = self.max_attempts
        for attempt in range(max_attempts):
            # Generate maze structure
            self.maze = self._wilson_algorithm()
            try:
                # Set start and end points
                self._set_start_end_points(
                    self.maze, self.valid_starts, self.valid_ends
                )

                # Find solution path
                self.solution_path = self._find_shortest_path(self.maze)

                if self.solution_path:  # Path found
                    break

            except ValueError:
                # This attempt failed, try again
                continue

            if attempt == max_attempts - 1:
                raise ValueError(
                    f"Could not generate valid maze after {max_attempts} attempts"
                )

        output_maze = self.maze.copy()
        if self.solution_path:
            for x, y in self.solution_path:
                if output_maze[y, x] not in [self.START, self.END]:
                    output_maze[y, x] = self.SOLUTION

        return TaskProblem(
            init_grid=self.maze.copy().tolist(),
            tgt_grid=output_maze.tolist(),
            task_specific_metadata=MazeSpecificMetadata(
                start=self.start,
                end=self.end,
                path=self.solution_path,
            ),
        )


if __name__ == "__main__":
    from base import TaskDatasetGenerator, TaskProblemSet

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
        ),
        dist_fn=path_distance,
    ).generate(n_train=1000, n_test=20, distance_threshold=0.5)

    # train_dataset[0].save("example.json")
    # loaded = MazeProblem.load("example.json")
    # print(train_dataset[0] == loaded)
    from render.styles import MazeBaseStyle

    TaskProblemSet(task_problems=test_dataset).save("test_set", MazeBaseStyle)
