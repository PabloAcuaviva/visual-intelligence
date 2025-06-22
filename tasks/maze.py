import random
from collections import deque
from typing import Dict, List, Tuple, TypedDict

import numpy as np

from tasks.base import Task, TaskProblem


class MazeSpecificMetadata(TypedDict):
    start: Tuple[int, int]
    end: Tuple[int, int]
    path: List[Tuple[int, int]]


class Maze(Task):
    # Ids grid
    WALL_BLOCK = 0
    PATH_BLOCK = 1
    START_BLOCK = 3
    END_BLOCK = 2
    SOLUTION_BLOCK = 4

    def __init__(
        self,
        width: int,
        height: int,
        seed: int = None,
        valid_starts=None,
        valid_ends=None,
        max_attempts: int = 100,
        generate_intermediate_states: bool = False,
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

        if seed is not None:
            random.seed(seed)

        self.generate_intermediate_states = generate_intermediate_states

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
        maze[2 * start_y + 1, 2 * start_x + 1] = self.PATH_BLOCK
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
                maze[2 * y + 1, 2 * x + 1] = self.PATH_BLOCK
                in_maze.add((x, y))

                # Connect to next cell in path
                if i < len(path) - 1:
                    next_x, next_y = path[i + 1]
                    wall_x = x + next_x + 1
                    wall_y = y + next_y + 1
                    maze[wall_y, wall_x] = self.PATH_BLOCK

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
                if maze[y, x] == self.PATH_BLOCK:
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
        maze[self.start[1], self.start[0]] = self.START_BLOCK
        maze[self.end[1], self.end[0]] = self.END_BLOCK

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
                    and maze[ny, nx] != self.WALL_BLOCK
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
                if output_maze[y, x] not in [self.START_BLOCK, self.END_BLOCK]:
                    output_maze[y, x] = self.SOLUTION_BLOCK

        intermediate_grids = None
        if self.generate_intermediate_states and self.solution_path:
            path_steps = [
                pos for pos in self.solution_path if pos not in [self.start, self.end]
            ]
            intermediate_grids = []
            current_grid = self.maze.copy()
            for x, y in path_steps:
                # Copy the previous grid
                grid_step = current_grid.copy()

                # Add the next solution block
                grid_step[y, x] = self.SOLUTION_BLOCK

                # Save this step
                intermediate_grids.append(grid_step.tolist())

                # Update current_grid for the next iteration
                current_grid = grid_step

        return TaskProblem(
            init_grid=self.maze.copy().tolist(),
            tgt_grid=output_maze.tolist(),
            intermediate_grids=intermediate_grids,
            task_specific_metadata=MazeSpecificMetadata(
                start=self.start,
                end=self.end,
                path=self.solution_path,
            ),
        )
