import random
from collections import deque
from typing import List, Tuple, TypedDict

import numpy as np

from visual_logic.tasks.base import Task, TaskProblem


class Navigation2DSpecificMetadata(TypedDict):
    start: Tuple[int, int]
    end: Tuple[int, int]
    path: List[Tuple[int, int]]


class Navigation2D(Task):
    # Ids grid
    WALL_BLOCK = 0
    WALKABLE_BLOCK = 1
    END_BLOCK = 2
    START_BLOCK = 3
    SOLUTION_BLOCK = 4

    def __init__(
        self,
        width: int,
        height: int,
        obstacle_density: float = 0.2,
        seed: int = None,
        valid_starts=None,
        valid_ends=None,
        add_border: bool = False,
        max_attempts: int = 100,
        generate_intermediate_states: bool = False,
        n_barriers: int = 2,
        barrier_orientation: str = "random",  # 'random', 'vertical', 'horizontal'
        n_blocks: int = 2,
        block_size_range: tuple = (2, 4),
        barrier_holes_range: tuple = (1, 4),
        min_manhattan_distance: int = 0,
    ):
        if width < 3 or height < 3:
            raise ValueError("Width and height must be at least 3")
        if not (0 <= obstacle_density < 1):
            raise ValueError("obstacle_density must be in [0, 1)")
        self.width = width
        self.height = height
        self.obstacle_density = obstacle_density
        self.add_border = add_border
        self.valid_starts = valid_starts
        self.valid_ends = valid_ends
        self.max_attempts = max_attempts
        self.generate_intermediate_states = generate_intermediate_states
        self.n_barriers = n_barriers
        self.barrier_orientation = barrier_orientation
        self.n_blocks = n_blocks
        self.block_size_range = block_size_range
        self.barrier_holes_range = barrier_holes_range
        self.min_manhattan_distance = min_manhattan_distance
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # Dynamic attributes
        self.grid: np.ndarray = None
        self.start: tuple[int, int] = None
        self.end: tuple[int, int] = None
        self.solution_path: list[tuple[int, int]] = None
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

    def _validate_position(self, pos: tuple, name: str):
        x, y = pos
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"{name} position {pos} is outside grid bounds")

    def _generate_obstacle_grid(self) -> np.ndarray:
        grid = np.full((self.height, self.width), self.WALKABLE_BLOCK, dtype=int)
        # Add border if needed
        if self.add_border:
            grid[0, :] = self.WALL_BLOCK
            grid[-1, :] = self.WALL_BLOCK
            grid[:, 0] = self.WALL_BLOCK
            grid[:, -1] = self.WALL_BLOCK
            border_mask = np.zeros_like(grid, dtype=bool)
            border_mask[0, :] = True
            border_mask[-1, :] = True
            border_mask[:, 0] = True
            border_mask[:, -1] = True
        else:
            border_mask = np.zeros_like(grid, dtype=bool)
        # Place obstacles randomly (not on border)
        n_cells = self.width * self.height
        n_obstacles = int(self.obstacle_density * n_cells)
        # Exclude border from obstacle placement
        possible = np.argwhere(~border_mask)
        if n_obstacles > len(possible):
            n_obstacles = len(possible)
        if n_obstacles > 0:
            chosen = possible[
                np.random.choice(len(possible), n_obstacles, replace=False)
            ]
            for y, x in chosen:
                grid[y, x] = self.WALL_BLOCK
        # Add structured barriers
        for _ in range(self.n_barriers):
            orientation = self.barrier_orientation
            if orientation == "random":
                orientation = random.choice(["vertical", "horizontal"])
            n_holes = random.randint(
                self.barrier_holes_range[0], self.barrier_holes_range[1]
            )
            if orientation == "vertical":
                col = random.randint(1, self.width - 2)
                possible_rows = list(range(1, self.height - 1))
                holes = random.sample(possible_rows, min(n_holes, len(possible_rows)))
                for row in possible_rows:
                    if row in holes:
                        continue
                    if not border_mask[row, col]:
                        grid[row, col] = self.WALL_BLOCK
            elif orientation == "horizontal":
                row = random.randint(1, self.height - 2)
                possible_cols = list(range(1, self.width - 1))
                holes = random.sample(possible_cols, min(n_holes, len(possible_cols)))
                for col in possible_cols:
                    if col in holes:
                        continue
                    if not border_mask[row, col]:
                        grid[row, col] = self.WALL_BLOCK
        # Add block obstacles
        if isinstance(self.n_blocks, int):
            n_blocks = self.n_blocks
        elif isinstance(self.n_blocks, (tuple, list)):
            if len(self.n_blocks) == 2:  # treat as range
                n_blocks = random.randint(self.n_blocks[0], self.n_blocks[1])
            else:  # treat as a set of possible values
                n_blocks = random.choice(self.n_blocks)
        else:
            raise ValueError("n_blocks must be int, tuple, or list")
        for _ in range(n_blocks):
            block_h = random.randint(self.block_size_range[0], self.block_size_range[1])
            block_w = random.randint(self.block_size_range[0], self.block_size_range[1])
            max_y = (
                self.height - block_h - 1 if self.add_border else self.height - block_h
            )
            max_x = (
                self.width - block_w - 1 if self.add_border else self.width - block_w
            )
            if max_y <= 0 or max_x <= 0:
                continue
            y0 = random.randint(1 if self.add_border else 0, max_y)
            x0 = random.randint(1 if self.add_border else 0, max_x)
            for dy in range(block_h):
                for dx in range(block_w):
                    yy, xx = y0 + dy, x0 + dx
                    if not border_mask[yy, xx]:
                        grid[yy, xx] = self.WALL_BLOCK
        return grid

    def _find_path_cells(self, grid: np.ndarray) -> List[Tuple[int, int]]:
        return [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if grid[y, x] == self.WALKABLE_BLOCK
        ]

    def _set_start_end_points(self, grid: np.ndarray, start_pos=None, end_pos=None):
        path_cells = self._find_path_cells(grid)
        if len(path_cells) < 2:
            raise ValueError("Not enough walkable cells for start and end points")
        # Handle start
        if start_pos is None:
            self.start = random.choice(path_cells)
        elif isinstance(start_pos, tuple):
            if start_pos not in path_cells:
                raise ValueError(
                    f"Start position {start_pos} is not a valid walkable cell"
                )
            self.start = start_pos
        elif isinstance(start_pos, list):
            valid_starts = [pos for pos in start_pos if pos in path_cells]
            if not valid_starts:
                raise ValueError("No valid start positions found in walkable cells")
            self.start = random.choice(valid_starts)
        else:
            raise AssertionError(
                "This should never be triggered, positions were checked in initialization."
            )
        # Handle end
        if end_pos is None:
            end_candidates = [cell for cell in path_cells if cell != self.start]
            if not end_candidates:
                raise ValueError("No valid end positions available")
            self.end = random.choice(end_candidates)
        elif isinstance(end_pos, tuple):
            if end_pos not in path_cells:
                raise ValueError(f"End position {end_pos} is not a valid walkable cell")
            if end_pos == self.start:
                raise ValueError("End position cannot be the same as start position")
            self.end = end_pos
        elif isinstance(end_pos, list):
            valid_ends = [
                pos for pos in end_pos if pos in path_cells and pos != self.start
            ]
            if not valid_ends:
                raise ValueError(
                    "No valid end positions found in walkable cells (different from start)"
                )
            self.end = random.choice(valid_ends)
        else:
            raise AssertionError(
                "This should never be triggered, positions were checked in initialization."
            )
        # Mark start and end
        grid[self.start[1], self.start[0]] = self.START_BLOCK
        grid[self.end[1], self.end[0]] = self.END_BLOCK

    def _find_shortest_path(self, grid: np.ndarray) -> list[tuple[int, int]]:
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
                    0 <= nx < grid.shape[1]
                    and 0 <= ny < grid.shape[0]
                    and (nx, ny) not in visited
                    and grid[ny, nx] != self.WALL_BLOCK
                ):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        return []  # No path found

    def generate(self) -> TaskProblem:
        max_attempts = self.max_attempts
        for attempt in range(max_attempts):
            self.grid = self._generate_obstacle_grid()
            try:
                self._set_start_end_points(
                    self.grid, self.valid_starts, self.valid_ends
                )
                if (
                    abs(self.start[0] - self.end[0]) + abs(self.start[1] - self.end[1])
                    < self.min_manhattan_distance
                ):
                    raise ValueError(
                        f"Start and end points are too close (Manhattan distance < {self.min_manhattan_distance})"
                    )  # This will be caught by the max_attempts loop
                self.solution_path = self._find_shortest_path(self.grid)
                if self.solution_path:
                    break
            except ValueError:
                continue
            if attempt == max_attempts - 1:
                raise ValueError(
                    f"Could not generate valid navigation problem after {max_attempts} attempts"
                )

        output_grid = self.grid.copy()
        if self.solution_path:
            for x, y in self.solution_path:
                if output_grid[y, x] not in [self.START_BLOCK, self.END_BLOCK]:
                    output_grid[y, x] = self.SOLUTION_BLOCK
        intermediate_grids = None
        if self.generate_intermediate_states and self.solution_path:
            path_steps = [
                pos for pos in self.solution_path if pos not in [self.start, self.end]
            ]
            intermediate_grids = []
            current_grid = self.grid.copy()
            for x, y in path_steps:
                grid_step = current_grid.copy()
                grid_step[y, x] = self.SOLUTION_BLOCK
                intermediate_grids.append(grid_step.tolist())
                current_grid = grid_step
        return TaskProblem(
            init_grid=self.grid.copy().tolist(),
            tgt_grid=output_grid.tolist(),
            intermediate_grids=intermediate_grids,
            task_specific_metadata=Navigation2DSpecificMetadata(
                start=self.start,
                end=self.end,
                path=self.solution_path,
            ),
        )
