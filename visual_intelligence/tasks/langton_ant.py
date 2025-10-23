import random
from typing import List, Optional, Tuple, TypedDict

import numpy as np

from visual_intelligence.tasks.base import Task, TaskProblem

# Cell states
WHITE = 0
BLACK = 1
ANT = 2

# Directions
DIRECTIONS = ["N", "E", "S", "W"]
DIR_DELTA = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}


class LangtonAntSpecificMetadata(TypedDict, total=False):
    ant_path: List[Tuple[int, int]]
    ant_initial_pos: Tuple[int, int]
    ant_initial_dir: str
    steps: int
    initialization: str
    seed: Optional[int]


class LangtonAnt(Task):
    def __init__(
        self,
        width: int,
        height: int,
        steps: int = 20,
        initialization: str = "random",  # or "random"
        ant_initial_pos: Optional[Tuple[int, int]] = None,
        ant_initial_dir: Optional[str] = None,
        seed: Optional[int] = None,
        init_grid_as: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.steps = steps
        self.initialization = initialization
        self.ant_initial_pos = ant_initial_pos
        self.ant_initial_dir = ant_initial_dir
        self.init_grid_as = init_grid_as
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _initialize_grid(self) -> np.ndarray:
        if self.initialization == "all_white":
            return np.zeros((self.height, self.width), dtype=int)
        elif self.initialization == "random":
            return np.random.choice([WHITE, BLACK], size=(self.height, self.width))
        else:
            raise ValueError(f"Unknown initialization: {self.initialization}")

    def _initialize_ant(self) -> Tuple[int, int, str]:
        x = self.ant_initial_pos[0] if self.ant_initial_pos else self.width // 2
        y = self.ant_initial_pos[1] if self.ant_initial_pos else self.height // 2
        direction = (
            self.ant_initial_dir if self.ant_initial_dir else random.choice(DIRECTIONS)
        )
        return x, y, direction

    def _turn_right(self, direction: str) -> str:
        idx = DIRECTIONS.index(direction)
        return DIRECTIONS[(idx + 1) % 4]

    def _turn_left(self, direction: str) -> str:
        idx = DIRECTIONS.index(direction)
        return DIRECTIONS[(idx - 1) % 4]

    def generate(self) -> TaskProblem:
        grid = self._initialize_grid()
        x, y, direction = self._initialize_ant()
        if self.init_grid_as is not None:
            grid[x, y] = self.init_grid_as
        ant_path = [(x, y)]
        history = []
        # Place ant in the initial grid
        grid_with_ant = grid.copy()
        grid_with_ant[y, x] = ANT
        history.append(grid_with_ant.copy())
        for _ in range(self.steps):
            # Remove ant from current cell (restore color)
            grid[y, x] = WHITE if grid[y, x] == ANT else grid[y, x]
            # Turn and flip
            if grid[y, x] == WHITE:
                direction = self._turn_right(direction)
                grid[y, x] = BLACK
            elif grid[y, x] == BLACK:
                direction = self._turn_left(direction)
                grid[y, x] = WHITE
            # Move forward
            dx, dy = DIR_DELTA[direction]
            x_new, y_new = x + dx, y + dy
            # Stay in bounds (wrap around)
            x, y = x_new % self.width, y_new % self.height
            ant_path.append((x, y))
            # Place ant in new cell
            grid_with_ant = grid.copy()
            grid_with_ant[y, x] = ANT
            history.append(grid_with_ant.copy())
        init_grid = history[0].tolist()
        tgt_grid = history[-1].tolist()
        intermediate_grids = (
            [g.tolist() for g in history[1:-1]] if len(history) > 2 else None
        )
        metadata: LangtonAntSpecificMetadata = {
            "ant_path": ant_path,
            "ant_initial_pos": ant_path[0],
            "ant_initial_dir": (
                self.ant_initial_dir if self.ant_initial_dir else direction
            ),
            "steps": self.steps,
            "initialization": self.initialization,
            "seed": self.seed,
        }
        return TaskProblem(
            init_grid=init_grid,
            tgt_grid=tgt_grid,
            intermediate_grids=intermediate_grids,
            task_specific_metadata=metadata,
        )
