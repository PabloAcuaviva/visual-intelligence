import random
import zipfile
from pathlib import Path
from typing import List, Optional, TypedDict

import numpy as np

from visual_logic.tasks.base import Task, TaskProblem

# Cell states
DEAD_STATE = 0
ALIVE_STATE = 1

# Load Patterns
_zip_path = Path(__file__).parent / "resources" / "game_of_life.zip"
_initial_patterns = dict()


def parse_cells_from_content(content: str) -> np.ndarray:
    lines = [line.rstrip() for line in content.splitlines() if not line.startswith("!")]
    grid = [
        [ALIVE_STATE if char == "O" else DEAD_STATE for char in line]
        for line in lines
        if line.strip()
    ]
    return np.array(grid, dtype=int)


def _load_initial_patterns():
    with zipfile.ZipFile(_zip_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            if not file_info.filename.endswith(".cells"):
                continue
            try:
                with zip_ref.open(file_info) as file:
                    pattern_name = file_info.filename.split(".")[0]
                    content = file.read().decode("utf-8")
                    grid = parse_cells_from_content(content)
                    _initial_patterns[pattern_name] = grid
            except Exception:
                pass


def expand_grid_size(grid: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w = grid.shape
    if h > H or w > W:
        raise ValueError(f"Current grid size ({h}, {w}) exceeds target size ({H}, {W})")
    h_left_margin = (H - h) // 2
    w_left_margin = (W - w) // 2
    h_right_margin = H - h - h_left_margin
    w_right_margin = W - w - w_left_margin
    return np.pad(
        grid, ((h_left_margin, h_right_margin), (w_left_margin, w_right_margin))
    )


class GameOfLifeSpecificMetadata(TypedDict, total=False):
    birth_rule: List[int]
    survival_rule: List[int]
    initialization: str
    pattern_name: Optional[str]
    density: Optional[float]
    steps: int


class GameOfLife(Task):
    DEAD = DEAD_STATE
    ALIVE = ALIVE_STATE

    def __init__(
        self,
        width: int,
        height: int,
        birth_rule: Optional[List[int]] = None,
        survival_rule: Optional[List[int]] = None,
        initialization: str = "random",
        pattern_name: Optional[str] = None,
        density: float = 0.3,
        steps: int = 10,
        seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.birth_rule = birth_rule if birth_rule is not None else [3]
        self.survival_rule = survival_rule if survival_rule is not None else [2, 3]
        self.initialization = initialization
        self.pattern_name = pattern_name
        self.density = density
        self.steps = steps
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _step(self, grid: np.ndarray) -> np.ndarray:
        new_grid = np.zeros_like(grid)
        for y in range(self.height):
            for x in range(self.width):
                neighbors = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if grid[ny, nx] == self.ALIVE:
                                neighbors += 1
                if grid[y, x] == self.ALIVE:
                    if neighbors in self.survival_rule:
                        new_grid[y, x] = self.ALIVE
                    else:
                        new_grid[y, x] = self.DEAD
                else:
                    if neighbors in self.birth_rule:
                        new_grid[y, x] = self.ALIVE
                    else:
                        new_grid[y, x] = self.DEAD
        return new_grid

    def _initialize_grid(self) -> np.ndarray:
        if self.initialization == "random":
            return np.random.choice(
                [self.DEAD, self.ALIVE],
                size=(self.height, self.width),
                p=[1 - self.density, self.density],
            )
        elif self.initialization == "pattern":
            if not _initial_patterns:
                _load_initial_patterns()  # So that we don't need to load all the patterns whenever library is imported. Only when used
            valid_patterns = {
                name: grid
                for name, grid in _initial_patterns.items()
                if grid.shape[0] <= self.height and grid.shape[1] <= self.width
            }
            pattern_name = self.pattern_name or random.choice(
                list(valid_patterns.keys())
            )
            if pattern_name not in valid_patterns:
                raise ValueError(f"Unknown pattern: {pattern_name}")
            initial_grid = valid_patterns[pattern_name]
            return expand_grid_size(initial_grid, H=self.height, W=self.width)
        else:
            raise ValueError(f"Unknown initialization: {self.initialization}")

    def generate(self) -> TaskProblem:
        grid = self._initialize_grid()
        history = [grid.copy()]
        for _ in range(self.steps):
            grid = self._step(grid)
            history.append(grid.copy())
        init_grid = history[0].tolist()
        tgt_grid = history[-1].tolist()
        intermediate_grids = (
            [g.tolist() for g in history[1:-1]] if len(history) > 2 else None
        )
        metadata: GameOfLifeSpecificMetadata = {
            "birth_rule": self.birth_rule,
            "survival_rule": self.survival_rule,
            "initialization": self.initialization,
            "pattern_name": self.pattern_name,
            "density": self.density,
            "steps": self.steps,
        }
        return TaskProblem(
            init_grid=init_grid,
            tgt_grid=tgt_grid,
            intermediate_grids=intermediate_grids,
            task_specific_metadata=metadata,
        )
