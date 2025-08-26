import random
from typing import List, Optional, TypedDict

import numpy as np

from visual_logic.tasks.base import Task, TaskProblem


class CellularAutomata1DSpecificMetadata(TypedDict, total=False):
    rule: int
    width: int
    initialization: str
    steps: int
    k: int
    seed: Optional[int]
    rule_binary: List[int]


class CellularAutomata1D(Task):
    def __init__(
        self,
        rule: int,
        width: int = 16,
        initialization: str = "random",  # "random", "zeros", "ones", or "single_center"
        steps: int = 16,
        k: int = 1,
        seed: Optional[int] = None,
        init_state: Optional[List[int]] = None,
    ):
        """
        Initialize a 1D Cellular Automaton task.

        Args:
            rule: The rule number (0-255)
            width: The width of the domain (number of cells)
            initialization: Initialization method ("random", "zeros", "ones", "single_center")
            steps: Number of time steps to simulate
            k: Interval for outputting time points (1 = every step)
            seed: Random seed for reproducibility
            init_state: Custom initial state as list of 0s and 1s
        """
        if not (0 <= rule <= 255):
            raise ValueError("Rule must be between 0 and 255")
        if width <= 0:
            raise ValueError("Width must be positive")
        if steps <= 0:
            raise ValueError("Steps must be positive")
        if k <= 0:
            raise ValueError("k must be positive")

        self.rule = rule
        self.width = width
        self.initialization = initialization
        self.steps = steps
        self.k = k
        self.init_state = init_state
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _get_rule_binary(self) -> np.ndarray:
        """Convert rule number to binary representation."""
        return np.array(
            [int(x) for x in np.binary_repr(self.rule, width=8)], dtype=np.uint8
        )

    def _initialize_state(self) -> np.ndarray:
        """Initialize the cellular automaton state."""
        if self.init_state is not None:
            if len(self.init_state) != self.width:
                raise ValueError(
                    f"init_state length {len(self.init_state)} must match width {self.width}"
                )
            return np.array(self.init_state, dtype=np.uint8)

        if self.initialization == "random":
            return np.random.randint(2, size=self.width, dtype=np.uint8)
        elif self.initialization == "zeros":
            return np.zeros(self.width, dtype=np.uint8)
        elif self.initialization == "ones":
            return np.ones(self.width, dtype=np.uint8)
        elif self.initialization == "single_center":
            state = np.zeros(self.width, dtype=np.uint8)
            state[self.width // 2] = 1
            return state
        else:
            raise ValueError(f"Unknown initialization: {self.initialization}")

    def _cellular_automaton_step(
        self, state: np.ndarray, rule_bin: np.ndarray
    ) -> np.ndarray:
        """Perform one step of the cellular automaton evolution."""
        new_state = np.zeros(self.width, dtype=np.uint8)

        for i in range(self.width):
            left = state[(i - 1) % self.width]
            center = state[i]
            right = state[(i + 1) % self.width]

            # Create neighborhood index (left, center, right as 3-bit number)
            neighborhood = (left << 2) | (center << 1) | right

            # Apply rule
            new_state[i] = rule_bin[7 - neighborhood]

        return new_state

    def generate(self) -> TaskProblem:
        """Generate a cellular automaton problem."""
        rule_bin = self._get_rule_binary()
        state = self._initialize_state()

        # Store all states (including initial)
        states = [state.copy()]

        # Evolve the cellular automaton
        for step in range(self.steps):
            state = self._cellular_automaton_step(state, rule_bin)
            states.append(state.copy())

        # Create T x W grids where T = steps + 1 (including initial state)
        # Value mapping: 0 -> empty/not computed, 1 -> cell state 0, 2 -> cell state 1
        total_steps = len(states)

        # Initialize grids with zeros (empty/not computed)
        init_grid = [[0 for _ in range(self.width)] for _ in range(total_steps)]
        tgt_grid = [[0 for _ in range(self.width)] for _ in range(total_steps)]

        # Fill init_grid with only the first row (initial state)
        for col in range(self.width):
            init_grid[0][col] = states[0][col] + 1  # Convert 0->1, 1->2

        # Fill tgt_grid with all computed states
        for row in range(total_steps):
            for col in range(self.width):
                tgt_grid[row][col] = states[row][col] + 1  # Convert 0->1, 1->2

        # Create intermediate grids showing partial evolution
        intermediate_grids = []
        if total_steps > 2:
            for step in range(1, total_steps - 1):
                intermediate_grid = [
                    [0 for _ in range(self.width)] for _ in range(total_steps)
                ]

                # Fill up to current step
                for row in range(step + 1):
                    for col in range(self.width):
                        intermediate_grid[row][col] = (
                            states[row][col] + 1
                        )  # Convert 0->1, 1->2

                intermediate_grids.append(intermediate_grid)

        # Create metadata
        metadata: CellularAutomata1DSpecificMetadata = {
            "rule": self.rule,
            "width": self.width,
            "initialization": self.initialization,
            "steps": self.steps,
            "k": self.k,
            "seed": self.seed,
            "rule_binary": rule_bin.tolist(),
        }

        return TaskProblem(
            init_grid=init_grid,
            tgt_grid=tgt_grid,
            intermediate_grids=intermediate_grids,
            task_specific_metadata=metadata,
        )
