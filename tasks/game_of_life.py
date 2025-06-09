import random
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

DEAD_STATE = 0
ALIVE_STATE = 1

###
# Load Patterns
###
_zip_path = Path(__file__).parent / "resources" / "game_of_life.zip"
_initial_patterns = dict()


def parse_cells_from_content(content):
    """Parse cells from string content"""
    lines = [line.rstrip() for line in content.splitlines() if not line.startswith("!")]
    grid = [
        [ALIVE_STATE if char == "O" else DEAD_STATE for char in line]
        for line in lines
        if line.strip()
    ]
    return np.array(grid, dtype=int)


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


###
# Utils
###
def expand_grid_size(grid, H: Optional[int] = None, W: Optional[int] = None):
    """Adjust the grid size to fit into target_size"""
    h, w = grid.shape
    H = H or H
    W = W or w
    if h > H or w > W:
        raise ValueError(f"Current grid size ({h}, {w}) exceeds target size ({H}, {W})")

    h_left_margin = (H - h) // 2
    w_left_margin = (W - w) // 2
    h_right_margin = H - h - h_left_margin
    w_right_margin = W - w - w_left_margin
    return np.pad(
        grid, ((h_left_margin, h_right_margin), (w_left_margin, w_right_margin))
    )


###
# Game of Life
###
class GameOfLifeGenerator:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = None
        self.history = []

        # Cell states
        self.DEAD = DEAD_STATE
        self.ALIVE = ALIVE_STATE

        # Default Conway's Game of Life rules (B3/S23)
        self.birth_rule = [3]  # Dead cell becomes alive with exactly 3 neighbors
        self.survival_rule = [2, 3]  # Live cell survives with 2 or 3 neighbors

    def set_rules(self, birth_rule: List[int] = None, survival_rule: List[int] = None):
        """Set custom rules for the Game of Life.

        Args:
            birth_rule: List of neighbor counts that cause a dead cell to become alive
            survival_rule: List of neighbor counts that keep an alive cell alive
        """
        if birth_rule is not None:
            self.birth_rule = birth_rule
        if survival_rule is not None:
            self.survival_rule = survival_rule

    def count_neighbors(self, grid: np.ndarray, x: int, y: int) -> int:
        """Count the number of alive neighbors for a cell."""
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if grid[ny, nx] == self.ALIVE:
                        count += 1
        return count

    def step(self, grid: np.ndarray) -> np.ndarray:
        """Apply one step of the Game of Life rules."""
        new_grid = np.zeros_like(grid)

        for y in range(self.height):
            for x in range(self.width):
                neighbors = self.count_neighbors(grid, x, y)

                if grid[y, x] == self.ALIVE:
                    # Living cell
                    if neighbors in self.survival_rule:
                        new_grid[y, x] = self.ALIVE
                    else:
                        new_grid[y, x] = self.DEAD
                else:
                    # Dead cell
                    if neighbors in self.birth_rule:
                        new_grid[y, x] = self.ALIVE
                    else:
                        new_grid[y, x] = self.DEAD

        return new_grid

    def initialize_random(self, density: float = 0.3, seed: int = None) -> np.ndarray:
        """Initialize grid with random alive cells.

        Args:
            density: Probability of each cell being alive (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        grid = np.random.choice(
            [self.DEAD, self.ALIVE],
            size=(self.height, self.width),
            p=[1 - density, density],
        )
        return grid

    def initialize_pattern(
        self,
        pattern_name: str = None,
        max_H: int = float("inf"),
        max_W: int = float("inf"),
    ) -> np.ndarray:
        """Initialize grid with a known Game of Life pattern. If no pattern is provided, choose a random one."""
        valid_initial_patterns = {
            pattern_name: grid
            for pattern_name, grid in _initial_patterns.items()
            if grid.shape[0] <= max_H and grid.shape[1] <= max_W
        }
        if pattern_name is None:
            pattern_name = random.choice(list(valid_initial_patterns.keys()))

        if pattern_name not in valid_initial_patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")

        initial_grid = valid_initial_patterns[pattern_name]
        return expand_grid_size(initial_grid, H=self.height, W=self.width)

    def initialize_from_grid(self, input_grid: List[List[int]]) -> np.ndarray:
        """Initialize from an input grid (for ARC-like format)."""
        initial_grid = np.array(input_grid, dtype=int)
        return expand_grid_size(initial_grid, H=self.height, W=self.width)

    def detect_cycle(self, max_history: int = 20) -> Tuple[bool, int, int]:
        """Detect if the system has entered a cycle.

        Returns:
            (is_cycle, cycle_start, cycle_length)
        """
        if len(self.history) < 3:
            return False, -1, -1

        # Check recent history for cycles
        recent_history = self.history[-max_history:]

        for cycle_len in range(1, len(recent_history) // 2 + 1):
            if cycle_len > len(recent_history) // 2:
                break

            # Check if last cycle_len grids repeat
            cycle_start = len(recent_history) - 2 * cycle_len
            if cycle_start < 0:
                continue

            is_cycle = True
            for i in range(cycle_len):
                grid1 = recent_history[cycle_start + i]
                grid2 = recent_history[cycle_start + i + cycle_len]
                if not np.array_equal(grid1, grid2):
                    is_cycle = False
                    break

            if is_cycle:
                actual_start = len(self.history) - 2 * cycle_len
                return True, actual_start, cycle_len

        return False, -1, -1

    def is_stable(self) -> bool:
        """Check if the current state is stable (no changes)."""
        if len(self.history) < 2:
            return False
        return np.array_equal(self.history[-1], self.history[-2])

    def is_extinct(self) -> bool:
        """Check if all cells are dead."""
        if not self.history:
            return False
        return np.sum(self.history[-1]) == 0

    def simulate(
        self,
        initial_grid: np.ndarray = None,
        max_steps: int = 100,
        initialization: str = "random",
        stop_at_cycle: bool = False,
        stop_at_extinct: bool = False,
        stop_at_stable: bool = False,
        init_kwargs: dict[str, Any] = None,
    ) -> List[np.ndarray]:
        """Run Game of Life simulation.

        Args:
            initial_grid: Starting grid (if None, will initialize based on initialization)
            max_steps: Maximum number of simulation steps
            initialization: 'random', 'pattern', or 'from_input'
            **init_kwargs: Additional arguments for initialization

        Returns:
            List of grids representing the evolution
        """
        init_kwargs = init_kwargs or {}
        # Initialize starting grid
        if initial_grid is not None:
            if isinstance(initial_grid, list):
                current_grid = self.initialize_from_grid(initial_grid)
            else:
                current_grid = initial_grid.copy()
        elif initialization == "random":
            current_grid = self.initialize_random(**init_kwargs)
        elif initialization == "pattern":
            current_grid = self.initialize_pattern(**init_kwargs)
        else:
            current_grid = np.zeros((self.height, self.width), dtype=int)

        self.grid = current_grid
        self.history = [current_grid.copy()]

        for step in range(max_steps):
            # Apply Game of Life rules
            next_grid = self.step(current_grid)
            self.history.append(next_grid.copy())

            # Check termination conditions
            if stop_at_extinct and self.is_extinct():
                break

            if stop_at_stable and self.is_stable():
                break

            if stop_at_cycle:
                is_cycle, cycle_start, cycle_length = self.detect_cycle()
                if is_cycle:
                    # Complete one more cycle to show the pattern
                    remaining_steps = min(cycle_length, max_steps - step - 1)
                    for _ in range(remaining_steps):
                        next_grid = self.step(next_grid)
                        self.history.append(next_grid.copy())
                    break

            current_grid = next_grid

        return self.history

    def to_arc_format(self) -> Dict:
        """Convert simulation to ARC-like JSON format.

        Returns:
            Dict with 'input' (initial grid) and 'output' (list of evolution grids)
        """
        if not self.history:
            return {}

        return {
            "input": self.history[0].tolist(),
            "output": [
                grid.tolist() for grid in self.history[1:]
            ],  # Evolution (excluding initial state)
        }

    def visualize_evolution(
        self, interval: int = 500, title: str = "Game of Life Evolution"
    ):
        """Create an animated visualization of the evolution."""
        if not self.history:
            return

        fig, ax = plt.subplots(figsize=(8, 8))

        # Create colormap: 0=white (dead), 1=black (alive)
        colors = ["white", "black"]
        cmap = ListedColormap(colors)

        im = ax.imshow(self.history[0], cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"{title} - Step 0")
        ax.set_xticks([])
        ax.set_yticks([])

        def animate(frame):
            if frame < len(self.history):
                im.set_array(self.history[frame])
                ax.set_title(f"{title} - Step {frame}")
            return [im]

        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(self.history),
            interval=interval,
            blit=True,
            repeat=True,
        )

        plt.tight_layout()
        plt.show()
        return anim

    def visualize_static(
        self, show_steps: List[int] = None, title: str = "Game of Life"
    ):
        """Show static visualization of selected steps."""
        if not self.history:
            return

        if show_steps is None:
            # Show first, middle, and last steps
            show_steps = [0, len(self.history) // 2, len(self.history) - 1]
            show_steps = [s for s in show_steps if s < len(self.history)]

        fig, axes = plt.subplots(1, len(show_steps), figsize=(4 * len(show_steps), 4))
        if len(show_steps) == 1:
            axes = [axes]

        colors = ["white", "black"]
        cmap = ListedColormap(colors)

        for i, step_idx in enumerate(show_steps):
            if step_idx < len(self.history):
                axes[i].imshow(self.history[step_idx], cmap=cmap, vmin=0, vmax=1)
                axes[i].set_title(f"Step {step_idx}")
                axes[i].set_xticks([])
                axes[i].set_yticks([])

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


class GameOfLifeDatasetGenerator:
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height

    def calculate_evolution_diversity(
        self, evolution1: List[List[List[int]]], evolution2: List[List[List[int]]]
    ) -> float:
        """Calculate diversity between two evolution sequences."""
        if not evolution1 or not evolution2:
            return 1.0

        # Compare final states and intermediate states
        min_len = min(len(evolution1), len(evolution2))
        if min_len == 0:
            return 1.0

        # Sample a few time points for comparison
        sample_points = (
            [0, min_len // 2, min_len - 1] if min_len > 2 else [0, min_len - 1]
        )
        sample_points = [p for p in sample_points if p < min_len]

        total_diff = 0
        total_cells = 0

        for t in sample_points:
            grid1 = np.array(evolution1[t])
            grid2 = np.array(evolution2[t])

            # Count different cells
            diff = np.sum(grid1 != grid2)
            total_diff += diff
            total_cells += grid1.size

        return total_diff / total_cells if total_cells > 0 else 1.0

    def generate_diverse_test_set(
        self,
        train_data: List[Dict],
        n_test: int,
        diversity_threshold: float = 0.3,
        max_attempts: int = 500,
        **generation_kwargs,
    ) -> List[Dict]:
        """Generate test set with diverse evolution patterns."""
        test_data = []
        attempts = 0

        while len(test_data) < n_test and attempts < max_attempts:
            attempts += 1

            # Generate new evolution
            generator = GameOfLifeGenerator(self.width, self.height)

            # Set rules if provided
            if "birth_rule" in generation_kwargs:
                generator.set_rules(
                    generation_kwargs["birth_rule"],
                    generation_kwargs.get("survival_rule", [2, 3]),
                )

            # Generate evolution
            generator.simulate(**generation_kwargs)
            arc_data = generator.to_arc_format()

            if not arc_data.get("output"):
                continue

            # Check diversity against training data
            min_diversity = float("inf")
            for train_sample in train_data:
                if train_sample.get("output"):
                    diversity = self.calculate_evolution_diversity(
                        arc_data["output"], train_sample["output"]
                    )
                    min_diversity = min(min_diversity, diversity)

            # Check diversity against existing test data
            for test_sample in test_data:
                if test_sample.get("output"):
                    diversity = self.calculate_evolution_diversity(
                        arc_data["output"], test_sample["output"]
                    )
                    min_diversity = min(min_diversity, diversity)

            if min_diversity >= diversity_threshold:
                test_data.append(arc_data)

        if len(test_data) < n_test:
            print(
                f"Warning: Only generated {len(test_data)}/{n_test} test samples "
                f"after {max_attempts} attempts"
            )

        return test_data

    def generate_dataset(
        self,
        n_train: int,
        n_test: int,
        max_steps: int = 50,
        diversity_threshold: float = 0.3,
        visualize_samples: bool = False,
        rule_variants: bool = False,
        **generation_kwargs,
    ) -> Dict:
        """Generate complete Game of Life dataset.

        Args:
            n_train: Number of training samples
            n_test: Number of test samples
            max_steps: Maximum evolution steps
            diversity_threshold: Minimum diversity between samples
            visualize_samples: Whether to show sample visualizations
            rule_variants: Whether to include different rule variants
            **generation_kwargs: Additional arguments for simulation
        """
        train_data = []

        # Generate training data
        for i in range(n_train):
            generator = GameOfLifeGenerator(self.width, self.height)

            # Optionally vary rules
            if rule_variants and random.random() < 0.3:
                rule_sets = [
                    ([3], [2, 3]),  # Classic Conway
                    ([3, 6], [2, 3]),  # HighLife
                    ([3], [1, 2, 3, 4, 5]),  # Life 34
                    ([3, 6, 8], [2, 4, 5]),  # Day & Night
                    ([2], [1, 2, 3, 4, 5]),  # Seeds
                ]
                birth_rule, survival_rule = random.choice(rule_sets)
                generator.set_rules(birth_rule, survival_rule)

            # Generate evolution
            generator.simulate(max_steps=max_steps, **generation_kwargs)
            arc_data = generator.to_arc_format()

            if arc_data.get("output"):  # Only add if evolution occurred
                train_data.append(arc_data)

        test_data = self.generate_diverse_test_set(
            train_data,
            n_test,
            diversity_threshold,
            max_steps=max_steps,
            **generation_kwargs,
        )

        if visualize_samples and train_data:
            # Show first training sample
            generator = GameOfLifeGenerator(self.width, self.height)
            generator.history = [np.array(train_data[0]["input"])] + [
                np.array(grid) for grid in train_data[0]["output"]
            ]
            generator.visualize_static(title="Sample Training Evolution")

            # Show first test sample if available
            if test_data:
                generator = GameOfLifeGenerator(self.width, self.height)
                generator.history = [np.array(test_data[0]["input"])] + [
                    np.array(grid) for grid in test_data[0]["output"]
                ]
                generator.visualize_static(title="Sample Test Evolution")

        return {"train": train_data, "test": test_data}


if __name__ == "__main__":
    ###
    # Random Game of Life
    ###
    # dataset_generator = GameOfLifeDatasetGenerator(width=16, height=16)

    # dataset = dataset_generator.generate_dataset(
    #     n_train=30,
    #     n_test=10,
    #     max_steps=10,
    #     diversity_threshold=0.4,
    #     visualize_samples=False,
    #     initialization="random",
    #     density=0.4,
    #     rule_variants=True,
    # )

    # # Save dataset
    # with open("gameoflife_dataset.json", "w") as f:
    #     json.dump(dataset, f, indent=2)

    ###
    # Game of Life - Patterns
    ###
    generator = GameOfLifeGenerator(30, 30)
    # Try different patterns
    n_patterns = 50
    for i in range(n_patterns):
        generator.simulate(
            max_steps=10,
            initialization="pattern",
            init_kwargs=dict(
                pattern_name=None,
                max_H=20,
                max_W=20,
            ),
        )
        print(f"\nPattern {i} evolution:")
