import json
import random
from collections import deque
from typing import Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class ObstacleNavigationGenerator:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = None
        self.start = None
        self.end = None
        self.solution_path = None

        self.OBSTACLE = 0  # Obstacle cells
        self.FREE = 1  # Free navigable space
        self.START = 3
        self.END = 2
        self.SOLUTION = 4

    def generate_obstacle_cluster(
        self, center_x: int, center_y: int, size: int, shape: str = "circular"
    ) -> List[Tuple[int, int]]:
        """Generate a cluster of obstacle cells around a center point.

        Args:
            center_x, center_y: Center of the obstacle cluster
            size: Approximate radius/size of the cluster
            shape: 'circular', 'rectangular', 'irregular', or 'elongated'
        """
        obstacle_cells = []

        if shape == "circular":
            # Circular obstacle with some randomness for natural appearance
            for y in range(
                max(0, center_y - size), min(self.height, center_y + size + 1)
            ):
                for x in range(
                    max(0, center_x - size), min(self.width, center_x + size + 1)
                ):
                    distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    # Add noise for irregular boundaries
                    noise_factor = random.uniform(0.7, 1.3)
                    if distance <= size * noise_factor:
                        obstacle_cells.append((x, y))

        elif shape == "rectangular":
            # Rectangular obstacle with some randomness
            for y in range(
                max(0, center_y - size), min(self.height, center_y + size + 1)
            ):
                for x in range(
                    max(0, center_x - size), min(self.width, center_x + size + 1)
                ):
                    if random.random() < 0.8:  # 80% density for irregular edges
                        obstacle_cells.append((x, y))

        elif shape == "elongated":
            # Elongated obstacle (randomly horizontal or vertical)
            if random.random() < 0.5:
                # Horizontal elongation
                width_factor = random.uniform(1.5, 2.5)
                height_factor = random.uniform(0.5, 0.8)
            else:
                # Vertical elongation
                width_factor = random.uniform(0.5, 0.8)
                height_factor = random.uniform(1.5, 2.5)

            for y in range(
                max(0, center_y - int(size * height_factor)),
                min(self.height, center_y + int(size * height_factor) + 1),
            ):
                for x in range(
                    max(0, center_x - int(size * width_factor)),
                    min(self.width, center_x + int(size * width_factor) + 1),
                ):
                    distance_x = abs(x - center_x) / (size * width_factor)
                    distance_y = abs(y - center_y) / (size * height_factor)
                    if distance_x + distance_y <= 1.0 + random.uniform(-0.2, 0.2):
                        obstacle_cells.append((x, y))

        else:  # irregular shape
            # Irregular obstacle using growth-based approach
            obstacle_cells = [(center_x, center_y)]
            current_size = 1
            target_size = size * size

            while current_size < target_size:
                if not obstacle_cells:
                    break

                # Select random boundary cell for expansion
                boundary_cell = random.choice(obstacle_cells)
                directions = [
                    (0, 1),
                    (1, 0),
                    (0, -1),
                    (-1, 0),
                    (1, 1),
                    (-1, -1),
                    (1, -1),
                    (-1, 1),
                ]

                # Add multiple cells per iteration for faster growth
                for _ in range(random.randint(1, 3)):
                    dx, dy = random.choice(directions)
                    new_x, new_y = boundary_cell[0] + dx, boundary_cell[1] + dy

                    if (
                        0 <= new_x < self.width
                        and 0 <= new_y < self.height
                        and (new_x, new_y) not in obstacle_cells
                    ):
                        obstacle_cells.append((new_x, new_y))
                        current_size += 1
                        if current_size >= target_size:
                            break

        return obstacle_cells

    def generate_obstacle_field(
        self, num_obstacles: int = None, obstacle_density: float = 0.15
    ) -> np.ndarray:
        """Generate a grid with randomly distributed obstacle clusters.

        Args:
            num_obstacles: Number of obstacle clusters (if None, calculated from density)
            obstacle_density: Approximate fraction of grid occupied by obstacles
        """
        # Initialize grid - all free space initially
        grid = np.ones((self.height, self.width), dtype=int) * self.FREE

        if num_obstacles is None:
            # Calculate number of obstacles based on density
            total_cells = self.width * self.height
            target_obstacle_cells = int(total_cells * obstacle_density)
            # Estimate obstacles needed (approximate)
            avg_obstacle_size = min(self.width, self.height) // 4
            num_obstacles = max(
                3, target_obstacle_cells // (avg_obstacle_size * avg_obstacle_size)
            )

        # Generate obstacle clusters
        for _ in range(num_obstacles):
            # Random placement
            center_x = random.randint(1, self.width - 2)
            center_y = random.randint(1, self.height - 2)

            # Random size (scaled to grid dimensions)
            size = random.randint(
                max(1, min(self.width, self.height) // 8),
                min(self.width, self.height) // 3,
            )

            # Random obstacle shape
            shape = random.choice(["circular", "rectangular", "elongated", "irregular"])

            # Generate obstacle cluster
            obstacle_cells = self.generate_obstacle_cluster(
                center_x, center_y, size, shape
            )

            # Place obstacles in grid
            for x, y in obstacle_cells:
                grid[y, x] = self.OBSTACLE

        return grid

    def find_free_cells(self, grid: np.ndarray) -> List[Tuple[int, int]]:
        """Find all free navigable cells in the grid."""
        free_cells = []
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] == self.FREE:
                    free_cells.append((x, y))
        return free_cells

    def set_start_end_points(self, grid: np.ndarray, start_pos=None, end_pos=None):
        """Set start and end points with flexible positioning options."""
        free_cells = self.find_free_cells(grid)

        if len(free_cells) < 2:
            raise ValueError("Insufficient free cells for start and end points")

        # Handle start position
        if start_pos is None:
            self.start = random.choice(free_cells)
        elif (
            isinstance(start_pos, (tuple, list))
            and len(start_pos) == 2
            and isinstance(start_pos[0], int)
        ):
            if start_pos not in free_cells:
                raise ValueError(f"Start position {start_pos} is not a valid free cell")
            self.start = start_pos
        elif isinstance(start_pos, list):
            valid_starts = [pos for pos in start_pos if pos in free_cells]
            if not valid_starts:
                raise ValueError("No valid start positions found in free cells")
            self.start = random.choice(valid_starts)
        else:
            raise ValueError(
                "start_pos must be None, a tuple (x, y), or a list of tuples"
            )

        # Handle end position
        if end_pos is None:
            end_candidates = [cell for cell in free_cells if cell != self.start]
            if not end_candidates:
                raise ValueError("No valid end positions available")
            self.end = random.choice(end_candidates)
        elif (
            isinstance(end_pos, (tuple, list))
            and len(end_pos) == 2
            and isinstance(end_pos[0], int)
        ):
            if end_pos not in free_cells:
                raise ValueError(f"End position {end_pos} is not a valid free cell")
            if end_pos == self.start:
                raise ValueError("End position cannot be the same as start position")
            self.end = end_pos
        elif isinstance(end_pos, list):
            valid_ends = [
                pos for pos in end_pos if pos in free_cells and pos != self.start
            ]
            if not valid_ends:
                raise ValueError(
                    "No valid end positions found in free cells (different from start)"
                )
            self.end = random.choice(valid_ends)
        else:
            raise ValueError(
                "end_pos must be None, a tuple (x, y), or a list of tuples"
            )

        # Mark start and end in grid
        grid[self.start[1], self.start[0]] = self.START
        grid[self.end[1], self.end[0]] = self.END

    def find_shortest_path(self, grid: np.ndarray) -> List[Tuple[int, int]]:
        """Find shortest path from start to end using BFS."""
        if self.start is None or self.end is None:
            return []

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
                    and grid[ny, nx] != self.OBSTACLE
                ):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))

        return []  # No path found

    def ensure_path_connectivity(
        self, grid: np.ndarray, max_attempts: int = 10
    ) -> bool:
        """Ensure path connectivity between start and end by removing obstacles if needed."""
        for attempt in range(max_attempts):
            path = self.find_shortest_path(grid)
            if path:
                return True

            # Remove some obstacles to create connectivity
            obstacle_cells = [
                (x, y)
                for y in range(self.height)
                for x in range(self.width)
                if grid[y, x] == self.OBSTACLE
            ]

            if not obstacle_cells:
                return False

            # Remove a small number of obstacles strategically
            num_to_remove = min(5, len(obstacle_cells))
            cells_to_remove = random.sample(obstacle_cells, num_to_remove)

            for x, y in cells_to_remove:
                grid[y, x] = self.FREE

        return False

    def generate_navigation_problem(
        self, start_pos=None, end_pos=None, num_obstacles=None, obstacle_density=0.15
    ) -> Dict:
        """Generate a complete obstacle navigation problem with solution.

        Args:
            start_pos: Start position control
            end_pos: End position control
            num_obstacles: Number of obstacle clusters (if None, calculated from density)
            obstacle_density: Density of obstacles in the grid
        """
        max_generation_attempts = 50

        for attempt in range(max_generation_attempts):
            # Generate obstacle field
            self.grid = self.generate_obstacle_field(num_obstacles, obstacle_density)

            # Set start and end points
            try:
                self.set_start_end_points(self.grid, start_pos, end_pos)
            except ValueError:
                continue  # Retry if no valid positions

            # Ensure path connectivity
            if self.ensure_path_connectivity(self.grid):
                # Find optimal solution path
                self.solution_path = self.find_shortest_path(self.grid)
                if self.solution_path:
                    break
        else:
            raise RuntimeError(
                f"Could not generate valid navigation problem after {max_generation_attempts} attempts"
            )

        return {
            "grid": self.grid.tolist(),
            "start": self.start,
            "end": self.end,
            "solution_path": self.solution_path,
        }

    def visualize_navigation_problem(
        self, show_solution: bool = True, title: str = "Obstacle Navigation"
    ):
        """Visualize the obstacle navigation problem using matplotlib."""
        if self.grid is None:
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        colors = {
            0: (71 / 255, 48 / 255, 45 / 255),  # Obstacle - dark brown
            1: (255 / 255, 255 / 255, 255 / 255),  # Free space - white
            2: (244 / 255, 96 / 255, 54 / 255),  # End - orange-red
            3: (72 / 255, 191 / 255, 132 / 255),  # Start - green
            4: (46 / 255, 134 / 255, 171 / 255),  # Solution - blue
        }

        # Create visualization copy
        viz_grid = self.grid.copy()

        # Mark solution path if requested
        if show_solution and self.solution_path:
            for x, y in self.solution_path:
                if viz_grid[y, x] not in [self.START, self.END]:
                    viz_grid[y, x] = self.SOLUTION

        # Render grid
        for y in range(viz_grid.shape[0]):
            for x in range(viz_grid.shape[1]):
                color = colors.get(viz_grid[y, x], "gray")
                rect = patches.Rectangle(
                    (x, viz_grid.shape[0] - y - 1), 1, 1, linewidth=0, facecolor=color
                )
                ax.add_patch(rect)

        ax.set_xlim(0, viz_grid.shape[1])
        ax.set_ylim(0, viz_grid.shape[0])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)

        plt.tight_layout()
        plt.show()

    def to_arc_format(self) -> Dict:
        """Convert obstacle navigation problem to ARC-like JSON format."""
        if self.grid is None:
            return {}

        # Input: grid with start and end points
        input_grid = self.grid.copy()

        # Output: grid with solution path
        output_grid = self.grid.copy()
        if self.solution_path:
            for x, y in self.solution_path:
                if output_grid[y, x] not in [self.START, self.END]:
                    output_grid[y, x] = self.SOLUTION

        return {"input": input_grid.tolist(), "output": output_grid.tolist()}


class ObstacleNavigationDatasetGenerator:
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height

    def calculate_path_similarity(
        self, path1: List[Tuple[int, int]], path2: List[Tuple[int, int]]
    ) -> float:
        """Calculate Jaccard similarity between two solution paths."""
        if not path1 or not path2:
            return 0.0

        set1 = set(path1)
        set2 = set(path2)

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def calculate_path_distance(
        self, path1: List[Tuple[int, int]], path2: List[Tuple[int, int]]
    ) -> float:
        """Calculate normalized distance between two solution paths (1 - similarity)."""
        return 1.0 - self.calculate_path_similarity(path1, path2)

    def generate_diverse_test_set(
        self,
        train_problems: List[Dict],
        n_test: int,
        diversity_threshold: float = 0.7,
        max_attempts: int = 1000,
        start_pos=None,
        end_pos=None,
        num_obstacles=None,
        obstacle_density=0.15,
    ) -> List[Dict]:
        """Generate test problems with sufficient diversity from training problems."""
        test_problems = []
        attempts = 0

        while len(test_problems) < n_test and attempts < max_attempts:
            attempts += 1

            # Generate new navigation problem
            generator = ObstacleNavigationGenerator(self.width, self.height)
            try:
                problem_data = generator.generate_navigation_problem(
                    start_pos=start_pos,
                    end_pos=end_pos,
                    num_obstacles=num_obstacles,
                    obstacle_density=obstacle_density,
                )
            except RuntimeError:
                continue  # Skip if generation failed

            # Check diversity from training problems
            min_distance = float("inf")
            for train_problem in train_problems:
                distance = self.calculate_path_distance(
                    problem_data["solution_path"], train_problem["solution_path"]
                )
                min_distance = min(min_distance, distance)

            # Add to test set if sufficiently diverse
            if min_distance >= diversity_threshold:
                test_problems.append(problem_data)

        if len(test_problems) < n_test:
            print(
                f"Warning: Only generated {len(test_problems)}/{n_test} test problems "
                f"after {max_attempts} attempts"
            )

        return test_problems

    def generate_dataset(
        self,
        n_train: int,
        n_test: int,
        diversity_threshold: float = 0.7,
        visualize_samples: bool = False,
        start_pos=None,
        end_pos=None,
        num_obstacles=None,
        obstacle_density=0.15,
    ) -> Dict:
        """Generate complete ARC-format dataset with train/test split.

        Args:
            n_train: Number of training samples
            n_test: Number of test samples
            diversity_threshold: Minimum diversity between train/test problems
            visualize_samples: Whether to show sample visualizations
            start_pos: Start position control
            end_pos: End position control
            num_obstacles: Number of obstacle clusters per problem
            obstacle_density: Density of obstacle coverage
        """
        # Generate training problems
        train_problems = []
        for i in range(n_train):
            generator = ObstacleNavigationGenerator(self.width, self.height)
            try:
                problem_data = generator.generate_navigation_problem(
                    start_pos=start_pos,
                    end_pos=end_pos,
                    num_obstacles=num_obstacles,
                    obstacle_density=obstacle_density,
                )
                train_problems.append(problem_data)
            except RuntimeError:
                print(f"Warning: Failed to generate training problem {i}")
                continue

        print(f"Generated {len(train_problems)} training problems")

        # Generate test problems
        test_problems = self.generate_diverse_test_set(
            train_problems,
            n_test,
            diversity_threshold,
            start_pos=start_pos,
            end_pos=end_pos,
            num_obstacles=num_obstacles,
            obstacle_density=obstacle_density,
        )

        # Convert to ARC format
        arc_data = {"train": [], "test": []}

        for problem_data in train_problems:
            generator = ObstacleNavigationGenerator(self.width, self.height)
            generator.grid = np.array(problem_data["grid"])
            generator.start = problem_data["start"]
            generator.end = problem_data["end"]
            generator.solution_path = problem_data["solution_path"]
            arc_data["train"].append(generator.to_arc_format())

        for problem_data in test_problems:
            generator = ObstacleNavigationGenerator(self.width, self.height)
            generator.grid = np.array(problem_data["grid"])
            generator.start = problem_data["start"]
            generator.end = problem_data["end"]
            generator.solution_path = problem_data["solution_path"]
            arc_data["test"].append(generator.to_arc_format())

        # Visualize samples if requested
        if visualize_samples and train_problems:
            # Show first training problem
            generator = ObstacleNavigationGenerator(self.width, self.height)
            generator.grid = np.array(train_problems[0]["grid"])
            generator.start = train_problems[0]["start"]
            generator.end = train_problems[0]["end"]
            generator.solution_path = train_problems[0]["solution_path"]
            generator.visualize_navigation_problem(
                show_solution=True, title="Sample Training Problem"
            )

            # Show first test problem if available
            if test_problems:
                generator = ObstacleNavigationGenerator(self.width, self.height)
                generator.grid = np.array(test_problems[0]["grid"])
                generator.start = test_problems[0]["start"]
                generator.end = test_problems[0]["end"]
                generator.solution_path = test_problems[0]["solution_path"]
                generator.visualize_navigation_problem(
                    show_solution=True, title="Sample Test Problem"
                )

        return arc_data


if __name__ == "__main__":
    start_pos = [(1, 1)]
    end_pos = [(19, 19)]

    dataset_generator = ObstacleNavigationDatasetGenerator(width=21, height=21)
    dataset = dataset_generator.generate_dataset(
        n_train=1000,
        n_test=50,
        diversity_threshold=0.5,
        visualize_samples=True,
        start_pos=start_pos,
        end_pos=end_pos,
        obstacle_density=0.2,
    )

    # Save dataset
    with open("obstacle_navigation_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(
        f"Generated dataset with {len(dataset['train'])} training and {len(dataset['test'])} test samples"
    )
