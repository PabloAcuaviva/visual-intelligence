import json
import random
from collections import deque
from typing import Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class MazeGenerator:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.maze = None
        self.start = None
        self.end = None
        self.solution_path = None

        # Ids grid
        self.WALL = 0
        self.PATH = 1
        self.START = 3
        self.END = 2
        self.SOLUTION = 4

    def wilson_algorithm(self) -> np.ndarray:
        """Generate a maze using Wilson's loop-erased random walk algorithm."""
        # Initialize grid - all walls initially
        maze = np.zeros((2 * self.height + 1, 2 * self.width + 1), dtype=int)

        # Set of cells that are part of the maze
        in_maze = set()

        # Add a random cell to start
        start_x, start_y = random.randrange(self.width), random.randrange(self.height)
        maze[2 * start_y + 1, 2 * start_x + 1] = self.PATH
        in_maze.add((start_x, start_y))

        # Get all cells not yet in maze
        remaining_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
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
                if 0 <= next_x < self.width and 0 <= next_y < self.height:
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
                for x in range(self.width)
                for y in range(self.height)
                if (x, y) not in in_maze
            ]

        return maze

    def find_path_cells(self, maze: np.ndarray) -> List[Tuple[int, int]]:
        """Find all path cells (non-wall cells) in the maze."""
        path_cells = []
        for y in range(1, maze.shape[0], 2):
            for x in range(1, maze.shape[1], 2):
                if maze[y, x] == self.PATH:
                    path_cells.append((x, y))
        return path_cells

    def set_start_end_points(self, maze: np.ndarray, start_pos=None, end_pos=None):
        """Set start and end points with flexible positioning options.

        Args:
            maze: The maze array
            start_pos: None (random), single tuple (fixed), or list of tuples (random choice from list)
            end_pos: None (random), single tuple (fixed), or list of tuples (random choice from list)
        """
        path_cells = self.find_path_cells(maze)

        if len(path_cells) < 2:
            raise ValueError("Not enough path cells for start and end points")

        # Handle start position
        if start_pos is None:
            # Random start from all path cells
            self.start = random.choice(path_cells)
        elif (
            isinstance(start_pos, (tuple, list))
            and len(start_pos) == 2
            and isinstance(start_pos[0], int)
        ):
            # Single fixed position (tuple)
            if start_pos not in path_cells:
                raise ValueError(f"Start position {start_pos} is not a valid path cell")
            self.start = start_pos
        elif isinstance(start_pos, list):
            # List of possible positions
            valid_starts = [pos for pos in start_pos if pos in path_cells]
            if not valid_starts:
                raise ValueError("No valid start positions found in path cells")
            self.start = random.choice(valid_starts)
        else:
            raise ValueError(
                "start_pos must be None, a tuple (x, y), or a list of tuples"
            )

        # Handle end position
        if end_pos is None:
            # Random end from remaining path cells (not start)
            end_candidates = [cell for cell in path_cells if cell != self.start]
            if not end_candidates:
                raise ValueError("No valid end positions available")
            self.end = random.choice(end_candidates)
        elif (
            isinstance(end_pos, (tuple, list))
            and len(end_pos) == 2
            and isinstance(end_pos[0], int)
        ):
            # Single fixed position (tuple)
            if end_pos not in path_cells:
                raise ValueError(f"End position {end_pos} is not a valid path cell")
            if end_pos == self.start:
                raise ValueError("End position cannot be the same as start position")
            self.end = end_pos
        elif isinstance(end_pos, list):
            # List of possible positions
            valid_ends = [
                pos for pos in end_pos if pos in path_cells and pos != self.start
            ]
            if not valid_ends:
                raise ValueError(
                    "No valid end positions found in path cells (different from start)"
                )
            self.end = random.choice(valid_ends)
        else:
            raise ValueError(
                "end_pos must be None, a tuple (x, y), or a list of tuples"
            )

        # Mark start and end in maze
        maze[self.start[1], self.start[0]] = self.START
        maze[self.end[1], self.end[0]] = self.END

    def find_shortest_path(self, maze: np.ndarray) -> List[Tuple[int, int]]:
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
                    0 <= nx < maze.shape[1]
                    and 0 <= ny < maze.shape[0]
                    and (nx, ny) not in visited
                    and maze[ny, nx] != self.WALL
                ):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))

        return []  # No path found

    def generate_maze(self, start_pos=None, end_pos=None) -> Dict:
        """Generate a complete maze with solution.

        Args:
            start_pos: None (random), tuple (x,y), or list of tuples for start position
            end_pos: None (random), tuple (x,y), or list of tuples for end position
        """
        # Generate maze structure
        self.maze = self.wilson_algorithm()

        # Set start and end points
        self.set_start_end_points(self.maze, start_pos, end_pos)

        # Find solution path
        self.solution_path = self.find_shortest_path(self.maze)

        return {
            "maze": self.maze.tolist(),
            "start": self.start,
            "end": self.end,
            "solution_path": self.solution_path,
        }

    def visualize_maze(self, show_solution: bool = True, title: str = "Maze"):
        """Visualize the maze using matplotlib."""
        if self.maze is None:
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        colors = {
            0: (71 / 255, 48 / 255, 45 / 255),  # Wall - dark brown
            1: (255 / 255, 255 / 255, 255 / 255),  # Path - white
            2: (244 / 255, 96 / 255, 54 / 255),  # End - orange-red
            3: (72 / 255, 191 / 255, 132 / 255),  # Start - green
            4: (46 / 255, 134 / 255, 171 / 255),  # Solution - blue
        }

        # Create a copy of maze for visualization
        viz_maze = self.maze.copy()

        # Mark solution path if requested
        if show_solution and self.solution_path:
            for x, y in self.solution_path:
                if viz_maze[y, x] not in [self.START, self.END]:
                    viz_maze[y, x] = self.SOLUTION

        # Draw maze
        for y in range(viz_maze.shape[0]):
            for x in range(viz_maze.shape[1]):
                color = colors.get(viz_maze[y, x], "gray")
                rect = patches.Rectangle(
                    (x, viz_maze.shape[0] - y - 1), 1, 1, linewidth=0, facecolor=color
                )
                ax.add_patch(rect)

        ax.set_xlim(0, viz_maze.shape[1])
        ax.set_ylim(0, viz_maze.shape[0])
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)

        plt.tight_layout()
        plt.show()

    def to_arc_format(self) -> Dict:
        """Convert maze to ARC-like JSON format."""
        if self.maze is None:
            return {}

        # Input: maze with start and end points
        input_maze = self.maze.copy()

        # Output: maze with solution path
        output_maze = self.maze.copy()
        if self.solution_path:
            for x, y in self.solution_path:
                if output_maze[y, x] not in [self.START, self.END]:
                    output_maze[y, x] = self.SOLUTION

        return {"input": input_maze.tolist(), "output": output_maze.tolist()}


class MazeDatasetGenerator:
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height

    def calculate_path_distance(
        self, path1: List[Tuple[int, int]], path2: List[Tuple[int, int]]
    ) -> float:
        """Calculate normalized intersection distance between two solution paths."""
        if not path1 or not path2:
            return 1.0  # Maximum distance if either path is empty

        set1 = set(path1)
        set2 = set(path2)

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:
            return 1.0

        # Return 1 - jaccard similarity (so 0 = identical, 1 = no overlap)
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity

    def generate_diverse_test_set(
        self,
        train_mazes: List[Dict],
        n_test: int,
        distance_threshold: float = 0.7,
        max_attempts: int = 1000,
        start_pos=None,
        end_pos=None,
    ) -> List[Dict]:
        """Generate test mazes that are sufficiently different from training mazes."""
        test_mazes = []
        attempts = 0

        while len(test_mazes) < n_test and attempts < max_attempts:
            attempts += 1

            # Generate new maze
            generator = MazeGenerator(self.width, self.height)
            maze_data = generator.generate_maze(start_pos=start_pos, end_pos=end_pos)

            # Check distance from all training mazes
            min_distance = float("inf")
            for train_maze in train_mazes:
                distance = self.calculate_path_distance(
                    maze_data["solution_path"], train_maze["solution_path"]
                )
                min_distance = min(min_distance, distance)

            # If sufficiently different, add to test set
            if min_distance >= distance_threshold:
                test_mazes.append(maze_data)

        if len(test_mazes) < n_test:
            print(
                f"Warning: Only generated {len(test_mazes)}/{n_test} test mazes "
                f"after {max_attempts} attempts"
            )

        return test_mazes

    def generate_dataset(
        self,
        n_train: int,
        n_test: int,
        distance_threshold: float = 0.7,
        visualize_samples: bool = False,
        start_pos=None,
        end_pos=None,
    ) -> Dict:
        """Generate complete ARC-format dataset with train/test split.

        Args:
            n_train: Number of training samples
            n_test: Number of test samples
            distance_threshold: Minimum distance between train/test mazes
            visualize_samples: Whether to show sample visualizations
            start_pos: Start position control (None, tuple, or list of tuples)
            end_pos: End position control (None, tuple, or list of tuples)
        """
        # Generate training mazes
        train_mazes = []
        for i in range(n_train):
            generator = MazeGenerator(self.width, self.height)
            maze_data = generator.generate_maze(start_pos=start_pos, end_pos=end_pos)
            train_mazes.append(maze_data)

        # Generate test mazes
        test_mazes = self.generate_diverse_test_set(
            train_mazes,
            n_test,
            distance_threshold,
            start_pos=start_pos,
            end_pos=end_pos,
        )

        # Convert to ARC format
        arc_data = {"train": [], "test": []}

        for maze_data in train_mazes:
            generator = MazeGenerator(self.width, self.height)
            generator.maze = np.array(maze_data["maze"])
            generator.start = maze_data["start"]
            generator.end = maze_data["end"]
            generator.solution_path = maze_data["solution_path"]
            arc_data["train"].append(generator.to_arc_format())

        for maze_data in test_mazes:
            generator = MazeGenerator(self.width, self.height)
            generator.maze = np.array(maze_data["maze"])
            generator.start = maze_data["start"]
            generator.end = maze_data["end"]
            generator.solution_path = maze_data["solution_path"]
            arc_data["test"].append(generator.to_arc_format())

        # Visualize some samples if requested
        if visualize_samples and train_mazes:
            # Show first training maze
            generator = MazeGenerator(self.width, self.height)
            generator.maze = np.array(train_mazes[0]["maze"])
            generator.start = train_mazes[0]["start"]
            generator.end = train_mazes[0]["end"]
            generator.solution_path = train_mazes[0]["solution_path"]
            generator.visualize_maze(show_solution=True, title="Sample Training Maze")

            # Show first test maze if available
            if test_mazes:
                generator = MazeGenerator(self.width, self.height)
                generator.maze = np.array(test_mazes[0]["maze"])
                generator.start = test_mazes[0]["start"]
                generator.end = test_mazes[0]["end"]
                generator.solution_path = test_mazes[0]["solution_path"]
                generator.visualize_maze(show_solution=True, title="Sample Test Maze")

        return arc_data


if __name__ == "__main__":
    start_pos = [(1, 1), (1, 11)]
    end_pos = [(11, 1), (11, 11)]
    start_pos = None
    end_pos = None
    dataset_generator = MazeDatasetGenerator(width=6, height=6)
    dataset = dataset_generator.generate_dataset(
        n_train=3,
        n_test=2,
        distance_threshold=0.5,
        visualize_samples=False,
        start_pos=start_pos,
        end_pos=end_pos,
    )

    with open("debug_output/maze_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
