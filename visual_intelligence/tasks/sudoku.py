import random
from typing import List, Optional, Tuple, TypedDict, Union

import numpy as np

from visual_intelligence.tasks.base import Task, TaskProblem

# Module-level dataset cache (lazy loading)
_sudoku_dataset = None

# Mapping from string difficulty to numeric ranges (min, max inclusive)
# HF dataset uses numeric difficulty where higher = harder
HF_DIFFICULTY_RANGES = {
    "easy": (0, 3),
    "medium": (4, 6),
    "hard": (7, 10),
}


def _load_sudoku_dataset():
    """Load the HuggingFace Sudoku dataset (lazy loading)."""
    global _sudoku_dataset
    if _sudoku_dataset is None:
        from datasets import load_dataset

        _sudoku_dataset = load_dataset("Ritvik19/Sudoku-Dataset")
    return _sudoku_dataset


def _parse_puzzle_string(puzzle_str: str) -> np.ndarray:
    """Convert puzzle string '530070000...' to 9x9 numpy array."""
    digits = [int(c) for c in puzzle_str]
    return np.array(digits).reshape(9, 9)


class SudokuSpecificMetadata(TypedDict, total=False):
    size: int  # 9 for standard, 4 for mini
    block_size: int  # 3 for standard, 2 for mini
    difficulty: str
    givens: int  # number of starting clues
    source: Optional[str]  # HF source field (None if generated)
    from_dataset: bool  # True if loaded from HF, False if generated


class Sudoku(Task):
    def __init__(
        self,
        difficulty: str = "easy",
        variant: str = "standard",
        seed: Optional[int] = None,
        # Dataset loading parameters
        initialization: str = "generate",  # "generate" or "dataset"
        hf_difficulty: Optional[
            Union[str, Tuple[int, int]]
        ] = None,  # "easy"/"medium"/"hard" or (min, max) tuple
        hf_source: Optional[str] = None,  # Filter by HF 'set' field (None = no filter)
        unique_solution_only: bool = True,  # Filter for unique solutions
    ):
        # Validate initialization mode
        if initialization not in ["generate", "dataset"]:
            raise ValueError("initialization must be 'generate' or 'dataset'")

        # Validate variant
        if variant not in ["standard", "mini"]:
            raise ValueError("Variant must be 'standard' or 'mini'")

        # Dataset mode requires standard variant (9x9)
        if initialization == "dataset" and variant != "standard":
            raise ValueError(
                f"HuggingFace Sudoku dataset only supports standard (9x9) variant. "
                f"Got variant='{variant}'"
            )

        # Validate difficulty only for generate mode
        if initialization == "generate" and difficulty not in [
            "easy",
            "medium",
            "hard",
        ]:
            raise ValueError("Difficulty must be 'easy', 'medium', or 'hard'")

        self.initialization = initialization
        self.difficulty = difficulty
        self.variant = variant
        self.hf_difficulty = hf_difficulty
        self.hf_source = hf_source
        self.unique_solution_only = unique_solution_only

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if variant == "standard":
            self.size = 9
            self.block_size = 3
        else:  # mini
            self.size = 4
            self.block_size = 2

        self.solution: np.ndarray = None  # type: ignore
        self.puzzle: np.ndarray = None  # type: ignore

        # Shuffled list of available indices (lazy initialized, pop as we use)
        self._available_indices: Optional[list[int]] = None
        self._current_source: Optional[str] = None  # Track source of current puzzle

    # ----------------- Generation Helpers -----------------

    def _pattern(self, r: int, c: int) -> int:
        """Base pattern for a valid Sudoku solution."""
        return (
            self.block_size * (r % self.block_size) + r // self.block_size + c
        ) % self.size

    def _shuffle(self, s: List[int]) -> List[int]:
        return random.sample(s, len(s))

    def _generate_full_solution(self) -> np.ndarray:
        """Generate a complete valid Sudoku grid using pattern + shuffling."""
        r_base = range(self.block_size)
        rows = [
            g * self.block_size + r
            for g in self._shuffle(list(r_base))
            for r in self._shuffle(list(r_base))
        ]
        cols = [
            g * self.block_size + c
            for g in self._shuffle(list(r_base))
            for c in self._shuffle(list(r_base))
        ]
        nums = self._shuffle(list(range(1, self.size + 1)))

        board = [[nums[self._pattern(r, c)] for c in cols] for r in rows]
        return np.array(board)

    # ----------------- Solver -----------------

    def _is_valid(self, board: np.ndarray, r: int, c: int, num: int) -> bool:
        if num in board[r, :]:
            return False
        if num in board[:, c]:
            return False
        br, bc = r - r % self.block_size, c - c % self.block_size
        if num in board[br : br + self.block_size, bc : bc + self.block_size]:
            return False
        return True

    def _count_solutions(self, board: np.ndarray, limit: int = 2) -> int:
        """Backtracking solver to count solutions, stop at `limit`."""
        for r in range(self.size):
            for c in range(self.size):
                if board[r, c] == 0:
                    for num in range(1, self.size + 1):
                        if self._is_valid(board, r, c, num):
                            board[r, c] = num
                            count = self._count_solutions(board, limit)
                            board[r, c] = 0
                            if count >= limit:
                                return count
                            if count:
                                return count
                    return 0
        return 1  # filled = solution found

    def _has_unique_solution(self, puzzle: np.ndarray) -> bool:
        """Check if a puzzle has exactly one solution."""
        return self._count_solutions(puzzle.copy(), limit=2) == 1

    # ----------------- Puzzle Creation -----------------

    def _remove_numbers(self, board: np.ndarray) -> np.ndarray:
        """Remove numbers from a filled Sudoku grid while ensuring uniqueness."""
        puzzle = board.copy()

        # Target givens depending on difficulty
        if self.difficulty == "easy":
            target_clues = int(self.size * self.size * 0.5)
        elif self.difficulty == "medium":
            target_clues = int(self.size * self.size * 0.4)
        else:  # hard
            target_clues = int(self.size * self.size * 0.3)

        cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        random.shuffle(cells)

        for r, c in cells:
            if np.count_nonzero(puzzle) <= target_clues:
                break

            backup = puzzle[r, c]
            puzzle[r, c] = 0
            if self._count_solutions(puzzle.copy(), limit=2) != 1:
                puzzle[r, c] = backup  # revert if uniqueness broken

        return puzzle

    # ----------------- Dataset Loading -----------------

    def _load_from_dataset(self) -> None:
        """Load a random puzzle from HF dataset (lazy filtering on-the-fly)."""
        ds = _load_sudoku_dataset()
        total = len(ds["train"])

        # Initialize shuffled indices on first call
        if not hasattr(self, "_available_indices") or self._available_indices is None:
            self._available_indices = list(range(total))
            random.shuffle(self._available_indices)

        # Try indices until we find a valid one or exhaust the dataset
        while self._available_indices:
            idx = self._available_indices.pop()
            entry = ds["train"][idx]

            # Check difficulty filter (supports string ranges or exact numeric match)
            if self.hf_difficulty is not None:
                entry_diff = entry["difficulty"]
                if isinstance(self.hf_difficulty, str):
                    # Map string to numeric range
                    if self.hf_difficulty not in HF_DIFFICULTY_RANGES:
                        raise ValueError(
                            f"Unknown difficulty '{self.hf_difficulty}'. "
                            f"Valid options: {list(HF_DIFFICULTY_RANGES.keys())} or (min, max) tuple."
                        )
                    min_diff, max_diff = HF_DIFFICULTY_RANGES[self.hf_difficulty]
                    if not (min_diff <= entry_diff <= max_diff):
                        continue
                elif isinstance(self.hf_difficulty, tuple):
                    # Custom (min, max) range
                    min_diff, max_diff = self.hf_difficulty
                    if not (min_diff <= entry_diff <= max_diff):
                        continue
                else:
                    raise ValueError(
                        f"hf_difficulty must be str or tuple, got {type(self.hf_difficulty)}"
                    )

            # Check source filter (HF dataset uses 'set' field)
            if self.hf_source is not None and entry.get("set") != self.hf_source:
                continue

            puzzle = _parse_puzzle_string(entry["puzzle"])

            # Check unique solution if required
            if self.unique_solution_only and not self._has_unique_solution(puzzle):
                continue

            # Found a valid puzzle
            self.puzzle = puzzle
            self.solution = _parse_puzzle_string(entry["solution"])
            self.difficulty = entry["difficulty"]
            self._current_source = entry.get("set")  # HF dataset uses 'set' field
            return

        raise ValueError(
            f"Exhausted all {total} puzzles in dataset without finding a match "
            f"(hf_difficulty={self.hf_difficulty}, hf_source={self.hf_source}, "
            f"unique_solution_only={self.unique_solution_only})"
        )

    # ----------------- Main Generation -----------------

    def generate(self) -> TaskProblem:
        if self.initialization == "generate":
            # Current behavior - generate new puzzle
            self.solution = self._generate_full_solution()
            self.puzzle = self._remove_numbers(self.solution)
            self._current_source = None
        else:  # initialization == "dataset"
            # Load from HF dataset
            self._load_from_dataset()

        return TaskProblem(
            init_grid=self.puzzle.tolist(),
            tgt_grid=self.solution.tolist(),
            intermediate_grids=None,
            task_specific_metadata=SudokuSpecificMetadata(
                size=self.size,
                block_size=self.block_size,
                difficulty=self.difficulty,
                givens=int(np.count_nonzero(self.puzzle)),
                source=self._current_source,
                from_dataset=(self.initialization == "dataset"),
            ),
        )
