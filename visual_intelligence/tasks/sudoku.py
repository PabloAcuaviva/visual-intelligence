import random
from typing import List, Optional, TypedDict

import numpy as np

from visual_intelligence.tasks.base import Task, TaskProblem


class SudokuSpecificMetadata(TypedDict):
    size: int  # 9 for standard, 4 for mini
    block_size: int  # 3 for standard, 2 for mini
    difficulty: str
    givens: int  # number of starting clues


class Sudoku(Task):
    def __init__(
        self,
        difficulty: str = "easy",
        variant: str = "standard",
        seed: Optional[int] = None,
    ):
        if difficulty not in ["easy", "medium", "hard"]:
            raise ValueError("Difficulty must be 'easy', 'medium', or 'hard'")
        if variant not in ["standard", "mini"]:
            raise ValueError("Variant must be 'standard' or 'mini'")

        self.difficulty = difficulty
        self.variant = variant
        if seed is not None:
            random.seed(seed)

        if variant == "standard":
            self.size = 9
            self.block_size = 3
        else:  # mini
            self.size = 4
            self.block_size = 2

        self.solution: np.ndarray = None  # type: ignore
        self.puzzle: np.ndarray = None  # type: ignore

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

    def generate(self) -> TaskProblem:
        self.solution = self._generate_full_solution()
        self.puzzle = self._remove_numbers(self.solution)

        return TaskProblem(
            init_grid=self.puzzle.tolist(),
            tgt_grid=self.solution.tolist(),
            intermediate_grids=None,
            task_specific_metadata=SudokuSpecificMetadata(
                size=self.size,
                block_size=self.block_size,
                difficulty=self.difficulty,
                givens=int(np.count_nonzero(self.puzzle)),
            ),
        )
