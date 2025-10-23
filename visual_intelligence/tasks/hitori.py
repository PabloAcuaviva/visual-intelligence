import random
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np

from visual_intelligence.tasks.base import Task, TaskProblem


class HitoriSpecificMetadata(TypedDict):
    size: int
    difficulty: str
    blacks: int
    unique: bool
    seed: Optional[int]


class Hitori(Task):
    """
    Generates Hitori puzzles with guaranteed UNIQUE solutions.

    Key approach:
    1. Generate a valid black/white pattern
    2. Assign numbers ensuring:
       - White cells have no duplicates in rows/columns
       - Black cells MUST have duplicates that force them to be black
       - No cell can be optionally black (no trivial additional solutions)
    3. Verify uniqueness with exhaustive solver
    """

    def __init__(
        self,
        difficulty: str = "easy",
        size: int = 5,
        seed: Optional[int] = None,
    ):
        if difficulty not in ["easy", "medium", "hard"]:
            raise ValueError("Difficulty must be 'easy', 'medium', or 'hard'")
        if not (3 <= size <= 14):
            raise ValueError("Size must be between 3 and 14")
        self.difficulty = difficulty
        self.size = size
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.grid: np.ndarray = None  # type: ignore
        self.solution_mask: np.ndarray = None  # type: ignore

    def generate(self) -> TaskProblem:
        n = self.size
        target_ratio = {"easy": 0.25, "medium": 0.30, "hard": 0.4}[self.difficulty]
        target_blacks = max(1, int(round(n * n * target_ratio)))

        max_attempts = 3000
        for attempt in range(max_attempts):
            # Generate a valid black pattern
            mask = self._generate_valid_mask(n, target_blacks)
            if mask is None:
                continue

            # Create puzzle that REQUIRES this exact mask
            grid = self._create_forced_puzzle(mask)
            if grid is None:
                continue

            # Verify uniqueness
            solutions = self._exhaustive_solve(grid, max_solutions=2)

            if len(solutions) == 1:
                self.grid = grid
                self.solution_mask = solutions[0]
                return TaskProblem(
                    init_grid=self.grid.tolist(),
                    tgt_grid=self.solution_mask.tolist(),
                    intermediate_grids=None,
                    task_specific_metadata=HitoriSpecificMetadata(
                        size=n,
                        difficulty=self.difficulty,
                        blacks=int(np.count_nonzero(self.solution_mask)),
                        unique=True,
                        seed=self.seed,
                    ),
                )

        # Fallback with simpler puzzle
        raise ValueError("Not able to generate in given attemps.")
        mask = self._generate_simple_valid_mask(n, max(1, target_blacks // 2))
        grid = self._create_forced_puzzle(mask)
        solutions = self._exhaustive_solve(grid, max_solutions=1)

        self.grid = grid
        self.solution_mask = solutions[0] if solutions else mask
        return TaskProblem(
            init_grid=self.grid.tolist(),
            tgt_grid=self.solution_mask.tolist(),
            intermediate_grids=None,
            task_specific_metadata=HitoriSpecificMetadata(
                size=n,
                difficulty=self.difficulty,
                blacks=int(np.count_nonzero(self.solution_mask)),
                unique=len(solutions) == 1,
                seed=self.seed,
            ),
        )

    def _generate_valid_mask(self, n: int, target_blacks: int) -> Optional[np.ndarray]:
        """Generate a valid black/white pattern."""
        for _ in range(100):
            mask = np.zeros((n, n), dtype=int)

            # Start with random positions
            positions = [(i, j) for i in range(n) for j in range(n)]
            random.shuffle(positions)

            for r, c in positions:
                if np.sum(mask) >= target_blacks:
                    break

                # Check if we can add black here
                temp_mask = mask.copy()
                temp_mask[r, c] = 1

                if self._is_valid_mask(temp_mask):
                    mask[r, c] = 1

            if self._is_valid_mask(mask) and np.sum(mask) >= max(1, target_blacks // 2):
                return mask

        return None

    def _generate_simple_valid_mask(self, n: int, target_blacks: int) -> np.ndarray:
        """Generate a simple but guaranteed valid mask."""
        mask = np.zeros((n, n), dtype=int)

        # Use checkerboard pattern with spacing
        positions = []
        for i in range(0, n, 2):
            for j in range(0, n, 2):
                if i + j > 0:  # Skip (0,0) to ensure connectivity
                    positions.append((i, j))

        random.shuffle(positions)
        for r, c in positions[: min(len(positions), target_blacks)]:
            temp_mask = mask.copy()
            temp_mask[r, c] = 1
            if self._is_valid_mask(temp_mask):
                mask[r, c] = 1

        return mask

    def _is_valid_mask(self, mask: np.ndarray) -> bool:
        """Check if mask satisfies all Hitori structural rules."""
        n = mask.shape[0]

        # Rule 1: No adjacent blacks
        for i in range(n):
            for j in range(n):
                if mask[i, j] == 1:
                    # Check all 4 neighbors
                    neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                    for ni, nj in neighbors:
                        if 0 <= ni < n and 0 <= nj < n and mask[ni, nj] == 1:
                            return False

        # Rule 2: All whites must be connected
        whites = [(i, j) for i in range(n) for j in range(n) if mask[i, j] == 0]
        if not whites:
            return False

        # BFS to check connectivity
        visited = set()
        queue = [whites[0]]
        visited.add(whites[0])

        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < n
                    and 0 <= nc < n
                    and mask[nr, nc] == 0
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return len(visited) == len(whites)

    def _create_forced_puzzle(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Create a puzzle where the given mask is the ONLY valid solution.
        Key: Every black cell must have duplicates that force it to be black.
        Every white cell that could potentially be black must have a reason not to be.
        """
        n = mask.shape[0]
        grid = np.zeros((n, n), dtype=int)

        # Step 1: Assign values to white cells ensuring uniqueness in rows/columns
        white_grid = self._create_white_assignment(mask)
        if white_grid is None:
            return None

        grid[mask == 0] = white_grid[mask == 0]

        # Step 2: Assign values to black cells that FORCE them to be black
        for i in range(n):
            for j in range(n):
                if mask[i, j] == 1:
                    value = self._get_forcing_value(grid, mask, i, j)
                    if value is None:
                        return None
                    grid[i, j] = value

        # Step 3: Verify no optional blacks exist
        if not self._verify_no_optional_blacks(grid, mask):
            return None

        return grid

    def _create_white_assignment(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Create valid number assignment for white cells."""
        n = mask.shape[0]
        grid = np.zeros((n, n), dtype=int)

        # Use constraint satisfaction to assign values
        white_cells = [(i, j) for i in range(n) for j in range(n) if mask[i, j] == 0]

        def backtrack(idx: int) -> bool:
            if idx == len(white_cells):
                return True

            r, c = white_cells[idx]
            used_values = set()

            # Check row
            for j in range(n):
                if j != c and mask[r, j] == 0 and grid[r, j] > 0:
                    used_values.add(grid[r, j])

            # Check column
            for i in range(n):
                if i != r and mask[i, c] == 0 and grid[i, c] > 0:
                    used_values.add(grid[i, c])

            # Try values
            available = [v for v in range(1, n + 1) if v not in used_values]
            random.shuffle(available)

            for val in available:
                grid[r, c] = val
                if backtrack(idx + 1):
                    return True
                grid[r, c] = 0

            return False

        if backtrack(0):
            return grid
        return None

    def _get_forcing_value(
        self, grid: np.ndarray, mask: np.ndarray, r: int, c: int
    ) -> Optional[int]:
        """
        Get a value for black cell (r,c) that FORCES it to be black.
        The value must create a duplicate in its row or column with a white cell.
        """
        n = grid.shape[0]

        # Get white values in same row and column
        row_white_vals = []
        col_white_vals = []

        for j in range(n):
            if j != c and mask[r, j] == 0:
                row_white_vals.append(grid[r, j])

        for i in range(n):
            if i != r and mask[i, c] == 0:
                col_white_vals.append(grid[i, c])

        # Best case: value that duplicates in both row and column
        common = set(row_white_vals) & set(col_white_vals)
        if common:
            return random.choice(list(common))

        # Good case: value that duplicates in row
        if row_white_vals:
            return random.choice(row_white_vals)

        # Acceptable case: value that duplicates in column
        if col_white_vals:
            return random.choice(col_white_vals)

        # Problem: This black cell has no whites in its row or column
        # This shouldn't happen with a valid mask, but handle it
        return None

    def _verify_no_optional_blacks(self, grid: np.ndarray, mask: np.ndarray) -> bool:
        """
        Verify that no white cell can be optionally turned black.
        A white cell can be turned black only if:
        1. It doesn't create adjacent blacks
        2. It doesn't break white connectivity
        3. It has a duplicate in its row or column
        """
        n = grid.shape[0]

        for i in range(n):
            for j in range(n):
                if mask[i, j] == 0:  # White cell
                    # Check if this white cell could be made black

                    # Would it create adjacent blacks?
                    has_black_neighbor = False
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n and mask[ni, nj] == 1:
                            has_black_neighbor = True
                            break

                    if has_black_neighbor:
                        continue  # Can't be made black due to adjacency

                    # Would it break connectivity?
                    temp_mask = mask.copy()
                    temp_mask[i, j] = 1
                    if not self._is_valid_mask(temp_mask):
                        continue  # Can't be made black due to connectivity

                    # Does it have NO duplicates? (if so, it CANNOT be black)
                    val = grid[i, j]
                    has_row_dup = any(grid[i, k] == val for k in range(n) if k != j)
                    has_col_dup = any(grid[k, j] == val for k in range(n) if k != i)

                    if not has_row_dup and not has_col_dup:
                        continue  # Good - this white cell cannot be made black

                    # Problem: This white cell COULD be made black!
                    # Try to fix by changing nearby black cells' values
                    if not self._fix_optional_black(grid, mask, i, j):
                        return False

        return True

    def _fix_optional_black(
        self, grid: np.ndarray, mask: np.ndarray, wr: int, wc: int
    ) -> bool:
        """Try to fix a white cell that could optionally be black."""
        n = grid.shape[0]
        val = grid[wr, wc]

        # Find black cells in same row/column that we could modify
        for j in range(n):
            if mask[wr, j] == 1 and grid[wr, j] != val:
                # Change this black cell to not duplicate with our white cell
                old_val = grid[wr, j]
                new_val = self._get_forcing_value(grid, mask, wr, j)
                if new_val and new_val != val:
                    grid[wr, j] = new_val
                    return True
                grid[wr, j] = old_val

        for i in range(n):
            if mask[i, wc] == 1 and grid[i, wc] != val:
                old_val = grid[i, wc]
                new_val = self._get_forcing_value(grid, mask, i, wc)
                if new_val and new_val != val:
                    grid[i, wc] = new_val
                    return True
                grid[i, wc] = old_val

        return False

    def _exhaustive_solve(
        self, grid: np.ndarray, max_solutions: int = 2
    ) -> List[np.ndarray]:
        """
        Exhaustively solve the puzzle using backtracking with strong pruning.
        Returns all solutions up to max_solutions.
        """
        n = grid.shape[0]
        solutions = []

        def is_valid_partial(mask: np.ndarray, decided: np.ndarray) -> bool:
            """Check if partial solution is valid so far."""
            # Check no adjacent blacks among decided cells
            for i in range(n):
                for j in range(n):
                    if decided[i, j] and mask[i, j] == 1:
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = i + di, j + dj
                            if (
                                0 <= ni < n
                                and 0 <= nj < n
                                and decided[ni, nj]
                                and mask[ni, nj] == 1
                            ):
                                return False

            # Check no duplicate whites in rows/columns
            for i in range(n):
                row_whites = {}
                col_whites = {}
                for j in range(n):
                    # Row check
                    if decided[i, j] and mask[i, j] == 0:
                        val = grid[i, j]
                        if val in row_whites:
                            return False
                        row_whites[val] = j
                    # Column check
                    if decided[j, i] and mask[j, i] == 0:
                        val = grid[j, i]
                        if val in col_whites:
                            return False
                        col_whites[val] = j

            # Check potential connectivity (whites + undecided can connect)
            whites = [
                (i, j)
                for i in range(n)
                for j in range(n)
                if decided[i, j] and mask[i, j] == 0
            ]
            if not whites:
                return True

            visited = set()
            queue = [whites[0]]
            visited.add(whites[0])

            while queue:
                r, c = queue.pop(0)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visited:
                        if (not decided[nr, nc]) or mask[nr, nc] == 0:
                            visited.add((nr, nc))
                            queue.append((nr, nc))

            return all(w in visited for w in whites)

        def get_forced_values(
            mask: np.ndarray, decided: np.ndarray
        ) -> Dict[Tuple[int, int], int]:
            """Determine cells that must have specific values."""
            forced = {}

            for i in range(n):
                for j in range(n):
                    if decided[i, j]:
                        continue

                    must_be_white = False
                    must_be_black = False

                    # Check if must be white (has black neighbor)
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if (
                            0 <= ni < n
                            and 0 <= nj < n
                            and decided[ni, nj]
                            and mask[ni, nj] == 1
                        ):
                            must_be_white = True
                            break

                    if must_be_white:
                        forced[(i, j)] = 0
                        continue

                    # Check if must be black (white with same value exists)
                    val = grid[i, j]
                    for k in range(n):
                        if (
                            k != j
                            and decided[i, k]
                            and mask[i, k] == 0
                            and grid[i, k] == val
                        ):
                            must_be_black = True
                            break
                        if (
                            k != i
                            and decided[k, j]
                            and mask[k, j] == 0
                            and grid[k, j] == val
                        ):
                            must_be_black = True
                            break

                    if must_be_black:
                        forced[(i, j)] = 1

            return forced

        def solve(mask: np.ndarray, decided: np.ndarray) -> None:
            nonlocal solutions

            if len(solutions) >= max_solutions:
                return

            # Apply forced moves
            changed = True
            while changed:
                changed = False
                forced = get_forced_values(mask, decided)
                for (r, c), val in forced.items():
                    if not decided[r, c]:
                        mask[r, c] = val
                        decided[r, c] = True
                        changed = True

                if not is_valid_partial(mask, decided):
                    return

            # Check if complete
            if np.all(decided):
                if self._is_valid_mask(mask):
                    solutions.append(mask.copy())
                return

            # Choose next cell (minimum remaining values heuristic)
            best_cell = None
            best_score = float("inf")

            for i in range(n):
                for j in range(n):
                    if not decided[i, j]:
                        score = 0
                        val = grid[i, j]

                        # Count duplicates (higher = more constrained)
                        for k in range(n):
                            if k != j and grid[i, k] == val:
                                score += 2
                            if k != i and grid[k, j] == val:
                                score += 2

                        # Prefer cells near decided cells
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < n and 0 <= nj < n and decided[ni, nj]:
                                score += 1

                        if score < best_score or (
                            score == best_score and random.random() < 0.5
                        ):
                            best_score = score
                            best_cell = (i, j)

            if best_cell is None:
                return

            r, c = best_cell

            # Try both values
            for value in [0, 1]:  # Try white first (usually more constrained)
                new_mask = mask.copy()
                new_decided = decided.copy()
                new_mask[r, c] = value
                new_decided[r, c] = True

                if is_valid_partial(new_mask, new_decided):
                    solve(new_mask, new_decided)

                if len(solutions) >= max_solutions:
                    return

        initial_mask = np.full((n, n), -1, dtype=int)
        decided = np.zeros((n, n), dtype=bool)
        solve(initial_mask, decided)

        return solutions
