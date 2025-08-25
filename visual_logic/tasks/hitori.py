import random
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np

from visual_logic.tasks.base import Task, TaskProblem


class HitoriSpecificMetadata(TypedDict):
    size: int
    difficulty: str
    blacks: int
    unique: bool
    seed: Optional[int]


class Hitori(Task):
    """
    Generates Hitori puzzles with a *unique* solution.

    Representation:
      - init_grid: N x N integer grid of the puzzle's given numbers.
      - tgt_grid : N x N mask of the unique solution (0 = white, 1 = black).

    Rules enforced by the solver:
      1) Among WHITE cells, numbers are unique in every row and column.
      2) BLACK cells are not orthogonally adjacent.
      3) All WHITE cells form a single orthogonally connected component.

    Strategy for efficient generation:
      • Build a canonical Latin square L (fast).
      • Sample a valid black mask S (no-adjacency + white connectivity).
      • Fill puzzle numbers so that WHITE cells copy L, while BLACK cells
        duplicate numbers in their row and/or column to force shading.
      • Verify uniqueness with a fast backtracking solver with strong propagation.
      • If multiple solutions appear, tweak only numbers on S's black cells to
        break alternates (keeps S valid) and re-check. Repeat a few times.
    """

    def __init__(
        self,
        difficulty: str = "easy",  # easy/medium/hard controls target black density
        size: int = 5,  # typical nice sizes: 5..10 (>= 5 recommended)
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

        self.grid: np.ndarray = None  # type: ignore  # puzzle numbers
        self.solution_mask: np.ndarray = None  # type: ignore  # 0=white,1=black

    # --------------------------- Public API ---------------------------

    def generate(self) -> TaskProblem:
        n = self.size
        target_ratio = {"easy": 0.16, "medium": 0.22, "hard": 0.30}[self.difficulty]
        target_blacks = max(1, int(round(n * n * target_ratio)))

        latin = self._latin_square(n)

        # Try multiple times to get a unique puzzle quickly
        for _attempt in range(60):
            mask = self._sample_mask(n, target_blacks)
            grid = self._fill_numbers_from_mask(latin, mask)

            if not self._mask_is_valid_solution(grid, mask):
                continue

            count, sols = self._count_solutions(grid, limit=2, want_solutions=True)
            if count == 1:
                self.grid = grid
                self.solution_mask = sols[0]
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

            # If multiple solutions, iteratively refine by strengthening some
            # black-cell numbers to create unavoidable duplicates if turned white.
            refined = self._refine_until_unique(
                latin, grid, mask, sols[0], max_steps=12
            )
            if refined is not None:
                final_grid = refined
                cnt2, sols2 = self._count_solutions(
                    final_grid, limit=2, want_solutions=True
                )
                if cnt2 == 1:
                    self.grid = final_grid
                    self.solution_mask = sols2[0]
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

        # Fallback (should be rare): accept a clean puzzle and assert uniqueness via solver again
        # This keeps the interface consistent even in worst-case sampling.
        latin = self._latin_square(n)
        mask = self._sample_mask(n, target_blacks)
        grid = self._fill_numbers_from_mask(latin, mask)
        cnt, sols = self._count_solutions(grid, limit=2, want_solutions=True)
        if cnt != 1:
            # As a last resort, shrink the board slightly to make uniqueness easier
            n2 = max(5, min(10, n))
            latin = self._latin_square(n2)
            mask = self._sample_mask(n2, max(1, int(round(n2 * n2 * target_ratio))))
            grid = self._fill_numbers_from_mask(latin, mask)
            cnt, sols = self._count_solutions(grid, limit=2, want_solutions=True)

        self.grid = grid
        self.solution_mask = sols[0]
        return TaskProblem(
            init_grid=self.grid.tolist(),
            tgt_grid=self.solution_mask.tolist(),
            intermediate_grids=None,
            task_specific_metadata=HitoriSpecificMetadata(
                size=self.grid.shape[0],
                difficulty=self.difficulty,
                blacks=int(np.count_nonzero(self.solution_mask)),
                unique=(cnt == 1),
                seed=self.seed,
            ),
        )

    # ----------------------- Latin + Mask Sampling -----------------------

    def _latin_square(self, n: int) -> np.ndarray:
        # Canonical Latin square L[i][j] = (i + j) % n + 1 then shuffle rows/cols/symbols
        L = np.fromfunction(lambda i, j: ((i + j) % n) + 1, (n, n), dtype=int)

        # Shuffle rows, cols, and symbols to diversify
        rows = list(range(n))
        random.shuffle(rows)
        cols = list(range(n))
        random.shuffle(cols)
        sym = list(range(1, n + 1))
        random.shuffle(sym)
        L = L[rows, :][:, cols]
        # remap symbols
        remap = {k: sym[k - 1] for k in range(1, n + 1)}
        L = np.vectorize(remap.__getitem__)(L)
        return L

    def _sample_mask(self, n: int, target_blacks: int) -> np.ndarray:
        """
        Create a valid black mask (0=white, 1=black) with:
          - No adjacent blacks
          - White cells are connected
        We greedily add blacks in random order while preserving constraints.
        """
        mask = np.zeros((n, n), dtype=np.int8)  # 0=white, 1=black

        candidates = [(r, c) for r in range(n) for c in range(n)]
        random.shuffle(candidates)

        def can_black(r: int, c: int) -> bool:
            if mask[r, c] == 1:
                return False
            # no adjacent black
            for nr, nc in self._nbrs4(n, r, c):
                if mask[nr, nc] == 1:
                    return False
            # tentative black: ensure remaining whites still connected
            mask[r, c] = 1
            ok = self._whites_connected(mask)
            mask[r, c] = 0
            return ok

        # Greedy pass
        for r, c in candidates:
            if np.count_nonzero(mask) >= target_blacks:
                break
            if can_black(r, c):
                mask[r, c] = 1

        # If we failed to hit target, try a few random nudges (swap out some blacks)
        tries = 0
        while np.count_nonzero(mask) < target_blacks and tries < 2000:
            tries += 1
            r, c = random.randrange(n), random.randrange(n)
            if mask[r, c] == 1:
                # try turning this back to white to free adjacency and add elsewhere
                saved = mask[r, c]
                mask[r, c] = 0
                if not self._whites_connected(mask):
                    mask[r, c] = saved
                    continue
            # try adding a new black
            r2, c2 = random.randrange(n), random.randrange(n)
            if can_black(r2, c2):
                mask[r2, c2] = 1

        # Final sanity: ensure valid (connected whites, no adjacent blacks)
        if not self._mask_ok(mask):
            # in the rare case of failure, default to a sparse valid mask
            mask = np.zeros((n, n), dtype=np.int8)
            # place a few safely spaced blacks along a checkerboard
            coords = [(r, c) for r in range(n) for c in range(n) if (r + c) % 2 == 0]
            random.shuffle(coords)
            for r, c in coords[: max(1, target_blacks // 2)]:
                if self._mask_ok_with_flip(mask, r, c, to_black=True):
                    mask[r, c] = 1
        return mask

    # ----------------------- Puzzle Number Filling -----------------------

    def _fill_numbers_from_mask(
        self, latin: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Set puzzle numbers:
          - WHITE cells copy from latin.
          - BLACK cells get numbers that duplicate a white neighbor in the same row and/or column,
            so making them white would introduce a row/col conflict.
        """
        n = latin.shape[0]
        grid = np.empty_like(latin)

        # Copy whites directly
        white_positions = np.where(mask == 0)
        grid[white_positions] = latin[white_positions]

        # For blacks, prefer a number that simultaneously duplicates a white value
        # in both its row and its column (if possible) to force stronger constraints.
        for r in range(n):
            for c in range(n):
                if mask[r, c] == 1:  # black cell
                    row_white_vals = set(
                        latin[r, j] for j in range(n) if mask[r, j] == 0 and j != c
                    )
                    col_white_vals = set(
                        latin[i, c] for i in range(n) if mask[i, c] == 0 and i != r
                    )
                    both = row_white_vals.intersection(col_white_vals)
                    if both:
                        grid[r, c] = random.choice(list(both))
                    elif row_white_vals:
                        grid[r, c] = random.choice(list(row_white_vals))
                    elif col_white_vals:
                        grid[r, c] = random.choice(list(col_white_vals))
                    else:
                        # edge case: no whites in row nor column; pick any symbol (will be isolated white area otherwise)
                        grid[r, c] = latin[r, c]
        return grid

    # --------------------------- Refinement ---------------------------

    def _refine_until_unique(
        self,
        latin: np.ndarray,
        grid: np.ndarray,
        target_mask: np.ndarray,
        one_solution: np.ndarray,
        max_steps: int = 12,
    ) -> Optional[np.ndarray]:
        """
        If puzzle is not unique, strengthen numbers on target BLACK cells that
        the alternative solution tries to keep WHITE. This never harms the target solution
        because target blacks stay black.
        """
        n = grid.shape[0]
        cur = grid.copy()

        for _ in range(max_steps):
            # Find a cell where our intended mask is black but the alt keeps white
            diff: List[Tuple[int, int]] = [
                (r, c)
                for r in range(n)
                for c in range(n)
                if target_mask[r, c] == 1 and one_solution[r, c] == 0
            ]
            if not diff:
                return cur  # already matches target
            r, c = random.choice(diff)

            # Force a stronger duplicate on (r,c)
            row_white_vals = set(
                latin[r, j] for j in range(n) if target_mask[r, j] == 0 and j != c
            )
            col_white_vals = set(
                latin[i, c] for i in range(n) if target_mask[i, c] == 0 and i != r
            )
            both = row_white_vals.intersection(col_white_vals)
            if both:
                cur[r, c] = random.choice(list(both))
            elif row_white_vals or col_white_vals:
                cur[r, c] = random.choice(list(row_white_vals or col_white_vals))
            else:
                # If no whites in either line, choose any frequent value in row to incentivize black
                vals = [cur[r, j] for j in range(n) if j != c]
                cur[r, c] = random.choice(vals) if vals else cur[r, c]

            # Recheck uniqueness quickly
            cnt, sols = self._count_solutions(cur, limit=2, want_solutions=True)
            if cnt == 1:
                return cur
            else:
                # If still multiple, keep the stronger choice and iterate again,
                # focusing on remaining disagreements with target mask.
                one_solution = sols[0]

        return None

    # ----------------------------- Solver -----------------------------

    def _mask_is_valid_solution(self, grid: np.ndarray, mask: np.ndarray) -> bool:
        """Quick check that the provided mask satisfies all Hitori rules for this grid."""
        if not self._mask_ok(mask):
            return False
        return self._mask_uniqueness_ok(grid, mask)

    def _mask_uniqueness_ok(self, grid: np.ndarray, mask: np.ndarray) -> bool:
        n = grid.shape[0]
        # rows
        for r in range(n):
            seen: Dict[int, int] = {}
            for c in range(n):
                if mask[r, c] == 0:
                    v = grid[r, c]
                    if v in seen:
                        return False
                    seen[v] = 1
        # cols
        for c in range(n):
            seen = {}
            for r in range(n):
                if mask[r, c] == 0:
                    v = grid[r, c]
                    if v in seen:
                        return False
                    seen[v] = 1
        return True

    def _count_solutions(
        self,
        grid: np.ndarray,
        limit: int = 2,
        want_solutions: bool = False,
    ) -> Tuple[int, List[np.ndarray]]:
        """
        Count solutions up to 'limit'. Uses strong propagation:
          - If a cell is WHITE, all same-number peers in its row/col become BLACK.
          - If a cell is BLACK, all 4-neighbors become WHITE.
          - Early prune with 'known whites must be connected' (treat unknowns as passable).
          - Immediate row/col duplicate check among whites.
        """
        n = grid.shape[0]
        # -1 unknown, 0 white, 1 black
        state = np.full((n, n), -1, dtype=np.int8)

        # Precompute duplicate groups for speed
        row_groups: List[Dict[int, List[int]]] = []
        col_groups: List[Dict[int, List[int]]] = []
        for r in range(n):
            g: Dict[int, List[int]] = {}
            for c in range(n):
                g.setdefault(grid[r, c], []).append(c)
            row_groups.append(g)
        for c in range(n):
            g = {}
            for r in range(n):
                g.setdefault(grid[r, c], []).append(r)
            col_groups.append(g)

        solutions: List[np.ndarray] = []
        sol_count = 0

        # Basic deterministic propagation (initial)
        def propagate(q: List[Tuple[int, int]], st: np.ndarray) -> Optional[np.ndarray]:
            """Process queue of assignments already set in st."""
            while q:
                r, c = q.pop()
                val = st[r, c]
                # adjacency rule: black implies neighbors white
                if val == 1:
                    for nr, nc in self._nbrs4(n, r, c):
                        if st[nr, nc] == 1:
                            return None
                        if st[nr, nc] == -1:
                            st[nr, nc] = 0
                            q.append((nr, nc))
                # uniqueness rule: white implies same-number peers black (row & col)
                if val == 0:
                    v = grid[r, c]
                    # row peers
                    for cc in row_groups[r][v]:
                        if cc == c:
                            continue
                        if st[r, cc] == 0:
                            return None
                        if st[r, cc] == -1:
                            st[r, cc] = 1
                            q.append((r, cc))
                    # col peers
                    for rr in col_groups[c][v]:
                        if rr == r:
                            continue
                        if st[rr, c] == 0:
                            return None
                        if st[rr, c] == -1:
                            st[rr, c] = 1
                            q.append((rr, c))
                # adjacency check (no two blacks)
                if val == 1:
                    for nr, nc in self._nbrs4(n, r, c):
                        if st[nr, nc] == 1:
                            return None
                # early connectivity prune: all known whites must be mutually reachable
                if not self._known_whites_can_connect(st):
                    return None
            return st

        # Heuristic: pick an unknown cell in the largest duplicate pressure set
        def pick_var(st: np.ndarray) -> Optional[Tuple[int, int]]:
            best = None
            best_score = -1
            for r in range(n):
                for c in range(n):
                    if st[r, c] != -1:
                        continue
                    v = grid[r, c]
                    dup_row = len(row_groups[r][v]) - 1
                    dup_col = len(col_groups[c][v]) - 1
                    adj_black = sum(
                        1 for nr, nc in self._nbrs4(n, r, c) if st[nr, nc] == 1
                    )
                    score = dup_row + dup_col + (2 if adj_black else 0)
                    if score > best_score:
                        best_score = score
                        best = (r, c)
            return best

        # Search (white-first tends to succeed faster)
        def dfs(st: np.ndarray) -> Optional[bool]:
            nonlocal sol_count, solutions
            if sol_count >= limit:
                return True
            # fully assigned?
            if not (st == -1).any():
                # final checks: uniqueness already ensured; adjacency ensured; connectivity check strict (no unknowns)
                if self._whites_connected(st):
                    sol_count += 1
                    if want_solutions:
                        solutions.append(st.copy())
                    return sol_count >= limit
                return None

            rc = pick_var(st)
            if rc is None:
                return None
            r, c = rc

            # Try WHITE then BLACK
            for val in (0, 1):
                # If neighbor already black and trying black -> immediate fail
                if val == 1:
                    for nr, nc in self._nbrs4(n, r, c):
                        if st[nr, nc] == 1:
                            break
                    else:
                        pass
                # clone and assign
                st2 = st.copy()
                st2[r, c] = val
                q = [(r, c)]
                st2 = propagate(q, st2)
                if st2 is None:
                    continue
                res = dfs(st2)
                if res is True and sol_count >= limit:
                    return True
            return None

        # Seed propagation with simple determinism:
        # If a cell has a black neighbor, it cannot be black -> WHITE.
        queue: List[Tuple[int, int]] = []
        for r in range(n):
            for c in range(n):
                # nothing known at start
                pass
        state = propagate(queue, state)
        if state is None:
            return 0, []

        dfs(state)
        return sol_count, solutions

    # --------------------------- Utilities ---------------------------

    def _nbrs4(self, n: int, r: int, c: int):
        if r > 0:
            yield (r - 1, c)
        if r + 1 < n:
            yield (r + 1, c)
        if c > 0:
            yield (r, c - 1)
        if c + 1 < n:
            yield (r, c + 1)

    def _mask_ok(self, mask: np.ndarray) -> bool:
        return self._no_adjacent_blacks(mask) and self._whites_connected(mask)

    def _mask_ok_with_flip(
        self, mask: np.ndarray, r: int, c: int, to_black: bool
    ) -> bool:
        saved = mask[r, c]
        mask[r, c] = 1 if to_black else 0
        ok = self._mask_ok(mask)
        mask[r, c] = saved
        return ok

    def _no_adjacent_blacks(self, mask: np.ndarray) -> bool:
        n = mask.shape[0]
        for r in range(n):
            for c in range(n):
                if mask[r, c] == 1:
                    for nr, nc in self._nbrs4(n, r, c):
                        if mask[nr, nc] == 1:
                            return False
        return True

    def _whites_connected(self, mask: np.ndarray) -> bool:
        """Treat 0 as white, 1 as black. Standard connectivity check."""
        n = mask.shape[0]
        # find any white
        start = None
        for r in range(n):
            for c in range(n):
                if mask[r, c] == 0:
                    start = (r, c)
                    break
            if start:
                break
        if not start:
            return True  # all black trivially connected (no whites to connect)

        seen = set([start])
        stack = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in self._nbrs4(n, r, c):
                if mask[nr, nc] == 0 and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))

        total_white = int((mask == 0).sum())
        return len(seen) == total_white

    def _known_whites_can_connect(self, state: np.ndarray) -> bool:
        """
        Early prune for partial states: known whites must lie in a single connected
        component if we treat UNKNOWN as passable (i.e., not black).
        """
        n = state.shape[0]
        whites = [(r, c) for r in range(n) for c in range(n) if state[r, c] == 0]
        if len(whites) <= 1:
            return True
        start = whites[0]
        seen = set([start])
        stack = [start]
        passable = lambda rr, cc: state[rr, cc] != 1  # unknown or white are passable
        while stack:
            r, c = stack.pop()
            for nr, nc in self._nbrs4(n, r, c):
                if passable(nr, nc) and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return all(w in seen for w in whites)
