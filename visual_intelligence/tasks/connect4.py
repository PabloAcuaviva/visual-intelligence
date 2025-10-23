import random
from typing import List, Optional, Tuple, TypedDict

import numpy as np

from visual_intelligence.tasks.base import Task, TaskProblem

ROWS = 6
COLS = 7
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2


class Connect4SpecificMetadata(TypedDict, total=False):
    winning_move_column: int


class Connect4(Task):
    def __init__(
        self, seed: Optional[int] = None, epsilon: float = 0.03, min_turns: int = 8
    ):
        self.rows = ROWS
        self.cols = COLS
        self.seed = seed
        self.epsilon = epsilon
        self.min_turns = min_turns
        if random.seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _get_next_open_row(self, grid: np.ndarray, col: int) -> Optional[int]:
        for r in range(self.rows - 1, -1, -1):
            if grid[r, col] == EMPTY:
                return r
        return None

    def _valid_columns(self, grid: np.ndarray) -> List[int]:
        return [
            c for c in range(self.cols) if self._get_next_open_row(grid, c) is not None
        ]

    def _drop_piece(
        self, grid: np.ndarray, col: int, player: int
    ) -> Tuple[np.ndarray, Optional[int]]:
        r = self._get_next_open_row(grid, col)
        if r is None:
            return grid, None
        g = grid.copy()
        g[r, col] = player
        return g, r

    def _check_win(self, grid: np.ndarray, player: int) -> bool:
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if all(grid[r, c + i] == player for i in range(4)):
                    return True
        for r in range(self.rows - 3):
            for c in range(self.cols):
                if all(grid[r + i, c] == player for i in range(4)):
                    return True
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if all(grid[r + i, c + i] == player for i in range(4)):
                    return True
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if all(grid[r - i, c + i] == player for i in range(4)):
                    return True
        return False

    def _check_win_if_drop(self, grid: np.ndarray, col: int, player: int) -> bool:
        r = self._get_next_open_row(grid, col)
        if r is None:
            return False
        g = grid.copy()
        g[r, col] = player
        return self._check_win(g, player)

    def _score_window(self, window: np.ndarray, player: int) -> int:
        opp = PLAYER1 if player == PLAYER2 else PLAYER2
        player_count = int(np.sum(window == player))
        opp_count = int(np.sum(window == opp))
        empty_count = int(np.sum(window == EMPTY))
        score = 0
        if player_count == 4:
            score += 100000
        elif player_count == 3 and empty_count == 1:
            score += 200
        elif player_count == 2 and empty_count == 2:
            score += 30
        if opp_count == 3 and empty_count == 1:
            score -= 220
        elif opp_count == 2 and empty_count == 2:
            score -= 25
        return score

    def _score_position(self, grid: np.ndarray, player: int) -> int:
        score = 0
        center_array = grid[:, self.cols // 2]
        score += int(np.sum(center_array == player)) * 6
        for r in range(self.rows):
            row_array = grid[r, :]
            for c in range(self.cols - 3):
                window = row_array[c : c + 4]
                score += self._score_window(window, player)
        for c in range(self.cols):
            col_array = grid[:, c]
            for r in range(self.rows - 3):
                window = col_array[r : r + 4]
                score += self._score_window(window, player)
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                window = np.array([grid[r + i, c + i] for i in range(4)])
                score += self._score_window(window, player)
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                window = np.array([grid[r - i, c + i] for i in range(4)])
                score += self._score_window(window, player)
        return score

    def _ordered_columns(self, cols: List[int]) -> List[int]:
        return sorted(cols, key=lambda c: abs(c - self.cols // 2))

    def _agent_move(self, grid: np.ndarray, player: int) -> int:
        valid_cols = self._valid_columns(grid)
        if not valid_cols:
            return random.randrange(self.cols)
        for c in valid_cols:
            if self._check_win_if_drop(grid, c, player):
                return c
        opp = PLAYER1 if player == PLAYER2 else PLAYER2
        for c in valid_cols:
            if self._check_win_if_drop(grid, c, opp):
                return c
        if random.random() < self.epsilon:
            return random.choice(valid_cols)
        scored = []
        for c in self._ordered_columns(valid_cols):
            g2, _ = self._drop_piece(grid, c, player)
            opp_valid = self._valid_columns(g2)
            gives_opp_win = any(
                self._check_win_if_drop(g2, oc, opp) for oc in opp_valid
            )
            scored.append((self._score_position(g2, player), not gives_opp_win, c))
        safe = [t for t in scored if t[1]]
        pool = safe if safe else scored
        max_score = max(t[0] for t in pool)
        best = [c for s, safe_flag, c in pool if s == max_score]
        return random.choice(best)

    def _simulate_game(
        self,
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]], Optional[int]]:
        grid = np.zeros((self.rows, self.cols), dtype=int)
        moves: List[Tuple[int, int, int]] = []
        current = PLAYER1
        while True:
            valid = self._valid_columns(grid)
            if not valid:
                return grid, moves, None
            col = self._agent_move(grid, current)
            if col not in valid:
                col = random.choice(valid)
            grid, row = self._drop_piece(grid, col, current)
            moves.append((row, col, current))
            if self._check_win(grid, current):
                return grid, moves, current
            current = PLAYER2 if current == PLAYER1 else PLAYER1

    def generate(self) -> TaskProblem:
        while True:
            final_grid, moves, winner = self._simulate_game()
            if winner != PLAYER1:
                continue
            if len(moves) < self.min_turns:
                continue
            last_row, last_col, last_player = moves[-1]
            prev_grid = final_grid.copy()
            prev_grid[last_row, last_col] = EMPTY
            if not self._check_win_if_drop(prev_grid, last_col, PLAYER1):
                continue
            metadata: Connect4SpecificMetadata = {"winning_move_column": last_col}
            return TaskProblem(
                init_grid=prev_grid.tolist(),
                tgt_grid=final_grid.tolist(),
                intermediate_grids=None,
                task_specific_metadata=metadata,
            )
