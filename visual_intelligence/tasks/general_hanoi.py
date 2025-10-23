import random
from collections import deque
from typing import List, Literal, Tuple, TypedDict

import numpy as np

from visual_intelligence.tasks.base import Task, TaskProblem


class TowerOfHanoiSpecificMetadata(TypedDict):
    moves: List[Tuple[int, int, int]]
    num_disks: int
    num_pegs: int


class GeneralHanoi(Task):
    BACKGROUND = 0
    NUM_PEGS = 3

    def __init__(
        self,
        num_disks: int = 3,
        seed: int | None = None,
        step: int | Literal["all"] = "all",
    ):
        if not (3 <= num_disks <= 11):
            raise ValueError("num_disks must be between 3 and 11 inclusive")
        self.num_disks = num_disks
        self.num_pegs = self.NUM_PEGS
        self.grid_height = num_disks
        self.disk_widths = [2 * i + 1 for i in range(num_disks)]
        self.peg_x = [0, 0, 0]
        self.grid_width = 3 * (2 * num_disks - 1) + 4
        peg_space = 2 * num_disks - 1 + 1
        self.peg_x = [1 + i * peg_space + (peg_space - 1) // 2 for i in range(3)]
        self._rng = random.Random(seed)

        self.step = step

    def _generate_empty_grid(self):
        return np.full((self.grid_height, self.grid_width), self.BACKGROUND, dtype=int)

    def _place_disks(self, grid: np.ndarray, peg: int, disks: List[int]):
        for i, disk in enumerate(reversed(disks)):
            width = self.disk_widths[disk - 1]
            center = self.peg_x[peg]
            y = self.grid_height - 1 - i
            left = center - width // 2
            right = center + width // 2 + 1
            grid[y, left:right] = disk

    def _state_to_pegs(self, state: Tuple[int, ...]) -> List[List[int]]:
        pegs = [[] for _ in range(self.num_pegs)]
        for d, p in enumerate(state, start=1):
            pegs[p].append(d)
        for p in range(self.num_pegs):
            pegs[p].sort()
        return pegs

    def _grid_from_state(self, state: Tuple[int, ...]) -> np.ndarray:
        grid = self._generate_empty_grid()
        pegs = self._state_to_pegs(state)
        for peg in range(self.num_pegs):
            self._place_disks(grid, peg, pegs[peg])
        return grid

    def _random_state(self) -> Tuple[int, ...]:
        while True:
            s = tuple(
                self._rng.randrange(0, self.num_pegs) for _ in range(self.num_disks)
            )
            if not all(p == self.num_pegs - 1 for p in s):
                return s

    def _neighbors(self, state: Tuple[int, ...]):
        top = [None] * self.num_pegs
        for d in range(1, self.num_disks + 1):
            p = state[d - 1]
            if top[p] is None:
                top[p] = d
        for u in range(self.num_pegs):
            du = top[u]
            if du is None:
                continue
            for v in range(self.num_pegs):
                if v == u:
                    continue
                dv = top[v]
                if dv is None or dv > du:
                    ns = list(state)
                    ns[du - 1] = v
                    yield tuple(ns), (du, u, v)

    def _shortest_moves(
        self, start: Tuple[int, ...], goal: Tuple[int, ...]
    ) -> List[Tuple[int, int, int]]:
        if start == goal:
            return []
        q = deque([start])
        prev = {start: None}
        move_rec = {}
        while q:
            s = q.popleft()
            for ns, mv in self._neighbors(s):
                if ns in prev:
                    continue
                prev[ns] = s
                move_rec[ns] = mv
                if ns == goal:
                    q.clear()
                    break
                q.append(ns)
        path_moves = []
        cur = goal
        while prev[cur] is not None:
            path_moves.append(move_rec[cur])
            cur = prev[cur]
        path_moves.reverse()
        return path_moves

    def _apply_moves_to_grids_from_state(
        self, start: Tuple[int, ...], moves: List[Tuple[int, int, int]]
    ) -> List[np.ndarray]:
        pegs = self._state_to_pegs(start)
        grids = []
        for disk, u, v in moves:
            pegs[u].remove(disk)
            pegs[v].append(disk)
            pegs[v].sort()
            grid = self._generate_empty_grid()
            for peg in range(self.num_pegs):
                self._place_disks(grid, peg, pegs[peg])
            grids.append(grid.copy())
        return grids

    def generate(self) -> TaskProblem:
        start_state = self._random_state()
        goal_state = tuple([self.num_pegs - 1] * self.num_disks)
        init_grid = self._grid_from_state(start_state)
        all_moves = self._shortest_moves(start_state, goal_state)

        # Case 1: full solution
        if self.step == "all":
            tgt_state = goal_state
            moves = all_moves
        else:
            # Case 2: step-limited
            num_steps = min(self.step, len(all_moves))
            moves = all_moves[:num_steps]

            # Apply moves to get final tgt_state
            state = list(start_state)
            for disk, u, v in moves:
                state[disk - 1] = v
            tgt_state = tuple(state)

        # Grids
        tgt_grid = self._grid_from_state(tgt_state)
        intermediate_grids = (
            self._apply_moves_to_grids_from_state(start_state, moves) if moves else None
        )
        intermediate_grids = (
            [g.tolist() for g in intermediate_grids] if intermediate_grids else None
        )

        metadata = TowerOfHanoiSpecificMetadata(
            moves=moves, num_disks=self.num_disks, num_pegs=self.num_pegs
        )
        return TaskProblem(
            init_grid=init_grid.tolist(),
            tgt_grid=tgt_grid.tolist(),
            intermediate_grids=intermediate_grids,
            task_specific_metadata=metadata,
        )
