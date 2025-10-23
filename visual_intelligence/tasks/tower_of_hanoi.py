from typing import List, Tuple, TypedDict

import numpy as np

from visual_intelligence.tasks.base import Task, TaskProblem


class TowerOfHanoiSpecificMetadata(TypedDict):
    moves: List[Tuple[int, int, int]]  # (disk, from_peg, to_peg)
    num_disks: int
    num_pegs: int


class TowerOfHanoi(Task):
    BACKGROUND = 0
    # Disk IDs: 1, 2, ..., num_disks
    NUM_PEGS = 3

    def __init__(self, num_disks: int = 3):
        if not (3 <= num_disks <= 11):
            raise ValueError("num_disks must be between 3 and 11 (inclusive)")
        self.num_disks = num_disks
        self.num_pegs = self.NUM_PEGS
        # Grid size: height = num_disks, width = enough for 3 pegs and largest disk
        self.grid_height = num_disks
        # Largest disk is 2*num_disks-1 wide, plus 2 columns between pegs
        self.disk_widths = [2 * i + 1 for i in range(num_disks)]  # smallest to largest
        self.peg_x = [0, 0, 0]  # will be set below
        self.grid_width = (
            3 * (2 * num_disks - 1) + 4
        )  # 3 pegs, 2 spaces between, 1 margin each side
        # Compute peg x positions (centered for each peg)
        peg_space = 2 * num_disks - 1 + 1  # disk width + 1 space
        self.peg_x = [1 + i * peg_space + (peg_space - 1) // 2 for i in range(3)]

    def _generate_empty_grid(self):
        return np.full((self.grid_height, self.grid_width), self.BACKGROUND, dtype=int)

    def _place_disks(self, grid: np.ndarray, peg: int, disks: List[int]):
        # Place disks (bottom to top) on peg
        for i, disk in enumerate(reversed(disks)):
            width = self.disk_widths[disk - 1]
            center = self.peg_x[peg]
            y = self.grid_height - 1 - i
            left = center - width // 2
            right = center + width // 2 + 1
            grid[y, left:right] = disk

    def _generate_initial_grid(self) -> np.ndarray:
        grid = self._generate_empty_grid()
        self._place_disks(grid, peg=0, disks=list(range(1, self.num_disks + 1)))
        return grid

    def _generate_target_grid(self) -> np.ndarray:
        grid = self._generate_empty_grid()
        self._place_disks(grid, peg=2, disks=list(range(1, self.num_disks + 1)))
        return grid

    def _hanoi_moves(self, n, source, target, auxiliary, moves):
        if n == 1:
            moves.append((n, source, target))
        else:
            self._hanoi_moves(n - 1, source, auxiliary, target, moves)
            moves.append((n, source, target))
            self._hanoi_moves(n - 1, auxiliary, target, source, moves)

    def _get_move_sequence(self) -> List[Tuple[int, int, int]]:
        moves = []
        self._hanoi_moves(self.num_disks, 0, 2, 1, moves)
        # moves: (disk, from_peg, to_peg), but disk is always n in recursion, so we need to track which disk is on top
        # We'll fix this below
        # Instead, let's simulate the pegs to get the actual disk number
        pegs = [list(range(self.num_disks, 0, -1)), [], []]
        real_moves = []
        for _, from_peg, to_peg in moves:
            disk = pegs[from_peg].pop()
            pegs[to_peg].append(disk)
            real_moves.append((disk, from_peg, to_peg))
        return real_moves

    def _apply_moves_to_grids(
        self, moves: List[Tuple[int, int, int]]
    ) -> List[np.ndarray]:
        # Start from initial grid and pegs
        pegs = [list(range(self.num_disks, 0, -1)), [], []]
        grids = []
        for disk, from_peg, to_peg in moves:
            # Move disk
            pegs[from_peg].remove(disk)
            pegs[to_peg].append(disk)
            # Create grid for this step
            grid = self._generate_empty_grid()
            for peg in range(3):
                self._place_disks(grid, peg, sorted(pegs[peg]))
            grids.append(grid.copy())
        return grids

    def generate(self) -> TaskProblem:
        init_grid = self._generate_initial_grid()
        tgt_grid = self._generate_target_grid()
        moves = self._get_move_sequence()
        intermediate_grids = self._apply_moves_to_grids(moves) if moves else None
        intermediate_grids = (
            [g.tolist() for g in intermediate_grids] if intermediate_grids else None
        )
        metadata = TowerOfHanoiSpecificMetadata(
            moves=moves,
            num_disks=self.num_disks,
            num_pegs=self.num_pegs,
        )
        return TaskProblem(
            init_grid=init_grid.tolist(),
            tgt_grid=tgt_grid.tolist(),
            intermediate_grids=intermediate_grids,
            task_specific_metadata=metadata,
        )
