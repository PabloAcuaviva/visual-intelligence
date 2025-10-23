import random
from typing import List, Optional, TypedDict

import numpy as np

from visual_intelligence.tasks.base import Task, TaskProblem

equiv_rules = {
    0: {0, 255},
    1: {1, 127},
    2: {2, 16, 191, 247},
    3: {3, 17, 63, 119},
    4: {4, 223},
    5: {5, 95},
    6: {6, 20, 159, 215},
    7: {7, 21, 31, 87},
    8: {8, 64, 239, 253},
    9: {9, 65, 111, 125},
    10: {10, 80, 175, 245},
    11: {11, 47, 81, 117},
    12: {12, 68, 207, 221},
    13: {13, 69, 79, 93},
    14: {14, 84, 143, 213},
    15: {15, 85},
    18: {18, 183},
    19: {19, 55},
    22: {22, 151},
    23: {23},
    24: {24, 66, 189, 231},
    25: {25, 61, 67, 103},
    26: {26, 82, 167, 181},
    27: {27, 39, 53, 83},
    28: {28, 70, 157, 199},
    29: {29, 71},
    30: {30, 86, 135, 149},
    32: {32, 251},
    33: {33, 123},
    34: {34, 48, 187, 243},
    35: {35, 49, 59, 115},
    36: {36, 219},
    37: {37, 91},
    38: {38, 52, 155, 211},
    40: {40, 96, 235, 249},
    41: {41, 97, 107, 121},
    42: {42, 112, 171, 241},
    43: {43, 113},
    44: {44, 100, 203, 217},
    45: {45, 75, 89, 101},
    46: {46, 116, 139, 209},
    50: {50, 179},
    51: {51},
    54: {54, 147},
    56: {56, 98, 185, 227},
    57: {57, 99},
    58: {58, 114, 163, 177},
    60: {60, 102, 153, 195},
    62: {62, 118, 131, 145},
    72: {72, 237},
    73: {73, 109},
    74: {74, 88, 173, 229},
    76: {76, 205},
    77: {77},
    78: {78, 92, 141, 197},
    90: {90, 165},
    94: {94, 133},
    104: {104, 233},
    105: {105},
    106: {106, 120, 169, 225},
    108: {108, 201},
    110: {110, 124, 137, 193},
    122: {122, 161},
    126: {126, 129},
    128: {128, 254},
    130: {130, 144, 190, 246},
    132: {132, 222},
    134: {134, 148, 158, 214},
    136: {136, 192, 238, 252},
    138: {138, 174, 208, 224},
    140: {140, 196, 206, 220},
    142: {142, 212},
    146: {146, 182},
    150: {150},
    152: {152, 188, 194, 230},
    154: {154, 166, 180, 210},
    156: {156, 198},
    160: {160, 250},
    162: {162, 176, 186, 242},
    164: {164, 218},
    168: {168, 224, 234, 248},
    170: {170, 240},
    172: {172, 202, 216, 228},
    178: {178},
    184: {184, 226},
    200: {200, 236},
    204: {204},
    232: {232},
}

wolfram_classes = {
    "class I": {0, 8, 32, 40, 128, 136, 160, 168},
    "class II": {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        19,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        33,
        34,
        35,
        36,
        37,
        38,
        42,
        43,
        44,
        46,
        50,
        51,
        56,
        57,
        58,
        62,
        72,
        73,
        74,
        76,
        77,
        78,
        94,
        104,
        108,
        130,
        132,
        134,
        138,
        140,
        142,
        152,
        154,
        156,
        162,
        164,
        170,
        172,
        178,
        184,
        200,
        204,
        232,
    },
    "class III": {18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150},
    "class IV": {41, 54, 106, 110},
}

wuensche_classes = {
    "symmetric": {
        0,
        1,
        4,
        5,
        18,
        19,
        22,
        23,
        32,
        33,
        36,
        37,
        50,
        51,
        54,
        72,
        73,
        76,
        77,
        90,
        94,
        104,
        105,
        108,
        122,
        126,
        128,
        132,
        146,
        150,
        160,
        164,
        178,
        200,
        204,
        232,
    },
    "semi-asymmetric": {
        2,
        3,
        6,
        7,
        8,
        9,
        12,
        13,
        26,
        27,
        30,
        34,
        35,
        38,
        40,
        41,
        44,
        45,
        58,
        62,
        74,
        78,
        106,
        110,
        130,
        134,
        136,
        140,
        154,
        162,
        168,
        172,
    },
    "full-asymmetric": {
        10,
        11,
        14,
        15,
        24,
        25,
        28,
        29,
        42,
        43,
        46,
        57,
        60,
        138,
        142,
        152,
        156,
        170,
        184,
    },
}


class CellularAutomata1DSpecificMetadata(TypedDict, total=False):
    rule: int
    width: int
    initialization: str
    steps: int
    k: int
    seed: Optional[int]
    rule_binary: List[int]


class CellularAutomata1D(Task):
    def __init__(
        self,
        rule: int,
        width: int = 16,
        initialization: str = "random",  # "random", "zeros", "ones", or "single_center"
        steps: int = 16,
        k: int = 1,
        seed: Optional[int] = None,
        init_state: Optional[List[int]] = None,
    ):
        """
        Initialize a 1D Cellular Automaton task.

        Args:
            rule: The rule number (0-255)
            width: The width of the domain (number of cells)
            initialization: Initialization method ("random", "zeros", "ones", "single_center")
            steps: Number of time steps to simulate
            k: Interval for outputting time points (1 = every step)
            seed: Random seed for reproducibility
            init_state: Custom initial state as list of 0s and 1s
        """
        if not (0 <= rule <= 255):
            raise ValueError("Rule must be between 0 and 255")
        if width <= 0:
            raise ValueError("Width must be positive")
        if steps <= 0:
            raise ValueError("Steps must be positive")
        if k <= 0:
            raise ValueError("k must be positive")

        self.rule = rule
        self.width = width
        self.initialization = initialization
        self.steps = steps
        self.k = k
        self.init_state = init_state
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _get_rule_binary(self) -> np.ndarray:
        """Convert rule number to binary representation."""
        return np.array(
            [int(x) for x in np.binary_repr(self.rule, width=8)], dtype=np.uint8
        )

    def _initialize_state(self) -> np.ndarray:
        """Initialize the cellular automaton state."""
        if self.init_state is not None:
            if len(self.init_state) != self.width:
                raise ValueError(
                    f"init_state length {len(self.init_state)} must match width {self.width}"
                )
            return np.array(self.init_state, dtype=np.uint8)

        if self.initialization == "random":
            return np.random.randint(2, size=self.width, dtype=np.uint8)
        elif self.initialization == "zeros":
            return np.zeros(self.width, dtype=np.uint8)
        elif self.initialization == "ones":
            return np.ones(self.width, dtype=np.uint8)
        elif self.initialization == "single_center":
            state = np.zeros(self.width, dtype=np.uint8)
            state[self.width // 2] = 1
            return state
        else:
            raise ValueError(f"Unknown initialization: {self.initialization}")

    def _cellular_automaton_step(
        self, state: np.ndarray, rule_bin: np.ndarray
    ) -> np.ndarray:
        """Perform one step of the cellular automaton evolution."""
        new_state = np.zeros(self.width, dtype=np.uint8)

        for i in range(self.width):
            left = state[(i - 1) % self.width]
            center = state[i]
            right = state[(i + 1) % self.width]

            # Create neighborhood index (left, center, right as 3-bit number)
            neighborhood = (left << 2) | (center << 1) | right

            # Apply rule
            new_state[i] = rule_bin[7 - neighborhood]

        return new_state

    def generate(self) -> TaskProblem:
        """Generate a cellular automaton problem."""
        rule_bin = self._get_rule_binary()
        state = self._initialize_state()

        # Store all states (including initial)
        states = [state.copy()]

        # Evolve the cellular automaton
        for step in range(self.steps):
            state = self._cellular_automaton_step(state, rule_bin)
            states.append(state.copy())

        # Create T x W grids where T = steps + 1 (including initial state)
        # Value mapping: 0 -> empty/not computed, 1 -> cell state 0, 2 -> cell state 1
        total_steps = len(states)

        # Initialize grids with zeros (empty/not computed)
        init_grid = [[0 for _ in range(self.width)] for _ in range(total_steps)]
        tgt_grid = [[0 for _ in range(self.width)] for _ in range(total_steps)]

        # Fill init_grid with only the first row (initial state)
        for col in range(self.width):
            init_grid[0][col] = states[0][col] + 1  # Convert 0->1, 1->2

        # Fill tgt_grid with all computed states
        for row in range(total_steps):
            for col in range(self.width):
                tgt_grid[row][col] = states[row][col] + 1  # Convert 0->1, 1->2

        # Create intermediate grids showing partial evolution
        intermediate_grids = []
        if total_steps > 2:
            for step in range(1, total_steps - 1):
                intermediate_grid = [
                    [0 for _ in range(self.width)] for _ in range(total_steps)
                ]

                # Fill up to current step
                for row in range(step + 1):
                    for col in range(self.width):
                        intermediate_grid[row][col] = (
                            states[row][col] + 1
                        )  # Convert 0->1, 1->2

                intermediate_grids.append(intermediate_grid)

        # Create metadata
        metadata: CellularAutomata1DSpecificMetadata = {
            "rule": self.rule,
            "width": self.width,
            "initialization": self.initialization,
            "steps": self.steps,
            "k": self.k,
            "seed": self.seed,
            "rule_binary": rule_bin.tolist(),
        }

        return TaskProblem(
            init_grid=init_grid,
            tgt_grid=tgt_grid,
            intermediate_grids=intermediate_grids,
            task_specific_metadata=metadata,
        )
