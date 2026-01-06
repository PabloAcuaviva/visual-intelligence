from visual_intelligence.tasks.problem_set import VideoConfig

from .base import (
    DISTANCE_FUNCTIONS,
    STYLES,
    ArcReducedStyle,
    BaseDatasetGenerator,
    get_distance_fn,
    get_style,
)
from .cellular_automata import CellularAutomata1DDatasetGenerator
from .chess_dataset import ChessMateDatasetGenerator
from .connect4_dataset import Connect4DatasetGenerator
from .game_of_life_dataset import GameOfLifeDatasetGenerator
from .general_hanoi_dataset import GeneralHanoiDatasetGenerator
from .hitori_dataset import HitoriDatasetGenerator
from .langton_ant_dataset import LangtonAntDatasetGenerator
from .maze_dataset import MazeDatasetGenerator, SmallMazeDatasetGenerator
from .navigation2d_dataset import (
    Navigation2DAnyToAnyDatasetGenerator,
    Navigation2DDatasetGenerator,
    ShortestPathDatasetGenerator,
)
from .sudoku_dataset import SudokuDatasetGenerator
from .tower_of_hanoi_dataset import TowerOfHanoiDatasetGenerator
