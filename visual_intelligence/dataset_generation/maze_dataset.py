from dataclasses import dataclass

from visual_intelligence.tasks.maze import Maze

from .base import BaseDatasetGenerator
from .registry import register_dataset


@dataclass(kw_only=True)
class MazeDatasetGenerator(BaseDatasetGenerator):
    """Maze dataset generator."""

    width: int = 21
    height: int = 21

    n_train: int = 1000
    style: str = "maze"
    distance_metric: str = "jaccard_path"
    distance_threshold: float = 0.5

    def create_task(self) -> Maze:
        generate_intermediate_states = (
            self.video is not None and self.video.frames_per_intermediate > 0
        )
        return Maze(
            width=self.width,
            height=self.height,
            seed=1,
            valid_starts=(1, 1),
            valid_ends=(self.width - 2, self.height - 2),
            generate_intermediate_states=generate_intermediate_states,
        )

    @property
    def dataset_name(self) -> str:
        return "maze"


@dataclass(kw_only=True)
class SmallMazeDatasetGenerator(MazeDatasetGenerator):
    """Small maze dataset generator."""

    width: int = 13
    height: int = 13
    image_width: int = 336
    image_height: int = 336

    @property
    def dataset_name(self) -> str:
        return "maze_small"


@register_dataset("maze")
def generate_maze_dataset(**kwargs):
    MazeDatasetGenerator(**kwargs).generate()


@register_dataset("maze_small")
def generate_small_maze_dataset(**kwargs):
    SmallMazeDatasetGenerator(**kwargs).generate()
