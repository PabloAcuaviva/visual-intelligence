from dataclasses import dataclass
from typing import Optional, Tuple

from visual_intelligence.tasks.navigation2d import Navigation2D

from .base import BaseDatasetGenerator
from .registry import register_dataset


@dataclass(kw_only=True)
class Navigation2DDatasetGenerator(BaseDatasetGenerator):
    """Navigation 2D dataset generator."""

    width: int = 21
    height: int = 21
    obstacle_density: float = 0.08
    n_barriers: int = 8
    barrier_holes_range: Tuple[int, int] = (3, 5)
    n_blocks: Tuple[int, int] = (1, 4)
    block_size_range: Tuple[int, int] = (2, 3)
    max_attempts: int = 4000
    valid_starts: Optional[Tuple[int, int]] = (1, 1)
    valid_ends: Optional[Tuple[int, int]] = None
    min_manhattan_distance: Optional[int] = None

    n_train: int = 1000
    n_test: int = 100
    style: str = "maze"
    distance_metric: str = "jaccard_path"
    distance_threshold: float = 0.5

    def __post_init__(self):
        if self.valid_ends is None:
            self.valid_ends = (self.width - 2, self.height - 2)

    def create_task(self) -> Navigation2D:
        generate_intermediate_states = (
            self.video is not None and self.video.frames_per_intermediate > 0
        )
        return Navigation2D(
            width=self.width,
            height=self.height,
            obstacle_density=self.obstacle_density,
            seed=42,
            add_border=True,
            valid_starts=self.valid_starts,
            valid_ends=self.valid_ends,
            n_barriers=self.n_barriers,
            barrier_holes_range=self.barrier_holes_range,
            n_blocks=self.n_blocks,
            block_size_range=self.block_size_range,
            generate_intermediate_states=generate_intermediate_states,
            max_attempts=self.max_attempts,
            min_manhattan_distance=self.min_manhattan_distance,
        )

    @property
    def dataset_name(self) -> str:
        return "navigation2d"


@dataclass(kw_only=True)
class Navigation2DAnyToAnyDatasetGenerator(Navigation2DDatasetGenerator):
    """Navigation 2D any-to-any dataset generator."""

    width: int = 15
    height: int = 15
    obstacle_density: float = 0.05
    n_barriers: int = 3
    barrier_holes_range: Tuple[int, int] = (8, 14)
    n_blocks: Tuple[int, int] = (3, 8)
    valid_starts: Optional[Tuple[int, int]] = None
    valid_ends: Optional[Tuple[int, int]] = None
    min_manhattan_distance: int = 15

    n_test: int = 200

    def __post_init__(self):
        pass

    @property
    def dataset_name(self) -> str:
        return "navigation2d_any_to_any"


@dataclass(kw_only=True)
class ShortestPathDatasetGenerator(Navigation2DDatasetGenerator):
    """Shortest path dataset generator."""

    width: int = 15
    height: int = 15
    obstacle_density: float = 0.05
    n_barriers: int = 0
    n_blocks: Tuple[int, int] = (3, 8)
    block_size_range: Tuple[int, int] = (2, 5)
    valid_starts: Optional[Tuple[int, int]] = None
    valid_ends: Optional[Tuple[int, int]] = None
    min_manhattan_distance: int = 16

    def __post_init__(self):
        pass

    @property
    def dataset_name(self) -> str:
        return "shortest_path"


@register_dataset("navigation2d")
def generate_navigation2d_dataset(**kwargs):
    Navigation2DDatasetGenerator(**kwargs).generate()


@register_dataset("navigation2d_any_to_any")
def generate_navigation2d_any_to_any_dataset(**kwargs):
    Navigation2DAnyToAnyDatasetGenerator(**kwargs).generate()


@register_dataset("shortest_path")
def generate_shortest_path_dataset(**kwargs):
    ShortestPathDatasetGenerator(**kwargs).generate()
