from dataclasses import dataclass
from typing import Optional

from visual_intelligence.tasks.game_of_life import GameOfLife

from .base import BaseDatasetGenerator
from .registry import register_dataset


@dataclass(kw_only=True)
class GameOfLifeDatasetGenerator(BaseDatasetGenerator):
    """Game of Life dataset generator."""

    # Task-specific params
    steps: int = 1
    gol_variant_name: str = "gol"
    width: int = 8
    height: int = 8
    density: float = 0.4
    survival_rule: Optional[list[int]] = None
    birth_rule: Optional[list[int]] = None

    # Override defaults
    image_width: int = 17 * 16
    image_height: int = 17 * 16
    distance_metric: str = "hamming_tgt"
    distance_threshold: float = 0.1
    attempts_multiplier: int = 500

    def create_task(self) -> GameOfLife:
        return GameOfLife(
            width=self.width,
            height=self.height,
            steps=self.steps,
            initialization="random",
            density=self.density,
            seed=42,
            survival_rule=self.survival_rule,
            birth_rule=self.birth_rule,
        )

    @property
    def dataset_name(self) -> str:
        return f"{self.gol_variant_name}_step{self.steps}"


@register_dataset("gol")
def generate_gol_dataset(**kwargs):
    GameOfLifeDatasetGenerator(**kwargs).generate()
