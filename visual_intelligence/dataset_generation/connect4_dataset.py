from dataclasses import dataclass

from visual_intelligence.tasks.connect4 import Connect4

from .base import BaseDatasetGenerator
from .registry import register_dataset


@dataclass(kw_only=True)
class Connect4DatasetGenerator(BaseDatasetGenerator):
    """Connect4 dataset generator."""

    # Override defaults
    image_width: int = 240
    image_height: int = 240
    distance_metric: str = "hamming_tgt"
    distance_threshold: float = 0.25
    attempts_multiplier: int = 5000

    def create_task(self) -> Connect4:
        return Connect4(seed=420)

    @property
    def dataset_name(self) -> str:
        return "connect4"


@register_dataset("connect4")
def generate_connect4_dataset(**kwargs):
    Connect4DatasetGenerator(**kwargs).generate()
