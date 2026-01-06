from dataclasses import dataclass
from typing import Optional

from visual_intelligence.tasks.cellular_automata_1d import CellularAutomata1D

from .base import BaseDatasetGenerator
from .registry import register_dataset


@dataclass(kw_only=True)
class CellularAutomata1DDatasetGenerator(BaseDatasetGenerator):
    """1D Cellular Automata dataset generator."""

    # Task-specific params (rule is required, no default)
    rule: int
    width: int = 16
    steps: int = 7
    initialization: str = "random"

    # Override defaults
    style: str = "maze"
    distance_metric: str = "hamming_tgt"
    attempts_multiplier: int = 1000

    def __post_init__(self):
        # Auto-calculate distance threshold if not explicitly set
        # This is a heuristic based on grid dimensions
        if self.distance_threshold == 0.3:  # Default value
            self.distance_threshold = 1 * (2 / self.width) * (1 / self.steps)

    def create_task(self) -> CellularAutomata1D:
        return CellularAutomata1D(
            rule=self.rule,
            width=self.width,
            steps=self.steps,
            initialization=self.initialization,
            seed=42,
        )

    @property
    def dataset_name(self) -> str:
        return f"cellular_automata_1d_rule{self.rule}_w{self.width}_s{self.steps}"


@register_dataset("cellular_automata_1d")
def generate_cellular_automata_1d_dataset(**kwargs):
    CellularAutomata1DDatasetGenerator(**kwargs).generate()
