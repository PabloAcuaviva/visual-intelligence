from dataclasses import dataclass

from visual_intelligence.tasks.chess_mate_in_n import ChessMate

from .base import BaseDatasetGenerator
from .registry import register_dataset


@dataclass(kw_only=True)
class ChessMateDatasetGenerator(BaseDatasetGenerator):
    """Chess mate-in-N dataset generator."""

    # Task-specific params
    mate_in: int = 1
    initial_turn: str = "w"

    # Override defaults
    style: str = "arc_extended"
    image_width: int = 272
    image_height: int = 272
    distance_metric: str = "hamming_tgt"
    distance_threshold: float = 0.01
    attempts_multiplier: int = 100

    def create_task(self) -> ChessMate:
        return ChessMate(
            mate_in=self.mate_in,
            initial_turn=self.initial_turn,
            generate_sequential=True,
        )

    @property
    def dataset_name(self) -> str:
        return f"chess_mate_in_{self.mate_in}_{self.initial_turn}"


@register_dataset("chess_mate_in_n")
def generate_chess_mate_in_n_dataset(**kwargs):
    ChessMateDatasetGenerator(**kwargs).generate()
