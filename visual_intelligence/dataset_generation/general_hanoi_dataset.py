from dataclasses import dataclass
from typing import Literal, Union

from visual_intelligence.tasks.general_hanoi import GeneralHanoi

from .base import BaseDatasetGenerator
from .registry import register_dataset


@dataclass(kw_only=True)
class GeneralHanoiDatasetGenerator(BaseDatasetGenerator):
    """General Hanoi dataset generator."""

    # Task-specific params
    num_disks: int = 5
    steps: Union[int, Literal["all"]] = "all"

    # Override defaults
    style: str = "arc_reduced"
    image_width: int = 320
    image_height: int = 64
    distance_threshold: float = 0.01

    def create_task(self) -> GeneralHanoi:
        return GeneralHanoi(num_disks=self.num_disks, step=self.steps)

    @property
    def dataset_name(self) -> str:
        return f"general_hanoi_step{self.steps}"


@register_dataset("general_hanoi")
def generate_general_hanoi_dataset(**kwargs):
    GeneralHanoiDatasetGenerator(**kwargs).generate()
