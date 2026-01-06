from dataclasses import dataclass

from visual_intelligence.tasks.langton_ant import LangtonAnt

from .base import BaseDatasetGenerator
from .registry import register_dataset


@dataclass(kw_only=True)
class LangtonAntDatasetGenerator(BaseDatasetGenerator):
    """Langton's Ant dataset generator."""

    # Task-specific params
    steps: int = 1
    width: int = 8
    height: int = 8
    ant_initial_dir: str = "N"

    # Override defaults
    style: str = "maze"
    distance_metric: str = "hamming_tgt"

    def create_task(self) -> LangtonAnt:
        return LangtonAnt(
            width=self.width,
            height=self.height,
            steps=self.steps,
            initialization="random",
            seed=123,
            init_grid_as=0,
            ant_initial_dir=self.ant_initial_dir,
        )

    @property
    def dataset_name(self) -> str:
        return f"langton_ant_step{self.steps}"


@register_dataset("langton_ant")
def generate_langton_ant_dataset(**kwargs):
    LangtonAntDatasetGenerator(**kwargs).generate()
