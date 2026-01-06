from dataclasses import dataclass

from visual_intelligence.tasks.hitori import Hitori
from visual_intelligence.tasks.render.schemas import ArcBaseStyle

from .base import BaseDatasetGenerator
from .registry import register_dataset


@dataclass(kw_only=True)
class HitoriDatasetGenerator(BaseDatasetGenerator):
    """Hitori puzzle dataset generator."""

    # Task-specific params
    size: int = 5
    difficulty: str = "easy"

    def create_task(self) -> Hitori:
        # Auto-calculate image size if not specified
        if self.image_width is None or self.image_height is None:
            orig_size = (
                ArcBaseStyle.cell_size + ArcBaseStyle.grid_border_size
            ) * self.size + ArcBaseStyle.grid_border_size
            calculated_size = 16 * (orig_size // 16 + (orig_size % 16 != 0))
            if self.image_width is None:
                self.image_width = calculated_size
            if self.image_height is None:
                self.image_height = calculated_size

        return Hitori(size=self.size, difficulty=self.difficulty, seed=123)

    @property
    def dataset_name(self) -> str:
        return f"hitori_{self.size}_{self.difficulty}"


@register_dataset("hitori")
def generate_hitori_dataset(**kwargs):
    HitoriDatasetGenerator(**kwargs).generate()
