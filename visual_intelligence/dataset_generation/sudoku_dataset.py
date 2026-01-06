from dataclasses import dataclass
from typing import Optional, Tuple, Union

from visual_intelligence.tasks.sudoku import Sudoku

from .base import BaseDatasetGenerator
from .registry import register_dataset


@dataclass(kw_only=True)
class SudokuDatasetGenerator(BaseDatasetGenerator):
    """Sudoku puzzle dataset generator."""

    # Task-specific params
    difficulty: str = "easy"
    variant: str = "standard"
    initialization: str = "generate"
    hf_difficulty: Optional[Union[str, Tuple[int, int]]] = None
    hf_source: Optional[str] = None
    unique_solution_only: bool = True

    def create_task(self) -> Sudoku:
        # Auto-calculate image size if not specified
        if self.image_width is None:
            self.image_width = 16 * (19 if self.variant == "standard" else 9)
        if self.image_height is None:
            self.image_height = self.image_width

        return Sudoku(
            difficulty=self.difficulty,
            variant=self.variant,
            seed=123,
            initialization=self.initialization,
            hf_difficulty=self.hf_difficulty,
            hf_source=self.hf_source,
            unique_solution_only=self.unique_solution_only,
        )

    @property
    def dataset_name(self) -> str:
        return f"sudoku_{self.variant}_{self.difficulty}"


@register_dataset("sudoku")
def generate_sudoku_dataset(**kwargs):
    SudokuDatasetGenerator(**kwargs).generate()
