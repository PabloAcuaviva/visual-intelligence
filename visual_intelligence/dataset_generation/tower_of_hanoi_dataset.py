import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from visual_intelligence.tasks.problem_set import TaskProblemSet
from visual_intelligence.tasks.tower_of_hanoi import TowerOfHanoi

from .base import get_style
from .registry import register_dataset


@dataclass(kw_only=True)
class TowerOfHanoiDatasetGenerator:
    """
    Tower of Hanoi dataset generator.

    Note: This doesn't inherit from BaseDatasetGenerator because
    it generates deterministic problems from num_disks rather than
    using TaskDatasetGenerator.
    """

    # Task-specific params
    train_num_disks: list[int] = field(default_factory=lambda: [3, 4, 5])
    test_num_disks: list[int] = field(default_factory=lambda: [6])

    # Common params
    out_dir: Union[str, Path] = "datasets"
    style: str = "arc_base"
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    def generate(self):
        train_problems = [
            TowerOfHanoi(num_disks=n).generate() for n in self.train_num_disks
        ]
        test_problems = [
            TowerOfHanoi(num_disks=n).generate() for n in self.test_num_disks
        ]

        resolved_style = get_style(self.style)

        save_kwargs: dict = {}
        if self.image_width is not None:
            save_kwargs["image_width"] = self.image_width
        if self.image_height is not None:
            save_kwargs["image_height"] = self.image_height

        out_dir = Path(self.out_dir) / "tower_of_hanoi"
        shutil.rmtree(out_dir, ignore_errors=True)

        TaskProblemSet(task_problems=train_problems).save(
            out_dir / "train",
            resolved_style,
            **save_kwargs,
        )
        TaskProblemSet(task_problems=test_problems).save(
            out_dir / "test",
            resolved_style,
            **save_kwargs,
        )

    def __call__(self):
        return self.generate()


@register_dataset("tower_of_hanoi")
def generate_tower_of_hanoi_dataset(**kwargs):
    TowerOfHanoiDatasetGenerator(**kwargs).generate()
