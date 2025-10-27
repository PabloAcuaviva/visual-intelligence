import shutil
from pathlib import Path
from typing import Union

from visual_intelligence.tasks.problem_set import TaskProblemSet
from visual_intelligence.tasks.render.schemas import ArcBaseStyle
from visual_intelligence.tasks.tower_of_hanoi import TowerOfHanoi

from .registry import register_dataset


@register_dataset("tower_of_hanoi")
def generate_tower_of_hanoi_dataset(out_dir: Union[str, Path] = "datasets"):
    train_problems = [TowerOfHanoi(num_disks=n).generate() for n in [3, 4, 5]]
    test_problems = [TowerOfHanoi(num_disks=6).generate()]

    out_dir = Path(out_dir)
    out_dir = out_dir / "tower_of_hanoi"
    shutil.rmtree(out_dir, ignore_errors=True)

    shutil.rmtree(out_dir, ignore_errors=True)
    TaskProblemSet(task_problems=train_problems).save(
        out_dir / "train",
        ArcBaseStyle,
    )
    TaskProblemSet(task_problems=test_problems).save(
        out_dir / "test",
        ArcBaseStyle,
    )
