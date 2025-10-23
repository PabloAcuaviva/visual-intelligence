import shutil

from visual_intelligence.tasks.problem_set import TaskProblemSet
from visual_intelligence.tasks.render.schemas import ArcBaseStyle
from visual_intelligence.tasks.tower_of_hanoi import TowerOfHanoi

from .registry import register_dataset


@register_dataset("tower_of_hanoi")
def generate_tower_of_hanoi_dataset():
    train_problems = [TowerOfHanoi(num_disks=n).generate() for n in [3, 4, 5]]
    test_problems = [TowerOfHanoi(num_disks=6).generate()]
    shutil.rmtree("datasets/tower_of_hanoi", ignore_errors=True)
    TaskProblemSet(task_problems=train_problems).save(
        "datasets/tower_of_hanoi/train",
        ArcBaseStyle,
    )
    TaskProblemSet(task_problems=test_problems).save(
        "datasets/tower_of_hanoi/test",
        ArcBaseStyle,
    )
