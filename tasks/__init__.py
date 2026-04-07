from tasks.task1 import grade as grade_task1, TASK_NAME as TASK1_NAME, TASK_DIFFICULTY as TASK1_DIFFICULTY
from tasks.task2 import grade as grade_task2, TASK_NAME as TASK2_NAME, TASK_DIFFICULTY as TASK2_DIFFICULTY
from tasks.task3 import grade as grade_task3, TASK_NAME as TASK3_NAME, TASK_DIFFICULTY as TASK3_DIFFICULTY

TASK_REGISTRY = {
    1: {"grade": grade_task1, "name": TASK1_NAME, "difficulty": TASK1_DIFFICULTY},
    2: {"grade": grade_task2, "name": TASK2_NAME, "difficulty": TASK2_DIFFICULTY},
    3: {"grade": grade_task3, "name": TASK3_NAME, "difficulty": TASK3_DIFFICULTY},
}

__all__ = ["TASK_REGISTRY"]