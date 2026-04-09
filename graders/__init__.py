"""
Universal grader runner
Guarantees score strictly within (0,1)
"""

import math

from tasks import TASK_REGISTRY


def safe_score(x):

    try:
        x = float(x)
    except:
        return 0.0001

    if math.isnan(x) or x <= 0:
        return 0.0001

    if x >= 1:
        return 0.9999

    return round(x, 6)


def run_grader(task_id, trajectory, final_state):

    if task_id not in TASK_REGISTRY:
        return 0.0001

    grade_fn = TASK_REGISTRY[task_id]["grade"]

    try:
        score = grade_fn(trajectory, final_state)
    except Exception:
        return 0.0001

    return safe_score(score)
