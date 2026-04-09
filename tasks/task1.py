"""
Task 1: Green Window Scheduling (EASY)
Score must be strictly within (0,1)
"""

import math

TASK_ID = 1
TASK_NAME = "Green Window Scheduling"
TASK_DIFFICULTY = "easy"

GREEN_THRESHOLD = 0.7
TOTAL_JOBS = 12


def safe_score(x):
    try:
        x = float(x)
    except:
        return 0.0001

    if math.isnan(x) or x <= 0:
        return 0.0001
    if x >= 1:
        return 0.9999

    return x


def grade(trajectory, final_state):

    if not trajectory:
        return 0.0001

    total_allocated = 0
    green_allocated = 0

    for step in trajectory:

        if step.get("action") == "allocate_jobs":

            amount = max(1, int(step.get("amount", 1)))
            renewable = step.get("renewable_ratio", 0)

            total_allocated += amount

            if renewable >= GREEN_THRESHOLD:
                green_allocated += amount

    jobs_completed = final_state.get("jobs_completed", 0)

    completion_ratio = min(jobs_completed / TOTAL_JOBS, 1.0)

    completion_bonus = 0.15 * completion_ratio

    if total_allocated == 0:
        return safe_score(0.001 + completion_bonus)

    green_ratio = green_allocated / total_allocated

    raw = green_ratio * 0.75 + completion_bonus

    return safe_score(raw)
