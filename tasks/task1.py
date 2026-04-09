"""
Task 1: Green Window Scheduling (EASY)
Score range: strictly (0.001, 0.999) — never exactly 0.0 or 1.0
"""

TASK_ID = 1
TASK_NAME = "Green Window Scheduling"
TASK_DIFFICULTY = "easy"
TASK_DESCRIPTION = (
    "Schedule all 12 jobs within 24 timesteps. "
    "Maximize jobs allocated during high-renewable windows."
)
GREEN_THRESHOLD = 0.7
TOTAL_JOBS = 12


def grade(trajectory: list, final_state: dict) -> float:
    if not trajectory:
        return 0.001

    total_allocated = 0
    green_allocated = 0

    for step in trajectory:
        if step["action"] == "allocate_jobs":
            amount = max(1, int(step.get("amount", 1)))
            renewable = step.get("renewable_ratio", 0.0)
            total_allocated += amount
            if renewable >= GREEN_THRESHOLD:
                green_allocated += amount

    jobs_completed = final_state.get("jobs_completed", 0)
    completion_ratio = min(jobs_completed / TOTAL_JOBS, 1.0)
    completion_bonus = 0.15 * completion_ratio  # reduced from 0.2

    if total_allocated == 0:
        return round(min(max(0.001 + completion_bonus * 0.1, 0.001), 0.999), 4)

    green_ratio = green_allocated / total_allocated

    # Max possible: 0.75 * 1.0 + 0.15 = 0.90 → never reaches 1.0
    raw = green_ratio * 0.75 + completion_bonus

    # Hard clamp — strictly between 0.001 and 0.999
    return round(min(max(raw, 0.001), 0.999), 4)
