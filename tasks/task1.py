"""
Task 1: Green Window Scheduling (EASY)
Score range: strictly (0.001, 0.999)
"""

TASK_ID = 1
TASK_NAME = "Green Window Scheduling"
TASK_DIFFICULTY = "easy"
TASK_DESCRIPTION = (
    "Schedule all 12 jobs within 24 timesteps. "
    "Maximize jobs allocated during high-renewable windows (renewable_ratio > 0.7)."
)
GREEN_THRESHOLD = 0.7
TOTAL_JOBS = 12


def grade(trajectory: list[dict], final_state: dict) -> float:
    if not trajectory:
        return 0.001  # never return exactly 0.0

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
    completion_bonus = 0.2 * completion_ratio

    if total_allocated == 0:
        # No jobs allocated at all — very low score but not exactly 0
        return round(min(max(0.001 + completion_bonus * 0.1, 0.001), 0.999), 4)

    green_ratio = green_allocated / total_allocated

    # Base score: green ratio (0–0.8) + completion bonus (0–0.2)
    raw = green_ratio * 0.8 + completion_bonus

    # Clamp strictly between 0.001 and 0.999
    return round(min(max(raw, 0.001), 0.999), 4)
