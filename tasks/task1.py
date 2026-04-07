"""
Task 1: Green Window Scheduling (EASY)
=======================================
Objective:
  Schedule all jobs preferentially during timesteps where renewable_ratio > 0.7.
  The agent must learn to wait for green windows and batch allocations.

Success criteria (deterministic):
  score = (jobs scheduled in green windows) / total_jobs_completed
  + completion_bonus if all jobs done

Score range: 0.0 – 1.0
Difficulty: EASY — renewable windows are wide and frequent in Task 1.
"""

TASK_ID = 1
TASK_NAME = "Green Window Scheduling"
TASK_DIFFICULTY = "easy"
TASK_DESCRIPTION = (
    "Schedule all 12 jobs within 24 timesteps. "
    "Maximize the proportion of jobs allocated during high-renewable windows "
    "(renewable_ratio > 0.7). Earn bonus for full completion."
)
GREEN_THRESHOLD = 0.7


def grade(trajectory: list[dict], final_state: dict) -> float:
    """
    Score the agent's trajectory.

    Args:
        trajectory: list of step dicts from env.get_trajectory()
        final_state: dict from env.state() after episode ends

    Returns:
        float in [0.0, 1.0]
    """
    if not trajectory:
        return 0.0

    total_allocated = 0
    green_allocated = 0

    for step in trajectory:
        if step["action"] == "allocate_jobs":
            amount = int(step.get("amount", 1))
            renewable = step.get("renewable_ratio", 0.0)
            total_allocated += amount
            if renewable >= GREEN_THRESHOLD:
                green_allocated += amount

    if total_allocated == 0:
        return 0.0

    green_ratio = green_allocated / total_allocated

    # Completion bonus: up to 0.2 extra if all jobs completed
    jobs_completed = final_state.get("jobs_completed", 0)
    total_jobs = 12  # Task 1 fixed
    completion_ratio = min(jobs_completed / total_jobs, 1.0)
    completion_bonus = 0.2 * completion_ratio

    # Base score is green ratio (0–0.8) + completion bonus (0–0.2)
    raw = green_ratio * 0.8 + completion_bonus
    return round(min(max(raw, 0.0), 1.0), 4)