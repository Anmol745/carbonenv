"""
Task 1: Green Window Scheduling (EASY)
Score: STRICTLY (0.001, 0.999) — enforced at every return point
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


def _clamp(value) -> float:
    """Convert anything to float strictly in (0.001, 0.999)."""
    try:
        s = float(value)
    except Exception:
        return 0.5
    if s != s:
        return 0.5
    if s <= 0.0:
        return 0.001
    if s >= 1.0:
        return 0.999
    result = max(0.001, min(0.999, s))
    return round(result, 4)


def grade(trajectory: list, final_state: dict) -> float:
    try:
        if not trajectory:
            return _clamp(0.001)

        total_allocated = 0
        green_allocated = 0

        for step in trajectory:
            try:
                action = step.get("action") or step.get("action_type", "")
                if action != "allocate_jobs":
                    continue

                amount = max(1, int(float(step.get("amount", 1))))

                renewable = 0.0
                if "renewable_ratio" in step:
                    renewable = float(step["renewable_ratio"])
                elif "obs" in step and step["obs"]:
                    renewable = float(step["obs"].get("renewable_ratio", 0.0))
                elif "observation" in step and step["observation"]:
                    renewable = float(
                        step["observation"].get("renewable_ratio", 0.0)
                    )

                total_allocated += amount
                if renewable >= GREEN_THRESHOLD:
                    green_allocated += amount

            except Exception:
                continue

        try:
            jobs_completed = int(
                final_state.get("jobs_completed")
                or final_state.get("observation", {}).get("jobs_completed")
                or 0
            )
        except Exception:
            jobs_completed = 0

        completion_ratio = min(float(jobs_completed) / float(TOTAL_JOBS), 1.0)
        completion_bonus = 0.15 * completion_ratio

        if total_allocated == 0:
            return _clamp(0.001 + completion_bonus * 0.1)

        green_ratio = float(green_allocated) / float(total_allocated)

        # Max = 0.75 * 1.0 + 0.15 = 0.90 — impossible to reach 1.0
        raw = green_ratio * 0.75 + completion_bonus

        return _clamp(raw)

    except Exception:
        return _clamp(0.5)
