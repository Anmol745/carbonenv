"""
Task 2: Budget-Constrained Execution (MEDIUM)
Score: STRICTLY (0.001, 0.999) — enforced at every return point
"""

TASK_ID = 2
TASK_NAME = "Budget-Constrained Execution"
TASK_DIFFICULTY = "medium"
TASK_DESCRIPTION = (
    "Complete all 30 jobs within 48 timesteps without exceeding 150 kg CO2."
)
TOTAL_JOBS = 30
CARBON_BUDGET = 150.0


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

        try:
            jobs_completed = int(
                final_state.get("jobs_completed")
                or final_state.get("observation", {}).get("jobs_completed")
                or 0
            )
        except Exception:
            jobs_completed = 0

        try:
            carbon_used = float(
                final_state.get("carbon_used")
                or final_state.get("observation", {}).get("carbon_used")
                or 0.0
            )
        except Exception:
            carbon_used = 0.0

        completion_score = min(
            float(jobs_completed) / float(TOTAL_JOBS), 1.0
        )

        if carbon_used > CARBON_BUDGET:
            overage_ratio = (carbon_used - CARBON_BUDGET) / CARBON_BUDGET
            cap = max(0.05, 0.28 - overage_ratio * 0.28)
            partial = completion_score * cap
            return _clamp(partial)

        # Cap at 0.97 so combined score can never reach 1.0
        budget_efficiency = (
            1.0 - (carbon_used / CARBON_BUDGET)
        ) * 0.97
        budget_score = max(0.001, budget_efficiency)

        # Max = 0.6 * 1.0 + 0.4 * 0.97 = 0.988 — impossible to reach 1.0
        raw = 0.6 * completion_score + 0.4 * budget_score

        return _clamp(raw)

    except Exception:
        return _clamp(0.5)
