"""
Task 2: Budget-Constrained Execution (MEDIUM)
Score range: strictly (0.001, 0.999)
"""

TASK_ID = 2
TASK_NAME = "Budget-Constrained Execution"
TASK_DIFFICULTY = "medium"
TASK_DESCRIPTION = (
    "Complete all 30 jobs within 48 timesteps without exceeding 150 kg CO2. "
    "Budget overruns cap your score at 0.3."
)

TOTAL_JOBS = 30
CARBON_BUDGET = 150.0


def grade(trajectory: list[dict], final_state: dict) -> float:
    if not trajectory:
        return 0.001

    jobs_completed = final_state.get("jobs_completed", 0)
    carbon_used = final_state.get("carbon_used", 0.0)

    completion_score = min(jobs_completed / TOTAL_JOBS, 1.0)

    if carbon_used > CARBON_BUDGET:
        # Budget exceeded — graduated penalty, capped at 0.3
        overage_ratio = (carbon_used - CARBON_BUDGET) / CARBON_BUDGET
        cap = max(0.05, 0.3 - overage_ratio * 0.3)  # min 0.05, never 0
        partial = completion_score * cap
        # Clamp strictly
        return round(min(max(partial, 0.001), 0.999), 4)

    # Budget respected
    budget_efficiency = 1.0 - (carbon_used / CARBON_BUDGET)
    budget_score = max(0.001, budget_efficiency)  # never exactly 0

    raw = 0.6 * completion_score + 0.4 * budget_score

    return round(min(max(raw, 0.001), 0.999), 4)
