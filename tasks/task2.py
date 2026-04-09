"""
Task 2: Budget-Constrained Execution (MEDIUM)
Score range: strictly (0.001, 0.999) — never exactly 0.0 or 1.0
"""

TASK_ID = 2
TASK_NAME = "Budget-Constrained Execution"
TASK_DIFFICULTY = "medium"
TASK_DESCRIPTION = (
    "Complete all 30 jobs within 48 timesteps without exceeding 150 kg CO2."
)

TOTAL_JOBS = 30
CARBON_BUDGET = 150.0


def grade(trajectory: list, final_state: dict) -> float:
    if not trajectory:
        return 0.001

    jobs_completed = final_state.get("jobs_completed", 0)
    carbon_used = final_state.get("carbon_used", 0.0)

    completion_score = min(jobs_completed / TOTAL_JOBS, 1.0)

    if carbon_used > CARBON_BUDGET:
        overage_ratio = (carbon_used - CARBON_BUDGET) / CARBON_BUDGET
        cap = max(0.05, 0.28 - overage_ratio * 0.28)
        partial = completion_score * cap
        return round(min(max(partial, 0.001), 0.999), 4)

    # Budget respected
    # Use 0.98 max for budget efficiency so raw never hits 1.0
    budget_efficiency = (1.0 - (carbon_used / CARBON_BUDGET)) * 0.98
    budget_score = max(0.001, budget_efficiency)

    # Max possible: 0.6 * 1.0 + 0.4 * 0.98 = 0.992 → never 1.0
    raw = 0.6 * completion_score + 0.4 * budget_score

    return round(min(max(raw, 0.001), 0.999), 4)
