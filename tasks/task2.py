"""
Task 2: Budget-Constrained Execution (MEDIUM)
==============================================
Objective:
  Complete all 30 jobs across 48 timesteps WITHOUT exceeding the carbon budget
  of 150 kg CO2. The agent must balance throughput vs emissions.
  Scheduling during high-renewable periods reduces per-job carbon cost.

Success criteria (deterministic):
  - If budget exceeded: score capped at 0.3 regardless of jobs
  - completion_score = jobs_completed / 30
  - budget_score = 1 - (carbon_used / carbon_budget) clamped to [0, 1]
  - final_score = 0.6 * completion_score + 0.4 * budget_score

Score range: 0.0 – 1.0
Difficulty: MEDIUM — budget is tight enough to require planning.
"""

TASK_ID = 2
TASK_NAME = "Budget-Constrained Execution"
TASK_DIFFICULTY = "medium"
TASK_DESCRIPTION = (
    "Complete all 30 jobs within 48 timesteps without exceeding 150 kg CO2. "
    "Scheduling during high-renewable windows reduces per-job emissions. "
    "Budget overruns cap your score at 0.3."
)

TOTAL_JOBS = 30
CARBON_BUDGET = 150.0


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

    jobs_completed = final_state.get("jobs_completed", 0)
    carbon_used = final_state.get("carbon_used", 0.0)

    # Hard cap for budget violation
    if carbon_used > CARBON_BUDGET:
        overage_ratio = (carbon_used - CARBON_BUDGET) / CARBON_BUDGET
        # Graduated penalty — small overage still gets something
        cap = max(0.0, 0.3 - overage_ratio * 0.3)
        completion_partial = (jobs_completed / TOTAL_JOBS) * cap
        return round(min(completion_partial, cap), 4)

    completion_score = min(jobs_completed / TOTAL_JOBS, 1.0)
    budget_efficiency = 1.0 - (carbon_used / CARBON_BUDGET)
    budget_score = max(0.0, budget_efficiency)

    raw = 0.6 * completion_score + 0.4 * budget_score
    return round(min(max(raw, 0.0), 1.0), 4)