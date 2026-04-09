"""
Task 2: Budget-Constrained Execution (MEDIUM)
Score must be strictly within (0,1)
"""

import math

TASK_ID = 2
TASK_NAME = "Budget-Constrained Execution"
TASK_DIFFICULTY = "medium"

TOTAL_JOBS = 30
CARBON_BUDGET = 150.0


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

    jobs_completed = final_state.get("jobs_completed", 0)
    carbon_used = final_state.get("carbon_used", 0)

    completion_score = min(jobs_completed / TOTAL_JOBS, 1.0)

    if carbon_used > CARBON_BUDGET:

        overage_ratio = (carbon_used - CARBON_BUDGET) / CARBON_BUDGET

        cap = max(0.05, 0.28 - overage_ratio * 0.28)

        raw = completion_score * cap

        return safe_score(raw)

    budget_efficiency = (1 - (carbon_used / CARBON_BUDGET)) * 0.98

    raw = 0.6 * completion_score + 0.4 * budget_efficiency

    return safe_score(raw)
