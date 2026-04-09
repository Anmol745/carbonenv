"""
Task 3: Profit-Aware Carbon Trading (HARD)
Score range: strictly (0.001, 0.999)
"""

import math

TASK_ID = 3
TASK_NAME = "Profit-Aware Carbon Trading"
TASK_DIFFICULTY = "hard"
TASK_DESCRIPTION = (
    "Complete all 50 jobs within 72 timesteps while staying under 120 kg CO2 "
    "AND generating net positive profit from carbon credit trading."
)

TOTAL_JOBS = 50
CARBON_BUDGET = 120.0


def _sigmoid(x: float) -> float:
    """Maps any real number to (0, 1). Centered at 0."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.001 if x < 0 else 0.999


def grade(trajectory: list[dict], final_state: dict) -> float:
    if not trajectory:
        return 0.001

    jobs_completed = final_state.get("jobs_completed", 0)
    carbon_used = final_state.get("carbon_used", 0.0)

    # ── Completion score ───────────────────────────────────────────────────────
    completion_score = min(jobs_completed / TOTAL_JOBS, 1.0)
    # Never exactly 0 — even 0 jobs gets tiny credit
    completion_score = max(completion_score, 0.001)

    # ── Budget score ───────────────────────────────────────────────────────────
    if carbon_used > CARBON_BUDGET:
        # Exceeded budget — graduated penalty, not hard 0
        overage = (carbon_used - CARBON_BUDGET) / CARBON_BUDGET
        budget_score = max(0.001, 0.1 - overage * 0.1)
    else:
        budget_score = max(0.001, 1.0 - carbon_used / CARBON_BUDGET)

    # ── Trading score ──────────────────────────────────────────────────────────
    buy_costs = []
    sell_revenues = []

    for step in trajectory:
        action = step.get("action", "")
        amount = step.get("amount", 0.0)
        obs = step.get("obs", {})
        credit_price = obs.get("carbon_credit_price", 25.0)

        if action == "buy_carbon_credits":
            cost = min(amount, 20.0) * credit_price
            buy_costs.append(cost)
        elif action == "sell_carbon_credits":
            credits_held_before = obs.get("credits_held", 0.0)
            actual_sold = min(amount, credits_held_before)
            revenue = actual_sold * credit_price
            sell_revenues.append(revenue)

    net_trading = sum(sell_revenues) - sum(buy_costs)

    # Sigmoid maps net profit to (0,1) — never hits exactly 0 or 1
    trading_score = _sigmoid(net_trading / 50.0)

    # ── Combined score ─────────────────────────────────────────────────────────
    raw = (
        0.40 * completion_score
        + 0.30 * budget_score
        + 0.30 * trading_score
    )

    # Strict clamp — never exactly 0.0 or 1.0
    return round(min(max(raw, 0.001), 0.999), 4)
