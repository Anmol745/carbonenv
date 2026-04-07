"""
Task 3: Profit-Aware Carbon Trading (HARD)
===========================================
Objective:
  Complete all 50 jobs within 72 timesteps, stay under 120 kg CO2 budget,
  AND generate net positive profit from carbon credit trading (buy low, sell high).
  
  This requires simultaneously:
  - Temporal planning (when to schedule jobs)
  - Emissions management (renewable-aware scheduling)
  - Financial strategy (carbon credit market timing)

Success criteria (deterministic):
  - completion_score = jobs_completed / 50              (weight: 0.40)
  - budget_score = 1 - (carbon_used / budget)          (weight: 0.30)
    → 0 if budget exceeded
  - trading_score = sigmoid(net_trading_profit / 50)   (weight: 0.30)
    → net profit from buy/sell operations

Score range: 0.0 – 1.0
Difficulty: HARD — requires multi-objective optimization simultaneously.
"""

import math

TASK_ID = 3
TASK_NAME = "Profit-Aware Carbon Trading"
TASK_DIFFICULTY = "hard"
TASK_DESCRIPTION = (
    "Complete all 50 jobs within 72 timesteps while staying under 120 kg CO2 "
    "AND generating net positive profit from carbon credit trading. "
    "All three objectives must be balanced simultaneously."
)

TOTAL_JOBS = 50
CARBON_BUDGET = 120.0


def _sigmoid(x: float) -> float:
    """Maps any real number to (0, 1). Centered at 0."""
    return 1.0 / (1.0 + math.exp(-x))


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

    # ── Completion score ───────────────────────────────────────────────────────
    completion_score = min(jobs_completed / TOTAL_JOBS, 1.0)

    # ── Budget score ───────────────────────────────────────────────────────────
    if carbon_used > CARBON_BUDGET:
        budget_score = 0.0
    else:
        budget_score = max(0.0, 1.0 - carbon_used / CARBON_BUDGET)

    # ── Trading score: reconstruct net profit from trajectory ──────────────────
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
            credits_held = obs.get("credits_held", 0.0)
            actual_sold = min(amount, credits_held)
            revenue = actual_sold * credit_price
            sell_revenues.append(revenue)

    net_trading = sum(sell_revenues) - sum(buy_costs)
    # Map profit to [0, 1] using sigmoid centered at 0, scaled by 50
    trading_score = _sigmoid(net_trading / 50.0)

    # ── Combined score ─────────────────────────────────────────────────────────
    raw = (
        0.40 * completion_score
        + 0.30 * budget_score
        + 0.30 * trading_score
    )

    return round(min(max(raw, 0.0), 1.0), 4)