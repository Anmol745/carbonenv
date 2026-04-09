"""
Task 3: Profit-Aware Carbon Trading (HARD)
Score: STRICTLY (0.001, 0.999) — enforced at every return point
"""

import math

TASK_ID = 3
TASK_NAME = "Profit-Aware Carbon Trading"
TASK_DIFFICULTY = "hard"
TASK_DESCRIPTION = (
    "Complete all 50 jobs within 72 timesteps while staying under "
    "120 kg CO2 AND generating net positive profit from carbon credit trading."
)
TOTAL_JOBS = 50
CARBON_BUDGET = 120.0


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


def _sigmoid(x: float) -> float:
    try:
        result = 1.0 / (1.0 + math.exp(-float(x)))
        return min(max(result, 0.002), 0.998)
    except Exception:
        return 0.5


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

        # Cap at 0.97 so combined score can never reach 1.0
        completion_score = min(
            float(jobs_completed) / float(TOTAL_JOBS), 0.97
        )
        completion_score = max(completion_score, 0.001)

        if carbon_used > CARBON_BUDGET:
            overage = (carbon_used - CARBON_BUDGET) / CARBON_BUDGET
            budget_score = max(0.001, 0.08 - overage * 0.08)
        else:
            budget_score = max(
                0.001,
                min(1.0 - carbon_used / CARBON_BUDGET, 0.96)
            )

        buy_costs = []
        sell_revenues = []

        for step in trajectory:
            try:
                action = step.get("action") or step.get("action_type", "")
                amount = float(step.get("amount", 0.0))

                obs = (
                    step.get("obs")
                    or step.get("observation")
                    or step
                )
                if not isinstance(obs, dict):
                    obs = step

                credit_price = float(
                    obs.get("carbon_credit_price", 25.0) or 25.0
                )
                credits_held = float(
                    obs.get("credits_held", 0.0) or 0.0
                )

                if action == "buy_carbon_credits":
                    buy_costs.append(min(amount, 20.0) * credit_price)
                elif action == "sell_carbon_credits":
                    actual_sold = min(amount, credits_held)
                    sell_revenues.append(actual_sold * credit_price)

            except Exception:
                continue

        net_trading = sum(sell_revenues) - sum(buy_costs)
        trading_score = _sigmoid(net_trading / 50.0)

        # Max = 0.40*0.97 + 0.30*0.96 + 0.30*0.998 = 0.9754
        raw = (
            0.40 * completion_score
            + 0.30 * budget_score
            + 0.30 * trading_score
        )

        return _clamp(raw)

    except Exception:
        return _clamp(0.5)
