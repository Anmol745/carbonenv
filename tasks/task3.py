"""
Task 3: Profit-Aware Carbon Trading (HARD)
Score must be strictly within (0,1)
"""

import math

TASK_ID = 3
TASK_NAME = "Profit-Aware Carbon Trading"
TASK_DIFFICULTY = "hard"

TOTAL_JOBS = 50
CARBON_BUDGET = 120.0


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


def sigmoid(x):

    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0001 if x < 0 else 0.9999


def grade(trajectory, final_state):

    if not trajectory:
        return 0.0001

    jobs_completed = final_state.get("jobs_completed", 0)
    carbon_used = final_state.get("carbon_used", 0)

    completion_score = min(jobs_completed / TOTAL_JOBS, 0.98)

    if carbon_used > CARBON_BUDGET:

        overage = (carbon_used - CARBON_BUDGET) / CARBON_BUDGET

        budget_score = max(0.001, 0.08 - overage * 0.08)

    else:

        budget_score = 1 - (carbon_used / CARBON_BUDGET)

    buy_cost = 0
    sell_revenue = 0

    for step in trajectory:

        action = step.get("action")

        obs = step.get("obs", {})

        price = obs.get("carbon_credit_price", 25)

        amount = step.get("amount", 0)

        if action == "buy_carbon_credits":
            buy_cost += amount * price

        if action == "sell_carbon_credits":
            sell_revenue += amount * price

    net_profit = sell_revenue - buy_cost

    trading_score = sigmoid(net_profit / 50)

    raw = (
        0.4 * completion_score
        + 0.3 * budget_score
        + 0.3 * trading_score
    )

    return safe_score(raw)
