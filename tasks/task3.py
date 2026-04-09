import math

TASK_ID = 3
TASK_NAME = "Profit-Aware Carbon Trading"
TASK_DIFFICULTY = "hard"
TOTAL_JOBS = 50
CARBON_BUDGET = 120.0

def _clamp(x, lo=0.001, hi=0.999):
    try:
        x = float(x)
    except:
        return lo
    if math.isnan(x) or math.isinf(x):
        return lo
    return max(lo, min(x, hi))

def _sigmoid(x):
    try:
        val = 1.0 / (1.0 + math.exp(-float(x)))
    except OverflowError:
        val = 0.0 if x < 0 else 1.0
    except:
        val = 0.5
    # Clamp sigmoid output itself
    return _clamp(val, 0.001, 0.999)

def _grade(trajectory, final_state):
    if not trajectory:
        return 0.001

    try:
        jobs_completed = float(final_state.get("jobs_completed", 0) or 0)
    except:
        jobs_completed = 0.0

    try:
        carbon_used = float(final_state.get("carbon_used", 0) or 0)
    except:
        carbon_used = 0.0

    # Clamp every component BEFORE combining
    completion_score = _clamp(jobs_completed / TOTAL_JOBS, 0.001, 0.979)
    # max capped at 0.979 so weighted sum never reaches 1

    if carbon_used > CARBON_BUDGET:
        overage = _clamp((carbon_used - CARBON_BUDGET) / CARBON_BUDGET, 0.001, 0.999)
        budget_score = _clamp(0.08 - overage * 0.08, 0.001, 0.079)
    else:
        carbon_ratio = _clamp(carbon_used / CARBON_BUDGET, 0.001, 0.999)
        budget_score = _clamp(1.0 - carbon_ratio, 0.001, 0.979)
        # max: 1 - 0.001 = 0.999 → clamped to 0.979 ✅

    buy_cost = 0.0
    sell_revenue = 0.0
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        obs = step.get("obs", {})
        if not isinstance(obs, dict):
            obs = {}
        try:
            price = float(obs.get("carbon_credit_price", 25) or 25)
        except:
            price = 25.0
        try:
            amount = float(step.get("amount", 0) or 0)
        except:
            amount = 0.0
        if action == "buy_carbon_credits":
            buy_cost += amount * price
        if action == "sell_carbon_credits":
            sell_revenue += amount * price

    net_profit = sell_revenue - buy_cost
    trading_score = _sigmoid(net_profit / 50.0)
    # sigmoid output already clamped to (0.001, 0.999)

    # Combine — worst case max: 0.4*0.979 + 0.3*0.979 + 0.3*0.999
    #                         = 0.3916 + 0.2937 + 0.2997 = 0.9850 < 1 ✅
    #           worst case min: 0.4*0.001 + 0.3*0.001 + 0.3*0.001 = 0.001 > 0 ✅
    raw = 0.4 * completion_score + 0.3 * budget_score + 0.3 * trading_score
    return _clamp(raw)

def grade(trajectory, final_state):
    try:
        if trajectory is None:
            trajectory = []
        if final_state is None:
            final_state = {}
        return _clamp(_grade(trajectory, final_state))
    except:
        return 0.001
