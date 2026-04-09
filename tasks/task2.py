import math

TASK_ID = 2
TASK_NAME = "Budget-Constrained Execution"
TASK_DIFFICULTY = "medium"
TOTAL_JOBS = 30
CARBON_BUDGET = 150.0

def _clamp(x, lo=0.001, hi=0.999):
    try:
        x = float(x)
    except:
        return lo
    if math.isnan(x) or math.isinf(x):
        return lo
    return max(lo, min(x, hi))

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
    completion_score = _clamp(jobs_completed / TOTAL_JOBS, 0.001, 0.999)

    if carbon_used > CARBON_BUDGET:
        overage_ratio = _clamp((carbon_used - CARBON_BUDGET) / CARBON_BUDGET, 0.001, 0.999)
        cap = _clamp(0.28 - overage_ratio * 0.28, 0.05, 0.27)
        raw = completion_score * cap
        # max: 0.999 * 0.27 = 0.26973 < 1 ✅
        # min: 0.001 * 0.05 = 0.00005 > 0 ✅
        return _clamp(raw)

    carbon_ratio = _clamp(carbon_used / CARBON_BUDGET, 0.001, 0.999)
    budget_efficiency = _clamp((1 - carbon_ratio) * 0.98, 0.001, 0.979)

    # Combine — max: 0.6*0.999 + 0.4*0.979 = 0.5994 + 0.3916 = 0.9910 < 1 ✅
    #           min: 0.6*0.001 + 0.4*0.001 = 0.001 > 0 ✅
    raw = 0.6 * completion_score + 0.4 * budget_efficiency
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
