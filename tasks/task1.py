import math

TASK_ID = 1
TASK_NAME = "Green Window Scheduling"
TASK_DIFFICULTY = "easy"
GREEN_THRESHOLD = 0.7
TOTAL_JOBS = 12

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

    total_allocated = 0
    green_allocated = 0

    for step in trajectory:
        if not isinstance(step, dict):
            continue
        if step.get("action") == "allocate_jobs":
            try:
                amount = max(1, int(step.get("amount", 1) or 1))
            except:
                amount = 1
            try:
                renewable = float(step.get("renewable_ratio", 0) or 0)
            except:
                renewable = 0.0
            total_allocated += amount
            if renewable >= GREEN_THRESHOLD:
                green_allocated += amount

    try:
        jobs_completed = float(final_state.get("jobs_completed", 0) or 0)
    except:
        jobs_completed = 0.0

    # Clamp every component individually BEFORE combining
    completion_ratio = _clamp(jobs_completed / TOTAL_JOBS, 0.001, 0.999)
    completion_bonus = 0.15 * completion_ratio  # range: (0.00015, 0.14985)

    if total_allocated == 0:
        return _clamp(0.001 + completion_bonus)

    green_ratio = _clamp(green_allocated / total_allocated, 0.001, 0.999)

    # Combine — max possible: 0.999*0.75 + 0.14985 = 0.89910 < 1 ✅
    #           min possible: 0.001*0.75 + 0.00015 = 0.00090 > 0 ✅
    raw = green_ratio * 0.75 + completion_bonus
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
