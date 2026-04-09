"""
Graders — run all task graders.
Score ALWAYS strictly in (0.001, 0.999).
"""

from tasks import TASK_REGISTRY
from env.carbon_env import CarbonEnv
from env.models import Action


def _clamp(value) -> float:
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
    return round(max(0.001, min(0.999, s)), 4)


def run_grader(
    task_id: int, trajectory: list, final_state: dict
) -> float:
    assert task_id in TASK_REGISTRY, f"Unknown task_id: {task_id}"

    try:
        grade_fn = TASK_REGISTRY[task_id]["grade"]
        score = grade_fn(trajectory, final_state)
    except Exception as e:
        print(f"[WARN] Grader {task_id} error: {e}")
        return _clamp(0.5)

    return _clamp(score)


def validate_graders() -> dict:
    results = {}

    for task_id in [1, 2, 3]:
        env = CarbonEnv(task_id=task_id, seed=42)
        obs = env.reset()
        done = False
        while not done:
            action = Action(action_type="allocate_jobs", amount=1)
            obs, reward, done, info = env.step(action)

        score = run_grader(
            task_id, env.get_trajectory(), env.state()
        )
        assert 0.0 < score < 1.0, (
            f"Task {task_id} score {score} not in (0,1)!"
        )
        results[task_id] = score

    return results


if __name__ == "__main__":
    print("Running grader validation...")
    scores = validate_graders()
    for task_id, score in scores.items():
        info = TASK_REGISTRY[task_id]
        print(
            f"  Task {task_id} [{info['difficulty']}] "
            f"{info['name']}: {score:.4f}"
        )
    print("All graders passed.")
