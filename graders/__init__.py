"""
Graders — validate and run all task graders.
Ensures scores are always in [0.0, 1.0] and deterministic.
"""

from tasks import TASK_REGISTRY
from env.carbon_env import CarbonEnv
from env.models import Action


def run_grader(task_id: int, trajectory: list[dict], final_state: dict) -> float:
    """
    Run the grader for a given task_id.
    Always returns a float in [0.0, 1.0].
    """
    assert task_id in TASK_REGISTRY, f"Unknown task_id: {task_id}"
    grade_fn = TASK_REGISTRY[task_id]["grade"]
    score = grade_fn(trajectory, final_state)
    # Enforce bounds — graders are deterministic but we clamp for safety
    return float(max(0.0, min(1.0, score)))


def validate_graders() -> dict:
    """
    Run a quick sanity check on all graders using a scripted trajectory.
    Returns dict of {task_id: score} — used for CI validation.
    """
    results = {}
    for task_id in [1, 2, 3]:
        env = CarbonEnv(task_id=task_id, seed=42)
        obs = env.reset()

        # Scripted agent: always allocate_jobs with amount=1
        for _ in range(env._config["max_timesteps"]):
            action = Action(action_type="allocate_jobs", amount=1)
            obs, reward, done, info = env.step(action)
            if done:
                break

        trajectory = env.get_trajectory()
        final_state = env.state()
        score = run_grader(task_id, trajectory, final_state)
        assert 0.0 <= score <= 1.0, f"Task {task_id} grader returned out-of-range score: {score}"
        results[task_id] = score

    return results


if __name__ == "__main__":
    print("Running grader validation...")
    scores = validate_graders()
    for task_id, score in scores.items():
        info = TASK_REGISTRY[task_id]
        print(f"  Task {task_id} [{info['difficulty']}] {info['name']}: {score:.4f}")
    print("All graders passed validation.")