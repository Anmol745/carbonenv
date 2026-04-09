"""
Graders — run all task graders.
Scores are always strictly in (0.001, 0.999) — never exactly 0.0 or 1.0.
"""

from tasks import TASK_REGISTRY
from env.carbon_env import CarbonEnv
from env.models import Action


def run_grader(task_id: int, trajectory: list[dict], final_state: dict) -> float:
    assert task_id in TASK_REGISTRY, f"Unknown task_id: {task_id}"

    grade_fn = TASK_REGISTRY[task_id]["grade"]
    score = grade_fn(trajectory, final_state)

    # Final safety clamp — strictly open interval
    score = float(score)
    score = max(0.001, min(0.999, score))

    return round(score, 4)


def validate_graders() -> dict:
    results = {}

    for task_id in [1, 2, 3]:
        env = CarbonEnv(task_id=task_id, seed=42)
        obs = env.reset()

        for _ in range(env._config["max_timesteps"]):
            action = Action(action_type="allocate_jobs", amount=1)
            obs, reward, done, info = env.step(action)
            if done:
                break

        trajectory = env.get_trajectory()
        final_state = env.state()
        score = run_grader(task_id, trajectory, final_state)

        # Must be STRICTLY between 0 and 1
        assert 0.0 < score < 1.0, (
            f"Task {task_id} score {score} is not strictly in (0, 1)!"
        )
        results[task_id] = score

    return results


if __name__ == "__main__":
    print("Running grader validation...")
    scores = validate_graders()
    for task_id, score in scores.items():
        info = TASK_REGISTRY[task_id]
        print(f"  Task {task_id} [{info['difficulty']}] {info['name']}: {score:.4f}")
    print("All graders passed.")
