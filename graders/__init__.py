"""
Graders — run all task graders.
Scores ALWAYS strictly in (0.001, 0.999).
Triple-safety: grader clamp + type check + final assert.
"""

from tasks import TASK_REGISTRY
from env.carbon_env import CarbonEnv
from env.models import Action


def run_grader(task_id: int, trajectory: list, final_state: dict) -> float:
    assert task_id in TASK_REGISTRY, f"Unknown task_id: {task_id}"

    grade_fn = TASK_REGISTRY[task_id]["grade"]

    try:
        score = grade_fn(trajectory, final_state)
    except Exception as e:
        print(f"[WARN] Grader {task_id} raised exception: {e}, returning 0.001")
        return 0.001

    # Convert to float — handles int, numpy float, etc.
    score = float(score)

    # Triple safety clamp
    if score <= 0.0 or score != score:   # handles 0, negative, NaN
        score = 0.001
    elif score >= 1.0:                    # handles 1.0 and above
        score = 0.999

    # Final strict clamp
    score = max(0.001, min(0.999, score))

    return round(score, 4)


def validate_graders() -> dict:
    """
    Run ALL edge cases the validator might try.
    If this passes locally, submission will pass Phase 2.
    """
    results = {}

    test_scenarios = [
        # (agent_action, label)
        ("allocate_jobs", "normal"),
        ("idle",          "idle_only"),
    ]

    for action_type, label in test_scenarios:
        for task_id in [1, 2, 3]:
            env = CarbonEnv(task_id=task_id, seed=42)
            obs = env.reset()
            done = False

            while not done:
                action = Action(action_type=action_type, amount=1)
                obs, reward, done, info = env.step(action)

            trajectory = env.get_trajectory()
            final_state = env.state()
            score = run_grader(task_id, trajectory, final_state)

            # This is the EXACT check Phase 2 uses
            assert 0.0 < score < 1.0, (
                f"FAIL: Task {task_id} [{label}] score={score} "
                f"not strictly in (0, 1)!"
            )
            results[f"task_{task_id}_{label}"] = score

    # Also test empty trajectory
    for task_id in [1, 2, 3]:
        score = run_grader(task_id, [], {})
        assert 0.0 < score < 1.0, (
            f"FAIL: Task {task_id} [empty] score={score} "
            f"not strictly in (0, 1)!"
        )
        results[f"task_{task_id}_empty"] = score

    # Test perfect agent (all jobs in green windows)
    for task_id in [1, 2, 3]:
        env = CarbonEnv(task_id=task_id, seed=42)
        obs = env.reset()
        done = False
        while not done:
            if obs.renewable_ratio > 0.5 and obs.jobs_remaining > 0:
                action = Action(action_type="allocate_jobs", amount=2)
            else:
                action = Action(action_type="idle", amount=1)
            obs, reward, done, info = env.step(action)
        score = run_grader(task_id, env.get_trajectory(), env.state())
        assert 0.0 < score < 1.0, (
            f"FAIL: Task {task_id} [perfect] score={score} "
            f"not strictly in (0, 1)!"
        )
        results[f"task_{task_id}_perfect"] = score

    return results


if __name__ == "__main__":
    print("Running grader validation...")
    scores = validate_graders()
    for key, score in scores.items():
        print(f"  {key}: {score:.4f}  {'✅' if 0 < score < 1 else '❌'}")
    print(f"\nAll {len(scores)} checks passed!")
