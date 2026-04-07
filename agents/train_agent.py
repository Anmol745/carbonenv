"""
training/train_agent.py
Run baseline agents against all 3 tasks and log results.
"""
import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.carbon_env import CarbonEnv
from env.models import Action
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from graders import run_grader
from tasks import TASK_REGISTRY


def run_episode(agent, task_id: int, seed: int = 42) -> dict:
    env = CarbonEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        steps += 1

    trajectory = env.get_trajectory()
    final_state = env.state()
    score = run_grader(task_id, trajectory, final_state)

    return {
        "task_id":       task_id,
        "score":         score,
        "total_reward":  round(total_reward, 4),
        "steps":         steps,
        "jobs_completed": final_state["jobs_completed"],
        "carbon_used":    round(final_state["carbon_used"], 3),
    }


def main():
    agents = {
        "random":    RandomAgent(seed=0),
        "heuristic": HeuristicAgent(),
    }

    os.makedirs("data", exist_ok=True)
    log_path = "data/training_log.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "agent", "task_id", "task_name", "difficulty",
            "score", "total_reward", "steps", "jobs_completed", "carbon_used"
        ])
        writer.writeheader()

        for agent_name, agent in agents.items():
            print(f"\n{'='*50}")
            print(f"Agent: {agent_name.upper()}")
            print(f"{'='*50}")
            for task_id in [1, 2, 3]:
                info = TASK_REGISTRY[task_id]
                result = run_episode(agent, task_id, seed=42)
                row = {
                    "agent":          agent_name,
                    "task_name":      info["name"],
                    "difficulty":     info["difficulty"],
                    **result
                }
                writer.writerow(row)
                print(
                    f"  Task {task_id} [{info['difficulty']:6s}] "
                    f"{info['name']:<35s} score={result['score']:.4f}"
                )

    print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    main()