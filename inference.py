"""
inference.py — Baseline LLM Agent for CarbonEnv
================================================
Uses OpenAI client with API_BASE_URL + MODEL_NAME from environment variables.
Emits structured stdout logs in [START] / [STEP] / [END] format.
Runs all 3 tasks and produces reproducible baseline scores.

Required env vars:
  API_BASE_URL   — LLM API endpoint
  MODEL_NAME     — model identifier
  HF_TOKEN       — Hugging Face / API key (used as OpenAI API key)
"""

import os
import sys
import json
import time
from openai import OpenAI
from env.carbon_env import CarbonEnv
from env.models import Action
from tasks import TASK_REGISTRY
from graders import run_grader

# ─── Load config from environment ─────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable not set.", file=sys.stderr)
    sys.exit(1)

# ─── OpenAI client pointed at HF Inference ────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

SYSTEM_PROMPT = """You are an AI agent controlling a carbon-aware data center job scheduler.
Your goal is to schedule compute jobs while minimizing carbon emissions and managing carbon credits.

At each step you will receive the current environment state and must choose ONE action.

Available actions:
  - allocate_jobs   : Schedule jobs now (amount = number of jobs, 1–5)
  - delay_jobs      : Wait / defer scheduling (amount = number to defer)
  - buy_carbon_credits  : Purchase carbon credits (amount = credits to buy, 1–5)
  - sell_carbon_credits : Sell held carbon credits (amount = credits to sell, 1–5)
  - idle            : Do nothing this timestep

Strategy hints:
  - Schedule jobs when renewable_ratio is HIGH (> 0.7) to reduce emissions
  - Buy carbon credits when price is LOW, sell when price is HIGH
  - Do NOT exceed the carbon_budget or you will be heavily penalized
  - Complete all jobs before the episode ends

Respond ONLY with a valid JSON object, no explanation:
{"action_type": "<action>", "amount": <number>}
"""


def obs_to_prompt(obs_dict: dict) -> str:
    """Convert observation dict to human-readable prompt for LLM."""
    return f"""Current environment state:
- Jobs remaining: {obs_dict['jobs_remaining']}
- Jobs completed: {obs_dict['jobs_completed']}
- Carbon budget remaining: {obs_dict['carbon_budget'] - obs_dict['carbon_used']:.1f} kg CO2
- Carbon used: {obs_dict['carbon_used']:.1f} kg CO2
- Renewable ratio: {obs_dict['renewable_ratio']:.2f} (higher = greener)
- Energy price: ${obs_dict['energy_price']:.1f}/MWh
- Carbon credit price: ${obs_dict['carbon_credit_price']:.2f}/credit
- Credits held: {obs_dict['credits_held']:.1f}
- Server load: {obs_dict['server_load']:.2f}
- Timestep: {obs_dict['timestep']} / {obs_dict['max_timesteps']}

Choose your action:"""


def query_llm(obs_dict: dict, retries: int = 3) -> Action:
    """Query the LLM and parse its action. Falls back to allocate_jobs on failure."""
    user_msg = obs_to_prompt(obs_dict)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=64,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)
            return Action(
                action_type=parsed.get("action_type", "allocate_jobs"),
                amount=float(parsed.get("amount", 1.0)),
            )

        except Exception as e:
            if attempt == retries - 1:
                # Final fallback
                return Action(action_type="allocate_jobs", amount=1.0)
            time.sleep(1.0)

    return Action(action_type="allocate_jobs", amount=1.0)


def run_task(task_id: int, seed: int = 42) -> dict:
    """
    Run one full episode with the LLM agent on task_id.
    Emits [START], [STEP], [END] logs to stdout.
    Returns result dict with score.
    """
    task_info = TASK_REGISTRY[task_id]
    env = CarbonEnv(task_id=task_id, seed=seed)
    obs = env.reset()

    # ── [START] log ────────────────────────────────────────────────────────────
    start_log = {
        "event":      "START",
        "task_id":    task_id,
        "task_name":  task_info["name"],
        "difficulty": task_info["difficulty"],
        "model":      MODEL_NAME,
        "seed":       seed,
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(f"[START] {json.dumps(start_log)}", flush=True)

    step_num = 0
    cumulative_reward = 0.0
    done = False

    while not done:
        obs_dict = obs.model_dump()
        action = query_llm(obs_dict)
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward.value
        step_num += 1

        # ── [STEP] log ─────────────────────────────────────────────────────────
        step_log = {
            "event":              "STEP",
            "task_id":            task_id,
            "step":               step_num,
            "action_type":        action.action_type,
            "amount":             action.amount,
            "reward":             round(reward.value, 4),
            "cumulative_reward":  round(cumulative_reward, 4),
            "jobs_remaining":     obs.jobs_remaining,
            "carbon_used":        round(obs.carbon_used, 3),
            "renewable_ratio":    obs.renewable_ratio,
            "done":               done,
        }
        print(f"[STEP] {json.dumps(step_log)}", flush=True)

    # ── Grade the episode ──────────────────────────────────────────────────────
    trajectory  = env.get_trajectory()
    final_state = env.state()
    score       = run_grader(task_id, trajectory, final_state)

    # ── [END] log ──────────────────────────────────────────────────────────────
    end_log = {
        "event":             "END",
        "task_id":           task_id,
        "task_name":         task_info["name"],
        "difficulty":        task_info["difficulty"],
        "score":             score,
        "cumulative_reward": round(cumulative_reward, 4),
        "total_steps":       step_num,
        "jobs_completed":    final_state["jobs_completed"],
        "carbon_used":       round(final_state["carbon_used"], 3),
        "model":             MODEL_NAME,
        "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(f"[END] {json.dumps(end_log)}", flush=True)

    return end_log


def main():
    """Run all 3 tasks and print a final summary."""
    results = []

    for task_id in [1, 2, 3]:
        result = run_task(task_id, seed=42)
        results.append(result)
        print("", flush=True)  # blank line between tasks

    # Final summary
    print("=" * 60, flush=True)
    print("BASELINE RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        print(
            f"  Task {r['task_id']} [{r['difficulty']:6s}] {r['task_name']:<35s} "
            f"score={r['score']:.4f}  steps={r['total_steps']}",
            flush=True,
        )
    avg = sum(r["score"] for r in results) / len(results)
    print(f"\n  Average score: {avg:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()