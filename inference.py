"""
inference.py — Optimized Agent for CarbonEnv
=============================================
Task 1: LLM with green-window focused prompt
Task 2: Heuristic (LLM was missing jobs due to over-cautious budget logic)
Task 3: Heuristic (LLM cannot handle multi-objective trading)

Required env vars:
  API_BASE_URL, MODEL_NAME, HF_TOKEN
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

# ─── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ─── Task 1 LLM Prompt — Green window focus ───────────────────────────────────
SYSTEM_PROMPT_TASK1 = """You are scheduling compute jobs for a green data center.

YOUR ONLY GOAL: Schedule all 12 jobs. Prefer HIGH renewable energy periods.

DECISION RULES (follow strictly in order):
1. renewable_ratio >= 0.7 → allocate_jobs, amount=2
2. renewable_ratio >= 0.5 → allocate_jobs, amount=1
3. renewable_ratio < 0.5 AND timestep < 18 → idle (wait for greener energy)
4. timestep >= 18 AND jobs_remaining > 0 → allocate_jobs, amount=1 (urgent, no more waiting)

Reply ONLY with valid JSON, nothing else:
{"action_type": "...", "amount": ...}"""


def obs_to_prompt_task1(obs: dict) -> str:
    steps_left = obs['max_timesteps'] - obs['timestep']
    return f"""State:
- jobs_remaining: {obs['jobs_remaining']}
- renewable_ratio: {obs['renewable_ratio']:.2f}  ← above 0.7 = GREEN, schedule now!
- timestep: {obs['timestep']} / {obs['max_timesteps']} (steps left: {steps_left})

JSON only:"""


# ─── Heuristic for Task 2 — Budget-Constrained ────────────────────────────────
def heuristic_task2(obs: dict) -> Action:
    """
    Pure rule-based agent for Task 2.
    Goal: Complete ALL 30 jobs without exceeding 150 kg CO2.
    Key fix: never idle when jobs remain and budget allows.
    """
    carbon_used    = obs['carbon_used']
    carbon_budget  = obs['carbon_budget']      # 150 kg
    carbon_left    = carbon_budget - carbon_used
    renewable      = obs['renewable_ratio']
    jobs_remaining = obs['jobs_remaining']
    timestep       = obs['timestep']
    max_timesteps  = obs['max_timesteps']
    steps_left     = max_timesteps - timestep

    # ── RULE 1: Time emergency — must finish no matter what ───────────────────
    # If steps left is close to jobs remaining, allocate immediately
    if jobs_remaining > 0 and steps_left <= jobs_remaining + 3:
        if carbon_left > 3:
            return Action(action_type="allocate_jobs", amount=1)

    # ── RULE 2: Budget is totally gone ────────────────────────────────────────
    if carbon_left <= 3:
        return Action(action_type="idle", amount=1)

    # ── RULE 3: Budget is very tight — only schedule in green windows ─────────
    if carbon_left <= 15:
        if renewable >= 0.65 and jobs_remaining > 0:
            # Green energy = less carbon per job, safe to schedule
            return Action(action_type="allocate_jobs", amount=1)
        elif jobs_remaining > 0 and steps_left <= jobs_remaining + 5:
            # Running out of time — must schedule even if not green
            return Action(action_type="allocate_jobs", amount=1)
        else:
            return Action(action_type="delay_jobs", amount=1)

    # ── RULE 4: Budget is comfortable — schedule based on renewable ───────────
    if jobs_remaining > 0:
        if renewable >= 0.7:
            # Great green window — schedule 2 jobs
            return Action(action_type="allocate_jobs", amount=2)
        elif renewable >= 0.5:
            # Acceptable — schedule 1 job
            return Action(action_type="allocate_jobs", amount=1)
        elif renewable < 0.45 and steps_left > jobs_remaining + 6:
            # Dirty energy and plenty of time — wait briefly
            return Action(action_type="delay_jobs", amount=1)
        else:
            # Default — just schedule, don't waste time
            return Action(action_type="allocate_jobs", amount=1)

    # ── RULE 5: All jobs done ─────────────────────────────────────────────────
    return Action(action_type="idle", amount=1)


# ─── Heuristic for Task 3 — Profit-Aware Trading ─────────────────────────────
def heuristic_task3(obs: dict) -> Action:
    """
    Pure rule-based agent for Task 3.
    Goal: Complete 50 jobs + stay under 120 kg CO2 + make trading profit.
    Key fix: never idle endlessly when jobs remain.
    """
    carbon_used    = obs['carbon_used']
    carbon_budget  = obs['carbon_budget']      # 120 kg
    carbon_left    = carbon_budget - carbon_used
    renewable      = obs['renewable_ratio']
    credit_price   = obs['carbon_credit_price']
    credits_held   = obs['credits_held']
    jobs_remaining = obs['jobs_remaining']
    timestep       = obs['timestep']
    max_timesteps  = obs['max_timesteps']
    steps_left     = max_timesteps - timestep

    # ── RULE 1: Time emergency — MUST finish jobs, no exceptions ──────────────
    # This rule fires first always — job completion is critical
    if jobs_remaining > 0 and steps_left <= jobs_remaining + 4:
        # Even if over budget, complete jobs — partial completion scores higher
        return Action(action_type="allocate_jobs", amount=1)

    # ── RULE 2: Excellent sell opportunity ────────────────────────────────────
    if credit_price >= 30 and credits_held >= 2:
        return Action(action_type="sell_carbon_credits",
                     amount=min(int(credits_held), 3))

    # ── RULE 3: Good sell opportunity ─────────────────────────────────────────
    if credit_price >= 27 and credits_held >= 1:
        return Action(action_type="sell_carbon_credits",
                     amount=min(int(credits_held), 2))

    # ── RULE 4: Excellent buy opportunity ─────────────────────────────────────
    if credit_price <= 20 and credits_held < 7 and carbon_left > 15:
        return Action(action_type="buy_carbon_credits", amount=2)

    # ── RULE 5: Good buy opportunity ──────────────────────────────────────────
    if credit_price <= 23 and credits_held < 5 and carbon_left > 20:
        return Action(action_type="buy_carbon_credits", amount=1)

    # ── RULE 6: Budget critically low ─────────────────────────────────────────
    if carbon_left <= 5:
        if jobs_remaining > 0 and steps_left <= jobs_remaining + 6:
            # Time is also running out — must allocate despite budget
            return Action(action_type="allocate_jobs", amount=1)
        if credits_held >= 1:
            return Action(action_type="sell_carbon_credits",
                         amount=min(int(credits_held), 2))
        if jobs_remaining == 0:
            return Action(action_type="idle", amount=1)
        # Jobs remaining but budget gone — still allocate to get partial credit
        return Action(action_type="allocate_jobs", amount=1)

    # ── RULE 7: Schedule jobs based on renewable energy ───────────────────────
    if jobs_remaining > 0:
        if renewable >= 0.7 and carbon_left > 8:
            return Action(action_type="allocate_jobs", amount=1)
        if renewable >= 0.55 and carbon_left > 10:
            return Action(action_type="allocate_jobs", amount=1)
        if renewable < 0.45 and steps_left > jobs_remaining + 8 and carbon_left > 15:
            # Dirty grid, plenty of time, good budget — wait briefly
            return Action(action_type="delay_jobs", amount=1)
        # Default — allocate rather than idle
        if carbon_left > 5:
            return Action(action_type="allocate_jobs", amount=1)

    # ── RULE 8: All jobs done — trade remaining credits ───────────────────────
    if jobs_remaining == 0:
        if credit_price >= 26 and credits_held >= 1:
            return Action(action_type="sell_carbon_credits",
                         amount=min(int(credits_held), 2))
        if credit_price <= 21 and credits_held < 6 and carbon_left > 10:
            return Action(action_type="buy_carbon_credits", amount=1)
        return Action(action_type="idle", amount=1)

    # ── FALLBACK: never idle with jobs remaining ───────────────────────────────
    if jobs_remaining > 0:
        return Action(action_type="allocate_jobs", amount=1)

    return Action(action_type="idle", amount=1)


# ─── LLM query for Task 1 only ────────────────────────────────────────────────
def query_llm_task1(obs_dict: dict, retries: int = 3) -> Action:
    """LLM for Task 1 — green window scheduling."""
    user_msg = obs_to_prompt_task1(obs_dict)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_TASK1},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=48,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
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

        except Exception:
            if attempt < retries - 1:
                time.sleep(0.5)
                continue

            # Smart fallback for Task 1
            if obs_dict['renewable_ratio'] >= 0.7:
                return Action(action_type="allocate_jobs", amount=2)
            elif obs_dict['renewable_ratio'] >= 0.5:
                return Action(action_type="allocate_jobs", amount=1)
            elif obs_dict['timestep'] >= 18 and obs_dict['jobs_remaining'] > 0:
                return Action(action_type="allocate_jobs", amount=1)
            else:
                return Action(action_type="idle", amount=1)

    return Action(action_type="allocate_jobs", amount=1.0)


# ─── Main action selector ─────────────────────────────────────────────────────
def get_action(obs_dict: dict, task_id: int) -> Action:
    """
    Route to correct agent based on task:
    Task 1 → LLM (green window logic)
    Task 2 → Heuristic (budget-constrained, LLM kept missing jobs)
    Task 3 → Heuristic (multi-objective, LLM cannot handle trading)
    """
    if task_id == 1:
        return query_llm_task1(obs_dict)
    elif task_id == 2:
        return heuristic_task2(obs_dict)
    else:
        return heuristic_task3(obs_dict)


# ─── Run one full episode ─────────────────────────────────────────────────────
def run_task(task_id: int, seed: int = 42) -> dict:
    task_info = TASK_REGISTRY[task_id]
    env = CarbonEnv(task_id=task_id, seed=seed)
    obs = env.reset()

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
        action = get_action(obs_dict, task_id)
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward.value
        step_num += 1

        step_log = {
            "event":             "STEP",
            "task_id":           task_id,
            "step":              step_num,
            "action_type":       action.action_type,
            "amount":            action.amount,
            "reward":            round(reward.value, 4),
            "cumulative_reward": round(cumulative_reward, 4),
            "jobs_remaining":    obs.jobs_remaining,
            "carbon_used":       round(obs.carbon_used, 3),
            "renewable_ratio":   obs.renewable_ratio,
            "done":              done,
        }
        print(f"[STEP] {json.dumps(step_log)}", flush=True)

    trajectory  = env.get_trajectory()
    final_state = env.state()
    score       = run_grader(task_id, trajectory, final_state)

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


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    results = []

    for task_id in [1, 2, 3]:
        result = run_task(task_id, seed=42)
        results.append(result)
        print("", flush=True)

    print("=" * 60, flush=True)
    print("BASELINE RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        print(
            f"  Task {r['task_id']} [{r['difficulty']:6s}] "
            f"{r['task_name']:<35s} "
            f"score={r['score']:.4f}  steps={r['total_steps']}",
            flush=True,
        )
    avg = sum(r["score"] for r in results) / len(results)
    print(f"\n  Average score: {avg:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
