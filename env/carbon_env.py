"""
CarbonEnv — Carbon-Aware Job Scheduling Environment
Full OpenEnv spec compliance: reset(), step(), state()
Simulates a real data center carbon credit + job scheduling problem.
"""

import random
import math
from typing import Optional
from env.models import Observation, Action, Reward, StepResult


# ─── Carbon intensity profile (simulates real grid data) ──────────────────────
def _generate_carbon_profile(steps: int, seed: int) -> list[float]:
    """
    Generates a realistic renewable ratio curve over episode steps.
    Peaks midday (solar), dips at night — mimics real grid behavior.
    """
    rng = random.Random(seed)
    profile = []
    for t in range(steps):
        base = 0.4 + 0.35 * math.sin(math.pi * t / steps)   # solar arc
        noise = rng.gauss(0, 0.05)
        profile.append(max(0.05, min(0.95, base + noise)))
    return profile


def _generate_price_profile(steps: int, seed: int) -> list[float]:
    """Energy price in $/MWh — inversely correlated with renewable ratio."""
    rng = random.Random(seed + 1)
    profile = []
    for t in range(steps):
        base = 80 - 30 * math.sin(math.pi * t / steps)
        noise = rng.gauss(0, 5)
        profile.append(max(20.0, base + noise))
    return profile


def _generate_credit_price_profile(steps: int, seed: int) -> list[float]:
    """Carbon credit price — random walk with mean reversion."""
    rng = random.Random(seed + 2)
    price = 25.0
    profile = []
    for _ in range(steps):
        price += rng.gauss(0, 1.5) + 0.05 * (25 - price)
        profile.append(max(5.0, round(price, 2)))
    return profile


# ─── Main Environment ──────────────────────────────────────────────────────────
class CarbonEnv:
    """
    Carbon-Aware Data Center Job Scheduling Environment.

    Three tasks with escalating difficulty:
      Task 1 (easy)   — Green Window Scheduling
      Task 2 (medium) — Budget-Constrained Execution
      Task 3 (hard)   — Profit-Aware Carbon Trading
    """

    TASK_CONFIGS = {
        1: {"max_timesteps": 24, "total_jobs": 12, "carbon_budget": 200.0, "initial_credits": 0.0},
        2: {"max_timesteps": 48, "total_jobs": 30, "carbon_budget": 150.0, "initial_credits": 5.0},
        3: {"max_timesteps": 72, "total_jobs": 50, "carbon_budget": 120.0, "initial_credits": 10.0},
    }

    CARBON_PER_JOB = 8.0        # kg CO2 per job scheduled (base)
    ENERGY_PER_JOB = 0.5        # MWh per job

    def __init__(self, task_id: int = 1, seed: int = 42):
        assert task_id in (1, 2, 3), "task_id must be 1, 2, or 3"
        self.task_id = task_id
        self.seed = seed
        self._config = self.TASK_CONFIGS[task_id]
        self._obs: Optional[Observation] = None
        self._done: bool = True
        self._trajectory: list[dict] = []

        # Pre-generate deterministic profiles
        self._renewable_profile = _generate_carbon_profile(
            self._config["max_timesteps"], seed
        )
        self._price_profile = _generate_price_profile(
            self._config["max_timesteps"], seed
        )
        self._credit_price_profile = _generate_credit_price_profile(
            self._config["max_timesteps"], seed
        )
        self._buy_price_history: list[float] = []

    # ── OpenEnv Interface ──────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment to initial state. Returns first observation."""
        cfg = self._config
        self._jobs_remaining = cfg["total_jobs"]
        self._jobs_completed = 0
        self._carbon_used = 0.0
        self._carbon_budget = cfg["carbon_budget"]
        self._credits_held = cfg["initial_credits"]
        self._timestep = 0
        self._done = False
        self._trajectory = []
        self._buy_price_history = []
        self._total_jobs = cfg["total_jobs"]

        # Re-generate profiles with same seed (deterministic)
        self._renewable_profile = _generate_carbon_profile(
            cfg["max_timesteps"], self.seed
        )
        self._price_profile = _generate_price_profile(
            cfg["max_timesteps"], self.seed
        )
        self._credit_price_profile = _generate_credit_price_profile(
            cfg["max_timesteps"], self.seed
        )

        self._obs = self._build_obs()
        return self._obs

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """
        Execute one action. Returns (observation, reward, done, info).
        Reward is a Reward model — use reward.value for scalar.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        reward = self._compute_reward(action)
        self._apply_action(action)
        self._timestep += 1

        # Episode ends when time runs out OR all jobs done
        if self._timestep >= self._config["max_timesteps"] or self._jobs_remaining <= 0:
            self._done = True

        self._obs = self._build_obs()
        info = {
            "task_id": self.task_id,
            "timestep": self._timestep,
            "jobs_completed": self._jobs_completed,
            "carbon_used": round(self._carbon_used, 3),
            "credits_held": round(self._credits_held, 2),
            "budget_remaining": round(self._carbon_budget - self._carbon_used, 3),
        }

        self._trajectory.append({
            "timestep": self._timestep,
            "action": action.action_type,
            "amount": action.amount,
            "obs": self._obs.model_dump(),
            "reward": reward.value,
            "renewable_ratio": self._obs.renewable_ratio,
        })

        return self._obs, reward, self._done, info

    def state(self) -> dict:
        """Returns full current environment state (OpenEnv spec)."""
        return {
            "observation": self._obs.model_dump() if self._obs else None,
            "done": self._done,
            "timestep": self._timestep,
            "task_id": self.task_id,
            "trajectory_length": len(self._trajectory),
            "jobs_completed": self._jobs_completed,
            "jobs_remaining": self._jobs_remaining,
            "carbon_used": self._carbon_used,
            "carbon_budget": self._carbon_budget,
            "credits_held": self._credits_held,
        }

    def get_trajectory(self) -> list[dict]:
        """Returns the full step trajectory for grading."""
        return self._trajectory

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_obs(self) -> Observation:
        t = min(self._timestep, self._config["max_timesteps"] - 1)
        return Observation(
            jobs_remaining=self._jobs_remaining,
            jobs_completed=self._jobs_completed,
            carbon_budget=round(self._carbon_budget, 3),
            carbon_used=round(self._carbon_used, 3),
            energy_price=round(self._price_profile[t], 2),
            renewable_ratio=round(self._renewable_profile[t], 3),
            carbon_credit_price=round(self._credit_price_profile[t], 2),
            server_load=round(
                (self._config["total_jobs"] - self._jobs_remaining)
                / max(self._config["total_jobs"], 1), 3
            ),
            credits_held=round(self._credits_held, 2),
            timestep=self._timestep,
            max_timesteps=self._config["max_timesteps"],
            task_id=self.task_id,
        )

    def _apply_action(self, action: Action):
        t = min(self._timestep, self._config["max_timesteps"] - 1)
        renewable = self._renewable_profile[t]
        credit_price = self._credit_price_profile[t]

        if action.action_type == "allocate_jobs":
            jobs = min(int(action.amount), self._jobs_remaining)
            carbon_factor = 1.0 - (0.5 * renewable)  # renewables reduce emissions
            emission = jobs * self.CARBON_PER_JOB * carbon_factor
            self._carbon_used += emission
            self._jobs_remaining -= jobs
            self._jobs_completed += jobs

        elif action.action_type == "delay_jobs":
            pass  # intentional no-op — penalty applied in reward

        elif action.action_type == "buy_carbon_credits":
            credits = min(action.amount, 20.0)
            self._credits_held += credits
            self._buy_price_history.append(credit_price)

        elif action.action_type == "sell_carbon_credits":
            credits = min(action.amount, self._credits_held)
            self._credits_held -= credits

        elif action.action_type == "idle":
            pass

    def _compute_reward(self, action: Action) -> Reward:
        t = min(self._timestep, self._config["max_timesteps"] - 1)
        renewable = self._renewable_profile[t]
        credit_price = self._credit_price_profile[t]

        job_reward = 0.0
        emission_penalty = 0.0
        delay_penalty = 0.0
        budget_penalty = 0.0
        trading_reward = 0.0
        green_bonus = 0.0

        if action.action_type == "allocate_jobs":
            jobs = min(int(action.amount), self._jobs_remaining)
            job_reward = jobs * 2.0

            carbon_factor = 1.0 - (0.5 * renewable)
            emission = jobs * self.CARBON_PER_JOB * carbon_factor
            emission_penalty = -emission * 0.05

            if renewable > 0.7:
                green_bonus = jobs * 1.5  # strong bonus for green scheduling

            projected_used = self._carbon_used + emission
            if projected_used > self._carbon_budget:
                budget_penalty = -10.0 * (projected_used - self._carbon_budget) / self._carbon_budget

        elif action.action_type == "delay_jobs":
            delay_penalty = -0.5 * min(int(action.amount), self._jobs_remaining)
            # Progressive urgency: penalty grows as deadline approaches
            urgency = self._timestep / self._config["max_timesteps"]
            delay_penalty *= (1 + urgency)

        elif action.action_type == "buy_carbon_credits":
            pass  # neutral; profit realized on sell

        elif action.action_type == "sell_carbon_credits":
            credits = min(action.amount, self._credits_held)
            if self._buy_price_history:
                avg_buy = sum(self._buy_price_history) / len(self._buy_price_history)
                profit = credits * (credit_price - avg_buy)
                trading_reward = profit * 0.1  # scaled
            else:
                trading_reward = credits * credit_price * 0.05

        elif action.action_type == "idle":
            delay_penalty = -0.3  # small penalty for idling when jobs remain
            if self._jobs_remaining == 0:
                delay_penalty = 0.0

        total = job_reward + emission_penalty + delay_penalty + budget_penalty + trading_reward + green_bonus

        return Reward(
            value=round(total, 4),
            job_reward=round(job_reward, 4),
            emission_penalty=round(emission_penalty, 4),
            delay_penalty=round(delay_penalty, 4),
            budget_penalty=round(budget_penalty, 4),
            trading_reward=round(trading_reward, 4),
            green_bonus=round(green_bonus, 4),
        )