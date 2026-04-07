"""
Typed Pydantic models for OpenEnv spec compliance.
Observation, Action, Reward — all strictly typed.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class Observation(BaseModel):
    """Full environment observation returned on every step/reset."""
    jobs_remaining: int = Field(..., description="Number of jobs yet to be scheduled")
    jobs_completed: int = Field(..., description="Jobs successfully completed this episode")
    carbon_budget: float = Field(..., description="Remaining carbon budget in kg CO2")
    carbon_used: float = Field(..., description="Total carbon emitted so far in kg CO2")
    energy_price: float = Field(..., description="Current energy price in $/MWh")
    renewable_ratio: float = Field(..., ge=0.0, le=1.0, description="Fraction of grid power from renewables (0–1)")
    carbon_credit_price: float = Field(..., description="Current carbon credit spot price in $/credit")
    server_load: float = Field(..., ge=0.0, le=1.0, description="Current server utilization (0–1)")
    credits_held: float = Field(..., description="Carbon credits currently held by agent")
    timestep: int = Field(..., description="Current timestep in episode")
    max_timesteps: int = Field(..., description="Total timesteps in this episode")
    task_id: int = Field(..., description="Which task is being evaluated (1, 2, or 3)")


class Action(BaseModel):
    """Discrete + continuous action space for the scheduling agent."""
    action_type: Literal[
        "allocate_jobs",
        "delay_jobs",
        "buy_carbon_credits",
        "sell_carbon_credits",
        "idle"
    ] = Field(..., description="The type of action to perform")
    amount: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Magnitude: jobs to schedule, or credits to buy/sell"
    )


class Reward(BaseModel):
    """Structured reward with full breakdown for transparency."""
    value: float = Field(..., description="Scalar reward for this step")
    job_reward: float = Field(default=0.0, description="Reward from completing jobs")
    emission_penalty: float = Field(default=0.0, description="Penalty for carbon emissions")
    delay_penalty: float = Field(default=0.0, description="Penalty for delaying jobs")
    budget_penalty: float = Field(default=0.0, description="Penalty for exceeding carbon budget")
    trading_reward: float = Field(default=0.0, description="Reward from profitable credit trading")
    green_bonus: float = Field(default=0.0, description="Bonus for scheduling during high renewable periods")


class StepResult(BaseModel):
    """Full result returned by step()."""
    observation: Observation
    reward: Reward
    done: bool
    info: dict