"""
Heuristic agent — strong rule-based policy.
Used as a meaningful comparison baseline:
  - Schedules jobs when renewable_ratio > 0.65
  - Buys credits when price < 22, sells when price > 28
  - Delays when grid is dirty AND budget is tight
"""
from env.models import Action, Observation


class HeuristicAgent:
    """Deterministic rule-based agent. Sets a strong baseline floor."""

    BUY_THRESHOLD  = 22.0   # buy credits if price below this
    SELL_THRESHOLD = 28.0   # sell credits if price above this
    GREEN_THRESHOLD = 0.65  # schedule jobs if renewable ratio above this

    def act(self, obs: Observation) -> Action:
        budget_remaining = obs.carbon_budget - obs.carbon_used
        budget_ratio = budget_remaining / max(obs.carbon_budget, 1.0)
        time_remaining = obs.max_timesteps - obs.timestep
        urgency = obs.jobs_remaining / max(time_remaining, 1)

        # Sell credits if price is high and we hold some
        if obs.carbon_credit_price > self.SELL_THRESHOLD and obs.credits_held >= 1.0:
            return Action(action_type="sell_carbon_credits", amount=min(obs.credits_held, 3.0))

        # Buy credits if price is low and budget allows
        if obs.carbon_credit_price < self.BUY_THRESHOLD and obs.credits_held < 5.0:
            return Action(action_type="buy_carbon_credits", amount=2.0)

        # Schedule jobs in green windows
        if obs.renewable_ratio >= self.GREEN_THRESHOLD and obs.jobs_remaining > 0:
            amount = min(3, obs.jobs_remaining)
            return Action(action_type="allocate_jobs", amount=float(amount))

        # Force scheduling if urgent (running out of time)
        if urgency > 1.5 and obs.jobs_remaining > 0:
            return Action(action_type="allocate_jobs", amount=1.0)

        # Delay if grid is dirty and we have time
        if obs.jobs_remaining > 0 and budget_ratio > 0.3:
            return Action(action_type="delay_jobs", amount=1.0)

        # Fallback: schedule one job
        if obs.jobs_remaining > 0:
            return Action(action_type="allocate_jobs", amount=1.0)

        return Action(action_type="idle", amount=0.0)