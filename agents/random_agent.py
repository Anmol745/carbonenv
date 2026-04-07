"""Random agent baseline — selects actions uniformly at random."""
import random
from env.models import Action, Observation

ACTIONS = ["allocate_jobs", "delay_jobs", "buy_carbon_credits", "sell_carbon_credits", "idle"]


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs: Observation) -> Action:
        action_type = self.rng.choice(ACTIONS)
        amount = round(self.rng.uniform(1.0, 3.0), 1)
        return Action(action_type=action_type, amount=amount)