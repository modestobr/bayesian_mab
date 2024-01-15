from typing import List, Dict
from baylatron.models import BayesianFinishIteration


class RewardTracker:
    def __init__(self) -> None:
        self.number_times_used = 0
        self.current_reward = 0
        self.brute_rewards: List[Dict[str, float]] = []
        self.average_rewards: List[Dict[str, float]] = []

    def update_reward_history(
        self, index: int, reward_value: int, average_reward: float
    ):
        self.brute_rewards.append([{"index": index, "reward_value": reward_value}])
        self.average_rewards.append([{"index": index, "reward_value": average_reward}])
        return


class FinisherTracker:
    def __init__(self) -> None:
        self.finisher_results: List[BayesianFinishIteration] = []

    def update_finisher_history(
        self,
        arm_comp,
        arm_base,
        prob: float,
        lift: float,
        iteration: int,
    ):
        finisher_result = BayesianFinishIteration(
            arm_comp=arm_comp.arm_name,
            arm_base=arm_base.arm_name,
            prob=prob,
            lift=lift,
            iteration=iteration,
        )
        self.finisher_results.extend([finisher_result])
        return
