from typing import Dict, List

from bayesian_mab.models import BayesianFinishIteration


class RewardTracker:
    """Track rewards for each arm"""

    def __init__(self) -> None:
        self.number_times_used = 0
        self.current_reward = 0
        self.brute_rewards: List[Dict[str, float]] = []
        self.average_rewards: List[Dict[str, float]] = []

    def update_reward_history(
        self, index: int, reward_value: int, average_reward: float
    ):
        """
        Update reward history

        Args:
            index: index of iteration
            reward_value: reward value
            average_reward: average reward
        """
        self.brute_rewards.append([{"index": index, "reward_value": reward_value}])
        self.average_rewards.append([{"index": index, "reward_value": average_reward}])
        return


class FinisherTracker:
    """Track finisher results"""

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
        """
        Update finisher history

        Args:
            arm_comp: arm name of challenger
            arm_base: arm name of base
            prob: probability of challenger being better than base
            lift: lift of challenger over base
            iteration: iteration number
        """

        finisher_result = BayesianFinishIteration(
            arm_comp=arm_comp.arm_name,
            arm_base=arm_base.arm_name,
            prob=prob,
            lift=lift,
            iteration=iteration,
        )
        self.finisher_results.extend([finisher_result])
        return
