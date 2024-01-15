from abc import abstractmethod
from typing import Union, List
from baylatron.reward import BinaryReward, DiscreteReward
from baylatron.arm import BayesianArm, BaseArm
import numpy as np
import pandas as pd
from baylatron.finisher import BayesianFinisher
from baylatron.contrib.logger import logger


class BaseMAB:
    def __init__(self, arms: BaseArm):
        self.arms = arms

    @property
    def number_times_used(self) -> int:
        return sum([len(arm.reward_tracker.brute_rewards) for arm in self.arms])

    @abstractmethod
    def sample_an_arm(self):
        pass

    @abstractmethod
    def update_arm(self):
        pass

    def _infer_index(self, iteration_index=None):
        """Check for all arms rewards,
        if missing, return index 0, if not, return
        bigger index observed"""

        if iteration_index != None:
            return iteration_index

        return sum([len(arm.reward_tracker.brute_rewards) for arm in self.arms])


class BayesianMAB(BaseMAB):
    def __init__(self, arms: List[BayesianArm]):
        super().__init__(arms=arms)

        self.finisher = BayesianFinisher()
        return

    def __repr__(self):
        rpr = "Bayesian Multi Armed Bandit\n   " + "\n   ".join(
            [str(c) for c in self.arms]
        )
        return rpr

    def sample_an_arm(self):
        """
        Choose an arm to sample from available arms given a sampled probability
        """
        arm_index = np.argmax([arm.sample_value() for arm in self.arms])

        return arm_index

    def update_arm(
        self,
        chosen_arm: Union[BayesianArm, int],
        reward_agent: Union[BinaryReward, DiscreteReward],
    ):
        """
        Update chosen arm, following some logic
        """
        reward_value = reward_agent.reward_value

        arm = self.arms[chosen_arm]

        if reward_agent.check_success():
            arm.record_success()
        else:
            arm.record_failure()

        arm.update_reward()

        arm.reward_tracker.update_reward_history(
            index=self.number_times_used,
            reward_value=reward_value,
            average_reward=arm.average(),
        )

        return

    def check_for_end(self, **kwargs):
        flg_end, current_winner = self.finisher.run(
            arms=self.arms, current_iteration=self.number_times_used, **kwargs
        )
        logger.info(f"flg_end: {flg_end}, Winner: {current_winner}")
        return flg_end


if __name__ == "__main__":
    binary_reward = BinaryReward()
    binary_reward.update_reward(1)

    bayesian_mab = BayesianMAB(
        arms=[
            BayesianArm(index=0),
            BayesianArm(index=1, arm_name="Arm 2"),
            BayesianArm(index=2, arm_name="Arm 3"),
        ]
    )
    logger.info(bayesian_mab)

    logger.info(bayesian_mab.sample_an_arm())

    for i in range(4):
        binary_reward.update_reward(np.random.binomial(1, p=0.9))
        bayesian_mab.update_arm(chosen_arm=1, reward_agent=binary_reward)

    for i in range(1500):
        binary_reward.update_reward(np.random.binomial(1, p=0.3))
        bayesian_mab.update_arm(chosen_arm=1, reward_agent=binary_reward)

    for i in range(1500):
        binary_reward.update_reward(np.random.binomial(1, p=0.9))
        bayesian_mab.update_arm(chosen_arm=2, reward_agent=binary_reward)

    logger.info(bayesian_mab)

    logger.info(bayesian_mab.check_for_end(winner_prob_threshold=0.80))
