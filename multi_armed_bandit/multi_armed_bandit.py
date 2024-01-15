from abc import abstractmethod
from typing import List, Union, Tuple

import numpy as np

from multi_armed_bandit.arm import BaseArm, BayesianArm
from multi_armed_bandit.contrib.logger import logger
from multi_armed_bandit.finisher import BayesianFinisher
from multi_armed_bandit.reward import BinaryReward, DiscreteReward


class BaseMAB:
    """
    Base class for Multi Armed Bandits
    """

    def __init__(self, arms: BaseArm):
        self.arms = arms
        self.current_winner = None
        self.minimum_interation = 500

    @property
    def number_times_used(self) -> int:
        """Number of times all arms have been used"""
        return sum([len(arm.reward_tracker.brute_rewards) for arm in self.arms])

    @abstractmethod
    def sample_an_arm(self):
        """Sample an arm from the available arms"""
        pass

    @abstractmethod
    def update_arm(self):
        """Update the arm with the reward"""
        pass


class BayesianMAB(BaseMAB):
    """
    Bayesian Multi Armed Bandit
    """

    def __init__(self, arms: List[BayesianArm]):
        super().__init__(arms=arms)

        self.finisher = BayesianFinisher()
        return

    def __repr__(self):
        rpr = "Bayesian Multi Armed Bandit\n   " + "\n   ".join(
            [str(c) for c in self.arms]
        )
        return rpr

    def sample_an_arm(self) -> int:
        """
        Choose an arm to sample from available arms given a sampled probability

        Returns:
            arm_index: index of arm to sample
        """
        arm_index = np.argmax([arm.sample_value() for arm in self.arms])

        return arm_index

    def update_arm(
        self,
        chosen_arm: Union[BayesianArm, int],
        reward_agent: Union[BinaryReward, DiscreteReward],
    ):
        """
        Update the arm with the reward

        Args:
            chosen_arm: arm to update
            reward_agent: reward agent
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

    def check_for_end(self, **kwargs) -> Tuple[bool, str]:
        """ "
        Run the finisher, checking for a winner between a set of arms given
        a probability threshold.

        Args:
            kwargs: kwargs for finisher

        Returns:
            flg_winner: flag for winner
            current_winner: current winner
        """
        if self.number_times_used < self.minimum_interation:
            logger.info(
                f"Number of times used {self.number_times_used} is less than minimum {self.minimum_interation}"
            )
            return False, None
        
        flg_end, self.current_winner = self.finisher.run(
            arms=self.arms, current_iteration=self.number_times_used, **kwargs
        )
        logger.info(f"flg_end: {flg_end}, Winner: {self.current_winner}")
        return flg_end, self.current_winner


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
        bayesian_mab.update_arm(chosen_arm=0, reward_agent=binary_reward)

    for i in range(10):
        binary_reward.update_reward(np.random.binomial(1, p=0.3))
        bayesian_mab.update_arm(chosen_arm=1, reward_agent=binary_reward)

    for i in range(500):
        binary_reward.update_reward(np.random.binomial(1, p=0.9))
        bayesian_mab.update_arm(chosen_arm=2, reward_agent=binary_reward)

    logger.info(bayesian_mab)

    logger.info(bayesian_mab.check_for_end(winner_prob_threshold=0.80))
