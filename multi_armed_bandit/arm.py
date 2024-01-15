from abc import abstractmethod

import names
import numpy as np

from multi_armed_bandit.tracker import RewardTracker


class BaseArm:
    """
    Base class for all arms
    """

    def __init__(self, index: int = None, arm_name: str = None):
        self.arm_name = arm_name if arm_name != None else names.get_first_name()
        self.index = index

        self.reward_tracker = RewardTracker()

        return

    @abstractmethod
    def __repr__(self):
        return "Generic Arm %s %s" % (self.index, self.arm_name)

    @abstractmethod
    def sample_value(self):
        """Sample a value from the arm"""
        return

    @abstractmethod
    def record_success(self):
        """Record a success for the arm"""
        return

    @abstractmethod
    def record_failure(self):
        """Record a failure for the arm"""
        return

    @abstractmethod
    def read_brute_reward(self):
        """Read the brute reward"""
        pass

    @abstractmethod
    def average(self):
        """Return the average reward"""
        pass

    @abstractmethod
    def update_reward(self):
        """Update the arm's reward"""
        pass


class BayesianArm(BaseArm):
    """
    Each arm's true click through rate is
    modeled by a beta distribution.
    """

    def __init__(self, index: int, arm_name: str = None, a: int = 1, b: int = 1):
        """
        Init with uniform prior.
        """
        super().__init__(index, arm_name)

        self.a = a
        self.b = b

    def __repr__(self):
        return "Bayesian Arm {} (alpha={}, beta={}, average={})".format(
            self.arm_name, self.a, self.b, round(self.average(), 3)
        )

    def record_success(self):
        """Record a success for the arm, updating alpha parameter from beta distribution"""
        self.a += 1

    def record_failure(self):
        """Record a failure for the arm, updating beta parameter from beta distribution"""
        self.b += 1

    def sample_value(self):
        """Sample a random value from the beta distribution"""
        return np.random.beta(self.a, self.b, 1)[0]

    def average(self):
        """Calculate the average reward"""
        return self.a / (self.a + self.b)

    def update_reward(self):
        """Update the arm's reward"""
        self.current_reward = self.average()
        return
