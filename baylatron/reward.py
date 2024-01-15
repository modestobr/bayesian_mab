from abc import abstractmethod
from typing import List
from baylatron.models import Reward


class BaseReward:
    """
    Base class for all rewards
    """

    @abstractmethod
    def __init__(self):
        self.reward_index = None
        self.reward_value = 0
        return

    def update_reward(self, reward_value):
        self.reward_value = reward_value
        return

    @abstractmethod
    def check_success(self):
        pass


class BinaryReward(BaseReward):
    """
    This Binary Reward is used in cases where the observations are results of the format:
        - clicked / not clicked
        - viewed / not viewed
    """

    def __init__(self, possible_values: List[int] = [0, 1]):
        super().__init__()

        self.possible_values = possible_values

        return

    def __repr__(self):
        return "Binary Reward %s" % (self.reward_value)

    def check_success(self):
        """
        If reward = 1, then success,
        Otherwise, failure
        """
        flg_success_reward = self.reward_value == max(self.possible_values)

        return flg_success_reward


class DiscreteReward(BaseReward):
    """
    Discrete Reward is used in cases where the observations are continuous and
    we can discretize them into binary results from a cutoff value, usually a central value
    such as the mean or median.

    Examples:
        - time viewing the page
        - number of ads clicked
        - number of shares
    """

    def __init__(self, average_reward=0.5):
        super().__init__()

        self.average_reward = average_reward

        return

    def __repr__(self):
        return "Discrete Reward (average reward) = %s" % (self.average_reward)

    def check_success(self):
        """
        If reward > average, then success,
        Otherwise, failure
        """
        flg_success_reward = self.reward_value > self.average_reward

        return flg_success_reward


class ContinuosReward(BaseReward):
    """
    The continuous reward is used in cases where the observations are continuous variables
    Examples:
        - time viewing the page
        - number of ads clicked
        - number of shares
    """

    def __init__(self):
        super().__init__()
        return
