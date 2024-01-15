from abc import abstractclassmethod
from typing import List, Tuple
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp

from baylatron.arm import BayesianArm
from scipy.stats import beta
import numpy as np
from baylatron.contrib.calc_bayesian_prob import calc_prob_between
from baylatron.contrib.logger import logger
from baylatron.tracker import FinisherTracker


class BaseFinisher:
    """
    End the execution of the tests, given the criteria
    of implemented stop
    """

    @abstractclassmethod
    def __init__(self):
        pass

    @abstractclassmethod
    def check_for_end(self):
        pass

    @abstractclassmethod
    def run(self):
        pass


class BayesianFinisher(BaseFinisher):
    def __init__(self):
        self.history_tracker = FinisherTracker()

    @staticmethod
    def check_for_end(a: int, b: int, c: int, d: int) -> Tuple[float, float]:
        """
        Beta(a,b)
        Beta(c,d)
        """
        # here we create the Beta functions for the two sets
        beta_C = beta(c, d)
        beta_T = beta(a, b)

        # calculating the lift
        lift = (beta_T.mean() - beta_C.mean()) / beta_C.mean()

        # calculating the probability for Test to be better than Control
        prob = calc_prob_between(beta_T, beta_C)

        logger.info(
            f"Test option lift Conversion Rates by {lift*100:2.2f}% with {prob*100:2.1f}% probability."
        )
        return prob, lift

    def check_for_winner(self, winner_prob_threshold: float):
        df = pd.DataFrame([c.dict() for c in self.history_tracker.finisher_results])
        df_check = df.groupby(["arm_comp"])["prob"].min().reset_index()

        flg_end = not df_check[df_check["prob"] >= winner_prob_threshold].empty

        return flg_end

    def get_current_winner(self):
        df = pd.DataFrame([c.dict() for c in self.history_tracker.finisher_results])
        df_check = df.groupby(["arm_comp"])["prob"].mean().reset_index()

        return df_check.sort_values(by="prob", ascending=False)["arm_comp"].iloc[0]

    def run(
        self,
        arms: List[BayesianArm],
        current_iteration: int,
        winner_prob_threshold: 0.80,
    ) -> bool:
        arms_combinations = [
            (arm_comp, arm_base)
            for arm_comp in arms
            for arm_base in arms
            if arm_comp.arm_name != arm_base.arm_name
        ]
        for arm_comp, arm_base in arms_combinations:
            prob, lift = self.check_for_end(
                arm_comp.a, arm_comp.b, arm_base.a, arm_base.b
            )

            self.history_tracker.update_finisher_history(
                arm_comp, arm_base, prob, lift, current_iteration
            )

        flg_winner = self.check_for_winner(winner_prob_threshold)
        current_winner = self.get_current_winner()

        print(flg_winner, current_winner)
        return flg_winner, current_winner
