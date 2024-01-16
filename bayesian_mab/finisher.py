from abc import abstractclassmethod
from typing import List, Tuple

import pandas as pd
from scipy.stats import beta, ttest_1samp

from bayesian_mab.arm import BayesianArm
from bayesian_mab.contrib.calc_bayesian_prob import calc_prob_between
from bayesian_mab.contrib.logger import logger
from bayesian_mab.tracker import FinisherTracker


class BaseFinisher:
    """
    Base class for Finishers
    """

    @abstractclassmethod
    def __init__(self):
        pass

    @abstractclassmethod
    def check_for_end(self):
        """Test used for determining finish"""
        pass

    @abstractclassmethod
    def run(self):
        """Main orchestration method for finish determination"""
        pass


class BayesianFinisher(BaseFinisher):
    """
    Aims to determine if there is a winner between a set of arms
    This uses Exact Calculation of Beta Inequalities from https://www.johndcook.com/UTMDABTR-005-05.pdf.
    """

    def __init__(self):
        self.history_tracker = FinisherTracker()

    @staticmethod
    def check_for_end(a: int, b: int, c: int, d: int) -> Tuple[float, float]:
        """
        Calculate the probability and lift of two bayesian arms using
        Exact Calculation of Beta Inequalities from https://www.johndcook.com/UTMDABTR-005-05.pdf.

        The comparison arm is given by Beta(a,b)
        The reference arm is given by Beta(c,d)

        Args:
            a: alpha of comparison arm
            b: beta of comparison arm
            c: alpha of reference arm
            d: beta of reference arm

        Returns:
            prob: probability of comparison arm being better than reference arm
            lift: lift of comparison arm over reference arm
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
        """
        Check if there is a winner based on the probability threshold
        and the bayesian arms probabilities
        """
        df = pd.DataFrame([c.dict() for c in self.history_tracker.finisher_results])
        df_check = df.groupby(["arm_comp"])["prob"].min().reset_index()

        flg_end = not df_check[df_check["prob"] >= winner_prob_threshold].empty

        return flg_end

    def get_current_winner(self) -> str:
        """
        Determine the current winner based on the results of exact calculation of beta inequalities

        Returns:
            current_winner: current winner
        """
        df = pd.DataFrame([c.dict() for c in self.history_tracker.finisher_results])
        df_check = df.groupby(["arm_comp"])["prob"].mean().reset_index()

        return df_check.sort_values(by="prob", ascending=False)["arm_comp"].iloc[0]

    def run(
        self,
        arms: List[BayesianArm],
        current_iteration: int,
        winner_prob_threshold: 0.80,
    ) -> Tuple[bool, str]:
        """ "
        Run the finisher, checking for a winner between a set of arms given
        a probability threshold.

        Args:
            arms: list of arms
            current_iteration: current iteration
            winner_prob_threshold: probability threshold for winner

        Returns:
            flg_winner: flag for winner
            current_winner: current winner
        """
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

        return flg_winner, current_winner
