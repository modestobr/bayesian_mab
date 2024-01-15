from typing import Dict, List

from pydantic import BaseModel


class BayesianFinishIteration(BaseModel):
    """Data model for finisher results"""

    arm_comp: str
    arm_base: str
    prob: float
    lift: float
    iteration: int
